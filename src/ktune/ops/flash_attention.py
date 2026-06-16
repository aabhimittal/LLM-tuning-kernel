"""FlashAttention-2 (forward **and** backward) — the IO-aware attention kernel.

Reference: :func:`ktune.utils.reference.attention` (the same math, materialised).

Standard attention forms the full ``[seq, seq]`` score matrix in HBM, softmaxes
it, then multiplies by ``V``. That matrix is quadratic in sequence length and is
the memory + bandwidth bottleneck of long-context training.

FlashAttention removes it with two ideas, both visible in the kernels below:

* **Tiling.** Each program owns a block of queries and streams *blocks* of keys
  and values through fast on-chip SRAM. The score matrix only ever exists one
  ``[BLOCK_M, BLOCK_N]`` tile at a time.
* **Online softmax.** As each key block arrives we update a running max ``m`` and
  running denominator ``l`` and rescale the partial output ``acc`` — so we get an
  exact softmax without ever seeing the whole row at once.

For causal attention we simply stop the key loop at the diagonal.

**Backward.** The forward also writes the per-row log-sum-exp ``L = m + log(l)``.
The backward (:func:`_flash_bwd_kernel`) reuses it to *recompute* the softmax
probabilities tile-by-tile — never materialising ``[seq, seq]`` — and forms the
gradients with the standard identities::

    D  = rowsum(dO ∘ O)
    P  = exp(S - L)                 # recomputed softmax, one tile at a time
    dV = Pᵀ @ dO
    dP = dO @ Vᵀ
    dS = P ∘ (dP - D)
    dQ = scale · dS @ K ,  dK = scale · dSᵀ @ Q

One program owns a query block and accumulates its ``dQ`` locally; the ``dK``/
``dV`` contributions to each key block are accumulated across query blocks with
``tl.atomic_add`` (a clean single-kernel design — the two-pass, atomic-free
formulation is the follow-up exercise in ``docs/07-flash-attention.md``).

Set :data:`USE_FUSED_BACKWARD` to ``False`` to fall back to the autograd-recompute
backward (handy for A/B-checking the fused kernel).
"""

from __future__ import annotations

import torch

from ktune.utils.reference import attention as _attention_ref
from ktune.utils.runtime import HAS_TRITON, use_triton

#: Use the fused Triton backward (True) or recompute via autograd on the
#: reference (False). Both are numerically correct; the fused path is memory-light.
USE_FUSED_BACKWARD = True

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _flash_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,  # all [B*H, seq, d], contiguous
        l_ptr,  # [B*H, seq] log-sum-exp, saved for the backward
        sm_scale,
        seq,
        CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,  # head_dim (power of 2)
    ):
        start_m = tl.program_id(0)  # which query block
        off_bh = tl.program_id(1)  # which (batch, head)
        base = off_bh * seq * BLOCK_D

        row = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # query positions
        d = tl.arange(0, BLOCK_D)
        row_mask = row < seq

        # Load this program's query block once; it stays in SRAM for the whole loop.
        q = tl.load(
            q_ptr + base + row[:, None] * BLOCK_D + d[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)  # running max
        l_i = tl.zeros([BLOCK_M], tl.float32)  # running denom
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)  # running output

        # For causal masking we only need keys up to this query block's last row.
        end_n = (start_m + 1) * BLOCK_M if CAUSAL else seq

        for start_n in range(0, end_n, BLOCK_N):
            col = start_n + tl.arange(0, BLOCK_N)  # key positions
            col_mask = col < seq
            k = tl.load(
                k_ptr + base + col[:, None] * BLOCK_D + d[None, :],
                mask=col_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_ptr + base + col[:, None] * BLOCK_D + d[None, :],
                mask=col_mask[:, None],
                other=0.0,
            ).to(tl.float32)

            qk = tl.dot(q, tl.trans(k)) * sm_scale  # [BLOCK_M, BLOCK_N] score tile
            qk = tl.where(col_mask[None, :], qk, -float("inf"))
            if CAUSAL:
                qk = tl.where(row[:, None] >= col[None, :], qk, -float("inf"))

            # --- online softmax update ---
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)  # rescale factor for what we accumulated so far
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, v)
            m_i = m_ij

        acc = acc / l_i[:, None]  # finalise the softmax denominator
        tl.store(
            o_ptr + base + row[:, None] * BLOCK_D + d[None, :],
            acc,
            mask=row_mask[:, None],
        )
        # Save log-sum-exp so the backward can recompute the probabilities exactly.
        tl.store(l_ptr + off_bh * seq + row, m_i + tl.log(l_i), mask=row_mask)

    @triton.jit
    def _flash_bwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        do_ptr,
        l_ptr,  # [B*H, seq] log-sum-exp from the forward
        dq_ptr,
        dk_ptr,  # fp32, zero-initialised (accumulated via atomics)
        dv_ptr,  # fp32, zero-initialised (accumulated via atomics)
        sm_scale,
        seq,
        CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        start_m = tl.program_id(0)  # this program owns one query block
        off_bh = tl.program_id(1)
        base = off_bh * seq * BLOCK_D

        row = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        d = tl.arange(0, BLOCK_D)
        row_mask = row < seq

        q = tl.load(
            q_ptr + base + row[:, None] * BLOCK_D + d[None, :], mask=row_mask[:, None], other=0.0
        ).to(tl.float32)
        do = tl.load(
            do_ptr + base + row[:, None] * BLOCK_D + d[None, :], mask=row_mask[:, None], other=0.0
        ).to(tl.float32)
        o = tl.load(
            o_ptr + base + row[:, None] * BLOCK_D + d[None, :], mask=row_mask[:, None], other=0.0
        ).to(tl.float32)
        L = tl.load(l_ptr + off_bh * seq + row, mask=row_mask, other=0.0)

        # D_i = rowsum(dO_i ∘ O_i) — the term that makes dS a centred softmax grad.
        Di = tl.sum(do * o, axis=1)  # [BLOCK_M]
        dq = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

        end_n = (start_m + 1) * BLOCK_M if CAUSAL else seq
        for start_n in range(0, end_n, BLOCK_N):
            col = start_n + tl.arange(0, BLOCK_N)
            col_mask = col < seq
            k = tl.load(
                k_ptr + base + col[:, None] * BLOCK_D + d[None, :],
                mask=col_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_ptr + base + col[:, None] * BLOCK_D + d[None, :],
                mask=col_mask[:, None],
                other=0.0,
            ).to(tl.float32)

            s = tl.dot(q, tl.trans(k)) * sm_scale
            s = tl.where(col_mask[None, :], s, -float("inf"))
            if CAUSAL:
                s = tl.where(row[:, None] >= col[None, :], s, -float("inf"))

            p = tl.exp(s - L[:, None])  # recomputed softmax tile [BLOCK_M, BLOCK_N]
            dp = tl.dot(do, tl.trans(v))  # [BLOCK_M, BLOCK_N]
            ds = p * (dp - Di[:, None])  # [BLOCK_M, BLOCK_N]

            dq += tl.dot(ds, k) * sm_scale  # accumulate this block's dQ locally
            # dK / dV for this key block, summed across query blocks via atomics.
            dk_c = tl.dot(tl.trans(ds), q) * sm_scale  # [BLOCK_N, BLOCK_D]
            dv_c = tl.dot(tl.trans(p), do)  # [BLOCK_N, BLOCK_D]
            tl.atomic_add(
                dk_ptr + base + col[:, None] * BLOCK_D + d[None, :], dk_c, mask=col_mask[:, None]
            )
            tl.atomic_add(
                dv_ptr + base + col[:, None] * BLOCK_D + d[None, :], dv_c, mask=col_mask[:, None]
            )

        tl.store(dq_ptr + base + row[:, None] * BLOCK_D + d[None, :], dq, mask=row_mask[:, None])

    def _flash_forward(q, k, v, causal, sm_scale):
        b, h, seq, d = q.shape
        q2 = q.reshape(b * h, seq, d).contiguous()
        k2 = k.reshape(b * h, seq, d).contiguous()
        v2 = v.reshape(b * h, seq, d).contiguous()
        o = torch.empty_like(q2)
        lse = torch.empty((b * h, seq), device=q.device, dtype=torch.float32)
        block_m = min(64, triton.next_power_of_2(seq))
        block_n = min(64, triton.next_power_of_2(seq))
        grid = (triton.cdiv(seq, block_m), b * h)
        _flash_fwd_kernel[grid](
            q2,
            k2,
            v2,
            o,
            lse,
            sm_scale,
            seq,
            CAUSAL=causal,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=d,
        )
        return o.reshape(b, h, seq, d), lse

    def _flash_backward(q, k, v, o, lse, do, causal, sm_scale):
        b, h, seq, d = q.shape
        q2 = q.reshape(b * h, seq, d).contiguous()
        k2 = k.reshape(b * h, seq, d).contiguous()
        v2 = v.reshape(b * h, seq, d).contiguous()
        o2 = o.reshape(b * h, seq, d).contiguous()
        do2 = do.reshape(b * h, seq, d).contiguous()
        # dK/dV are accumulated with atomics, so they must start at zero in fp32.
        dq = torch.zeros_like(q2, dtype=torch.float32)
        dk = torch.zeros_like(k2, dtype=torch.float32)
        dv = torch.zeros_like(v2, dtype=torch.float32)
        block_m = min(64, triton.next_power_of_2(seq))
        block_n = min(64, triton.next_power_of_2(seq))
        grid = (triton.cdiv(seq, block_m), b * h)
        _flash_bwd_kernel[grid](
            q2,
            k2,
            v2,
            o2,
            do2,
            lse,
            dq,
            dk,
            dv,
            sm_scale,
            seq,
            CAUSAL=causal,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=d,
        )
        return (
            dq.reshape(b, h, seq, d).to(q.dtype),
            dk.reshape(b, h, seq, d).to(k.dtype),
            dv.reshape(b, h, seq, d).to(v.dtype),
        )


def _recompute_backward(q, k, v, do, causal, sm_scale):
    """Fallback backward: autograd through the reference attention formula.

    Correct but re-materialises the score matrix; used when
    :data:`USE_FUSED_BACKWARD` is False, and as the baseline the fused kernel is
    checked against.
    """
    with torch.enable_grad():
        qd = q.detach().requires_grad_(True)
        kd = k.detach().requires_grad_(True)
        vd = v.detach().requires_grad_(True)
        out = _attention_ref(qd, kd, vd, causal=causal, sm_scale=sm_scale)
        return torch.autograd.grad(out, (qd, kd, vd), do)


class _FlashAttnTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        out, lse = _flash_forward(q, k, v, causal, sm_scale)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse = ctx.saved_tensors
        do = do.contiguous()
        if USE_FUSED_BACKWARD:
            dq, dk, dv = _flash_backward(q, k, v, out, lse, do, ctx.causal, ctx.sm_scale)
        else:
            dq, dk, dv = _recompute_backward(q, k, v, do, ctx.causal, ctx.sm_scale)
        return dq, dk, dv, None, None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """FlashAttention with fused forward + backward. Triton on CUDA, reference on CPU.

    ``q``/``k``/``v`` are ``[batch, heads, seq, head_dim]`` with ``head_dim`` a
    power of two. Returns the attention output of the same shape.
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    if use_triton(q, k, v):
        return _FlashAttnTriton.apply(q, k, v, causal, sm_scale)
    return _attention_ref(q, k, v, causal=causal, sm_scale=sm_scale)
