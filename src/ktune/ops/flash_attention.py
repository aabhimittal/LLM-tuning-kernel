"""FlashAttention-2 (forward) — the IO-aware attention kernel.

Reference: :func:`ktune.utils.reference.attention` (the same math, materialised).

Standard attention forms the full ``[seq, seq]`` score matrix in HBM, softmaxes
it, then multiplies by ``V``. That matrix is quadratic in sequence length and is
the memory + bandwidth bottleneck of long-context training.

FlashAttention removes it with two ideas, both visible in the kernel below:

* **Tiling.** Each program owns a block of queries and streams *blocks* of keys
  and values through fast on-chip SRAM. The score matrix only ever exists one
  ``[BLOCK_M, BLOCK_N]`` tile at a time.
* **Online softmax.** As each key block arrives we update a running max ``m`` and
  running denominator ``l`` and rescale the partial output ``acc`` — so we get an
  exact softmax without ever seeing the whole row at once.

For causal attention we simply stop the key loop at the diagonal.

This module ships the **forward kernel** (the thing worth learning). The backward
recomputes gradients via autograd on the reference formula — correct and a clean
starting point; a fully-fused Triton backward is the natural next exercise (see
``docs/07-flash-attention.md``).
"""

from __future__ import annotations

import torch

from ktune.utils.reference import attention as _attention_ref
from ktune.utils.runtime import HAS_TRITON, use_triton

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _flash_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,  # all [B*H, seq, d], contiguous
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

    def _flash_forward(q, k, v, causal, sm_scale):
        b, h, seq, d = q.shape
        q2 = q.reshape(b * h, seq, d).contiguous()
        k2 = k.reshape(b * h, seq, d).contiguous()
        v2 = v.reshape(b * h, seq, d).contiguous()
        o = torch.empty_like(q2)
        block_m = min(64, triton.next_power_of_2(seq))
        block_n = min(64, triton.next_power_of_2(seq))
        grid = (triton.cdiv(seq, block_m), b * h)
        _flash_fwd_kernel[grid](
            q2,
            k2,
            v2,
            o,
            sm_scale,
            seq,
            CAUSAL=causal,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=d,
        )
        return o.reshape(b, h, seq, d)


class _FlashAttnTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        out = _flash_forward(q, k, v, causal, sm_scale)
        ctx.save_for_backward(q, k, v)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx, do):
        # Backward via autograd on the reference formula (correct; not yet fused).
        q, k, v = ctx.saved_tensors
        with torch.enable_grad():
            qd = q.detach().requires_grad_(True)
            kd = k.detach().requires_grad_(True)
            vd = v.detach().requires_grad_(True)
            out = _attention_ref(qd, kd, vd, causal=ctx.causal, sm_scale=ctx.sm_scale)
            dq, dk, dv = torch.autograd.grad(out, (qd, kd, vd), do)
        return dq, dk, dv, None, None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """FlashAttention forward. Triton kernel on CUDA, reference on CPU.

    ``q``/``k``/``v`` are ``[batch, heads, seq, head_dim]`` with ``head_dim`` a
    power of two. Returns the attention output of the same shape.
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    if use_triton(q, k, v):
        return _FlashAttnTriton.apply(q, k, v, causal, sm_scale)
    return _attention_ref(q, k, v, causal=causal, sm_scale=sm_scale)
