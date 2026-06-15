"""RoPE — Rotary Position Embedding (Su et al., 2021).

Reference: :func:`ktune.utils.reference.apply_rope`.

RoPE encodes position by *rotating* pairs of channels by an angle that grows
with position. Concretely, for the "rotate-half" layout used by Llama/Qwen::

    y = x * cos + rotate_half(x) * sin        rotate_half([x1, x2]) = [-x2, x1]

Like SwiGLU this is element-wise and memory-bound; the kernel reads each row of
q and k once and writes it once, folding the four multiply-adds together. The
backward is the transpose of the same rotation::

    dx = dy * cos + rotate_half_back(dy * sin) rotate_half_back([a1, a2]) = [a2, -a1]

We map **one program to one (batch, head, position) row** and split the row into
its two halves on chip.
"""

from __future__ import annotations

import torch

from ktune.utils.reference import apply_rope as _apply_rope_ref
from ktune.utils.runtime import HAS_TRITON, use_triton

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _rope_kernel(
        x_ptr,  # [n_rows, head_dim]
        cos_ptr,  # [seq, head_dim]
        sin_ptr,  # [seq, head_dim]
        out_ptr,  # [n_rows, head_dim]
        seq_len,
        head_dim,
        half,
        BACKWARD: tl.constexpr,  # transpose the rotation for the backward pass
        BLOCK: tl.constexpr,  # >= half
    ):
        row = tl.program_id(0)
        seq_idx = row % seq_len  # position within the sequence selects cos/sin row
        x_ptr += row * head_dim
        out_ptr += row * head_dim
        cos_ptr += seq_idx * head_dim
        sin_ptr += seq_idx * head_dim

        cols = tl.arange(0, BLOCK)
        mask = cols < half

        x1 = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + half + cols, mask=mask, other=0.0).to(tl.float32)
        cos1 = tl.load(cos_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        cos2 = tl.load(cos_ptr + half + cols, mask=mask, other=0.0).to(tl.float32)
        sin1 = tl.load(sin_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sin2 = tl.load(sin_ptr + half + cols, mask=mask, other=0.0).to(tl.float32)

        if BACKWARD:
            # adjoint of the forward rotation
            y1 = x1 * cos1 + x2 * sin2
            y2 = -x1 * sin1 + x2 * cos2
        else:
            y1 = x1 * cos1 - x2 * sin1
            y2 = x2 * cos2 + x1 * sin2

        tl.store(out_ptr + cols, y1, mask=mask)
        tl.store(out_ptr + half + cols, y2, mask=mask)

    def _rope_apply(x, cos, sin, backward):
        orig_shape = x.shape
        head_dim = orig_shape[-1]
        seq_len = cos.shape[0]
        x2d = x.contiguous().view(-1, head_dim)
        out = torch.empty_like(x2d)
        half = head_dim // 2
        block = triton.next_power_of_2(half)
        _rope_kernel[(x2d.shape[0],)](
            x2d,
            cos.contiguous(),
            sin.contiguous(),
            out,
            seq_len,
            head_dim,
            half,
            BACKWARD=backward,
            BLOCK=block,
        )
        return out.view(orig_shape)


class _RoPETriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        ctx.save_for_backward(cos, sin)
        q_rot = _rope_apply(q, cos, sin, backward=False)
        k_rot = _rope_apply(k, cos, sin, backward=False)
        return q_rot, k_rot

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        dq_in = _rope_apply(dq, cos, sin, backward=True)
        dk_in = _rope_apply(dk, cos, sin, backward=True)
        return dq_in, dk_in, None, None


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to ``q`` and ``k``. Triton on CUDA, reference on CPU.

    ``q``/``k`` are ``[batch, heads, seq, head_dim]``; ``cos``/``sin`` are
    ``[seq, head_dim]``.
    """
    if use_triton(q, k, cos, sin):
        return _RoPETriton.apply(q, k, cos, sin)
    return _apply_rope_ref(q, k, cos, sin)
