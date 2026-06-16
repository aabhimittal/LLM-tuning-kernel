"""RMSNorm — the "hello world" of LLM Triton kernels.

Read :func:`ktune.utils.reference.rms_norm` first; this file is that same math,
tiled so that **one Triton program handles one row** of the input.

Why this is the canonical first kernel:

* It is a **reduction along the last dimension** (the mean of squares), which is
  the pattern underneath softmax, layernorm, cross-entropy and more. Learn it
  here once.
* The whole row fits in fast on-chip SRAM, so the kernel reads the row from HBM
  once, does all the math on-chip, and writes the row back once. That single
  load/store is why a fused kernel beats a chain of PyTorch ops that each
  round-trips through HBM.

The public entry point is :func:`rms_norm`, which dispatches to the Triton
kernel on CUDA tensors and to the reference on CPU.
"""

from __future__ import annotations

import torch

from ktune.utils.reference import rms_norm as _rms_norm_ref
from ktune.utils.runtime import HAS_TRITON, use_triton

if HAS_TRITON:
    import triton
    import triton.language as tl

    # These row-wise kernels must size BLOCK to cover the whole row (one program
    # per row), so BLOCK is fixed per call — but the best *number of warps* still
    # depends on the row width, so we autotune over that, keyed on n_cols.
    _WARP_CONFIGS = [triton.Config({}, num_warps=w) for w in (1, 2, 4, 8)]

    @triton.autotune(configs=_WARP_CONFIGS, key=["n_cols"])
    @triton.jit
    def _rmsnorm_fwd_kernel(
        x_ptr,  # [n_rows, n_cols] input
        w_ptr,  # [n_cols] gain
        y_ptr,  # [n_rows, n_cols] output
        rstd_ptr,  # [n_rows] saved 1/rms for the backward pass
        x_row_stride,
        y_row_stride,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program == one row. `program_id(0)` indexes the row.
        row = tl.program_id(0)
        x_ptr += row * x_row_stride
        y_ptr += row * y_row_stride

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        # Load the whole row into SRAM (masked tail reads as 0).
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # Reduction: mean of squares -> reciprocal RMS. Done entirely on-chip.
        mean_sq = tl.sum(x * x, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(mean_sq + eps)
        tl.store(rstd_ptr + row, rstd)  # cache for backward

        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(y_ptr + cols, y, mask=mask)

    @triton.autotune(configs=_WARP_CONFIGS, key=["n_cols"])
    @triton.jit
    def _rmsnorm_bwd_kernel(
        dy_ptr,  # incoming gradient [n_rows, n_cols]
        x_ptr,  # input  [n_rows, n_cols]
        w_ptr,  # gain   [n_cols]
        rstd_ptr,  # saved 1/rms [n_rows]
        dx_ptr,  # grad wrt x [n_rows, n_cols]
        dw_partial_ptr,  # per-row partial grad wrt w [n_rows, n_cols], summed later
        x_row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        dy_ptr += row * x_row_stride
        x_ptr += row * x_row_stride
        dx_ptr += row * x_row_stride
        dw_partial_ptr += row * x_row_stride

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        dy = tl.load(dy_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row)

        x_hat = x * rstd  # normalised input
        # dL/dx for y = x_hat * w. The middle term is the correction from x_hat's
        # own dependence on the row's mean-square (the reduction couples columns).
        c1 = tl.sum(dy * w * x_hat, axis=0) / n_cols
        dx = (dy * w - x_hat * c1) * rstd
        tl.store(dx_ptr + cols, dx, mask=mask)

        # dL/dw for this row; rows are summed afterwards with a cheap torch.sum.
        dw = dy * x_hat
        tl.store(dw_partial_ptr + cols, dw, mask=mask)


class _RMSNormTriton(torch.autograd.Function):
    """Autograd wrapper that calls the Triton kernels above."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        orig_shape = x.shape
        x = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x.shape
        y = torch.empty_like(x)
        rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(n_cols)
        _rmsnorm_fwd_kernel[(n_rows,)](
            x, weight, y, rstd, x.stride(0), y.stride(0), n_cols, eps, BLOCK_SIZE=block
        )
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        dy = dy.contiguous().view(-1, x.shape[-1])
        n_rows, n_cols = x.shape
        dx = torch.empty_like(x)
        dw_partial = torch.empty_like(x)
        block = triton.next_power_of_2(n_cols)
        _rmsnorm_bwd_kernel[(n_rows,)](
            dy, x, weight, rstd, dx, dw_partial, x.stride(0), n_cols, BLOCK_SIZE=block
        )
        dw = dw_partial.sum(dim=0).to(weight.dtype)
        return dx.view(ctx.orig_shape), dw, None


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm. Triton on CUDA, pure-PyTorch reference on CPU.

    Args:
        x: input, normalised over the last dimension.
        weight: per-channel gain, shape ``[x.shape[-1]]``.
        eps: numerical-stability epsilon added to the mean-square.
    """
    if use_triton(x, weight):
        return _RMSNormTriton.apply(x, weight, eps)
    return _rms_norm_ref(x, weight, eps)
