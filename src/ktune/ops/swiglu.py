"""SwiGLU — the fused gated MLP activation (Llama/Qwen/Mistral).

Reference: :func:`ktune.utils.reference.swiglu`. Computes ``silu(gate) * up``.

This is a pure element-wise op, so it is **memory-bound**: there is no reduction
and barely any arithmetic per element. The win from a custom kernel is purely
about IO — a naive ``F.silu(gate) * up`` writes the intermediate ``silu(gate)``
to HBM and reads it back, whereas the fused kernel reads ``gate`` and ``up``
once and writes the result once. The backward fuses the same way.
"""

from __future__ import annotations

import torch

from ktune.utils.reference import swiglu as _swiglu_ref
from ktune.utils.runtime import HAS_TRITON, use_triton

if HAS_TRITON:
    import triton
    import triton.language as tl

    # Pure 1-D elementwise: both the block size and the warp count are free to
    # tune, keyed on the total element count.
    _CONFIGS = [
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w)
        for bs in (1024, 2048, 4096)
        for w in (2, 4, 8)
    ]

    @triton.jit
    def _silu(x):
        # SiLU / swish: x * sigmoid(x).
        return x * tl.sigmoid(x)

    @triton.autotune(configs=_CONFIGS, key=["n_elements"])
    @triton.jit
    def _swiglu_fwd_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, _silu(gate) * up, mask=mask)

    @triton.autotune(configs=_CONFIGS, key=["n_elements"])
    @triton.jit
    def _swiglu_bwd_kernel(
        dout_ptr, gate_ptr, up_ptr, dgate_ptr, dup_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        dout = tl.load(dout_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        sig = tl.sigmoid(gate)
        silu = gate * sig
        # d(silu)/d(gate) = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
        dsilu = sig * (1.0 + gate * (1.0 - sig))
        tl.store(dgate_ptr + offs, dout * up * dsilu, mask=mask)
        tl.store(dup_ptr + offs, dout * silu, mask=mask)


class _SwiGLUTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.contiguous()
        out = torch.empty_like(gate)
        n = gate.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)  # noqa: E731
        _swiglu_fwd_kernel[grid](gate, up, out, n)  # BLOCK_SIZE chosen by autotune
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, dout):
        gate, up = ctx.saved_tensors
        dout = dout.contiguous()
        dgate = torch.empty_like(gate)
        dup = torch.empty_like(up)
        n = gate.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)  # noqa: E731
        _swiglu_bwd_kernel[grid](dout, gate, up, dgate, dup, n)  # BLOCK_SIZE via autotune
        return dgate, dup


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused ``silu(gate) * up``. Triton on CUDA, reference on CPU."""
    if use_triton(gate, up):
        return _SwiGLUTriton.apply(gate, up)
    return _swiglu_ref(gate, up)
