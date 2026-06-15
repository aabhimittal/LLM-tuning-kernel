"""ktune — Triton kernels for fast, memory-efficient LLM fine-tuning, built to learn.

Quick start::

    import torch
    from ktune.ops import rms_norm, swiglu, flash_attention, fused_linear_cross_entropy

    x = torch.randn(4, 512, 2048, device="cuda")        # runs the Triton kernel
    w = torch.ones(2048, device="cuda")
    y = rms_norm(x, w)                                    # ...or the CPU reference on CPU

Every op has a readable PyTorch reference (``ktune.utils.reference``) used as the
correctness oracle, and a hand-written Triton kernel (``ktune.ops``) used on GPU.
Drop-in ``nn.Module``s live in ``ktune.nn`` and a HuggingFace patcher in
``ktune.integrations``.
"""

from ktune.ops import (
    apply_rope,
    cross_entropy,
    flash_attention,
    fused_linear_cross_entropy,
    rms_norm,
    swiglu,
)
from ktune.utils.runtime import HAS_TRITON, triton_available

__version__ = "0.1.0"

__all__ = [
    "rms_norm",
    "apply_rope",
    "swiglu",
    "cross_entropy",
    "fused_linear_cross_entropy",
    "flash_attention",
    "HAS_TRITON",
    "triton_available",
    "__version__",
]
