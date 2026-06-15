"""The kernels. Each op dispatches to Triton on CUDA and a PyTorch reference on CPU.

Read these alongside :mod:`ktune.utils.reference` and the matching page in ``docs/``.
"""

from ktune.ops.cross_entropy import cross_entropy
from ktune.ops.flash_attention import flash_attention
from ktune.ops.fused_linear_ce import fused_linear_cross_entropy
from ktune.ops.rmsnorm import rms_norm
from ktune.ops.rope import apply_rope
from ktune.ops.swiglu import swiglu

__all__ = [
    "rms_norm",
    "apply_rope",
    "swiglu",
    "cross_entropy",
    "fused_linear_cross_entropy",
    "flash_attention",
]
