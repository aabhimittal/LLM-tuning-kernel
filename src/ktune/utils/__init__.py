"""Shared helpers: reference math oracles and runtime/Triton detection."""

from ktune.utils.runtime import HAS_TRITON, triton_available, use_triton

__all__ = ["HAS_TRITON", "triton_available", "use_triton"]
