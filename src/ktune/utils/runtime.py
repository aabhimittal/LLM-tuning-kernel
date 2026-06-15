"""Runtime detection: is Triton importable, and should we use it for a tensor?

Every public op in ``ktune.ops`` dispatches through :func:`use_triton`:

* On a CUDA tensor with Triton installed -> run the hand-written Triton kernel.
* Otherwise (CPU tensor, or no Triton) -> fall back to the pure-PyTorch
  reference in :mod:`ktune.utils.reference`.

This is what lets the *exact same code path* be correct on a laptop CPU and fast
on a GPU, and it is why the whole library is importable in a GPU-less CI.
"""

from __future__ import annotations

import torch

try:  # Triton only ships meaningful kernels on CUDA platforms.
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised on CPU-only machines
    HAS_TRITON = False


def triton_available() -> bool:
    """True if Triton is importable *and* a CUDA device is present."""
    return HAS_TRITON and torch.cuda.is_available()


def use_triton(*tensors: torch.Tensor) -> bool:
    """Decide whether to dispatch to a Triton kernel for these tensors.

    Requires Triton to be importable, a CUDA runtime, and every tensor to live
    on a CUDA device. Any CPU tensor forces the readable reference path.
    """
    if not triton_available():
        return False
    return all(t.is_cuda for t in tensors if isinstance(t, torch.Tensor))
