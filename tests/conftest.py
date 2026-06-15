"""Shared pytest fixtures/helpers.

Tests come in two flavours:

* **CPU reference tests** (run everywhere, including GPU-less CI): check that the
  pure-PyTorch references in :mod:`ktune.utils.reference` match independent
  ground truth (``torch.nn.functional``) and that gradients are correct.
* **GPU kernel tests** (``@pytest.mark.gpu``): check that each Triton kernel
  matches its reference within tolerance. These are skipped automatically when
  no CUDA device is present.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    """Seed before every test so unseeded random tensors are reproducible."""
    torch.manual_seed(0)


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a CUDA GPU + Triton to run the kernels",
)


def assert_close(a, b, atol=1e-2, rtol=1e-2, msg=""):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=msg or None)
