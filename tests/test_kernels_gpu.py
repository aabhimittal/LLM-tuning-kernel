"""GPU kernel tests — each Triton kernel vs its PyTorch reference.

These are gated behind ``@pytest.mark.gpu`` + a CUDA check, so they are collected
but **skipped** on CPU-only machines (including default CI). Run them on a GPU:

    pytest tests/test_kernels_gpu.py -m gpu -v

Tolerances are loose-ish because kernels run in fp16/bf16 like real training.
"""

from __future__ import annotations

import pytest
import torch

from ktune.ops import (
    apply_rope,
    cross_entropy,
    flash_attention,
    fused_linear_cross_entropy,
    rms_norm,
    swiglu,
)
from ktune.utils import reference as ref

pytestmark = pytest.mark.gpu

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA + Triton")


@CUDA
def test_rmsnorm_kernel():
    x = torch.randn(8, 2048, device="cuda", dtype=torch.float16, requires_grad=True)
    w = torch.randn(2048, device="cuda", dtype=torch.float16, requires_grad=True)
    y = rms_norm(x, w)
    y_ref = ref.rms_norm(x.detach().float(), w.detach().float())
    torch.testing.assert_close(y.float(), y_ref, atol=1e-2, rtol=1e-2)
    y.sum().backward()
    assert x.grad is not None and w.grad is not None


@CUDA
def test_swiglu_kernel():
    g = torch.randn(16, 4096, device="cuda", dtype=torch.float16, requires_grad=True)
    u = torch.randn(16, 4096, device="cuda", dtype=torch.float16, requires_grad=True)
    out = swiglu(g, u)
    torch.testing.assert_close(out.float(), ref.swiglu(g.float(), u.float()), atol=1e-2, rtol=1e-2)
    out.sum().backward()
    assert g.grad is not None and u.grad is not None


@CUDA
@pytest.mark.parametrize("causal", [True, False])
def test_flash_attention_kernel(causal):
    q = torch.randn(2, 8, 256, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 8, 256, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 8, 256, 64, device="cuda", dtype=torch.float16)
    out = flash_attention(q, k, v, causal=causal)
    out_ref = ref.attention(q.float(), k.float(), v.float(), causal=causal)
    torch.testing.assert_close(out.float(), out_ref, atol=2e-2, rtol=2e-2)


@CUDA
def test_cross_entropy_kernel():
    logits = torch.randn(512, 32000, device="cuda", dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, 32000, (512,), device="cuda")
    loss = cross_entropy(logits, targets)
    loss_ref = ref.cross_entropy(logits.detach(), targets)
    torch.testing.assert_close(loss, loss_ref, atol=1e-3, rtol=1e-3)
    loss.backward()
    assert logits.grad is not None


@CUDA
def test_rope_kernel():
    seq, hd = 128, 64
    pos = torch.arange(seq, device="cuda", dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device="cuda").float() / hd))
    emb = torch.cat([torch.outer(pos, inv)] * 2, dim=-1)
    cos, sin = emb.cos(), emb.sin()
    q = torch.randn(2, 8, seq, hd, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 8, seq, hd, device="cuda", dtype=torch.float16)
    qr, kr = apply_rope(q, k, cos, sin)
    qr_ref, kr_ref = ref.apply_rope(q.float(), k.float(), cos, sin)
    torch.testing.assert_close(qr.float(), qr_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(kr.float(), kr_ref, atol=1e-2, rtol=1e-2)


@CUDA
def test_flce_kernel_memory_and_values():
    hidden = torch.randn(2048, 1024, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(32000, 1024, device="cuda", dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, 32000, (2048,), device="cuda")

    loss = fused_linear_cross_entropy(hidden, weight, targets, chunk_size=512)
    loss_ref = ref.fused_linear_cross_entropy(hidden.detach(), weight.detach(), targets)
    torch.testing.assert_close(loss, loss_ref, atol=1e-2, rtol=1e-2)
    loss.backward()
    assert hidden.grad is not None and weight.grad is not None
