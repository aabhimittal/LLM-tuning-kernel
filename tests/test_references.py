"""CPU reference tests — run everywhere, no GPU required.

These pin the *math*: each reference must match independent ground truth from
``torch.nn.functional``, including the backward pass. The Triton kernels are then
checked against these same references in ``test_kernels_gpu.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ktune.ops import (
    apply_rope,
    cross_entropy,
    flash_attention,
    fused_linear_cross_entropy,
    rms_norm,
    swiglu,
)


def test_rmsnorm_matches_manual():
    # The reference upcasts to fp32 internally (like HF Llama RMSNorm), so test
    # in fp32 — the precision the real path actually uses.
    x = torch.randn(4, 32, 64)
    w = torch.randn(64)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    torch.testing.assert_close(rms_norm(x, w), expected, atol=1e-5, rtol=1e-5)


def test_swiglu_matches_manual_and_grad():
    g = torch.randn(8, 64, dtype=torch.float64, requires_grad=True)
    u = torch.randn(8, 64, dtype=torch.float64, requires_grad=True)
    out = swiglu(g, u)
    torch.testing.assert_close(out, F.silu(g) * u)
    out.sum().backward()
    assert g.grad is not None and u.grad is not None


def test_cross_entropy_matches_F():
    logits = torch.randn(16, 100)
    targets = torch.randint(0, 100, (16,))
    torch.testing.assert_close(cross_entropy(logits, targets), F.cross_entropy(logits, targets))


def test_cross_entropy_ignore_index():
    logits = torch.randn(16, 100)
    targets = torch.randint(0, 100, (16,))
    targets[:4] = -100
    got = cross_entropy(logits, targets, ignore_index=-100)
    exp = F.cross_entropy(logits, targets, ignore_index=-100)
    torch.testing.assert_close(got, exp)


@pytest.mark.parametrize("causal", [True, False])
def test_attention_matches_sdpa(causal):
    q = torch.randn(2, 4, 32, 16)
    k = torch.randn(2, 4, 32, 16)
    v = torch.randn(2, 4, 32, 16)
    got = flash_attention(q, k, v, causal=causal)
    exp = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(got, exp, atol=1e-4, rtol=1e-4)


def test_rope_matches_manual():
    seq, hd = 8, 16
    pos = torch.arange(seq, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
    emb = torch.cat([torch.outer(pos, inv)] * 2, dim=-1)
    cos, sin = emb.cos(), emb.sin()
    q = torch.randn(2, 3, seq, hd)
    k = torch.randn(2, 3, seq, hd)
    qr, kr = apply_rope(q, k, cos, sin)

    def rotate_half(x):
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

    torch.testing.assert_close(qr, q * cos + rotate_half(q) * sin)
    torch.testing.assert_close(kr, k * cos + rotate_half(k) * sin)


@pytest.mark.parametrize("chunk_size", [4, 16, 1000])
def test_flce_matches_unfused_forward_and_grad(chunk_size):
    h = torch.randn(40, 24, dtype=torch.float64, requires_grad=True)
    W = torch.randn(80, 24, dtype=torch.float64, requires_grad=True)
    t = torch.randint(0, 80, (40,))

    loss = fused_linear_cross_entropy(h, W, t, chunk_size=chunk_size)
    ref = F.cross_entropy(F.linear(h, W), t)
    torch.testing.assert_close(loss.double(), ref)

    gh, gW = torch.autograd.grad(loss, (h, W))
    gh_ref, gW_ref = torch.autograd.grad(ref, (h, W))
    torch.testing.assert_close(gh, gh_ref)
    torch.testing.assert_close(gW, gW_ref)


def test_flce_chunking_is_invariant_to_chunk_size():
    h = torch.randn(33, 16, dtype=torch.float64)
    W = torch.randn(50, 16, dtype=torch.float64)
    t = torch.randint(0, 50, (33,))
    a = fused_linear_cross_entropy(h, W, t, chunk_size=5)
    b = fused_linear_cross_entropy(h, W, t, chunk_size=33)
    torch.testing.assert_close(a, b)
