"""CPU tests for the nn modules and the HuggingFace-style patcher.

No GPU and no `transformers` needed — we build a tiny Llama-shaped module tree by
hand and check the patcher rewires it to the ktune ops.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ktune.integrations import apply_ktune_to_model, summarize_patchable
from ktune.nn import KTuneFusedLinearCrossEntropy, KTuneRMSNorm, KTuneSwiGLUMLP


def test_ktune_rmsnorm_module_matches_reference():
    m = KTuneRMSNorm(64)
    x = torch.randn(3, 8, 64)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + m.variance_epsilon) * m.weight
    torch.testing.assert_close(m(x), expected, atol=1e-5, rtol=1e-5)


def test_ktune_swiglu_mlp_shapes_and_grad():
    m = KTuneSwiGLUMLP(32, 64)
    x = torch.randn(4, 32, requires_grad=True)
    out = m(x)
    assert out.shape == (4, 32)
    out.sum().backward()
    assert x.grad is not None


def test_ktune_flce_module_matches_unfused():
    m = KTuneFusedLinearCrossEntropy(chunk_size=8)
    hidden = torch.randn(20, 16, dtype=torch.float64)
    weight = torch.randn(40, 16, dtype=torch.float64)
    targets = torch.randint(0, 40, (20,))
    loss = m(hidden, weight, targets)
    ref = F.cross_entropy(F.linear(hidden, weight), targets)
    torch.testing.assert_close(loss.double(), ref)


class _FakeRMSNorm(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(h))
        self.variance_epsilon = 1e-6

    def forward(self, x):  # deliberately a no-op so we can detect patching
        return x


class _FakeMLP(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x):  # deliberately wrong (no activation) to detect patching
        return self.down_proj(self.up_proj(x))


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.ModuleDict({"norm": _FakeRMSNorm(16), "mlp": _FakeMLP(16, 32)}) for _ in range(2)]
        )


def test_patcher_counts_and_rewires():
    model = _FakeModel()
    assert summarize_patchable(model) == {"rmsnorm": 2, "mlp": 2}

    apply_ktune_to_model(model, verbose=False)

    x = torch.randn(2, 16)
    mlp = model.blocks[0]["mlp"]
    expected = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    torch.testing.assert_close(mlp(x), expected, atol=1e-5, rtol=1e-5)

    # RMSNorm was a no-op before; after patching it must actually normalise.
    norm = model.blocks[0]["norm"]
    y = norm(x)
    assert not torch.allclose(y, x)
