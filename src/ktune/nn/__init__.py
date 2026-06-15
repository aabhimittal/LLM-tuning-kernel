"""Drop-in ``nn.Module``s wrapping the ktune ops.

These mirror the interfaces of the corresponding HuggingFace/Llama modules so
they can be swapped in directly (see :mod:`ktune.integrations.patch`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ktune.ops import fused_linear_cross_entropy, rms_norm, swiglu


class KTuneRMSNorm(nn.Module):
    """RMSNorm with a learnable per-channel gain. Matches Llama ``RMSNorm``."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.variance_epsilon)


class KTuneSwiGLUMLP(nn.Module):
    """Llama-style gated MLP: ``down(silu(gate(x)) * up(x))`` with a fused middle."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class KTuneFusedLinearCrossEntropy(nn.Module):
    """LM-head projection fused with cross-entropy (flat memory in vocab size).

    Pass the *hidden states* (not logits) plus the LM-head weight. Use this in
    place of ``lm_head`` + ``F.cross_entropy`` to cut the logits-memory spike.
    """

    def __init__(self, ignore_index: int = -100, chunk_size: int = 1024):
        super().__init__()
        self.ignore_index = ignore_index
        self.chunk_size = chunk_size

    def forward(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        targets: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return fused_linear_cross_entropy(
            hidden,
            lm_head_weight,
            targets,
            bias,
            ignore_index=self.ignore_index,
            chunk_size=self.chunk_size,
        )


__all__ = ["KTuneRMSNorm", "KTuneSwiGLUMLP", "KTuneFusedLinearCrossEntropy"]
