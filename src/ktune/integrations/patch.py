"""Patch a HuggingFace model in place to use ktune kernels.

This is the "application" layer: take a stock Llama/Qwen/Mistral model and swap
the hot-path ``forward`` methods for ktune's fused versions, without touching the
weights. It mirrors how Liger-Kernel and Unsloth monkey-patch model modules.

We patch the two transforms that are robust across model versions and give most
of the win:

* every **RMSNorm** module  ->  ktune fused RMSNorm,
* every **gated MLP** (``gate_proj``/``up_proj``/``down_proj``)  ->  fused SwiGLU.

Attention is left to the model's own fused backend — load with
``attn_implementation="sdpa"`` (built into PyTorch, always available) or
``"flash_attention_2"`` if you've installed the flash-attn package. The LM-head
loss can be fused separately with :class:`ktune.nn.KTuneFusedLinearCrossEntropy`.

No hard dependency on ``transformers``; everything is duck-typed so importing
this module never fails on a machine without it.
"""

from __future__ import annotations

import types

import torch

from ktune.ops import rms_norm, swiglu


def _patched_rmsnorm_forward(self, x: torch.Tensor) -> torch.Tensor:
    eps = getattr(self, "variance_epsilon", None) or getattr(self, "eps", 1e-6)
    return rms_norm(x, self.weight, eps)


def _patched_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


def _is_rmsnorm(module) -> bool:
    name = type(module).__name__.lower()
    return "rmsnorm" in name and hasattr(module, "weight")


def _is_gated_mlp(module) -> bool:
    return all(hasattr(module, p) for p in ("gate_proj", "up_proj", "down_proj"))


def summarize_patchable(model) -> dict[str, int]:
    """Count how many modules ktune *would* patch — handy for a dry run."""
    counts = {"rmsnorm": 0, "mlp": 0}
    for module in model.modules():
        if _is_rmsnorm(module):
            counts["rmsnorm"] += 1
        elif _is_gated_mlp(module):
            counts["mlp"] += 1
    return counts


def apply_ktune_to_model(model, *, rmsnorm: bool = True, mlp: bool = True, verbose: bool = True):
    """Monkey-patch ``model`` in place to use ktune kernels. Returns the model.

    Args:
        model: a HuggingFace ``PreTrainedModel`` (Llama/Qwen/Mistral family).
        rmsnorm: patch RMSNorm modules.
        mlp: patch gated (SwiGLU) MLP modules.
        verbose: print a one-line summary of what was patched.
    """
    patched = {"rmsnorm": 0, "mlp": 0}
    for module in model.modules():
        if rmsnorm and _is_rmsnorm(module):
            module.forward = types.MethodType(_patched_rmsnorm_forward, module)
            patched["rmsnorm"] += 1
        elif mlp and _is_gated_mlp(module):
            module.forward = types.MethodType(_patched_mlp_forward, module)
            patched["mlp"] += 1
    if verbose:
        print(
            f"[ktune] patched {patched['rmsnorm']} RMSNorm and {patched['mlp']} "
            f"SwiGLU-MLP modules. (Load with attn_implementation='sdpa' (or "
            f"'flash_attention_2' if installed) and use KTuneFusedLinearCrossEntropy "
            f"for the loss to fuse the rest.)"
        )
    return model
