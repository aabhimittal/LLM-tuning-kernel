"""Pure-PyTorch reference implementations of every op in ``ktune``.

These are the **correctness oracle** and the **teaching baseline**:

* They are written for *readability*, not speed — each one is the plain
  mathematical definition of the operation, in a handful of lines.
* They run on **CPU**, so they (and the tests that compare against them) work
  in environments without a GPU. The Triton kernels in ``ktune.ops`` are checked
  against these on a GPU.
* When you read a Triton kernel in this repo, read the matching function here
  first. The kernel is "this math, but tiled and fused to avoid moving data
  through GPU HBM". Keeping the two side by side is the whole point.

Nothing here imports Triton, so this module is always importable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# RMSNorm
# --------------------------------------------------------------------------- #
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Root-mean-square layer norm (Zhang & Sennrich, 2019).

    ``y = x / sqrt(mean(x**2, dim=-1) + eps) * weight``

    Unlike LayerNorm there is no mean-subtraction and no bias — only a
    reciprocal-RMS rescale followed by a per-channel gain. That single
    reduction along the last dimension is exactly what the Triton kernel
    parallelises one row per program.
    """
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x.to(dtype)).to(dtype)


# --------------------------------------------------------------------------- #
# Rotary Position Embedding (RoPE)
# --------------------------------------------------------------------------- #
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dim: ``[a, b] -> [-b, a]``."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    ``q_rot = q * cos + rotate_half(q) * sin`` (and likewise for ``k``).

    ``q``/``k`` are ``[batch, heads, seq, head_dim]``; ``cos``/``sin`` are
    ``[seq, head_dim]`` and broadcast over batch and heads. This is an
    element-wise op — no reduction — so the kernel is memory-bound and the win
    comes from fusing the multiply-adds into a single read/write of q and k.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# --------------------------------------------------------------------------- #
# SwiGLU (the gated MLP activation used by Llama/Qwen/Mistral)
# --------------------------------------------------------------------------- #
def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: ``silu(gate) * up`` where ``silu(x) = x * sigmoid(x)``.

    In a Llama-style MLP this sits between two projections:
    ``down( silu(gate_proj(x)) * up_proj(x) )``. Fusing the ``silu`` and the
    element-wise product into one kernel avoids writing the intermediate
    ``silu(gate)`` tensor to HBM only to read it straight back.
    """
    return F.silu(gate) * up


# --------------------------------------------------------------------------- #
# Cross entropy
# --------------------------------------------------------------------------- #
def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mean token-level cross-entropy over a ``[N, vocab]`` logits tensor.

    Numerically this is ``logsumexp(logits) - logits[target]``, averaged over
    non-ignored tokens. The fused kernel computes the softmax statistics and
    writes the gradient back into the logits buffer in a single pass.
    """
    return F.cross_entropy(logits.float(), targets, ignore_index=ignore_index)


# --------------------------------------------------------------------------- #
# Fused Linear + Cross Entropy (the flagship memory kernel)
# --------------------------------------------------------------------------- #
def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Reference for the fused LM-head projection + cross-entropy.

    Computes ``logits = hidden @ weight.T (+ bias)`` then cross-entropy.

    The *whole reason this op exists* is the memory cost of that intermediate
    ``logits`` tensor: for a 128k-vocab model with a long sequence it dwarfs the
    rest of the activation memory. The reference materialises it (so you can see
    the definition); the real kernel never does — it walks ``hidden`` in chunks,
    projects + reduces each chunk, and accumulates the loss/gradient, so peak
    memory stays flat in the vocab size. Compare with ``ops/fused_linear_ce.py``.
    """
    logits = F.linear(hidden, weight, bias)
    return F.cross_entropy(logits.float(), targets, ignore_index=ignore_index)


# --------------------------------------------------------------------------- #
# Attention (the thing FlashAttention computes, without the IO trick)
# --------------------------------------------------------------------------- #
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Standard scaled dot-product attention — the math FlashAttention fuses.

    ``softmax(q @ k.T * scale + mask) @ v``

    This materialises the full ``[seq, seq]`` score matrix, which is the
    quadratic-memory bottleneck FlashAttention removes via tiling + the online
    softmax. ``q``/``k``/``v`` are ``[batch, heads, seq, head_dim]``.
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    if causal:
        seq = q.shape[-2]
        mask = torch.triu(torch.ones(seq, seq, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)
