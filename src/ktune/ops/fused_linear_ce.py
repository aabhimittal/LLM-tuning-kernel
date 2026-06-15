"""Fused Linear + Cross Entropy (FLCE) — the flagship memory kernel.

Reference: :func:`ktune.utils.reference.fused_linear_cross_entropy`.

**The problem.** The last step of an LLM forward is ``logits = hidden @ Wᵀ``
followed by cross-entropy. For a 128k-vocab model and a long batch, that
``logits`` tensor (``[batch·seq, vocab]``, plus an equal-sized gradient) is often
the single largest activation in the whole network — bigger than everything in
the transformer blocks combined.

**The fix (Liger-Kernel's idea).** Never hold all the logits at once. Walk the
``hidden`` rows in **chunks**; for each chunk:

1. project just that chunk to logits (``h_chunk @ Wᵀ``),
2. run cross-entropy on it (our fused kernel) to get the loss and ``d_logits``,
3. immediately turn ``d_logits`` back into ``d_hidden`` and accumulate ``d_W``,
4. discard the chunk's logits.

Peak memory then scales with ``chunk_size · vocab`` instead of ``N · vocab`` —
a >4x reduction at 128k vocab — while the math is bit-for-bit the standard loss.

The chunk loop here (the matmuls + accumulation) is plain PyTorch so it is
correct and testable on CPU; on a GPU the per-chunk cross-entropy step runs the
Triton kernel from :mod:`ktune.ops.cross_entropy`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ktune.utils.reference import fused_linear_cross_entropy as _flce_ref
from ktune.utils.runtime import HAS_TRITON, use_triton

if HAS_TRITON:
    import triton

    from ktune.ops.cross_entropy import _ce_kernel


def _chunk_ce_and_grad(logits, targets, inv_n_valid, ignore_index):
    """Cross-entropy loss + ``d_logits`` for one chunk.

    Returns ``(loss_sum, d_logits)`` where both are already scaled by the global
    ``inv_n_valid`` so summing across chunks yields the mean loss / its gradient.
    On CUDA this calls the fused Triton kernel (which writes the gradient back
    into the logits buffer); on CPU it uses a readable PyTorch equivalent.
    """
    if use_triton(logits):
        n_rows, vocab = logits.shape
        loss_per_row = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
        block = min(triton.next_power_of_2(vocab), 16384)
        logits = logits.contiguous()
        _ce_kernel[(n_rows,)](
            logits,
            targets,
            loss_per_row,
            n_rows,
            vocab,
            inv_n_valid,
            ignore_index,
            BLOCK=block,
        )
        return loss_per_row.sum(), logits  # logits now holds d_logits

    # CPU reference: identical math, written for clarity. Upcast low-precision
    # inputs to fp32 for numerical stability, but never *down*cast (so fp64 stays
    # exact — the GPU kernel likewise accumulates in fp32).
    logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
    logp = F.log_softmax(logits, dim=-1)
    valid = targets != ignore_index
    safe_targets = targets.clamp(min=0)
    nll = -logp.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    loss_sum = nll.sum() * inv_n_valid

    d_logits = logp.exp()
    d_logits.scatter_add_(1, safe_targets.unsqueeze(1), -torch.ones_like(nll).unsqueeze(1))
    d_logits = d_logits * valid.unsqueeze(1)
    return loss_sum, d_logits * inv_n_valid


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, bias, ignore_index, chunk_size):
        orig_shape = hidden.shape
        hidden = hidden.contiguous().view(-1, orig_shape[-1])  # [N, H]
        targets = targets.contiguous().view(-1)  # [N]
        n_tokens = hidden.shape[0]
        n_valid = (targets != ignore_index).sum().clamp(min=1).item()
        inv_n_valid = 1.0 / n_valid

        loss = hidden.new_zeros(())
        d_hidden = torch.zeros_like(hidden)
        d_weight = torch.zeros_like(weight)
        d_bias = torch.zeros_like(bias) if bias is not None else None

        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            h_c = hidden[start:end]  # [c, H]
            t_c = targets[start:end]  # [c]
            logits_c = F.linear(h_c, weight, bias)  # [c, V] — the only big tensor, one chunk wide

            loss_c, d_logits_c = _chunk_ce_and_grad(logits_c, t_c, inv_n_valid, ignore_index)
            loss = loss + loss_c

            # Map the chunk's logit-gradient back to inputs and accumulate.
            d_logits_c = d_logits_c.to(weight.dtype)
            d_hidden[start:end] = d_logits_c @ weight
            d_weight += d_logits_c.t() @ h_c
            if d_bias is not None:
                d_bias += d_logits_c.sum(dim=0)

        saved_bias = d_bias if d_bias is not None else weight.new_zeros(0)
        ctx.save_for_backward(d_hidden.view(orig_shape), d_weight, saved_bias)
        ctx.has_bias = bias is not None
        return loss

    @staticmethod
    def backward(ctx, dloss):
        d_hidden, d_weight, d_bias = ctx.saved_tensors
        db = (dloss * d_bias) if ctx.has_bias else None
        return dloss * d_hidden, dloss * d_weight, None, db, None, None


def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Memory-efficient fused LM-head projection + cross-entropy.

    Args:
        hidden: ``[..., hidden_dim]`` hidden states feeding the LM head.
        weight: LM-head weight ``[vocab, hidden_dim]``.
        targets: ``[...]`` integer targets (``ignore_index`` to mask).
        bias: optional ``[vocab]`` bias.
        ignore_index: target value to skip in the loss.
        chunk_size: rows of ``hidden`` processed per chunk — the memory/speed knob.

    Always uses the chunked path (so peak memory stays flat in ``vocab``);
    the per-chunk cross-entropy runs in Triton on CUDA and in PyTorch on CPU.
    """
    if hidden.numel() == 0:
        return _flce_ref(hidden, weight, targets, bias, ignore_index)
    return _FusedLinearCE.apply(hidden, weight, targets, bias, ignore_index, chunk_size)
