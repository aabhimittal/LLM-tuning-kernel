"""Fused causal-LM loss — FusedLinearCrossEntropy with the standard next-token shift.

This is what lets the flagship memory kernel actually fire in a real fine-tune.
A normal HuggingFace forward computes ``logits = lm_head(hidden)`` (the huge
``[batch·seq, vocab]`` tensor) and then cross-entropy. Here we hand the
**hidden states** and the LM-head weight straight to
:func:`ktune.ops.fused_linear_cross_entropy`, which chunks the projection so the
full logits tensor is never materialised — see ``docs/06-fused-linear-ce.md``.

Use :func:`fused_causal_lm_loss` directly if you control the training loop, or see
``examples/finetune_qlora.py`` for a drop-in ``Trainer`` that wires it into the
HuggingFace ``Trainer`` via the model's output-embedding.
"""

from __future__ import annotations

import torch

from ktune.ops import fused_linear_cross_entropy


def fused_causal_lm_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Next-token cross-entropy fused with the LM-head projection.

    Applies the usual causal shift (position ``t`` predicts token ``t+1``) and runs
    FusedLinearCrossEntropy on the hidden states, so peak memory stays flat in the
    vocabulary size.

    Args:
        hidden_states: ``[batch, seq, hidden]`` — exactly what the model feeds to
            its ``lm_head`` (i.e. post-final-norm hidden states).
        lm_head_weight: ``[vocab, hidden]`` LM-head weight.
        labels: ``[batch, seq]`` token ids; use ``ignore_index`` for non-loss
            positions (padding, prompt tokens, ...).
        bias: optional ``[vocab]`` LM-head bias.
        ignore_index: label value to skip in the loss.
        chunk_size: rows of hidden states processed per chunk (the memory knob).

    Returns:
        Scalar mean cross-entropy, differentiable w.r.t. ``hidden_states`` (and the
        LM-head weight/bias).
    """
    # Shift: drop the last position's hidden state and the first label.
    shift_hidden = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return fused_linear_cross_entropy(
        shift_hidden.view(-1, shift_hidden.shape[-1]),
        lm_head_weight,
        shift_labels.view(-1),
        bias=bias,
        ignore_index=ignore_index,
        chunk_size=chunk_size,
    )
