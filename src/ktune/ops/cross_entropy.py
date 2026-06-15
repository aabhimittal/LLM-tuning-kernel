"""Cross-entropy — fused softmax + negative-log-likelihood.

Reference: :func:`ktune.utils.reference.cross_entropy`.

Two ideas make the kernel fast and memory-light:

1. **Online softmax.** We never materialise the softmax probabilities. A single
   streaming pass over the vocab tracks the running max ``m`` and the running sum
   of ``exp(x - m)``, giving the log-sum-exp in one go (the same trick
   FlashAttention uses over keys).
2. **Gradient-in-place.** The gradient of cross-entropy w.r.t. the logits is just
   ``softmax(x) - onehot(target)``. We write it straight back into the logits
   buffer during the forward pass, so the backward is a free scale — no second
   softmax, no extra big tensor.

One program handles one row (one token's vocab distribution), looping over the
vocab in blocks so arbitrarily large vocabularies work.
"""

from __future__ import annotations

import torch

from ktune.utils.reference import cross_entropy as _ce_ref
from ktune.utils.runtime import HAS_TRITON, use_triton

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _ce_kernel(
        x_ptr,  # [n_rows, vocab] logits (gradient written back here)
        target_ptr,  # [n_rows] int targets
        loss_ptr,  # [n_rows] per-row loss out
        n_rows,
        vocab,
        inv_n_valid,  # 1 / number-of-non-ignored-tokens (for the mean)
        ignore_index,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_ptr += row * vocab
        target = tl.load(target_ptr + row)

        if target == ignore_index:
            # masked token: zero loss and zero gradient over the whole row
            for start in range(0, vocab, BLOCK):
                cols = start + tl.arange(0, BLOCK)
                tl.store(x_ptr + cols, tl.zeros([BLOCK], tl.float32), mask=cols < vocab)
            tl.store(loss_ptr + row, 0.0)
            return

        # Pass 1: online max + sum of exp over the vocab.
        m = -float("inf")
        s = 0.0
        for start in range(0, vocab, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < vocab
            x = tl.load(x_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
            block_max = tl.max(x, axis=0)
            new_m = tl.maximum(m, block_max)
            s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
            m = new_m

        lse = m + tl.log(s)  # log-sum-exp
        x_target = tl.load(x_ptr + target).to(tl.float32)
        tl.store(loss_ptr + row, (lse - x_target) * inv_n_valid)

        # Pass 2: write gradient = (softmax - onehot) * (1 / n_valid).
        for start in range(0, vocab, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < vocab
            x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            grad = tl.exp(x - lse)
            grad = tl.where(cols == target, grad - 1.0, grad)
            tl.store(x_ptr + cols, grad * inv_n_valid, mask=mask)


class _CrossEntropyTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index):
        logits2d = logits.contiguous().view(-1, logits.shape[-1]).float()
        targets = targets.contiguous().view(-1)
        n_rows, vocab = logits2d.shape
        n_valid = (targets != ignore_index).sum().clamp(min=1).item()
        loss_per_row = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
        block = min(triton.next_power_of_2(vocab), 16384)
        # The kernel overwrites logits2d with the gradient — keep it for backward.
        _ce_kernel[(n_rows,)](
            logits2d,
            targets,
            loss_per_row,
            n_rows,
            vocab,
            1.0 / n_valid,
            ignore_index,
            BLOCK=block,
        )
        ctx.save_for_backward(logits2d)
        ctx.input_shape = logits.shape
        return loss_per_row.sum()

    @staticmethod
    def backward(ctx, dloss):
        (grad,) = ctx.saved_tensors
        return (dloss * grad).view(ctx.input_shape), None, None


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Fused mean cross-entropy. Triton on CUDA, reference on CPU."""
    if use_triton(logits):
        return _CrossEntropyTriton.apply(logits, targets, ignore_index)
    return _ce_ref(logits, targets, ignore_index)
