# 05 · Cross-entropy — online softmax + in-place gradient

**Code:** `src/ktune/ops/cross_entropy.py` · **Reference:** `cross_entropy`

This kernel introduces the **online softmax**, the single most important trick in
this repo — it reappears, two-dimensionally, in FlashAttention.

## The math

For a row of logits `x` (width = vocab) and target class `t`:

```
loss = logsumexp(x) - x[t]        where logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
grad = softmax(x) - onehot(t)
```

The `max` subtraction is mandatory for numerical stability — `exp` of a raw logit
overflows fast.

## Trick 1: online softmax (one streaming pass)

A naive softmax needs the max over the whole row *before* it can exponentiate,
implying multiple passes (and, if you're not careful, materialising the row). The
online algorithm keeps a running max `m` and running sum `s` and fixes up `s`
whenever the max increases:

```
for each block of the row:
    m_new = max(m, max(block))
    s     = s * exp(m - m_new) + sum(exp(block - m_new))
    m     = m_new
lse = m + log(s)
```

One pass, O(1) state, exact result. This lets the vocab be arbitrarily large —
the kernel just loops over it in `BLOCK`-sized pieces.

## Trick 2: gradient in place

The gradient of CE w.r.t. the logits is exactly `softmax(x) - onehot(t)`. We
compute it in the **forward** pass and write it *back into the logits buffer*. The
`backward` then just scales that stored tensor by the incoming `dloss` — no second
softmax, no extra vocab-sized allocation.

`ignore_index` tokens get zero loss and zero gradient; the mean is taken over the
count of non-ignored tokens (`inv_n_valid`).

## Why this kernel matters

For large vocabularies the logits gradient is a huge tensor. Fusing it away
(here) and, better, never forming the full logits at all (next doc) is where the
biggest fine-tuning memory savings come from.

## Exercises

1. Add label smoothing — how do the loss and the in-place gradient change?
2. Add a `z_loss` / logit-regularisation term (used in PaLM/Gemma training).
3. Verify the online softmax matches a two-pass softmax to fp32 precision for a
   row with one enormous logit.

Next: [06 · Fused Linear Cross-Entropy](06-fused-linear-ce.md).
