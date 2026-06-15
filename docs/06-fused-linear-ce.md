# 06 · Fused Linear Cross-Entropy — the flagship memory kernel

**Code:** `src/ktune/ops/fused_linear_ce.py` · **Reference:**
`fused_linear_cross_entropy`

This is the highest-impact technique in the repo for fine-tuning, and the headline
result of [Liger-Kernel](https://arxiv.org/abs/2410.10989).

## The problem: logits are enormous

The last step of an LLM forward is the LM head:

```
logits = hidden @ W.T        # [tokens, hidden] @ [hidden, vocab] -> [tokens, vocab]
loss   = cross_entropy(logits, targets)
```

For Llama-3 (vocab ≈ 128k) with 8k tokens in a batch, `logits` in fp32 is
`8192 × 128256 × 4 bytes ≈ 4.0 GB` — and the gradient `d_logits` is another 4 GB.
That single intermediate often **dwarfs every other activation in the model
combined**, and it's the thing that OOMs your fine-tune.

## The fix: chunk and never materialise

Process the `hidden` rows in **chunks**. For each chunk you only ever hold
`chunk_size × vocab` logits, do the loss + gradient for that chunk, convert the
gradient back to input space, accumulate, and throw the chunk's logits away:

```
for chunk in hidden.split(chunk_size):
    logits_c   = chunk @ W.T                 # small: [chunk, vocab]
    loss_c, dlogits_c = cross_entropy(logits_c, targets_c)   # online-softmax kernel
    d_hidden[chunk] = dlogits_c @ W          # back to input space
    d_W            += dlogits_c.T @ chunk    # accumulate weight grad
    # logits_c is freed here
```

Peak logits memory drops from `tokens × vocab` to `chunk_size × vocab` — a **>4×
reduction at 128k vocab** with no change to the numerical result. The loss is
bit-for-bit the ordinary cross-entropy.

## Where the kernels live

The chunk loop (the matmuls and accumulation) is plain PyTorch — that's what makes
it correct and testable on CPU, and it's already enough to deliver the flat-memory
property. The *inner* cross-entropy step calls the online-softmax Triton kernel
from [05 · Cross-entropy](05-cross-entropy.md), which writes `dlogits_c` in place.

We precompute `d_hidden` and `d_W` during the forward and stash them, so backward
is just a scale by the upstream gradient. (For a loss, that upstream is the scalar
1.0 in the usual case.)

## The knob: `chunk_size`

Smaller chunk → less peak memory, slightly more kernel-launch overhead. Larger
chunk → more memory, better matmul efficiency. It's the central memory/speed
trade-off; sweep it in `benchmarks/bench_flce.py`.

## Exercises

1. Measure peak memory vs `chunk_size` and find the knee of the curve for your GPU.
2. Push the input-gradient matmul (`dlogits_c @ W`) into a Triton kernel too and
   fuse it with the CE step.
3. Extend to a tied-embedding model (LM head shares weights with the input
   embedding) — what changes for `d_W`?

Next: [07 · FlashAttention](07-flash-attention.md).
