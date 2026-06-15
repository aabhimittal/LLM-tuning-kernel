# 00 · Why kernels? The GPU memory wall

If you remember one thing: **on modern GPUs, most LLM ops are limited by how fast
you can move data, not how fast you can do math.** Custom kernels win by moving
less data. Everything in this repo is an instance of that idea.

## The memory hierarchy

A GPU has a steep memory pyramid. Rough numbers for an NVIDIA A100:

| Level | Size | Bandwidth | Analogy |
|-------|------|-----------|---------|
| Registers | ~256 KB / SM | ~tens of TB/s | your hands |
| **SRAM** (shared mem / L1) | ~192 KB / SM | ~19 TB/s | the desk |
| L2 cache | ~40 MB | ~7 TB/s | the shelf |
| **HBM** (global / "VRAM") | 40–80 GB | ~1.5–2 TB/s | the warehouse |

HBM is where your tensors live. SRAM is ~10× faster but ~100,000× smaller. The
whole game is: **load a tile from HBM into SRAM once, do as much work on it as
possible, write the result back once.**

## Compute-bound vs memory-bound

Every op has an *arithmetic intensity* = FLOPs ÷ bytes moved.

- A big matmul does O(N³) math for O(N²) data → **compute-bound**. cuBLAS already
  nails these; you won't beat it. Leave them alone.
- An elementwise op (add, SiLU, RMSNorm, RoPE) does O(N) math for O(N) data →
  **memory-bound**. Here the runtime is dominated by HBM traffic, and *fusing*
  several ops so the data is read/written once is a real, large win.

This is why this repo fuses things like `silu(gate) * up` (one kernel, one
read/write) instead of running them as three separate PyTorch ops (three
round-trips through HBM).

## The two tricks you'll see everywhere

1. **Fusion** — merge a chain of memory-bound ops into a single kernel so
   intermediates never touch HBM. (RMSNorm, SwiGLU, RoPE, the in-place CE grad.)
2. **Tiling + online algorithms** — when an intermediate is *too big to keep*
   (the `[seq, seq]` attention matrix, the `[tokens, vocab]` logits), process it
   in blocks and maintain running statistics so you never materialise the whole
   thing. (FlashAttention's online softmax, FusedLinearCrossEntropy's chunking.)

## Why this matters for *fine-tuning* specifically

Fine-tuning is memory-starved: you want the biggest model and longest context
that fit on the GPU you have. Two activations dominate and are both *removable*
with the tricks above:

- the attention score matrix → killed by **FlashAttention**,
- the LM-head logits (huge for 128k+ vocabularies) → killed by
  **FusedLinearCrossEntropy**.

Remove those and you can fit a bigger LoRA/QLoRA run on the same card. That is
exactly what Unsloth and Liger-Kernel do, and what you'll rebuild here.

Next: [01 · Triton 101](01-triton-101.md).
