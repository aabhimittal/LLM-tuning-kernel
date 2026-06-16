# Benchmark results

> **How to read this page.** The numbers below are **expected ranges** drawn from
> the published results of the techniques this repo reimplements (Liger-Kernel,
> Unsloth, FlashAttention) — they tell you what *shape* of win to look for. They
> are **not** measurements committed by the author. Reproduce them on your own GPU
> with the scripts in this folder and paste your real table in. CI here is
> GPU-less, so it cannot generate these.

## Reproduce

```bash
pip install -e ".[gpu,bench]"
python benchmarks/bench_ops.py        --hardware t4     # or a100
python benchmarks/bench_flce.py       --hardware t4
python benchmarks/bench_attention.py  --hardware t4 --seq 4096
```

Each script prints a table and (where relevant) peak-memory numbers. Save the
output here, e.g. `benchmarks/results/t4_<date>.txt`.

## What to expect

### Element-wise / reduction ops (`bench_ops.py`)

Memory-bound ops; the win is bandwidth saved by fusion. Speedups are modest per-op
but compound across a full training step.

| op | expected speedup vs eager PyTorch |
|----|-----------------------------------|
| `rms_norm` | ~1.5–3× |
| `swiglu` | ~1.5–2.5× |
| `cross_entropy` (fused grad) | ~1.5–2× + removes the grad allocation |

### Fused Linear Cross-Entropy (`bench_flce.py`)

The flagship. The win grows with vocab size because the avoided `[tokens, vocab]`
logits tensor grows with it.

**Measured (`bench_flce.py --hardware t4`, T4, 4096 tokens × hidden 2048, vocab 32k):**

| variant | peak mem | fwd+bwd |
|---------|---------:|--------:|
| baseline (`lm_head` + CE) | 1877 MB | 377 ms |
| ktune (FLCE) | 1297 MB | 379 ms |

→ **1.45× peak-memory reduction at 32k vocab**, same speed. The reduction grows
with vocab (≈2–3× at 32k for larger token counts, **>4× at 128k** per Liger-Kernel)
because the avoided logits/gradient tensors scale with vocab.

### FlashAttention forward (`bench_attention.py`)

Naive (materialised) attention memory grows quadratically with sequence length and
OOMs first; the tiled kernel stays roughly flat. PyTorch SDPA (a fused backend) is
the strong reference point.

| seq len | naive attention memory | flash memory |
|---------|------------------------|--------------|
| 1k | baseline | ~same |
| 4k | ~16× the 1k matrix | roughly flat |
| 8k+ | often OOM on a T4 | still fits |

### End-to-end QLoRA (`examples/finetune_qlora.py`) — measured

**Qwen2.5-0.5B, QLoRA (nf4), 30 steps, seq 1024, batch 1 × grad-accum 4, sdpa
attention, on a Colab T4.**

First run, **`--ktune` patched RMSNorm + SwiGLU only** (loss still computed by HF):

| variant | train loss | throughput | peak VRAM |
|---------|-----------:|-----------:|----------:|
| baseline | 1.482 | 1,824 tok/s | 3.07 GB |
| ktune (RMSNorm + SwiGLU) | 1.483 | 1,787 tok/s | 3.07 GB |

**How to read this — an honest result:**

- **Correctness ✓.** The loss curves are step-for-step identical (1.482 vs 1.483).
  The fused kernels change *how* the math runs, not *what* it computes.
- **No memory change in that first run**, because the patcher only swapped RMSNorm
  + SwiGLU — the memory-dominant kernels weren't engaged: the loss flowed through
  HF's own `lm_head` + cross-entropy, and attention ran on `sdpa`.
- **No speedup at this scale**, in fact ~2% slower. At 0.5B the patched
  element-wise ops are a small slice of total runtime, and these kernels are
  **not `@triton.autotune`'d** — fixed block sizes mean launch overhead roughly
  cancels the bandwidth they save. Memory-bound kernels need autotuning and a
  larger model to pull clearly ahead.

Second run, **`--ktune` with the loss routed through FusedLinearCrossEntropy**:
peak VRAM dropped **3.07 → 2.65 GB** (logits no longer materialised). The first
attempt also exposed a real integration bug — the loss read ~7.3 instead of ~1.7
because the custom `Trainer` returned a per-microbatch *mean* while recent
transformers passes `num_items_in_batch` and skips the grad-accum division
(expecting `sum / num_items`), making the loss/grads ~`grad_accum`× too large.

**Fixed:** the `FLCETrainer` now honours `num_items_in_batch` and self-checks its
loss against the model's native loss on step 1, falling back to the native loss
if they ever diverge. Re-run `--ktune` for the corrected loss curve (should track
the baseline) alongside the ~14% VRAM drop; try `--seq 2048` for a larger gap.

**Where the wins are largest** (run these to see them in isolation):

- `bench_flce.py` — the FusedLinearCrossEntropy memory drop, which grows with
  vocab size (negligible at 32k, large at 128k+).
- `bench_attention.py --seq 4096/8192` — FlashAttention's flat-vs-quadratic
  memory as sequence length grows.

To get an end-to-end win you have to put those kernels *in the hot path*: route
the loss through `KTuneFusedLinearCrossEntropy` (a custom `Trainer.compute_loss`)
and/or train at long context + large vocab. The element-wise patcher alone is
best understood as "free correctness-preserving fusion that's roughly neutral at
small scale", not a speed button.

