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

| vocab | expected peak-memory reduction (loss step) |
|-------|--------------------------------------------|
| 32k (Qwen2.5) | ~2–3× |
| 128k (Llama-3) | **~4×+** |

Liger-Kernel reports **>4× memory reduction at 128k vocab** for this kernel.

### FlashAttention forward (`bench_attention.py`)

Naive (materialised) attention memory grows quadratically with sequence length and
OOMs first; the tiled kernel stays roughly flat. PyTorch SDPA (a fused backend) is
the strong reference point.

| seq len | naive attention memory | flash memory |
|---------|------------------------|--------------|
| 1k | baseline | ~same |
| 4k | ~16× the 1k matrix | roughly flat |
| 8k+ | often OOM on a T4 | still fits |

### End-to-end QLoRA (`examples/finetune_qlora.py`)

Across the whole fine-tune, the combination (fused RMSNorm + SwiGLU + FLCE +
FlashAttention-2 backend) is in the ballpark Unsloth reports: **~1.5–2× faster
training and ~40–60% less VRAM** for small models on a single GPU — with an
essentially identical loss curve.

> Replace this section with your measured `[baseline]` vs `[ktune]` output.
