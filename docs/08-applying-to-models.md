# 08 · Applying the kernels to real models

You've built the kernels. Now use them to make a real fine-tune faster and
lighter. This is the "application" half of the repo.

## Two ways to use ktune

### 1. Drop-in modules (`ktune.nn`)

If you're building a model yourself, use the modules directly:

```python
from ktune.nn import KTuneRMSNorm, KTuneSwiGLUMLP, KTuneFusedLinearCrossEntropy

norm = KTuneRMSNorm(hidden_size=2048)
mlp  = KTuneSwiGLUMLP(hidden_size=2048, intermediate_size=5632)

# Fuse the LM head + loss (pass hidden states, NOT logits):
loss_fn = KTuneFusedLinearCrossEntropy(ignore_index=-100)
loss = loss_fn(hidden_states, model.lm_head.weight, labels)
```

### 2. Patch an existing HuggingFace model (`ktune.integrations`)

For a stock Llama/Qwen/Mistral, monkey-patch the hot path in place — no weight
changes:

```python
from transformers import AutoModelForCausalLM
from ktune.integrations import apply_ktune_to_model, summarize_patchable

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B", attn_implementation="sdpa",
)
print(summarize_patchable(model))     # dry run: what would change
apply_ktune_to_model(model)           # swaps RMSNorm + SwiGLU-MLP forwards
```

This swaps every RMSNorm and gated MLP for ktune's fused versions. Attention is
left to the model's own fused backend — `"sdpa"` is built into PyTorch and always
available, so it's the safe default; use `"flash_attention_2"` instead only if
you've installed the separate `flash-attn` package. You fuse the LM-head loss
separately with `KTuneFusedLinearCrossEntropy`. This mirrors how Liger-Kernel and
Unsloth patch models.

## The end-to-end example

`examples/finetune_qlora.py` runs a real **QLoRA** fine-tune of a small model and
reports tokens/s and peak VRAM **with vs without** the ktune patch:

```bash
python examples/finetune_qlora.py --model Qwen/Qwen2.5-0.5B --steps 30          # baseline
python examples/finetune_qlora.py --model Qwen/Qwen2.5-0.5B --steps 30 --ktune  # patched
```

Or open `examples/finetune_qlora.ipynb` in Colab (badge in the README) and run
both on a free T4.

## How to read the benchmark numbers

- **Loss curve.** Should be essentially identical to the baseline — these kernels
  change *how* the math is computed, not *what* is computed. If the loss diverges,
  that's a bug, not a feature. (Measured: 1.482 vs 1.483 — see `RESULTS.md`.)
- **Throughput (tokens/s).** Be calibrated: the element-wise patcher (RMSNorm +
  SwiGLU) is roughly **neutral at small scale** — on a 0.5B model those ops are a
  small slice of runtime, and the kernels here aren't `@triton.autotune`'d, so
  launch overhead cancels the bandwidth they save. A clear speedup needs
  autotuning and/or a larger model.
- **Peak VRAM.** The big wins live in **FusedLinearCrossEntropy** and
  **FlashAttention**, and they only help when they're in the hot path. The example
  now routes the loss through FLCE by default under `--ktune` (see the
  `FLCETrainer` in `examples/finetune_qlora.py`, built on
  `ktune.integrations.fused_causal_lm_loss`), so the `[tokens, vocab]` logits are
  never materialised. The size of the drop scales with `vocab × seq` — modest at a
  32k vocab + short context, large at 128k vocab or long context. Attention is
  still left to `sdpa` here; run `bench_attention.py` to see FlashAttention's
  separate memory win.

See `benchmarks/RESULTS.md` for the full measured numbers and interpretation.
