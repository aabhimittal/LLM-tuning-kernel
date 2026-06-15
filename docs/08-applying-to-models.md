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
    "Qwen/Qwen2.5-0.5B", attn_implementation="flash_attention_2",
)
print(summarize_patchable(model))     # dry run: what would change
apply_ktune_to_model(model)           # swaps RMSNorm + SwiGLU-MLP forwards
```

This swaps every RMSNorm and gated MLP for ktune's fused versions. Attention is
left to the model's own FlashAttention-2 backend (hence
`attn_implementation="flash_attention_2"`), and you fuse the LM-head loss
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

- **Throughput (tokens/s).** The fused element-wise kernels (RMSNorm, SwiGLU,
  RoPE) cut HBM traffic; expect a modest but real end-to-end speedup.
- **Peak VRAM.** This is where the big wins live — FusedLinearCrossEntropy and
  FlashAttention remove the two largest activations. Lower peak VRAM means you can
  raise batch size or sequence length, which *then* raises throughput again.
- **Loss curve.** Should be essentially identical to the baseline — these kernels
  change *how* the math is computed, not *what* is computed. If the loss diverges,
  that's a bug, not a feature.

See `benchmarks/RESULTS.md` for the harness output and reference numbers.
