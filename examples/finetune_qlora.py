"""End-to-end QLoRA fine-tune, with vs without ktune kernels.

This is the "application" payoff: take a small instruct model, fine-tune it with
4-bit QLoRA, and measure tokens/s + peak VRAM. Pass ``--ktune`` to patch the model
with ktune's fused kernels and compare.

    # baseline
    python examples/finetune_qlora.py --model Qwen/Qwen2.5-0.5B --steps 30
    # with ktune fused kernels (also fuses the loss via FusedLinearCrossEntropy)
    python examples/finetune_qlora.py --model Qwen/Qwen2.5-0.5B --steps 30 --ktune

``--ktune`` patches RMSNorm + SwiGLU and, by default, also routes the loss through
FusedLinearCrossEntropy so the LM-head logits are never materialised (the real
memory win). Toggle that independently with ``--flce`` / ``--no-flce``.

Defaults are tuned for a **free Colab T4**. Requires the ``[app]`` extra:
    pip install -e ".[app]"
and a CUDA GPU (QLoRA / bitsandbytes is GPU-only).

The point isn't the resulting model — it's the side-by-side resource numbers.
Run it both ways and read docs/08-applying-to-models.md for interpretation.
"""

from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--dataset", default="yahma/alpaca-cleaned")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--ktune", action="store_true", help="patch the model with ktune kernels")
    ap.add_argument(
        "--flce",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="route the loss through FusedLinearCrossEntropy so the LM-head logits "
        "are never materialised (the real memory win). Defaults to ON when --ktune.",
    )
    ap.add_argument("--bf16", action="store_true", help="use bf16 (A100); default fp16 (T4)")
    ap.add_argument(
        "--attn",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="attention backend. sdpa (default) is built into PyTorch and always "
        "available; flash_attention_2 needs the separately-built flash-attn package.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.flce is None:  # default: fuse the loss whenever we're using ktune
        args.flce = args.ktune

    # Imports are inside main so `--help` works without the heavy [app] deps.
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency: {e}. Install the app extra:  pip install -e '.[app]'"
        ) from e

    if not torch.cuda.is_available():
        raise SystemExit("This example needs a CUDA GPU (QLoRA/bitsandbytes is GPU-only).")

    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # ---- 4-bit base model (QLoRA) ----
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        quantization_config=bnb,
        attn_implementation=args.attn,
        device_map={"": 0},
    )
    # `torch_dtype` was renamed to `dtype` in recent transformers; support both.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=compute_dtype, **model_kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=compute_dtype, **model_kwargs
        )
    model = prepare_model_for_kbit_training(model)

    # ---- optionally patch in ktune fused kernels ----
    if args.ktune:
        from ktune.integrations import apply_ktune_to_model, summarize_patchable

        print("[ktune] patchable modules:", summarize_patchable(model))
        apply_ktune_to_model(model)

    # ---- LoRA adapters ----
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ---- data ----
    n_examples = args.steps * args.batch_size * args.grad_accum + 64
    ds = load_dataset(args.dataset, split=f"train[:{n_examples}]")

    def fmt(ex):
        instr, inp, out = ex.get("instruction", ""), ex.get("input", ""), ex.get("output", "")
        prompt = f"### Instruction:\n{instr}\n\n"
        if inp:
            prompt += f"### Input:\n{inp}\n\n"
        prompt += f"### Response:\n{out}"
        return tokenizer(prompt, truncation=True, max_length=args.seq_len, padding="max_length")

    ds = ds.map(fmt, remove_columns=ds.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    targs = TrainingArguments(
        output_dir="/tmp/ktune-qlora",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.steps,
        learning_rate=2e-4,
        logging_steps=5,
        bf16=args.bf16,
        fp16=not args.bf16,
        optim="paged_adamw_8bit",
        report_to=[],
        gradient_checkpointing=True,
    )
    if args.flce:
        from ktune.integrations import fused_causal_lm_loss

        class FLCETrainer(Trainer):
            """Computes the LM loss with FusedLinearCrossEntropy, with a safety net.

            We swap the model's output embedding (``lm_head``) for a hook that
            captures its *input* (the hidden states) and returns dummy logits, so
            the model never forms the full ``[batch, seq, vocab]`` tensor; the loss
            is then computed from the hidden states + LM-head weight via the chunked
            FLCE kernel.

            On the **first step** we also compute the model's native loss and
            compare: if they disagree (some model wiring doesn't expose hidden
            states the way we assume), we print both and permanently fall back to
            the native loss so training stays correct — you just don't get the
            memory win. When they agree, FLCE is used for every step.
            """

            _flce_ok = None  # None = untested, True = use FLCE, False = fall back

            def _flce_loss(self, model, inputs):
                lm_head = model.get_output_embeddings()
                if lm_head is None:
                    raise RuntimeError("model has no output embeddings; can't fuse the loss")
                captured = {}
                original_forward = lm_head.forward

                def capture(x, *a, **k):
                    captured["hidden"] = x
                    return x[..., :1]  # dummy logits — the body never sees full vocab

                lm_head.forward = capture
                try:
                    model(**{k: v for k, v in inputs.items() if k != "labels"})
                finally:
                    lm_head.forward = original_forward
                if "hidden" not in captured:
                    raise RuntimeError("FLCE loss: the model never called its output embedding")
                return fused_causal_lm_loss(
                    captured["hidden"],
                    lm_head.weight,
                    inputs["labels"],
                    bias=getattr(lm_head, "bias", None),
                    ignore_index=-100,
                    chunk_size=1024,
                )

            def _validate(self, model, inputs):
                # Validate the FLCE math on a SHORT slice so we never materialise
                # full-size logits (which would re-inflate peak VRAM and hide the
                # win). Compares the FLCE mean loss to the model's native mean loss.
                k = min(128, inputs["labels"].shape[1])
                sub = {
                    key: (v[:, :k] if torch.is_tensor(v) and v.dim() >= 2 else v)
                    for key, v in inputs.items()
                }
                with torch.no_grad():
                    flce = self._flce_loss(model, sub)
                    native = super().compute_loss(model, dict(sub))
                rel = (flce - native).abs() / native.abs().clamp(min=1e-6)
                ok = bool(rel < 0.02)
                verdict = (
                    "match — using FLCE (memory-efficient)"
                    if ok
                    else "MISMATCH — falling back to the model's native loss"
                )
                print(
                    f"[ktune] FLCE check (first {k} tokens): "
                    f"flce={flce.item():.4f} native={native.item():.4f} -> {verdict}"
                )
                return ok

            def compute_loss(
                self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs
            ):
                if FLCETrainer._flce_ok is None:  # one-time, cheap validation
                    FLCETrainer._flce_ok = self._validate(model, inputs)
                if FLCETrainer._flce_ok is False:  # validated-bad: native (correct) loss
                    return super().compute_loss(
                        model, inputs, return_outputs, num_items_in_batch=num_items_in_batch
                    )

                loss = self._flce_loss(model, inputs)  # mean over valid tokens in microbatch
                # Recent transformers passes num_items_in_batch and then does NOT divide
                # the loss by gradient_accumulation_steps — it expects sum/num_items, not a
                # per-microbatch mean. Convert ours to match, else loss/grads are
                # ~grad_accum× too large.
                if num_items_in_batch is not None:
                    n_valid = (inputs["labels"][..., 1:] != -100).sum().to(loss.dtype)
                    loss = loss * n_valid / num_items_in_batch
                return (loss, {}) if return_outputs else loss

        print("[ktune] loss fused via FusedLinearCrossEntropy (self-checked on step 1)")
        trainer = FLCETrainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    else:
        trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0

    tokens = args.steps * args.batch_size * args.grad_accum * args.seq_len
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    tag = "ktune" if args.ktune else "baseline"
    print("\n" + "=" * 60)
    print(f"[{tag}] steps={args.steps}  time={elapsed:.1f}s")
    print(f"[{tag}] throughput = {tokens / elapsed:,.0f} tokens/s")
    print(f"[{tag}] peak VRAM  = {peak_gb:.2f} GB")
    print("=" * 60)
    print("Run again with the opposite --ktune setting and compare.")


if __name__ == "__main__":
    main()
