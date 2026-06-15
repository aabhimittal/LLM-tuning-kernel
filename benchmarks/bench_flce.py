"""Benchmark the flagship: Fused Linear + Cross Entropy memory savings.

    python benchmarks/bench_flce.py --hardware t4

Compares peak GPU memory and latency of:
  * baseline: ``lm_head(hidden)`` -> ``F.cross_entropy`` (materialises all logits)
  * ktune:    ``fused_linear_cross_entropy`` (chunked, logits never fully formed)

The gap widens with vocab size — this is where ktune earns its keep on
large-vocab models (Llama-3 128k, Gemma 256k).
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from _bench_utils import banner, device, peak_memory_mb, time_ms
from ktune.ops import fused_linear_cross_entropy

CONFIGS = {
    "t4": dict(tokens=4096, hidden=2048, vocab=32000, chunk=1024),
    "a100": dict(tokens=8192, hidden=4096, vocab=128256, chunk=2048),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hardware", choices=list(CONFIGS), default="t4")
    args = ap.parse_args()
    cfg = CONFIGS[args.hardware]
    dev = device()

    banner(f"Fused Linear + Cross Entropy — {args.hardware.upper()} config")
    hidden = torch.randn(cfg["tokens"], cfg["hidden"], device=dev, requires_grad=True)
    weight = torch.randn(cfg["vocab"], cfg["hidden"], device=dev, requires_grad=True)
    targets = torch.randint(0, cfg["vocab"], (cfg["tokens"],), device=dev)

    def baseline():
        loss = F.cross_entropy(F.linear(hidden, weight), targets)
        loss.backward()

    def ktune():
        loss = fused_linear_cross_entropy(hidden, weight, targets, chunk_size=cfg["chunk"])
        loss.backward()

    base_mem = peak_memory_mb(baseline)
    ours_mem = peak_memory_mb(ktune)
    base_ms = time_ms(baseline, warmup=3, iters=10)
    ours_ms = time_ms(ktune, warmup=3, iters=10)

    print(f"tokens={cfg['tokens']} hidden={cfg['hidden']} vocab={cfg['vocab']}\n")
    print(f"{'variant':<16}{'peak mem (MB)':>16}{'fwd+bwd (ms)':>16}")
    print("-" * 48)
    print(f"{'baseline':<16}{base_mem:>16.1f}{base_ms:>16.2f}")
    print(f"{'ktune (FLCE)':<16}{ours_mem:>16.1f}{ours_ms:>16.2f}")
    if ours_mem > 0:
        print(f"\nmemory reduction: {base_mem / ours_mem:.2f}x")


if __name__ == "__main__":
    main()
