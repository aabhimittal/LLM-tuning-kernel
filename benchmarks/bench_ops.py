"""Benchmark the element-wise / reduction ops vs their PyTorch baselines.

    python benchmarks/bench_ops.py --hardware t4
    python benchmarks/bench_ops.py --hardware a100

Reports median latency (ms) for ktune kernels against the eager-PyTorch way of
doing the same thing. On CPU it still runs but prints a "not representative"
notice (kernels need a GPU).
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from _bench_utils import banner, device, time_ms
from ktune.ops import cross_entropy, rms_norm, swiglu

# T4-friendly vs A100 problem sizes (batch*seq, hidden / vocab).
CONFIGS = {
    "t4": dict(tokens=4096, hidden=2048, inter=5632, vocab=32000),
    "a100": dict(tokens=16384, hidden=4096, inter=14336, vocab=128256),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hardware", choices=list(CONFIGS), default="t4")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    cfg = CONFIGS[args.hardware]
    dev = device()
    dtype = getattr(torch, args.dtype)
    banner(f"ktune ops — {args.hardware.upper()} config, {args.dtype}")
    print(f"{'op':<28}{'baseline (ms)':>16}{'ktune (ms)':>14}{'speedup':>10}")
    print("-" * 68)

    # RMSNorm
    x = torch.randn(cfg["tokens"], cfg["hidden"], device=dev, dtype=dtype)
    w = torch.randn(cfg["hidden"], device=dev, dtype=dtype)
    base = time_ms(lambda: F.rms_norm(x, (cfg["hidden"],), w))
    ours = time_ms(lambda: rms_norm(x, w))
    print(f"{'rms_norm':<28}{base:>16.3f}{ours:>14.3f}{base / ours:>9.2f}x")

    # SwiGLU
    g = torch.randn(cfg["tokens"], cfg["inter"], device=dev, dtype=dtype)
    u = torch.randn(cfg["tokens"], cfg["inter"], device=dev, dtype=dtype)
    base = time_ms(lambda: F.silu(g) * u)
    ours = time_ms(lambda: swiglu(g, u))
    print(f"{'swiglu':<28}{base:>16.3f}{ours:>14.3f}{base / ours:>9.2f}x")

    # CrossEntropy (note: ktune writes grad in fwd; this is fwd-only latency)
    logits = torch.randn(cfg["tokens"], cfg["vocab"], device=dev, dtype=torch.float32)
    targets = torch.randint(0, cfg["vocab"], (cfg["tokens"],), device=dev)
    base = time_ms(lambda: F.cross_entropy(logits, targets))
    ours = time_ms(lambda: cross_entropy(logits, targets))
    print(f"{'cross_entropy':<28}{base:>16.3f}{ours:>14.3f}{base / ours:>9.2f}x")


if __name__ == "__main__":
    main()
