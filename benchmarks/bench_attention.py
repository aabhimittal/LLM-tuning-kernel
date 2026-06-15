"""Benchmark the FlashAttention forward kernel vs eager attention + SDPA.

    python benchmarks/bench_attention.py --hardware t4 --seq 2048

Shows the quadratic-memory blowup of naive attention against the flat memory of
the tiled FlashAttention kernel as sequence length grows. PyTorch's SDPA (which
itself uses a fused backend) is included as a strong reference point.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from _bench_utils import banner, device, peak_memory_mb, time_ms
from ktune.ops import flash_attention
from ktune.utils.reference import attention as naive_attention

CONFIGS = {
    "t4": dict(batch=2, heads=16, head_dim=64),
    "a100": dict(batch=4, heads=32, head_dim=128),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hardware", choices=list(CONFIGS), default="t4")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    args = ap.parse_args()
    cfg = CONFIGS[args.hardware]
    dev = device()
    dtype = getattr(torch, args.dtype)

    banner(f"FlashAttention fwd — {args.hardware.upper()}, seq={args.seq}")
    shape = (cfg["batch"], cfg["heads"], args.seq, cfg["head_dim"])
    q = torch.randn(shape, device=dev, dtype=dtype)
    k = torch.randn(shape, device=dev, dtype=dtype)
    v = torch.randn(shape, device=dev, dtype=dtype)

    variants = {
        "naive (materialised)": lambda: naive_attention(q, k, v, causal=True),
        "torch SDPA": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
        "ktune flash": lambda: flash_attention(q, k, v, causal=True),
    }
    print(f"{'variant':<24}{'peak mem (MB)':>16}{'fwd (ms)':>12}")
    print("-" * 52)
    for name, fn in variants.items():
        try:
            mem = peak_memory_mb(fn)
            ms = time_ms(fn, warmup=5, iters=20)
            print(f"{name:<24}{mem:>16.1f}{ms:>12.3f}")
        except RuntimeError as e:  # naive attention can OOM at long seq — that's the point
            print(f"{name:<24}{'OOM/err':>16}{'  (' + str(e)[:20] + '...)':>12}")


if __name__ == "__main__":
    main()
