"""Tiny benchmarking helpers shared by the ``bench_*`` scripts.

Measures wall-clock latency (CUDA-event timed, with warmup) and peak GPU memory
for a callable. Everything degrades gracefully on CPU so the scripts at least
*run* without a GPU — they just print a notice instead of real GPU numbers.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable

import torch


def cuda_available() -> bool:
    return torch.cuda.is_available()


def time_ms(fn: Callable[[], object], warmup: int = 10, iters: int = 50) -> float:
    """Median latency of ``fn`` in milliseconds."""
    if not cuda_available():
        # CPU fallback: rough timing, no events.
        import time

        for _ in range(warmup):
            fn()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(times)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends, strict=True))


def peak_memory_mb(fn: Callable[[], object]) -> float:
    """Peak CUDA memory (MB) used while running ``fn`` once. 0.0 on CPU."""
    if not cuda_available():
        fn()
        return 0.0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def device() -> str:
    return "cuda" if cuda_available() else "cpu"


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    if not cuda_available():
        print("[!] No CUDA device — numbers below are CPU and NOT representative.")
        print("    Run on a GPU (e.g. free Colab T4) for meaningful results.\n")
