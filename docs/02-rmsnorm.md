# 02 · RMSNorm — your first kernel

**Code:** `src/ktune/ops/rmsnorm.py` · **Reference:** `rms_norm` in
`src/ktune/utils/reference.py`

RMSNorm is the normalisation used by Llama/Qwen/Mistral. It's the perfect first
kernel: the math is three lines, but it teaches the **reduction** pattern that
underlies softmax, LayerNorm, and cross-entropy.

## The math

For a row `x` of width `H` and a learned gain vector `w`:

```
rms(x) = sqrt(mean(x_i^2) + eps)
y_i    = x_i / rms(x) * w_i
```

No mean subtraction, no bias (that's the difference from LayerNorm). The only
coupling between columns is the single mean-of-squares reduction.

## Kernel design

- **One program per row.** `program_id(0)` is the row index; the program loads the
  whole row into SRAM, does the reduction on-chip, and writes the row back. One
  HBM read, one HBM write — versus several for a naive `pow → mean → rsqrt → mul`.
- We cache `1/rms` (`rstd`) from the forward pass; backward needs it and
  recomputing is wasteful.

## The backward (worth understanding once)

With `x̂ = x / rms` and `y = x̂ * w`:

```
dL/dw_i = sum_over_rows( dy_i * x̂_i )
dL/dx_i = (1/rms) * ( dy_i * w_i  -  x̂_i * mean_j(dy_j * w_j * x̂_j) )
```

That second term is the correction because every output depends on the row's
mean-square — change one input and you nudge the normaliser for the whole row.
Each program computes its row's `dx` and a *partial* `dw`; the per-row `dw`
partials are summed afterwards with a cheap `torch.sum` (a reduction across rows,
which a single program can't do alone).

## Try it

```python
import torch
from ktune.ops import rms_norm
x = torch.randn(8, 2048, device="cuda", dtype=torch.float16, requires_grad=True)
w = torch.ones(2048, device="cuda", dtype=torch.float16, requires_grad=True)
rms_norm(x, w).sum().backward()   # runs the Triton fwd + bwd kernels
```

On CPU the exact same call runs the reference. Compare the two files line by line.

## Exercises

1. Fuse an optional `residual` add (`x = x + residual`) into the kernel so the
   residual stream never round-trips through HBM.
2. Add `@triton.autotune` over `num_warps ∈ {1,2,4,8}` and plot latency vs `H`.
3. Implement LayerNorm by adding the mean-subtraction — what extra reduction and
   what extra backward term appear?

Next: [03 · RoPE](03-rope.md).
