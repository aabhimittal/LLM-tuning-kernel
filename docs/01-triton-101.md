# 01 · Triton 101

[Triton](https://triton-lang.org) lets you write GPU kernels in Python. You think
in terms of **blocks of data**, not individual threads; Triton compiles your code
to efficient PTX and handles the within-block parallelism for you. It's the sweet
spot for this repo: close enough to the metal to teach the concepts, high-level
enough that the kernels read like the math.

## The mental model

A Triton kernel is a function that runs many times in parallel. You launch a
**grid** of *programs*; each program gets an id and is responsible for one tile of
the output.

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                 # which program am I?
    offs = pid * BLOCK + tl.arange(0, BLOCK)   # the elements I own
    mask = offs < n                        # guard the ragged tail
    x = tl.load(x_ptr + offs, mask=mask)   # HBM -> SRAM
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)  # SRAM -> HBM

# launch: one program per BLOCK-sized chunk
grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
add_kernel[grid](x, y, out, n, BLOCK=1024)
```

## The five things in (almost) every kernel here

1. **`tl.program_id(axis)`** — this program's coordinate in the grid. We usually
   map *one program to one row* (a token, a sequence position, a query block).
2. **`tl.arange(0, BLOCK)` + a base offset** — the indices this program touches.
   `BLOCK` is a `tl.constexpr` (known at compile time) so the compiler can
   vectorise.
3. **`mask`** — because real sizes aren't multiples of `BLOCK`, every load/store
   is masked so out-of-range lanes are skipped (`other=` gives a fill value).
4. **`tl.load` / `tl.store`** — the *only* HBM traffic. Minimising these is the
   whole point (see [00 · Why kernels](00-why-kernels.md)).
5. **Reductions** — `tl.sum`, `tl.max` along an axis, done on-chip. These power
   RMSNorm, softmax, and cross-entropy.

## Autograd: kernels need a backward

PyTorch won't differentiate through a raw kernel. We wrap each op in a
`torch.autograd.Function`: `forward` calls the kernel(s) and saves what's needed;
`backward` calls the gradient kernel(s). See `src/ktune/ops/rmsnorm.py` for the
canonical example, and [02 · RMSNorm](02-rmsnorm.md) for the math.

## How *this repo* is set up to learn

Every op exists **twice**:

- a pure-PyTorch **reference** in `src/ktune/utils/reference.py` — the readable
  math, runs on CPU, and is the correctness oracle;
- a **Triton kernel** in `src/ktune/ops/` — the same math, tiled and fused.

The public function dispatches to the kernel on CUDA and the reference on CPU, so
you can read them side by side and run the whole library without a GPU. Tests
compare kernel-vs-reference on a GPU (`pytest -m gpu`) and reference-vs-truth on
CPU (the default).

## Two more tools you'll meet

- **`@triton.autotune`** — let Triton try several `BLOCK`/`num_warps` configs and
  cache the fastest for each input shape.
- **`triton.next_power_of_2(n)`** — pick a `BLOCK` that covers a row of width `n`.

Now pick an op and read its doc + its two implementations. Start with RMSNorm.

Next: [02 · RMSNorm](02-rmsnorm.md).
