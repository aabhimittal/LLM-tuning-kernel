# 03 · RoPE — rotary position embeddings

**Code:** `src/ktune/ops/rope.py` · **Reference:** `apply_rope`

RoPE injects position information by *rotating* pairs of channels by an angle that
grows with the token's position. Crucially, the dot product of a rotated query and
a rotated key depends only on their **relative** distance — which is why RoPE
generalises across positions so well.

## The math (rotate-half layout)

Split the head dimension in half. With per-position `cos`/`sin` vectors:

```
rotate_half([x1, x2]) = [-x2, x1]
y = x * cos + rotate_half(x) * sin
```

Written per half (the form the kernel uses):

```
y1 = x1*cos1 - x2*sin1
y2 = x2*cos2 + x1*sin2
```

This is a pure **element-wise** op — no reduction. So like SwiGLU it is
memory-bound, and the kernel's job is simply to read each row of `q` and `k` once
and write it once, folding the four multiply-adds together.

## Kernel design

- **One program per (batch, head, position) row.** The sequence position selects
  which `cos`/`sin` row to use (`seq_idx = row % seq_len`).
- The row is split into its two halves on chip; we load `x1`, `x2` and the matching
  cos/sin halves, compute `y1`, `y2`, and store.

## The backward is the transpose

Because the forward is a rotation (an orthogonal linear map), the backward is its
transpose — the *opposite* rotation:

```
rotate_half_back([a1, a2]) = [a2, -a1]
dx = dy * cos + rotate_half_back(dy * sin)
```

In the kernel this is the `BACKWARD` branch: same loads, signs flipped to apply
the adjoint. No saved activations are needed beyond `cos`/`sin`.

## Exercises

1. Many implementations duplicate `cos`/`sin` across the two halves
   (`cos1 == cos2`). Specialise the kernel for that case — does it get faster?
2. Fuse RoPE *into* the QKV projection's output write, so q/k are rotated before
   they ever land in HBM.
3. Add support for a position-offset (KV-cache / inference) and verify against the
   reference.

Next: [04 · SwiGLU](04-swiglu.md).
