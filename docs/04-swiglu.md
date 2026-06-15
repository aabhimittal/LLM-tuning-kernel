# 04 · SwiGLU — the gated MLP activation

**Code:** `src/ktune/ops/swiglu.py` · **Reference:** `swiglu`

SwiGLU is the activation in the Llama/Qwen/Mistral MLP block. The full block is:

```
MLP(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )      silu(z) = z * sigmoid(z)
```

The middle — `silu(gate) * up` — is what we fuse.

## Why fuse it

Done naively in PyTorch, `F.silu(gate) * up` is **two** kernels: one writes the
`silu(gate)` tensor to HBM, the next reads it straight back to multiply by `up`.
For a 4096-token × 14336-intermediate tensor in fp16 that intermediate is ~117 MB
of pointless round-trip traffic. The fused kernel reads `gate` and `up` once,
computes `silu(gate) * up` in registers, and writes the result once.

It's a pure element-wise op, so this is the simplest possible "fusion saves
bandwidth" demonstration — read [00 · Why kernels](00-why-kernels.md) and this
side by side.

## Kernel design

- Flatten everything to 1-D and give each program a `BLOCK`-sized chunk
  (`program_id(0) * BLOCK + arange(BLOCK)`), masked at the tail.
- Forward: `silu(g) * u` where `silu(g) = g * sigmoid(g)`.

## Backward (one chain-rule step, fused)

With `s = sigmoid(g)`:

```
d(silu)/dg = s * (1 + g*(1 - s))
d_gate = dout * up * d(silu)/dg
d_up   = dout * silu(g)
```

Both gradients are written in a single kernel pass over `gate`, `up`, `dout`.

## Exercises

1. Fuse the *entire* MLP middle including the `down_proj` matmul boundary — what
   can and can't be fused, and why? (Hint: the matmul is compute-bound.)
2. Swap SiLU for GELU and re-derive the backward.
3. Benchmark fused vs unfused at several intermediate sizes; confirm the gap
   tracks bytes-moved, not FLOPs.

Next: [05 · Cross-entropy](05-cross-entropy.md).
