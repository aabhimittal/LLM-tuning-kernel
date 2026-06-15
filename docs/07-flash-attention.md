# 07 · FlashAttention — tiling + 2-D online softmax

**Code:** `src/ktune/ops/flash_attention.py` · **Reference:** `attention`

This is the centerpiece. It combines everything: tiling, the online softmax from
[05](05-cross-entropy.md), and careful IO. Read those first.

## The problem: the score matrix is quadratic

Standard attention:

```
S = Q @ K.T * scale      # [seq, seq]   <-- quadratic in sequence length
P = softmax(S)           # [seq, seq]
O = P @ V                # [seq, head_dim]
```

That `[seq, seq]` matrix `S`/`P` is the bottleneck. At seq = 8192 it's 67M
elements *per head* — too big to keep in SRAM, expensive to push through HBM, and
the reason naive attention OOMs on long context.

## The fix: never form the full matrix

FlashAttention tiles the computation so the score matrix only exists one
`[BLOCK_M, BLOCK_N]` tile at a time, and uses an **online softmax over key
blocks** to get an exact result without seeing a full row at once.

Each program owns a block of `BLOCK_M` queries. It keeps three running quantities
in SRAM — the row max `m`, the softmax denominator `l`, and the partial output
`acc` — and streams key/value blocks through:

```
for each key block (K_j, V_j):
    S_ij   = Q_i @ K_j.T * scale         # one small tile
    m_new  = max(m, rowmax(S_ij))
    P_ij   = exp(S_ij - m_new)
    alpha  = exp(m - m_new)              # how much to rescale what we have so far
    l      = l * alpha + rowsum(P_ij)
    acc    = acc * alpha + P_ij @ V_j    # rescale + accumulate
    m      = m_new
O_i = acc / l                            # finalise the denominator at the end
```

`alpha` is the magic: when a later key block raises the running max, it
retroactively rescales the partial output and denominator so the math stays
exactly equal to a full softmax. The `[seq, seq]` matrix is **never** written to
HBM. Memory becomes linear in sequence length.

## Causal masking, cheaply

For causal attention, query block `i` can only attend to key positions `≤` its
last row, so the key loop stops at the diagonal (`end_n = (start_m+1)*BLOCK_M`).
That alone halves the work; within the boundary tile we mask `col > row`.

## What this repo ships

The **forward kernel** is implemented in full — it's the part worth learning, and
it's checked against the reference on GPU in `tests/test_kernels_gpu.py`.

The **backward** currently recomputes gradients via autograd on the reference
formula. That is numerically correct and a clean baseline, but it re-materialises
the score matrix, so it is *not* memory-optimal. Fusing the backward is the
natural capstone exercise (below).

## Exercises

1. Store the per-row logsumexp `L = m + log(l)` from the forward (the kernel
   already computes `m` and `l`) — you'll need it for a fused backward.
2. Implement the FlashAttention-2 **backward** in Triton: compute `D = rowsum(dO ∘
   O)`, then a `dK`/`dV` loop and a `dQ` loop, all tiled. Verify against the
   reference and benchmark the memory drop vs the recompute path.
3. Add support for non-power-of-two `head_dim` via masking on the `D` axis.
4. Add a sliding-window / block-sparse mask and skip key blocks entirely outside
   the window.

Next: [08 · Applying to models](08-applying-to-models.md).
