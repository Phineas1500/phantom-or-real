# Hint-Delta Program — Verdict (Job 456999)

Pre-registered analysis of `docs/hint_delta_27b_property_manifest.json` (672
generations, 1h41 batched) against the outcome grid in
`docs/causal_handle_directions.md`. Geometry uses
`hint_delta_27b_property_manifest_deltas.npz` and the saved INLP stacks.

## Arms (row-paired bootstrap, 14 rows, k=8)

| arm | P(strong) | dP vs baseline (95% CI) |
| --- | --- | --- |
| baseline | 0.161 | — |
| delta_add_s1 | 0.152 | -0.009 [-0.107, +0.089] |
| delta_add_s2 | 0.143 | -0.018 [-0.161, +0.143] |
| **subset_concept** | **0.411** | **+0.250 [+0.107, +0.420]** |
| subset_random | 0.205 | +0.045 [-0.009, +0.098] |
| cross_row_delta | 0.188 | +0.027 [-0.018, +0.080]; targets donor concept 0.027 |

## Outcome-grid bin: delta_add(-) / subset_concept(+)

**The focus state is positionally localized but high-rank: patchable spans,
no single summarizing vector.** Patching only the concept-mention token
positions reproduces about half of the full-patch repair (+0.250 vs +0.491
in job 456990, same rows), while matched-size random position subsets do
nothing. The per-layer mean delta is causally null at scales 1-2, and
cross-row transplant is inert (donor-concept targeting 0.027) — the deltas
are context-bound, not transportable concept pointers.

## Geometry

- **Outside the readable subspace.** Projection of row and mean deltas onto
  the full 9-direction INLP readable subspace: 0.009-0.021, below the 0.041
  chance level for random vectors at d=5376; cosine to the canonical probe
  axis |cos| <= 0.008 (chance ~0.014). The hint-conditioned shift is
  geometrically disjoint from everything the correctness probes can read —
  the strongest version of the gauge/lever dissociation.
- **Shared-but-inert component.** Per-row deltas cluster (pairwise cos
  0.23-0.38, far above chance), yet that shared component is exactly what
  mean-delta adds, and it repairs nothing. The causal content is the
  row-specific, position-specific residual at concept mentions. The
  pre-registered geometric/behavioral cross-check thus disagrees in the
  informative direction: geometry shows a common hint signature, behavior
  shows only the specific content matters.

## Scope and owed analyses (per pre-registration)

- Null branch licenses only "rank-1 insufficient at tested scales": a wider
  scale sweep and a rank-k PCA add over per-position deltas are owed before
  any "not low-rank" sentence.
- The conditional random-add control was NOT triggered (its trigger was a
  positive delta_add); subset_concept carries its own matched control
  (subset_random, null).
- Necessity remains open: erase-delta-from-hinted (does the 1.000 collapse?)
  pairs with the KV transplant as the next cluster job.

## One-line thesis update

The model's focus is stored where the concept is mentioned, not as a movable
direction — and none of it lives in the subspace the probes read.
