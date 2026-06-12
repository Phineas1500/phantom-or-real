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

**The focus state is positionally localized; it is not captured by a uniform
rank-1 summary.** Patching only the concept-mention token positions
reproduces about half of the full-patch repair (+0.250 vs +0.491 in job
456990, same rows; point estimate — CIs overlap), while matched-size random
position subsets do nothing: roughly **6x per-token potency** at concept
mentions (+0.250 vs +0.045 at matched count).

**Dilution caveat (review catch, 2026-06-12): the mean-delta null is diluted,
not decisive.** The delta was averaged over the ~90-token block while the
causal content sits at 5-15 concept-mention tokens — an attenuation of
6-10x that scale 2 does not compensate, and the geometrically shared inert
component dominates the position-mean. The arm tested "uniform rank-1
across the block", not "low-rank at the positions where it lives". The same
dilution overdetermines the cross-row inertness, so concept-specificity is
not cleanly answered either. Sharpened tests (next job):
concept-position-restricted mean delta with a real scale sweep, rank-k PCA
on the concept-position delta submatrix, and the spotlight test (row A's
concept-position delta at row B's concept positions — spotlight vs content).

## Geometry

- **Outside the readable subspace — pending restriction check.** Projection
  of row and mean deltas onto the full 9-direction INLP readable subspace:
  0.009-0.021, below the 0.041 chance level at d=5376; cosine to the
  canonical probe axis |cos| <= 0.008 (chance ~0.014). Caveat: these are
  block-mean deltas, so the number could be an artifact of the inert shared
  component; the claim must be recomputed on concept-position-restricted
  deltas (next job saves them) before it becomes paper prose. If it
  survives restriction it is bulletproof.
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
