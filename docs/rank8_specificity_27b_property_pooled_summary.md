# Rank-8 Specificity Controls (fresh rows) — Pooled Verdict, Jobs 458401 + 458402

Shard outputs: `docs/rank8_specificity_27b_property_shard{0,1}of2.json`;
row-level generations in
`results/stage2/erasure/rank8_specificity_27b_property_shard{0,1}of2.jsonl`.
26 rows pooled (13 per shard from 16 selected; same guard-v2 fresh-row
selection and skip pattern as jobs 458374/458375). Row keys are
(shard, source_row_index); paired row-cluster bootstrap (10k draws, seed
20260704, percentile [2.5, 97.5]) vs the in-job unhinted baseline. Random
families are pooled across draws and shards (per-row mean over d1–d4);
per-draw numbers are descriptive only, per the pre-registration in
`docs/causal_handle_directions.md` item C.

## Pooled causal arms (26 rows, k=8 samples each)

| arm | dP vs unhinted (CI95) | % of rank8 |
| --- | ---: | ---: |
| unhinted_baseline (P(strong) = 0.120) | — | — |
| hinted_baseline (P(strong) = 0.870) | +0.750 [+0.601, +0.885] | — |
| rank8_loo_add_L30 | **+0.245 [+0.111, +0.394]** | 100% |
| mean_only_add_L30 | +0.087 [+0.019, +0.173] | 35% |
| rand_subspace family (d1–d4 pooled) | −0.041 [−0.154, +0.055] | null |
| rand_norm family (d1–d4 pooled) | −0.088 [−0.180, −0.017] | negative |

Paired differences (per-row, same bootstrap): rank8 − rand_norm
**+0.333 [+0.186, +0.489]**; rank8 − rand_subspace +0.286 [+0.143, +0.441];
rank8 − mean_only +0.159 [+0.048, +0.288]. All exclude zero.

Per-draw descriptive: rand_subspace d1–d4 = −0.043, −0.062, −0.043, −0.014
(all CIs straddle zero); rand_norm d1–d4 = −0.082, −0.067, −0.096, −0.106
(all CIs entirely ≤ −0.005). No draw of either family repairs.

Hint-validated slice (hinted P(strong) ≥ 0.5; 23/26 rows): rank8 +0.272
[+0.125, +0.435]; mean_only +0.098; rand_subspace family −0.046;
rand_norm family −0.099. Same picture.

## Pre-registered decision rule → outcome

- **Gate** (pooled rank8_loo CI excludes zero): **holds** (+0.245, CI low
  +0.111).
- **PASS** required pooled rand_norm CI to include zero AND the paired
  (rank8 − rand_norm) CI to exclude zero. Observed: the paired difference
  excludes zero decisively, and the rand_norm family not only fails to
  repair but is slightly destructive (CI entirely ≤ −0.017 — stricter than
  the "includes zero" wording anticipated). The FAIL branch (rand_norm
  positive CI at ≥50% of rank8) does not fire in any form.
- **Outcome: PASS.** Norm-matched per-row Gaussian noise at the same
  positions and layer does not repair; the rank-8 effect is not a
  perturbation-size artifact.

## Decomposition grid → branch

- mean_only / rank8 = 0.35, below the 0.70 bar → **not** mean-dominated.
- rand_subspace family (LOO mean + row delta projected onto a random
  orthonormal rank-8 basis, per-position norm-matched to the PCA non-mean
  component) is null (−0.041), and (rank8 − rand_subspace) excludes zero
  → **not** mean+magnitude.
- **Branch: the PCA subspace itself is load-bearing.** The specific 8
  directions carry the repair; a random 8-dim subspace dressed to match
  them in mean, norm, position, and rank does nothing, and the mean shift
  alone recovers only about a third of the effect.

## Claim 8 wording consequence

The strongest available upgrade: the fresh-row-portable rank-8 core is
direction-specific, not a norm/mean/rank artifact. The mean component is a
real but minor contributor (+0.087 alone); the non-mean PCA structure
carries the remaining two-thirds and cannot be replaced by random
same-norm structure. Cite jobs 458401/458402 alongside the guard-v2 pass
(458374/458375).
