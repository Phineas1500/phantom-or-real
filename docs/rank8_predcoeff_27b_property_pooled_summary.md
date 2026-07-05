# Predicted-Coefficient Repair (item E) — Pooled Verdict, Jobs 458409 + 458410

Shard outputs: `docs/rank8_predcoeff_27b_property_shard{0,1}of2.json`;
row-level generations in
`results/stage2/erasure/rank8_predcoeff_27b_property_shard{0,1}of2.jsonl`.
26 rows pooled (13/shard; same guard-v2 fresh-row selection as items A/C);
row-cluster bootstrap, 10k draws, seed 20260704, paired vs the in-job
unhinted baseline. All four causal arms share the dev basis (rank-8 PCA of
the 13 composite rows' deltas, EVR 0.685) and differ only in coefficient
source. Exploratory (item E): no current-paper claim moves.

## Pooled arms (26 rows, k=8)

| arm | coefficient source | dP vs unhinted (CI95) |
| --- | --- | ---: |
| unhinted_baseline (P=0.120) | — | — |
| hinted_baseline (P=0.870) | — | +0.750 [+0.601,+0.885] |
| rank8_dev_add_L30 (ceiling) | row's own donor delta | +0.135 [+0.043,+0.245] |
| mean_only_dev_add_L30 (floor) | none (dev mean tiled) | +0.091 [+0.010,+0.197] |
| rank8_pred_add_L30 | ridge from row's UNHINTED states | +0.154 [+0.062,+0.264] |
| rank8_shufpred_add_L30 (control) | shuffled-pairing ridge | +0.139 [+0.062,+0.231] |

Paired differences: pred − mean_only **+0.062 [+0.000,+0.149]** (grazes
zero); pred − shufpred **+0.014 [−0.034,+0.072]** (null); dev − mean_only
+0.043 [−0.029,+0.144] (null).

Predictor diagnostics (in-job, per row): cos(predicted, true coefficients)
= +0.564 shard 0 / +0.608 shard 1 on rows the ridge never saw (every row
positive, range +0.36..+0.84); shuffled control −0.015/−0.027 ≈ 0.
Coefficient scales matched.

## Pre-registered rules → outcome

- **Gate (rank8_dev CI excludes zero): HOLDS** (+0.135, CI low +0.043) —
  the dev basis transfers, so the arms are interpretable.
- **SUCCESS fails**: it required (pred − shufpred) to exclude zero; the
  observed paired difference is +0.014 [−0.034,+0.072].
- **Outcome: FAIL branch** (shufpred ≈ pred): coefficient decodability
  does not convert to behavioral repair at this design. The PARTIAL
  condition (pred − mean_only straddling zero) is also met at the boundary
  (+0.000); either reading gives the same substantive conclusion —
  **the ridge's row-specific information adds nothing behavioral beyond
  generic content in the dev subspace.**

## What the experiment DID establish

1. **The predictor works as a predictor.** Out-of-sample cos ≈ +0.59 vs
   ≈ 0 for the shuffle — step 1's decodability result transfers to fresh
   rows. The information pipeline is not the failure point.
2. **The ceiling collapsed, and that is the binding constraint.** True
   coefficients on the dev basis give +0.135 where the same rows under a
   fresh-row LOO basis gave +0.245 (item C). Basis provenance is
   first-order: the dev basis under-transfers, compressing ceiling, floor,
   pred, and control into an indistinguishable +0.09..+0.15 band with no
   headroom for coefficient quality to matter.
3. **Subspace ≫ coefficients.** Every arm writing into the dev rank-8
   subspace repairs with CI excluding zero (+0.091..+0.154) — including
   shuffled coefficients — while item C's norm-matched random subspaces
   sat at −0.041/−0.088. Which subspace is written into carries the
   effect; the coordinates within it matter far less than the step-1
   cosine suggested. This is the design input for item F: the class-mean
   arm must use a fresh-row basis, and the deployment-relevant question
   becomes "right subspace + right scale," not per-row prediction.

## Wording consequences

- `docs/hint_free_repair_direction.md` step 2 outcome recorded; step 3
  (gated deployment test) does NOT unlock; the ladder folds into item F.
- No claim-table changes (exploratory by pre-registration).
