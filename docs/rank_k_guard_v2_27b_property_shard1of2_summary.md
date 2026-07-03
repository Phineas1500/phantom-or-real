# Rank-k Guard v2 (fresh rows) - Job 458375 - shard 1 of 2

Output JSON: `docs/rank_k_guard_v2_27b_property_shard1of2.json`
Rows: 13 prepared from 16 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| L30_concept_replace | 0.423 | +0.298 [+0.087, +0.529] | unhinted_baseline |
| L30_random_replace | 0.173 | +0.048 [-0.010, +0.135] | unhinted_baseline |
| hinted_baseline | 0.865 | +0.740 [+0.519, +0.923] | unhinted_baseline |
| rank4_loo_add_L30 | 0.327 | +0.202 [+0.019, +0.394] | unhinted_baseline |
| rank8_loo_add_L30 | 0.413 | +0.288 [+0.096, +0.510] | unhinted_baseline |
| unhinted_baseline | 0.125 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 12.

Reading rule: pooled-shard rank4/rank8 LOO CI excluding zero at >=70% of the
pooled in-job concept-replace effect confirms claim 8 on fresh rows; a null
concept-replace on fresh rows scopes the compact-core claim to
recognition-gap-style rows rather than failing the guard.
