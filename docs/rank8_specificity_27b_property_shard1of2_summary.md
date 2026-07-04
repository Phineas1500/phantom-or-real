# Rank-8 Specificity Controls (fresh rows) - Job 458402 - shard 1 of 2

Output JSON: `docs/rank8_specificity_27b_property_shard1of2.json`
Rows: 13 prepared from 16 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| hinted_baseline | 0.865 | +0.740 [+0.519, +0.923] | unhinted_baseline |
| mean_only_add_L30 | 0.192 | +0.067 [+0.000, +0.154] | unhinted_baseline |
| rand_norm_add_L30_d1 | 0.000 | -0.125 [-0.308, +0.000] | unhinted_baseline |
| rand_norm_add_L30_d2 | 0.029 | -0.096 [-0.240, +0.000] | unhinted_baseline |
| rand_norm_add_L30_d3 | 0.029 | -0.096 [-0.240, +0.000] | unhinted_baseline |
| rand_norm_add_L30_d4 | 0.000 | -0.125 [-0.317, +0.000] | unhinted_baseline |
| rand_subspace_add_L30_d1 | 0.000 | -0.125 [-0.308, +0.000] | unhinted_baseline |
| rand_subspace_add_L30_d2 | 0.010 | -0.115 [-0.308, +0.000] | unhinted_baseline |
| rand_subspace_add_L30_d3 | 0.000 | -0.125 [-0.308, +0.000] | unhinted_baseline |
| rand_subspace_add_L30_d4 | 0.058 | -0.067 [-0.279, +0.115] | unhinted_baseline |
| rank8_loo_add_L30 | 0.423 | +0.298 [+0.096, +0.529] | unhinted_baseline |
| unhinted_baseline | 0.125 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 12.

Reading rule: Gate: pooled rank8_loo CI must exclude zero. Specificity passes if pooled rand_norm CI includes zero and the paired (rank8 - rand_norm) difference CI excludes zero; fails if rand_norm CI excludes zero at >=50% of the rank8 effect. mean_only and rand_subspace decompose the carrier per the wording grid in docs/causal_handle_directions.md item C.
