# Rank-k Guard v2 (fresh rows) - Job 458431 - shard 0 of 1

Output JSON: `docs/classmean_c_deployment_27b_property_shard0of1.json`
Rows: 42 prepared from 48 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| correct_fixednorm_add_L30 | 0.992 | +0.266 [+0.086, +0.469] | correct_unhinted_baseline |
| correct_unhinted_baseline | 0.727 | - | none |
| fixednorm_allpos_add_L30 | 0.149 | +0.029 [-0.053, +0.120] | unhinted_baseline |
| fixednorm_proj_add_L30 | 0.567 | +0.447 [+0.293, +0.611] | unhinted_baseline |
| unhinted_baseline | 0.120 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: POSITION-FREE if fixednorm_allpos CI excludes zero AND >=50% of fixednorm_proj (paired). COLLATERAL-SAFE if correct-side dP point >= -0.10 AND CI low >= -0.20; HARMFUL if CI entirely below -0.10. Rules in docs/causal_handle_directions.md item F(ii)-c. Exploratory.
