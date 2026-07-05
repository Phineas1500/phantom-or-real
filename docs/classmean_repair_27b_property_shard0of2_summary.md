# Rank-k Guard v2 (fresh rows) - Job 458418 - shard 0 of 2

Output JSON: `docs/classmean_repair_27b_property_shard0of2.json`
Rows: 13 prepared from 16 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| class_mean_proj_add_L30 | 0.356 | +0.240 [+0.087, +0.433] | unhinted_baseline |
| class_mean_raw_add_L30 | 0.240 | +0.125 [-0.154, +0.394] | unhinted_baseline |
| hinted_baseline | 0.875 | +0.760 [+0.538, +0.942] | unhinted_baseline |
| rand_norm_add_L30_d1 | 0.077 | -0.038 [-0.106, +0.000] | unhinted_baseline |
| rank8_loo_add_L30 | 0.308 | +0.192 [+0.048, +0.375] | unhinted_baseline |
| unhinted_baseline | 0.115 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 11.

Reading rule: Gate: rank8_loo CI excludes zero. Natural-delta CAUSAL if class_mean_raw CI excludes zero AND paired (class_mean_raw - rand_norm_d1) CI excludes zero. Channel dissociation read per docs/causal_handle_directions.md item F(ii). Exploratory: no current-paper claim moves.
