# Rank-k Guard v2 (fresh rows) - Job 458419 - shard 1 of 2

Output JSON: `docs/classmean_repair_27b_property_shard1of2.json`
Rows: 13 prepared from 16 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| class_mean_proj_add_L30 | 0.567 | +0.442 [+0.231, +0.663] | unhinted_baseline |
| class_mean_raw_add_L30 | 0.087 | -0.038 [-0.221, +0.087] | unhinted_baseline |
| hinted_baseline | 0.865 | +0.740 [+0.519, +0.923] | unhinted_baseline |
| rand_norm_add_L30_d1 | 0.010 | -0.115 [-0.279, +0.000] | unhinted_baseline |
| rank8_loo_add_L30 | 0.423 | +0.298 [+0.087, +0.538] | unhinted_baseline |
| unhinted_baseline | 0.125 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 12.

Reading rule: Gate: rank8_loo CI excludes zero. Natural-delta CAUSAL if class_mean_raw CI excludes zero AND paired (class_mean_raw - rand_norm_d1) CI excludes zero. Channel dissociation read per docs/causal_handle_directions.md item F(ii). Exploratory: no current-paper claim moves.
