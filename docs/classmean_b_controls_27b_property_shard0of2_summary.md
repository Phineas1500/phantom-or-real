# Rank-k Guard v2 (fresh rows) - Job 458424 - shard 0 of 2

Output JSON: `docs/classmean_b_controls_27b_property_shard0of2.json`
Rows: 13 prepared from 16 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| fixednorm_proj_add_L30 | 0.385 | +0.269 [+0.077, +0.481] | unhinted_baseline |
| shuflabel_proj_add_L30_d1 | 0.067 | -0.048 [-0.173, +0.029] | unhinted_baseline |
| shuflabel_proj_add_L30_d2 | 0.135 | +0.019 [+0.000, +0.058] | unhinted_baseline |
| shuflabel_proj_add_L30_d3 | 0.115 | +0.000 [+0.000, +0.000] | unhinted_baseline |
| shuflabel_proj_add_L30_d4 | 0.010 | -0.106 [-0.269, +0.000] | unhinted_baseline |
| signflip_proj_add_L30 | 0.000 | -0.115 [-0.288, +0.000] | unhinted_baseline |
| unhinted_baseline | 0.115 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Integrity gate: in-job unhinted_baseline must reproduce F(ii)'s per-row outcomes. LABEL-SPECIFIC if paired (F(ii) real_proj - shuflabel family) CI excludes zero AND family < 50% of real_proj. GENERIC if the paired CI includes zero. Sign-flip and fixednorm are riders. Rules in docs/causal_handle_directions.md item F(ii)-b.
