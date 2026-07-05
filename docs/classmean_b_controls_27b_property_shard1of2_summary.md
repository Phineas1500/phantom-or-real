# Rank-k Guard v2 (fresh rows) - Job 458425 - shard 1 of 2

Output JSON: `docs/classmean_b_controls_27b_property_shard1of2.json`
Rows: 13 prepared from 16 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| fixednorm_proj_add_L30 | 0.654 | +0.529 [+0.317, +0.750] | unhinted_baseline |
| shuflabel_proj_add_L30_d1 | 0.067 | -0.058 [-0.154, +0.000] | unhinted_baseline |
| shuflabel_proj_add_L30_d2 | 0.154 | +0.029 [-0.029, +0.115] | unhinted_baseline |
| shuflabel_proj_add_L30_d3 | 0.067 | -0.058 [-0.163, +0.019] | unhinted_baseline |
| shuflabel_proj_add_L30_d4 | 0.000 | -0.125 [-0.308, +0.000] | unhinted_baseline |
| signflip_proj_add_L30 | 0.000 | -0.125 [-0.317, +0.000] | unhinted_baseline |
| unhinted_baseline | 0.125 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Integrity gate: in-job unhinted_baseline must reproduce F(ii)'s per-row outcomes. LABEL-SPECIFIC if paired (F(ii) real_proj - shuflabel family) CI excludes zero AND family < 50% of real_proj. GENERIC if the paired CI includes zero. Sign-flip and fixednorm are riders. Rules in docs/causal_handle_directions.md item F(ii)-b.
