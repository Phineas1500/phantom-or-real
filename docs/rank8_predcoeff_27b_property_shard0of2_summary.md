# Rank-8 Predicted-Coefficient Repair (fresh rows) - Job 458409 - shard 0 of 2

Output JSON: `docs/rank8_predcoeff_27b_property_shard0of2.json`
Rows: 13 prepared from 16 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| hinted_baseline | 0.875 | +0.760 [+0.548, +0.942] | unhinted_baseline |
| mean_only_dev_add_L30 | 0.192 | +0.077 [+0.000, +0.212] | unhinted_baseline |
| rank8_dev_add_L30 | 0.183 | +0.067 [+0.000, +0.192] | unhinted_baseline |
| rank8_pred_add_L30 | 0.221 | +0.106 [+0.010, +0.240] | unhinted_baseline |
| rank8_shufpred_add_L30 | 0.250 | +0.135 [+0.029, +0.269] | unhinted_baseline |
| unhinted_baseline | 0.115 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 11.

Reading rule: Gate: pooled rank8_dev CI must exclude zero (dev-basis transfer). SUCCESS if pooled rank8_pred CI excludes zero AND paired (pred - mean_only_dev) CI excludes zero AND paired (pred - shufpred) CI excludes zero. PARTIAL if pred excludes zero but (pred - mean_only_dev) straddles zero. FAIL otherwise. Exploratory: no current-paper claim moves; rules in docs/causal_handle_directions.md item E.
