# Rank-k Guard v2 (fresh rows) - Job 461632 - shard 3 of 8

Output JSON: `docs/selfaddress_27b_property_shard3of8.json`
Rows: 7 prepared from 7 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| matched_bestofN_unsteered | 0.168 | -0.011 [-0.031, +0.007] | unhinted_baseline |
| percand_fire_L30 | 0.152 | -0.026 [-0.083, +0.014] | unhinted_baseline |
| unhinted_baseline | 0.161 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Item L rules in docs/causal_handle_directions.md. L0 gates: natural AUC >= 0.75; selection-signal paired (gold - mean nongold) CI > 0. L1: oracle gate first (gold-branch dP CI > 0), then PRIMARY = gauge-select beats baseline AND matched-bestofN (paired CIs > 0), selectors evaluated offline at verdict time. No registered prediction. Exploratory.
