# Rank-k Guard v2 (fresh rows) - Job 461633 - shard 7 of 8

Output JSON: `docs/selfaddress_27b_property_shard7of8.json`
Rows: 8 prepared from 8 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| matched_bestofN_unsteered | 0.022 | -0.004 [-0.013, +0.000] | unhinted_baseline |
| percand_fire_L30 | 0.044 | +0.022 [+0.000, +0.067] | unhinted_baseline |
| unhinted_baseline | 0.031 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Item L rules in docs/causal_handle_directions.md. L0 gates: natural AUC >= 0.75; selection-signal paired (gold - mean nongold) CI > 0. L1: oracle gate first (gold-branch dP CI > 0), then PRIMARY = gauge-select beats baseline AND matched-bestofN (paired CIs > 0), selectors evaluated offline at verdict time. No registered prediction. Exploratory.
