# Rank-k Guard v2 (fresh rows) - Job 461630 - shard 2 of 4

Output JSON: `docs/selfaddress_27b_property_shard2of4.json`
Rows: 12 prepared from 12 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| matched_bestofN_unsteered | 0.051 | +0.025 [-0.019, +0.094] | unhinted_baseline |
| percand_fire_L30 | 0.046 | +0.021 [-0.031, +0.094] | unhinted_baseline |
| unhinted_baseline | 0.021 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Item L rules in docs/causal_handle_directions.md. L0 gates: natural AUC >= 0.75; selection-signal paired (gold - mean nongold) CI > 0. L1: oracle gate first (gold-branch dP CI > 0), then PRIMARY = gauge-select beats baseline AND matched-bestofN (paired CIs > 0), selectors evaluated offline at verdict time. No registered prediction. Exploratory.
