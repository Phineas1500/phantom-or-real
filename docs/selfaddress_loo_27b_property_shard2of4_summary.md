# Rank-k Guard v2 (fresh rows) - Job 462044 - shard 2 of 4

Output JSON: `docs/selfaddress_loo_27b_property_shard2of4.json`
Rows: 16 prepared from 16 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| percand_loo_fire_L30 | 0.132 | +0.042 [-0.004, +0.081] | unhinted_baseline |
| unhinted_baseline | 0.102 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Item L rules in docs/causal_handle_directions.md. L0 gates: natural AUC >= 0.75; selection-signal paired (gold - mean nongold) CI > 0. L1: oracle gate first (gold-branch dP CI > 0), then PRIMARY = gauge-select beats baseline AND matched-bestofN (paired CIs > 0), selectors evaluated offline at verdict time. No registered prediction. Exploratory.
