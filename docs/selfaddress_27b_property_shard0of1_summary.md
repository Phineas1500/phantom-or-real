# Rank-k Guard v2 (fresh rows) - Job 462046 - shard 0 of 1

Output JSON: `docs/selfaddress_27b_property_shard0of1.json`
Rows: 58 prepared from 58 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| percand_fire_L30 | 0.606 | +0.513 [+0.399, +0.625] | unhinted_baseline |
| unhinted_baseline | 0.093 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Item L rules in docs/causal_handle_directions.md. L0 gates: natural AUC >= 0.75; selection-signal paired (gold - mean nongold) CI > 0. L1: oracle gate first (gold-branch dP CI > 0), then PRIMARY = gauge-select beats baseline AND matched-bestofN (paired CIs > 0), selectors evaluated offline at verdict time. No registered prediction. Exploratory.
