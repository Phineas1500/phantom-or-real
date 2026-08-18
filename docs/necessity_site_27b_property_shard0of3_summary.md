# Rank-k Guard v2 (fresh rows) - Job 462075 - shard 0 of 3

Output JSON: `docs/necessity_site_27b_property_shard0of3.json`
Rows: 15 prepared from 16 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| correct_meanablate_nongold_L30 | 0.808 | +0.025 [-0.117, +0.167] | correct_unhinted_baseline |
| correct_meanablate_random_L30 | 0.475 | -0.308 [-0.550, -0.050] | correct_unhinted_baseline |
| correct_unhinted_baseline | 0.783 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Claim 8 survives if pooled rank4_loo or rank8_loo CI excludes zero and reaches >=70% of the pooled in-job L30_concept_replace effect. A null concept_replace on fresh rows scopes the compact-core claim to recognition-gap-style rows rather than failing the guard.
