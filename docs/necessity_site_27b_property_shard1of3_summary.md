# Rank-k Guard v2 (fresh rows) - Job 462076 - shard 1 of 3

Output JSON: `docs/necessity_site_27b_property_shard1of3.json`
Rows: 16 prepared from 16 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| correct_meanablate_nongold_L30 | 0.555 | -0.242 [-0.406, -0.094] | correct_unhinted_baseline |
| correct_meanablate_random_L30 | 0.484 | -0.312 [-0.469, -0.164] | correct_unhinted_baseline |
| correct_unhinted_baseline | 0.797 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Claim 8 survives if pooled rank4_loo or rank8_loo CI excludes zero and reaches >=70% of the pooled in-job L30_concept_replace effect. A null concept_replace on fresh rows scopes the compact-core claim to recognition-gap-style rows rather than failing the guard.
