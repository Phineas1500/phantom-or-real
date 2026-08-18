# Rank-k Guard v2 (fresh rows) - Job 462067 - shard 1 of 2

Output JSON: `docs/sjoc_m2_27b_property_shard1of2.json`
Rows: 21 prepared from 23 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| fixednorm_proj_add_L30 | 0.518 | +0.464 [+0.292, +0.637] | unhinted_baseline |
| unhinted_baseline | 0.054 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Claim 8 survives if pooled rank4_loo or rank8_loo CI excludes zero and reaches >=70% of the pooled in-job L30_concept_replace effect. A null concept_replace on fresh rows scopes the compact-core claim to recognition-gap-style rows rather than failing the guard.
