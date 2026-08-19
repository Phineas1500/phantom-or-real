# Rank-k Guard v2 (fresh rows) - Job 462362 - shard 0 of 1

Output JSON: `docs/fr1_rank6rem_27b_property_shard0of1.json`
Rows: 26 prepared from 32 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| fixednorm_proj_add_L30 | 0.567 | +0.447 [+0.288, +0.611] | unhinted_baseline |
| fixednorm_rank6rem_L30 | 0.606 | +0.486 [+0.317, +0.649] | unhinted_baseline |
| unhinted_baseline | 0.120 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Claim 8 survives if pooled rank4_loo or rank8_loo CI excludes zero and reaches >=70% of the pooled in-job L30_concept_replace effect. A null concept_replace on fresh rows scopes the compact-core claim to recognition-gap-style rows rather than failing the guard.
