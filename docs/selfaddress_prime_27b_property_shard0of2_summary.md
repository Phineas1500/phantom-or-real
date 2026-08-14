# Rank-k Guard v2 (fresh rows) - Job 461648 - shard 0 of 2

Output JSON: `docs/selfaddress_prime_27b_property_shard0of2.json`
Rows: 27 prepared from 32 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| fixednorm_proj_add_L30 | 0.417 | +0.315 [+0.176, +0.458] | unhinted_baseline |
| unhinted_baseline | 0.102 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Claim 8 survives if pooled rank4_loo or rank8_loo CI excludes zero and reaches >=70% of the pooled in-job L30_concept_replace effect. A null concept_replace on fresh rows scopes the compact-core claim to recognition-gap-style rows rather than failing the guard.
