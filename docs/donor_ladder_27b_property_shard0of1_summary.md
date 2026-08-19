# Rank-k Guard v2 (fresh rows) - Job 462363 - shard 0 of 1

Output JSON: `docs/donor_ladder_27b_property_shard0of1.json`
Rows: 57 prepared from 64 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| fixednorm_donor10_L30 | 0.510 | +0.333 [+0.135, +0.552] | unhinted_baseline |
| fixednorm_donor20_L30 | 0.656 | +0.479 [+0.260, +0.708] | unhinted_baseline |
| fixednorm_donor40_L30 | 0.656 | +0.479 [+0.240, +0.719] | unhinted_baseline |
| fixednorm_donor5_L30 | 0.479 | +0.302 [+0.094, +0.531] | unhinted_baseline |
| unhinted_baseline | 0.177 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Claim 8 survives if pooled rank4_loo or rank8_loo CI excludes zero and reaches >=70% of the pooled in-job L30_concept_replace effect. A null concept_replace on fresh rows scopes the compact-core claim to recognition-gap-style rows rather than failing the guard.
