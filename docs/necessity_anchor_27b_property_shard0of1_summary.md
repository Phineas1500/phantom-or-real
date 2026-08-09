# Item K Anchor Replication (guard rows, verbatim gates) - Job 459839 - shard 0 of 1

Output JSON: `docs/necessity_anchor_27b_property_shard0of1.json`
Rows: 26 prepared from 32 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| fixednorm_proj_add_L30 | 0.567 | +0.447 [+0.288, +0.611] | unhinted_baseline |
| unhinted_baseline | 0.120 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Item K rules in docs/causal_handle_directions.md. Gates: anchor arms verbatim vs 458431; pooled correct baseline >= 0.55; parse-fail < 5% per arm (>20% voids the arm). All branches scored on dP(strong) with unparsed = not-strong. K-PRIMARY branch partition (channel-in-use / write-only port / projection-damage / breaks-specificity-unresolved / inverse-specificity / catch-all) over (ablate x paired x rand8) sign-status; perm8 is the flag layer. Prediction (i), the item's ONLY registered prediction: signflip_100 CI < 0 AND paired (signflip_100 - rand_norm family) CI < 0. Exploratory; no section-1 claim moves.
