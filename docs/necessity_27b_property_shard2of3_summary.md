# Item K Necessity Ablation (naturally-correct rows) - Job 459838 - shard 2 of 3

Output JSON: `docs/necessity_27b_property_shard2of3.json`
Rows: 15 prepared from 16 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| correct_ablate_perm8_gold_L30 | 0.700 | -0.008 [-0.058, +0.033] | correct_unhinted_baseline |
| correct_ablate_rand8_gold_L30_d1 | 0.700 | -0.008 [-0.058, +0.042] | correct_unhinted_baseline |
| correct_ablate_rand8_gold_L30_d2 | 0.725 | +0.017 [-0.058, +0.108] | correct_unhinted_baseline |
| correct_ablate_rank8_gold_L30 | 0.067 | -0.642 [-0.842, -0.425] | correct_unhinted_baseline |
| correct_fixednorm_100 | 0.900 | +0.192 [-0.033, +0.417] | correct_unhinted_baseline |
| correct_rand_norm_gold_d1 | 0.400 | -0.308 [-0.500, -0.133] | correct_unhinted_baseline |
| correct_rand_norm_gold_d2 | 0.375 | -0.333 [-0.558, -0.125] | correct_unhinted_baseline |
| correct_signflip_fixednorm_100 | 0.100 | -0.608 [-0.808, -0.400] | correct_unhinted_baseline |
| correct_signflip_fixednorm_200 | 0.000 | -0.708 [-0.892, -0.500] | correct_unhinted_baseline |
| correct_unhinted_baseline | 0.708 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Item K rules in docs/causal_handle_directions.md. Gates: anchor arms verbatim vs 458431; pooled correct baseline >= 0.55; parse-fail < 5% per arm (>20% voids the arm). All branches scored on dP(strong) with unparsed = not-strong. K-PRIMARY branch partition (channel-in-use / write-only port / projection-damage / breaks-specificity-unresolved / inverse-specificity / catch-all) over (ablate x paired x rand8) sign-status; perm8 is the flag layer. Prediction (i), the item's ONLY registered prediction: signflip_100 CI < 0 AND paired (signflip_100 - rand_norm family) CI < 0. Exploratory; no section-1 claim moves.
