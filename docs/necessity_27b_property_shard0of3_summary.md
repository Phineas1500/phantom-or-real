# Item K Necessity Ablation (naturally-correct rows) - Job 459836 - shard 0 of 3

Output JSON: `docs/necessity_27b_property_shard0of3.json`
Rows: 15 prepared from 16 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| correct_ablate_perm8_gold_L30 | 0.750 | -0.033 [-0.133, +0.042] | correct_unhinted_baseline |
| correct_ablate_rand8_gold_L30_d1 | 0.700 | -0.083 [-0.150, -0.017] | correct_unhinted_baseline |
| correct_ablate_rand8_gold_L30_d2 | 0.692 | -0.092 [-0.267, +0.058] | correct_unhinted_baseline |
| correct_ablate_rank8_gold_L30 | 0.092 | -0.692 [-0.858, -0.500] | correct_unhinted_baseline |
| correct_fixednorm_100 | 0.975 | +0.192 [+0.033, +0.375] | correct_unhinted_baseline |
| correct_rand_norm_gold_d1 | 0.417 | -0.367 [-0.533, -0.200] | correct_unhinted_baseline |
| correct_rand_norm_gold_d2 | 0.433 | -0.350 [-0.575, -0.142] | correct_unhinted_baseline |
| correct_signflip_fixednorm_100 | 0.050 | -0.733 [-0.883, -0.558] | correct_unhinted_baseline |
| correct_signflip_fixednorm_200 | 0.000 | -0.783 [-0.925, -0.617] | correct_unhinted_baseline |
| correct_unhinted_baseline | 0.783 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Item K rules in docs/causal_handle_directions.md. Gates: anchor arms verbatim vs 458431; pooled correct baseline >= 0.55; parse-fail < 5% per arm (>20% voids the arm). All branches scored on dP(strong) with unparsed = not-strong. K-PRIMARY branch partition (channel-in-use / write-only port / projection-damage / breaks-specificity-unresolved / inverse-specificity / catch-all) over (ablate x paired x rand8) sign-status; perm8 is the flag layer. Prediction (i), the item's ONLY registered prediction: signflip_100 CI < 0 AND paired (signflip_100 - rand_norm family) CI < 0. Exploratory; no section-1 claim moves.
