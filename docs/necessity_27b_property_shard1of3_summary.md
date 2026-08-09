# Item K Necessity Ablation (naturally-correct rows) - Job 459837 - shard 1 of 3

Output JSON: `docs/necessity_27b_property_shard1of3.json`
Rows: 16 prepared from 16 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| correct_ablate_perm8_gold_L30 | 0.727 | -0.070 [-0.125, -0.023] | correct_unhinted_baseline |
| correct_ablate_rand8_gold_L30_d1 | 0.586 | -0.211 [-0.328, -0.102] | correct_unhinted_baseline |
| correct_ablate_rand8_gold_L30_d2 | 0.750 | -0.047 [-0.172, +0.070] | correct_unhinted_baseline |
| correct_ablate_rank8_gold_L30 | 0.133 | -0.664 [-0.812, -0.500] | correct_unhinted_baseline |
| correct_fixednorm_100 | 0.898 | +0.102 [+0.000, +0.203] | correct_unhinted_baseline |
| correct_rand_norm_gold_d1 | 0.383 | -0.414 [-0.586, -0.234] | correct_unhinted_baseline |
| correct_rand_norm_gold_d2 | 0.469 | -0.328 [-0.492, -0.164] | correct_unhinted_baseline |
| correct_signflip_fixednorm_100 | 0.055 | -0.742 [-0.860, -0.609] | correct_unhinted_baseline |
| correct_signflip_fixednorm_200 | 0.000 | -0.797 [-0.898, -0.688] | correct_unhinted_baseline |
| correct_unhinted_baseline | 0.797 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Item K rules in docs/causal_handle_directions.md. Gates: anchor arms verbatim vs 458431; pooled correct baseline >= 0.55; parse-fail < 5% per arm (>20% voids the arm). All branches scored on dP(strong) with unparsed = not-strong. K-PRIMARY branch partition (channel-in-use / write-only port / projection-damage / breaks-specificity-unresolved / inverse-specificity / catch-all) over (ablate x paired x rand8) sign-status; perm8 is the flag layer. Prediction (i), the item's ONLY registered prediction: signflip_100 CI < 0 AND paired (signflip_100 - rand_norm family) CI < 0. Exploratory; no section-1 claim moves.
