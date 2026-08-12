# Item K' Energy-Matched Necessity Controls - Job 459850 - shard 1 of 3

Output JSON: `docs/necessity_prime_27b_property_shard1of3.json`
Rows: 16 prepared from 16 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| correct_ablate_dose012_gold_L30 | 0.648 | -0.148 [-0.250, -0.055] | correct_unhinted_baseline |
| correct_ablate_rank1_gold_L30 | 0.180 | -0.617 [-0.773, -0.445] | correct_unhinted_baseline |
| correct_ablate_rank2_gold_L30 | 0.078 | -0.719 [-0.852, -0.570] | correct_unhinted_baseline |
| correct_ablate_rank4_gold_L30 | 0.062 | -0.734 [-0.859, -0.586] | correct_unhinted_baseline |
| correct_ablate_rank8_gold_L30 | 0.133 | -0.664 [-0.812, -0.500] | correct_unhinted_baseline |
| correct_keeponly8_gold_L30 | 0.000 | -0.797 [-0.898, -0.688] | correct_unhinted_baseline |
| correct_meanablate_gold_L30 | 0.000 | -0.797 [-0.891, -0.688] | correct_unhinted_baseline |
| correct_statepca8_ablate_gold_L30 | 0.000 | -0.797 [-0.898, -0.695] | correct_unhinted_baseline |
| correct_unhinted_baseline | 0.797 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Item K' rules in docs/causal_handle_directions.md. Gates: baseline + ablate_rank8 verbatim vs the matching item-K shard; parse and baseline gates per item K. K'-PRIMARY partition on (meanablate sign x paired (meanablate - ablate_rank8) sign): content-necessity / partial / energy-account / catch-all. statepca8, keep-only, ladder, dose012 are descriptive riders (statepca8 interpretable per registered pre-rule, overlap 0.472; no MEAN-FAR flag, 0.175). No registered prediction. Exploratory; resolves only item K's deferred wording.
