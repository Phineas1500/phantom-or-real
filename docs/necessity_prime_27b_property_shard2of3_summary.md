# Item K' Energy-Matched Necessity Controls - Job 459847 - shard 2 of 3

Output JSON: `docs/necessity_prime_27b_property_shard2of3.json`
Rows: 15 prepared from 16 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| correct_ablate_dose012_gold_L30 | 0.658 | -0.050 [-0.133, +0.025] | correct_unhinted_baseline |
| correct_ablate_rank1_gold_L30 | 0.100 | -0.608 [-0.800, -0.400] | correct_unhinted_baseline |
| correct_ablate_rank2_gold_L30 | 0.067 | -0.642 [-0.842, -0.425] | correct_unhinted_baseline |
| correct_ablate_rank4_gold_L30 | 0.042 | -0.667 [-0.850, -0.467] | correct_unhinted_baseline |
| correct_ablate_rank8_gold_L30 | 0.067 | -0.642 [-0.850, -0.425] | correct_unhinted_baseline |
| correct_keeponly8_gold_L30 | 0.000 | -0.708 [-0.892, -0.500] | correct_unhinted_baseline |
| correct_meanablate_gold_L30 | 0.000 | -0.708 [-0.892, -0.500] | correct_unhinted_baseline |
| correct_statepca8_ablate_gold_L30 | 0.000 | -0.708 [-0.892, -0.500] | correct_unhinted_baseline |
| correct_unhinted_baseline | 0.708 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Item K' rules in docs/causal_handle_directions.md. Gates: baseline + ablate_rank8 verbatim vs the matching item-K shard; parse and baseline gates per item K. K'-PRIMARY partition on (meanablate sign x paired (meanablate - ablate_rank8) sign): content-necessity / partial / energy-account / catch-all. statepca8, keep-only, ladder, dose012 are descriptive riders (statepca8 interpretable per registered pre-rule, overlap 0.472; no MEAN-FAR flag, 0.175). No registered prediction. Exploratory; resolves only item K's deferred wording.
