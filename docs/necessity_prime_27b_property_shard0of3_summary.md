# Item K' Energy-Matched Necessity Controls - Job 459849 - shard 0 of 3

Output JSON: `docs/necessity_prime_27b_property_shard0of3.json`
Rows: 15 prepared from 16 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| correct_ablate_dose012_gold_L30 | 0.675 | -0.108 [-0.192, -0.042] | correct_unhinted_baseline |
| correct_ablate_rank1_gold_L30 | 0.200 | -0.583 [-0.783, -0.367] | correct_unhinted_baseline |
| correct_ablate_rank2_gold_L30 | 0.158 | -0.625 [-0.808, -0.433] | correct_unhinted_baseline |
| correct_ablate_rank4_gold_L30 | 0.067 | -0.717 [-0.883, -0.517] | correct_unhinted_baseline |
| correct_ablate_rank8_gold_L30 | 0.092 | -0.692 [-0.858, -0.500] | correct_unhinted_baseline |
| correct_keeponly8_gold_L30 | 0.000 | -0.783 [-0.933, -0.608] | correct_unhinted_baseline |
| correct_meanablate_gold_L30 | 0.000 | -0.783 [-0.925, -0.617] | correct_unhinted_baseline |
| correct_statepca8_ablate_gold_L30 | 0.000 | -0.783 [-0.925, -0.617] | correct_unhinted_baseline |
| correct_unhinted_baseline | 0.783 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Item K' rules in docs/causal_handle_directions.md. Gates: baseline + ablate_rank8 verbatim vs the matching item-K shard; parse and baseline gates per item K. K'-PRIMARY partition on (meanablate sign x paired (meanablate - ablate_rank8) sign): content-necessity / partial / energy-account / catch-all. statepca8, keep-only, ladder, dose012 are descriptive riders (statepca8 interpretable per registered pre-rule, overlap 0.472; no MEAN-FAR flag, 0.175). No registered prediction. Exploratory; resolves only item K's deferred wording.
