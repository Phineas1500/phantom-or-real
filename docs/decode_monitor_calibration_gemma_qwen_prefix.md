# Decode Monitor Calibration

Generated: `2026-06-06T23:42:34.619087+00:00`

Purpose: calibrate a conservative prompt-margin gate for the next decode-time intervention, using the completed Gemma prefix-conditioned trajectory and Qwen comparison as predictive evidence.

## Recommendation

Use `gold_vs_foil_logprob_margin lt -15.0` as the first prompt-margin gated intervention trigger. On the Gemma manifest checkpoint-0 rows it triggers `8/11` regenerated-wrong rows and `0/3` regenerated-correct rows.

This is a planning gate, not a manuscript-level statistical claim. It is calibrated on only 14 Gemma recognition-gap rows and should be interpreted through matched raw, orthogonal, and Gaussian intervention outcomes.

## Prompt-Margin Candidates

| candidate | triggered | wrong triggered | correct triggered | precision | recall | specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gold_vs_foil_lt_0` | 13/14 | 11/11 | 2/3 | 0.846 | 1.000 | 0.333 |
| `gold_vs_foil_lt_neg1` | 13/14 | 11/11 | 2/3 | 0.846 | 1.000 | 0.333 |
| `gold_vs_foil_lt_neg2` | 13/14 | 11/11 | 2/3 | 0.846 | 1.000 | 0.333 |
| `gold_vs_foil_lt_neg5` | 11/14 | 9/11 | 2/3 | 0.818 | 0.818 | 0.333 |
| `gold_vs_foil_lt_neg10` | 11/14 | 9/11 | 2/3 | 0.818 | 0.818 | 0.333 |
| `gold_vs_foil_lt_neg15` | 8/14 | 8/11 | 0/3 | 1.000 | 0.727 | 1.000 |
| `gold_vs_foil_lt_neg20` | 5/14 | 5/11 | 0/3 | 1.000 | 0.455 | 1.000 |
| `selected_vs_gold_ge_0` | 13/14 | 10/11 | 3/3 | 0.769 | 0.909 | 0.000 |
| `selected_vs_gold_ge_1` | 12/14 | 10/11 | 2/3 | 0.833 | 0.909 | 0.333 |
| `selected_vs_gold_ge_2` | 12/14 | 10/11 | 2/3 | 0.833 | 0.909 | 0.333 |
| `selected_vs_gold_ge_5` | 11/14 | 9/11 | 2/3 | 0.818 | 0.818 | 0.333 |
| `selected_vs_gold_ge_10` | 10/14 | 8/11 | 2/3 | 0.800 | 0.727 | 0.333 |
| `selected_vs_gold_ge_15` | 9/14 | 8/11 | 1/3 | 0.889 | 0.727 | 0.667 |
| `selected_vs_gold_ge_20` | 6/14 | 6/11 | 0/3 | 1.000 | 0.545 | 1.000 |

## Historical Raw-Z Gate Context

The earlier raw-projection decode gate is retained as a negative calibration example: `z < 0` fired on nearly every decode trajectory and did not separate regenerated-correct from regenerated-wrong outputs.

| layer | z gate | rows triggered | wrong triggered | correct triggered | mean decode trigger fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| L45 | `zlt_0` | 14 | 11 | 3 | 0.976 |
| L45 | `zlt_neg1` | 14 | 11 | 3 | 0.931 |
| L45 | `zlt_neg2` | 14 | 11 | 3 | 0.856 |
| L53 | `zlt_0` | 14 | 11 | 3 | 0.980 |
| L53 | `zlt_neg1` | 14 | 11 | 3 | 0.928 |
| L53 | `zlt_neg2` | 14 | 11 | 3 | 0.807 |

## Qwen Context

Qwen is used as cross-model predictive support only: its h4 subtype subset has `selected>=gold` on `14/14` rows and `gold>=foil` on `0/14` rows at checkpoint 0. It is not a matched intervention rowset.

## Next Run

Run `scripts/stage2_prompt_margin_gated_decode_correction_27b_L45_property_manifest.sbatch` with the recommended threshold. Interpret a repair claim only if raw false-to-true repairs exceed matched Gaussian noise and orthogonal controls under the existing preflight criteria.

## Causal-Abstraction Claim

Predictive calibration only. It selects a prompt-margin gate over `gold_vs_foil_margin` for a future decode-time intervention on `commitment_state`; it does not itself intervene. A causal claim requires false-to-true repairs above orthogonal and matched-Gaussian controls.
