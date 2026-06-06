# Gemma Manifest Decode Commitment Analysis

Generated: `2026-06-06T16:28:49.943340+00:00`

Purpose: analyze the completed manifest-selected Gemma decode trace run at row level before launching another GPU job.

## Summary

- Rows: `14` free-form-wrong, hard-foil-recognition-correct Gemma property rows.
- Regenerated strong accuracy: `3/14` (21.4%); parse failures: `1/14`.
- Prompt gold-vs-foil margin stayed below zero on `13/14` rows.
- Selected-vs-gold prompt margin was nonnegative on `13/13` parsed rows.
- L45/L53 prompt-trained correctness z stayed strongly negative through decoding for both regenerated-correct and regenerated-wrong rows.

Interpretation: this is stronger evidence that the prompt-trained correctness direction is not a decode-time commitment monitor. The useful next measurement is prefix-conditioned margin scoring: ask when the generated/selected hypothesis becomes more likely than gold as the output prefix accumulates.

## Aggregate Metrics

| metric | value |
| --- | --- |
| strong accuracy | 21.4% |
| weak accuracy | 64.3% |
| parse fail rate | 7.1% |
| gold-vs-foil margin mean | -11.650 |
| gold-vs-foil below 0 | 13/14 |
| selected-vs-gold margin mean | 17.318 |
| selected-vs-gold nonnegative | 13/13 |

## Decode Projection Summary

| layer | decode z mean | z<0 | wrong-row mean | correct-row mean | first decode mean | last decode mean |
| --- | --- | --- | --- | --- | --- | --- |
| L45 | -3.799 | 97.6% | -3.749 | -3.978 | -3.241 | -4.201 |
| L53 | -3.302 | 98.0% | -3.238 | -3.535 | -3.965 | -4.689 |

## Row-Level Pattern

| row | example | h | strong | parse fail | gold-vs-foil | selected-vs-gold | L45 mean z | L53 mean z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3073 | property_h3_00073 | 3 | False | False | -20.501 | 17.075 | -3.734 | -3.210 |
| 3290 | property_h3_00290 | 3 | True | False | 64.931 | 0.000 | -4.288 | -3.780 |
| 3415 | property_h3_00415 | 3 | False | False | -4.034 | 7.339 | -3.163 | -2.764 |
| 4322 | property_h3_01322 | 3 | True | False | -14.906 | 15.243 | -3.939 | -3.567 |
| 4675 | property_h3_01675 | 3 | False | False | -19.346 | 19.455 | -3.761 | -3.457 |
| 5292 | property_h3_02292 | 3 | False | True | -18.613 | NA | -3.679 | -4.212 |
| 6188 | property_h4_00188 | 4 | False | False | -28.312 | 28.312 | -3.519 | -3.243 |
| 6327 | property_h4_00327 | 4 | False | False | -4.445 | 4.445 | -4.248 | -3.255 |
| 8035 | property_h4_02035 | 4 | False | False | -22.326 | 22.326 | -4.361 | -3.766 |
| 8298 | property_h4_02298 | 4 | False | False | -24.994 | 27.686 | -4.299 | -3.346 |
| 8874 | property_h4_02874 | 4 | False | False | -18.807 | 22.251 | -3.820 | -3.083 |
| 9549 | property_h4_03549 | 4 | False | False | -26.689 | 26.689 | -3.694 | -2.959 |
| 10079 | property_h4_04079 | 4 | False | False | -12.238 | 21.518 | -2.966 | -2.326 |
| 10714 | property_h4_04714 | 4 | True | False | -12.828 | 12.800 | -3.709 | -3.257 |

## Next Job Recommendation

Run `prefix_conditioned_margin_trajectory_gemma_manifest` before a Qwen decode trace. Use the same 14 Gemma manifest rows and score `gold_hypothesis`, `hard_foil_hypothesis`, and `selected_hypothesis` after generated-prefix checkpoints `[0, 1, 4, 8, 16, 32, 64, 'final']`.

Primary readout: the first prefix checkpoint where selected-vs-gold becomes positive, and whether it stays positive on regenerated-wrong rows. This directly tests `commitment_state`; the current prompt-trained z readout does not.

## Causal-Abstraction Claim

The manifest-selected Gemma rows preserve the recognition-vs-generation gap: forced-choice recognition was correct by construction, but regenerated free-form strong correctness remained low. Prompt-trained correctness projections stay negative through decoding and do not distinguish regenerated-correct rows. Prompt gold-vs-foil scoring remains wrong on almost all rows, while the selected hypothesis is at least as prompt-likely as gold on every parsed row; this points toward testing prefix-conditioned margins rather than reusing the prompt-trained z gate.

