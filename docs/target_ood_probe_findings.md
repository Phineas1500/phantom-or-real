# Target/OOD Probe Findings

Generated from:

- `docs/target_ood_preflight.json`
- `docs/target_ood_raw_probe_27b_weak_main_fullc_bootstrap.json`
- `docs/target_ood_raw_probe_27b_strong_height_fullc_bootstrap.json`

## Summary

The target/OOD preflight confirms that `is_correct_weak` is the useful alternate binary target. It differs from strong correctness on thousands of parsed rows, especially for Qwen. In contrast, `quality_score == 1.0` is identical to strong correctness for Qwen and nearly identical for Gemma, so a binary perfect-quality target is mostly a sanity check. A true quality/parsimony test should use the graded score directly.

The full-C validation used `C in {0.01, 0.1, 1.0, 10.0}` with 1000 bootstrap samples. Weak correctness remains linearly readable on S1/S3, but height extrapolation is much weaker than the corresponding strong-correctness height extrapolation. That makes depth/difficulty a live confound for weak validity and argues against treating weak correctness as a drop-in replacement for the paper's strong label.

All validation probes selected `C=0.01`, so the fixed-`C=1.0` first pass should be treated as superseded by these reports.

## Label Audit

| Source | Parsed n | Strong+ | Weak+ | Quality=1+ | Strong!=weak | Strong!=quality=1 |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma property | 10025 | 4610 | 6413 | 4631 | 1803 | 21 |
| Gemma subtype | 10737 | 2243 | 4486 | 2279 | 2243 | 36 |
| Qwen property | 10997 | 6148 | 10559 | 6148 | 4411 | 0 |
| Qwen subtype | 10999 | 4533 | 9352 | 4533 | 4819 | 0 |

## Weak-Correctness Raw Probes

AUCs are test AUC with bootstrap 95% intervals.

| Model / task | S1 AUC | S3 AUC | h1/h2 -> h3/h4 AUC | best C |
| --- | --- | --- | --- | --- |
| Gemma 3 27B L45 / property | 0.842 [0.823, 0.862] | 0.843 [0.824, 0.862] | 0.667 [0.654, 0.680] | 0.01 |
| Gemma 3 27B L45 / subtype | 0.801 [0.778, 0.822] | 0.772 [0.749, 0.796] | 0.623 [0.609, 0.636] | 0.01 |
| Qwen3.5 27B L53 / property | 0.806 [0.746, 0.861] | 0.774 [0.709, 0.830] | 0.649 [0.620, 0.676] | 0.01 |
| Qwen3.5 27B L53 / subtype | 0.857 [0.833, 0.880] | 0.873 [0.851, 0.896] | 0.694 [0.681, 0.708] | 0.01 |

## Height Extrapolation Comparison

| Model / task | Weak correctness AUC | Strong correctness AUC | Strong-minus-weak gap |
| --- | --- | --- | --- |
| Gemma 3 27B L45 / property | 0.667 [0.654, 0.680] | 0.749 [0.737, 0.760] | 0.081 |
| Gemma 3 27B L45 / subtype | 0.623 [0.609, 0.636] | 0.754 [0.735, 0.773] | 0.131 |
| Qwen3.5 27B L53 / property | 0.649 [0.620, 0.676] | 0.863 [0.854, 0.870] | 0.213 |
| Qwen3.5 27B L53 / subtype | 0.694 [0.681, 0.708] | 0.818 [0.809, 0.827] | 0.124 |

## Interpretation

- `is_correct_weak` is readable, but its height generalization is poor. The full-C height AUCs are only 0.623-0.694 for weak correctness.
- Strong correctness generalizes across height substantially better: 0.749/0.754 for Gemma and 0.863/0.818 for Qwen.
- The strong-minus-weak height gap is largest for Qwen property at 0.213, which reinforces that weak correctness is not merely a relaxed version of strong correctness.
- Binary perfect quality does not add much: for Qwen it is exactly the strong label, and for Gemma it only differs on 21 property rows and 36 subtype rows after parse filtering.
- Existing Gemma name-scramble artifacts already show raw-probe degradation under natural/nonce scrambling. No Qwen name-scramble artifacts are present yet.

## Next Checks

- Use the full-C/bootstrap reports for target/OOD numerical claims; keep the fixed-C reports as historical planning artifacts only.
- If we want a real parsimony target, add a graded quality-score regression or ordinal probe instead of `quality_score == 1.0`.
- If cross-model OOD controls become important, generate Qwen name-scramble activations; the current name-scramble OOD evidence is Gemma-only.
