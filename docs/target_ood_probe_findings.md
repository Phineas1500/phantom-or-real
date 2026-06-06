# Target/OOD Probe Findings

Generated from:

- `docs/target_ood_preflight.json`
- `docs/target_ood_raw_probe_27b_weak_main.json`
- `docs/target_ood_raw_probe_27b_strong_height.json`

## Summary

The target/OOD preflight confirms that `is_correct_weak` is the useful alternate binary target. It differs from strong correctness on thousands of parsed rows, especially for Qwen. In contrast, `quality_score == 1.0` is identical to strong correctness for Qwen and nearly identical for Gemma, so a binary perfect-quality target is mostly a sanity check. A true quality/parsimony test should use the graded score directly.

The first raw-probe pass used fixed `C=1.0` logistic probes as a fast planning run. Weak correctness is linearly readable on S1/S3, but height extrapolation is much weaker than the corresponding strong-correctness height extrapolation. That makes depth/difficulty a live confound for weak validity and argues against treating weak correctness as a drop-in replacement for the paper's strong label.

## Label Audit

| Source | Parsed n | Strong+ | Weak+ | Quality=1+ | Strong!=weak | Strong!=quality=1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma property | 10025 | 4610 | 6413 | 4631 | 1803 | 21 |
| Gemma subtype | 10737 | 2243 | 4486 | 2279 | 2243 | 36 |
| Qwen property | 10997 | 6148 | 10559 | 6148 | 4411 | 0 |
| Qwen subtype | 10999 | 4533 | 9352 | 4533 | 4819 | 0 |

## Weak-Correctness Raw Probes

| Model | Task | S1 AUC | S3 AUC | h1/h2 -> h3/h4 AUC |
| --- | --- | ---: | ---: | ---: |
| Gemma 3 27B L45 | property | 0.808 | 0.802 | 0.644 |
| Gemma 3 27B L45 | subtype | 0.757 | 0.711 | 0.608 |
| Qwen3.5 27B L53 | property | 0.794 | 0.747 | 0.606 |
| Qwen3.5 27B L53 | subtype | 0.822 | 0.836 | 0.655 |

## Strong-Correctness Height Extrapolation

| Model | Task | h1/h2 -> h3/h4 AUC |
| --- | --- | ---: |
| Gemma 3 27B L45 | property | 0.725 |
| Gemma 3 27B L45 | subtype | 0.725 |
| Qwen3.5 27B L53 | property | 0.838 |
| Qwen3.5 27B L53 | subtype | 0.786 |

## Interpretation

- `is_correct_weak` is readable, but its height generalization is poor. This suggests weak correctness mixes correctness with depth/difficulty and should not replace the strong target without a separate analysis.
- Strong correctness generalizes across height substantially better, especially for Qwen. This supports keeping `is_correct_strong` as the main paper target.
- Binary perfect quality does not add much: for Qwen it is exactly the strong label, and for Gemma it only differs on 21 property rows and 36 subtype rows after parse filtering.
- Existing Gemma name-scramble artifacts already show raw-probe degradation under natural/nonce scrambling. No Qwen name-scramble artifacts are present yet.

## Next Checks

- Treat the fixed-`C=1.0` target/OOD probes as planning results. Use the full C grid and bootstrap CIs before manuscript-level claims.
- If we want a real parsimony target, add a graded quality-score regression or ordinal probe instead of `quality_score == 1.0`.
- If cross-model OOD controls become important, generate Qwen name-scramble activations; the current OOD evidence is Gemma-only.
