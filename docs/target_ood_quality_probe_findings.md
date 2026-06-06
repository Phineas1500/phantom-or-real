# Graded Quality-Score Probe Findings

Generated from `docs/target_ood_quality_probe_27b_main.json`.

## Summary

This run tests whether raw activations read out the graded `quality_score` target, not just binary strong correctness. It uses Ridge regression over Gemma 3 27B L45 and Qwen3.5 27B L53 residual activations, with `alpha in {10, 100, 1000, 10000}` selected by validation Spearman and 1000 bootstrap samples for test intervals.

Main result: direct graded quality prediction is dominated by the binary strong-correctness baseline. In every model/task/split, the strong-label mean baseline has higher quality Spearman than raw activation regression. However, raw activations still predict quality residuals after subtracting the strong-label baseline. That residual signal is modest for Gemma and substantially stronger for Qwen.

## All Splits

Spearman values are test correlations. Bracketed values are bootstrap 95% intervals.

| Model/task | Split | Raw quality Spearman | Strong baseline Spearman | Weak baseline Spearman | Raw residual-after-strong Spearman | Raw R2 | Strong baseline R2 | Residual raw R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma property | s1 | 0.688 [0.661, 0.714] | 0.922 | 0.827 | 0.475 [0.430, 0.515] | 0.485 | 0.935 | 0.265 |
| Gemma property | s3 | 0.677 [0.649, 0.703] | 0.929 | 0.825 | 0.448 [0.406, 0.490] | 0.461 | 0.932 | 0.217 |
| Gemma property | height_h12_to_h34 | 0.441 [0.421, 0.461] | 0.864 | 0.805 | 0.342 [0.323, 0.362] | 0.056 | 0.887 | 0.078 |
| Gemma subtype | s1 | 0.616 [0.581, 0.651] | 0.748 | 0.767 | 0.472 [0.430, 0.512] | 0.533 | 0.891 | 0.252 |
| Gemma subtype | s3 | 0.605 [0.568, 0.640] | 0.731 | 0.746 | 0.452 [0.409, 0.494] | 0.519 | 0.862 | 0.230 |
| Gemma subtype | height_h12_to_h34 | 0.214 [0.191, 0.237] | 0.525 | 0.678 | 0.270 [0.250, 0.291] | 0.040 | 0.636 | -0.019 |
| Qwen property | s1 | 0.684 [0.658, 0.709] | 0.954 | 0.344 | 0.692 [0.662, 0.720] | 0.450 | 0.727 | 0.488 |
| Qwen property | s3 | 0.663 [0.633, 0.690] | 0.959 | 0.340 | 0.685 [0.655, 0.713] | 0.439 | 0.727 | 0.499 |
| Qwen property | height_h12_to_h34 | 0.560 [0.544, 0.576] | 0.939 | 0.379 | 0.625 [0.610, 0.639] | 0.237 | 0.699 | 0.143 |
| Qwen subtype | s1 | 0.653 [0.623, 0.681] | 0.893 | 0.584 | 0.677 [0.646, 0.709] | 0.430 | 0.641 | 0.516 |
| Qwen subtype | s3 | 0.641 [0.612, 0.670] | 0.878 | 0.608 | 0.626 [0.588, 0.662] | 0.409 | 0.646 | 0.498 |
| Qwen subtype | height_h12_to_h34 | 0.478 [0.460, 0.495] | 0.842 | 0.652 | 0.566 [0.548, 0.583] | 0.185 | 0.558 | 0.298 |

## Height Extrapolation Only

| Model/task | Split | Raw quality Spearman | Strong baseline Spearman | Weak baseline Spearman | Raw residual-after-strong Spearman | Raw R2 | Strong baseline R2 | Residual raw R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma property | height_h12_to_h34 | 0.441 [0.421, 0.461] | 0.864 | 0.805 | 0.342 [0.323, 0.362] | 0.056 | 0.887 | 0.078 |
| Gemma subtype | height_h12_to_h34 | 0.214 [0.191, 0.237] | 0.525 | 0.678 | 0.270 [0.250, 0.291] | 0.040 | 0.636 | -0.019 |
| Qwen property | height_h12_to_h34 | 0.560 [0.544, 0.576] | 0.939 | 0.379 | 0.625 [0.610, 0.639] | 0.237 | 0.699 | 0.143 |
| Qwen subtype | height_h12_to_h34 | 0.478 [0.460, 0.495] | 0.842 | 0.652 | 0.566 [0.548, 0.583] | 0.185 | 0.558 | 0.298 |

## Interpretation

- The graded score is not a better replacement target for the current paper's main raw probe. Strong correctness alone explains graded quality better than raw activation regression in direct prediction.
- The residual probe is still informative: after removing the strong-label mean, raw activations predict remaining quality variation with Spearman about 0.45-0.48 for Gemma S1/S3 and 0.63-0.69 for Qwen S1/S3.
- Height transfer remains weaker for Gemma, especially subtype. Qwen retains stronger residual quality signal under height extrapolation: property residual Spearman 0.625 and subtype 0.566.
- This supports keeping `is_correct_strong` as the main target while mentioning that graded quality contains additional raw-readable structure, particularly in Qwen.

## Next Checks

- Do not replace the main paper target with graded quality. Use this as a limitation/appendix result unless we decide to build a separate parsimony-focused section.
- If pursuing parsimony further, the next defensible version is a residual-quality analysis conditioned on strong correctness and height, not a binary `quality_score == 1.0` probe.
