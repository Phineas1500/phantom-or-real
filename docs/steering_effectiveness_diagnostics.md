# Steering-Effectiveness Diagnostics

This is the Track 2 artifact preflight. It summarizes existing probe, metadata, steering, and LAP/logit-lens artifacts where available.

LAP/logit-lens artifacts are complete for the target rows. Entropy/branching/KL predictors are still pending; confidence/effect correlations are available for the regenerated Gemma property row and require sidecar fields for historical or untested rows.

Gemma L45 raw/error steering and the first optimized-vector intervention now have regenerated baselines, positive-control coverage, and matched Gaussian controls; Qwen property steering remains a historical pilot without Gaussian controls.

## Diagnostic Table

| Model | Task | Site | Raw AUC | Best B0 | Raw-B0 | Metadata-adjusted proxy | LAP profile | Prior steering | Planning regime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 3 27B | `infer_property` | `L45` | 0.897 | 0.743 (`b0_prompt`) | 0.153 | exact; +raw delta=0.154; resid AUC=0.802 | L45 peak 0.685; row 0.685 (first-diff margin) | n=8, raw F->T max=0, raw changed max=0; Gaussian ctrl | `linearly_readable_matched_control_raw_steering_null` |
| Gemma 3 27B | `infer_subtype` | `L45` | 0.914 | 0.841 (`b0_height`) | 0.073 | exact; +raw delta=0.078; resid AUC=0.684 | L15 peak 0.649; row 0.599 (first-diff margin) | untested | `linear_readout_steerability_untested` |
| Qwen3.5 27B | `infer_property` | `L45` | 0.918 | 0.602 (`b0_namefreq`) | 0.316 | exact; +raw delta=0.317; resid AUC=0.871 | L53 peak 0.452; row 0.421 (first-diff margin) | n=4, raw F->T max=0, raw changed max=0 | `linearly_readable_low_lap_accessibility_historical_raw_steering_null` |
| Qwen3.5 27B | `infer_property` | `L53` | 0.940 | 0.602 (`b0_namefreq`) | 0.339 | proxy L45; +raw delta=0.317; resid AUC=0.871 | L53 peak 0.452; row 0.452 (first-diff margin) | n=4, raw F->T max=0, raw changed max=0 | `linearly_readable_low_lap_accessibility_historical_raw_steering_null` |
| Qwen3.5 27B | `infer_subtype` | `L45` | 0.901 | 0.675 (`b0_prompt`) | 0.226 | exact; +raw delta=0.226; resid AUC=0.817 | L16 peak 0.514; row 0.410 (first-diff margin) | untested | `strong_activation_over_metadata_low_lap_accessibility_steerability_untested` |
| Qwen3.5 27B | `infer_subtype` | `L53` | 0.920 | 0.675 (`b0_prompt`) | 0.246 | proxy L45; +raw delta=0.226; resid AUC=0.817 | L16 peak 0.514; row 0.332 (first-diff margin) | untested | `strong_activation_over_metadata_low_lap_accessibility_steerability_untested` |

## Optimized-Vector Intervention

Teacher-forced gold-continuation L45 vector on the same balanced Gemma property h3/h4 set is a controlled null.

| Method | Job | Baseline n | Baseline strong acc | Positive control | Optimized F->T max | Optimized changed max | Control F->T max | Control changed max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `optimized_gold_continuation_vector` | `456693` | 8 | 0.375 | passed | 0 | 0 | 0 | 1 |

Claim: Teacher-forced gold-continuation L45 vector produced no false-to-true repair and no optimized-condition accuracy changes on the balanced h3/h4 set; a matched Gaussian control degraded one row.

## Qwen Metadata Takeaway

Qwen strengthens the activation-over-metadata result because Qwen B0 baselines are much weaker than raw readouts.

| Task | Qwen L53 raw AUC | Qwen best B0 AUC | Raw-B0 |
| --- | --- | --- | --- |
| `infer_property` | 0.940 | 0.602 | 0.339 |
| `infer_subtype` | 0.920 | 0.675 | 0.246 |

Gemma subtype context: raw AUC=0.914, best B0=0.841, raw-B0=0.073. Gemma subtype remains predictive, but its best metadata baseline is much closer to raw than Qwen's.

## Control Status

| Row | Gaussian control | Positive control | Confidence/effect correlation |
| --- | --- | --- | --- |
| Gemma 3 27B `infer_property` `L45` | yes | passed | available |
| Gemma 3 27B `infer_subtype` `L45` | untested | passed | missing_steering_rows |
| Qwen3.5 27B `infer_property` `L45` | no | not_configured | needs_probe_score_sidecar |
| Qwen3.5 27B `infer_property` `L53` | no | not_configured | needs_probe_score_sidecar |
| Qwen3.5 27B `infer_subtype` `L45` | untested | not_configured | missing_steering_rows |
| Qwen3.5 27B `infer_subtype` `L53` | untested | not_configured | missing_steering_rows |

## Interpretation

- Gemma property L45 now has matched-control raw-direction, error-subspace, and optimized-vector nulls; rows with no steering report remain predictive-only.
- Qwen property steering pilots still need regenerated matched Gaussian/noise controls before causal interpretation.
- Qwen L53 has the strongest activation-over-metadata margin, but its metadata-residualization proxy is currently L45.
- Probe-confidence vs steering-effect correlation is now available for Gemma property L45; Qwen historical pilots and untested rows still need projection-enabled reruns if we want comparable causal diagnostics.
- The optimized-vector null supports `causally inaccessible under tested methods` more than `causally distributed`; DAS/distributed interventions with passing controls are required before using `causally distributed`.

## Next Jobs

- Gemma property projection-enabled raw/error and optimized-vector reruns are complete on the balanced h3/h4 set.
- Add entropy/branching/KL-style steerability predictors where the required logits are available.
- Refresh Qwen property steering with regenerated baseline, orthogonal, Gaussian, and Qwen positive-control gates if Qwen causal interpretation is needed.
- Move to the next stronger intervention family: DAS/distributed interchange, decode-time correction, or AtP* localization with exact patch validation.
