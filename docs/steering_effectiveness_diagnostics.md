# Steering-Effectiveness Diagnostics

This is the Track 2 artifact preflight. It summarizes existing probe, metadata, steering, and LAP/logit-lens artifacts where available.

Entropy/branching/KL predictors, regenerated baselines, matched-noise controls, and positive-control steering are still pending.

## Diagnostic Table

| Model | Task | Site | Raw AUC | Best B0 | Raw-B0 | Metadata-adjusted proxy | LAP profile | Prior steering | Planning regime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 3 27B | `infer_property` | `L45` | 0.897 | 0.743 (`b0_prompt`) | 0.153 | exact; +raw delta=0.154; resid AUC=0.802 | L45 peak 0.685; row 0.685 (first-diff margin) | n=8, raw F->T max=0, raw changed max=0 | `linearly_readable_activation_over_metadata_historical_raw_steering_null` |
| Gemma 3 27B | `infer_subtype` | `L45` | 0.914 | 0.841 (`b0_height`) | 0.073 | exact; +raw delta=0.078; resid AUC=0.684 | L15 peak 0.649; row 0.599 (first-diff margin) | untested | `linear_readout_steerability_untested` |
| Qwen3.5 27B | `infer_property` | `L45` | 0.918 | 0.602 (`b0_namefreq`) | 0.316 | exact; +raw delta=0.317; resid AUC=0.871 | L53 peak 0.452; row 0.421 (first-diff margin) | n=4, raw F->T max=0, raw changed max=0 | `linearly_readable_activation_over_metadata_historical_raw_steering_null` |
| Qwen3.5 27B | `infer_property` | `L53` | 0.940 | 0.602 (`b0_namefreq`) | 0.339 | proxy L45; +raw delta=0.317; resid AUC=0.871 | L53 peak 0.452; row 0.452 (first-diff margin) | n=4, raw F->T max=0, raw changed max=0 | `linearly_readable_activation_over_metadata_historical_raw_steering_null` |
| Qwen3.5 27B | `infer_subtype` | `L45` | 0.901 | 0.675 (`b0_prompt`) | 0.226 | exact; +raw delta=0.226; resid AUC=0.817 | L16 peak 0.514; row 0.410 (first-diff margin) | untested | `strong_activation_over_metadata_steerability_untested` |
| Qwen3.5 27B | `infer_subtype` | `L53` | 0.920 | 0.675 (`b0_prompt`) | 0.246 | proxy L45; +raw delta=0.226; resid AUC=0.817 | L16 peak 0.514; row 0.332 (first-diff margin) | untested | `strong_activation_over_metadata_steerability_untested` |

## Qwen Metadata Takeaway

Qwen strengthens the activation-over-metadata result because Qwen B0 baselines are much weaker than raw readouts.

| Task | Qwen L53 raw AUC | Qwen best B0 AUC | Raw-B0 |
| --- | --- | --- | --- |
| `infer_property` | 0.940 | 0.602 | 0.339 |
| `infer_subtype` | 0.920 | 0.675 | 0.246 |

Gemma subtype context: raw AUC=0.914, best B0=0.841, raw-B0=0.073. Gemma subtype remains predictive, but its best metadata baseline is much closer to raw than Qwen's.

## Interpretation

- Existing Gemma and Qwen property steering rows remain historical raw-direction nulls, not final controllability tests.
- Rows with no steering report are predictive-only until regenerated baseline and controls exist.
- Qwen L53 has the strongest activation-over-metadata margin, but its metadata-residualization proxy is currently L45.
- Do not claim `causally distributed` from this table; DAS/distributed interventions with passing controls are required.

## Next Jobs

- Run or refresh GPU LAP/logit-lens profiles for Gemma scanned layers and Qwen L53/scanned layers.
- Regenerate balanced baseline rows before interpreting new causal interventions.
- Run a known positive-control steering task before treating any new steering null as evidence.
- Add matched Gaussian/noise controls and orthogonal controls to optimized/DAS/decode-time intervention jobs.
