# Qwen vs Gemma Experiment Coverage

This is the working coverage audit for the active goal: mirror the Gemma Stage 2
experiments on Qwen as far as the current Qwen/Qwen Scope artifacts allow.

## Current Status

As of 2026-05-23, Qwen3.5-27B has completed the core Gemma-style behavioral,
L45 raw-probe, residual-SAE, reconstruction/error, forced-choice, patching, and
raw layer-scan passes. The raw layer scan over Qwen layers 16/31/40/53 is
complete for S1 and S3, with L53 strongest for both tasks on both splits. The
matching dtype sanity, L53 Qwen Scope residual-SAE feature extraction, L45/L53
raw free-form steering, answer-property steering, L45/L53 Qwen Scope
sparse-bundle steering, L53 local feature-stability audits, and dependent L53
probe/reconstruction diagnostics have completed.

## Coverage Matrix

| Gemma experiment family | Qwen status | Qwen artifacts / notes |
|---|---|---|
| Stage 1 inference for `infer_property` and `infer_subtype` | Complete | `results/full/with_errortype/qwen35_27b_infer_property.jsonl`, `results/full/with_errortype/qwen35_27b_infer_subtype.jsonl`; both have 11,000 rows. |
| Stage 1 validation / merge | Complete | `docs/qwen35_27b_inference_merge_summary.json`; Qwen thinking preambles are validated away by `scripts/stage2_validate_qwen_outputs.py`. |
| S1/S3 split reuse | Complete | Qwen rows are attached to the repo's `results/stage2/splits.jsonl`; S2 remains non-evaluable for the same topology reason as Gemma. |
| Metadata B0 baselines | Complete | `docs/qwen35_27b_b0_summary.json`; best B0 AUCs are much weaker than raw activations. |
| Raw L45 residual extraction | Complete | `results/stage2/activations/qwen35_27b_{task}_L45.*`. |
| Raw L45 S1/S3 probes | Complete | `docs/qwen_scope_raw_probe_27b_l45_s1.json`, `docs/qwen_scope_raw_probe_27b_l45_s3.json`. |
| Label-shuffle control | Complete for S1 | `docs/qwen_scope_raw_probe_27b_l45_s1_label_shuffle.json`; Qwen stays near chance. |
| Cross-task transfer | Complete | `docs/qwen_scope_raw_probe_transfer_27b_l45_s1.json`, `docs/qwen_scope_raw_probe_transfer_27b_l45_s3.json`. |
| Metadata residualization | Complete | `docs/qwen_scope_raw_probe_metadata_residualization_27b_l45.json`; raw score retains substantial signal after residualizing against `b0_namefreq`. |
| Residual SAE sparse probes | Complete for Qwen Scope residual SAEs | `docs/qwen_scope_sae_probe_27b_l45_w80k_l0_50_{s1,s3}.json` and `docs/qwen_scope_sae_probe_27b_l45_w80k_l0_100_{s1,s3}.json`. |
| Residual SAE reconstruction/error diagnostic | Complete | `docs/qwen_scope_reconstruction_probe_27b_l45_w80k_l0_50_{s1,s3}.json` and `docs/qwen_scope_reconstruction_probe_27b_l45_w80k_l0_100_{s1,s3}.json`. |
| Raw layer scan over `{16,31,40,53}` | Complete | Slurm job `456259`; reports `docs/qwen_scope_raw_probe_27b_layers_16_31_40_53_s1.json` and `docs/qwen_scope_raw_probe_27b_layers_16_31_40_53_s3.json` landed. L53 is best on S1 property/subtype (`0.9400`/`0.9205`) and S3 property/subtype (`0.9325`/`0.9148`). |
| Residual sparse probes at non-L45 layers | Complete for L53 S1/S3 | Raw scan makes L53 the natural follow-up. Qwen Scope publishes `layer53.sae.pt` for W80K L0_50 and L0_100; feature extraction job `456264` completed for both tasks/SAEs and probe/reconstruction job `456265` completed. Full S1 reports match the local audit: raw L53 property/subtype test AUC `0.9400`/`0.9205`, L0_100 sparse `0.8687`/`0.8576`, and L0_50 sparse `0.8363`/`0.8397`. S3 raw L53 is `0.9325`/`0.9148`; S3 L0_100 sparse is `0.8356`/`0.8537`, and S3 L0_50 sparse is `0.7846`/`0.8322`. L53 S1 L0_50 reconstruction/error reached reconstruction AUC `0.8397`/`0.8428`, error AUC `0.9374`/`0.9213`, and energy explained `0.6761`/`0.6738`; L0_100 reconstruction/error reached `0.8760`/`0.8641`, error AUC `0.9337`/`0.9183`, and energy `0.7313`/`0.7284`. S3 L0_50 reconstruction/error reached `0.7866`/`0.8330`, error AUC `0.9269`/`0.9125`, and energy `0.6761`/`0.6738`; S3 L0_100 reconstruction/error reached `0.8462`/`0.8550`, error AUC `0.9260`/`0.9127`, and energy `0.7313`/`0.7284`. |
| Sparse feature-family concat | Pending / artifact-limited | Gemma concat mixes residual SAEs, exact MLP-output SAE, and exact transcoders. A live HF inventory check found Qwen-owned residual SAE repos but no Qwen transcoder/crosscoder/MLP-SAE repos, so a one-for-one concat needs new Qwen artifact support or local training. |
| Exact MLP-output SAE | Not directly mirrored / no published Qwen artifact found | No Qwen-side MLP-output SAE adapter/artifact is currently wired in this repo, and Qwen-owned HF searches for `MLP`/`SAE-MLP` did not return matching artifact repos. |
| Exact skip-transcoder / affine transcoder | Not directly mirrored / no published Qwen artifact found | Gemma used Gemma Scope 2 transcoders; Qwen-owned HF search for `transcoder` returned no matching artifact repos, so the branch currently targets residual Qwen Scope SAEs only. |
| Crosscoder pilot | Not directly mirrored / no published Qwen artifact found | Gemma used a Gemma Scope crosscoder over layers 16/31/40/53; Qwen-owned HF search for `crosscoder` returned no matching artifact repos. |
| Dense-active sparse scaling check | Complete for Qwen residual SAEs | `docs/qwen_scope_dense_active_sae_probe_27b_l45_s1.json`, `docs/qwen_scope_dense_active_sae_probe_27b_l45_s3.json`; dense-active AUCs track standard sparse probes. |
| bf16-vs-fp32 sparse encoding sanity | Complete | `docs/qwen_scope_sparse_dtype_sanity_27b.json`; fp32 first-512 re-encoding exactly matched L0 counts, active-set Jaccard was about `0.991`-`0.995`, and top-1 feature match rate was `0.994`-`1.000`. |
| Feature mini-dashboard / Neuronpedia audit | Complete for Qwen residual SAE local audit at L45 and L53 | L45 reports: `docs/qwen_scope_feature_stability_27b_l45_w80k_l0_50_s1.json`, `docs/qwen_scope_feature_stability_27b_l45_w80k_l0_100_s1.json`, and `docs/qwen_scope_feature_mini_dashboard_27b_l45_w80k_l0_100_top8.md`. L53 reports: `docs/qwen_scope_feature_stability_27b_l53_w80k_l0_50_s1.json`, `docs/qwen_scope_feature_stability_27b_l53_w80k_l0_100_s1.json`, and `docs/qwen_scope_feature_mini_dashboard_27b_l53_w80k_l0_100_top8.md`. L53 L0_100 improves the sparse local fit and has stronger task overlap, but its top shared candidates still mix dense correctness/depth features with a few low-height sparse features. |
| Free-form steering | Complete L45 and L53 raw-direction smokes | `docs/qwen35_raw_steering_27b_l45_property_pilot.json` and `docs/qwen35_raw_steering_27b_l53_property_pilot.json`; the L45 property direction was predictive (S1 test AUC `0.9179`) and L53 was stronger (S1 test AUC `0.9400`, projection std `0.5555`), but neither caused flips on the 4-row smoke. Baseline, raw +/-1sd, and orthogonal +/-1sd all stayed at strong accuracy `0.5`, weak accuracy `1.0`, parse-fail rate `0.0`, with zero paired output-correctness changes. |
| Answer-property steering | Complete L45 answer-polarity smoke | `docs/qwen35_answer_property_steering_27b_l45_polarity_smoke.json`; the L45 gold-polarity direction had val/test AUC `1.0000`/`1.0000`, but caused no answer-content changes or strong-accuracy flips on the 8-row smoke. All conditions stayed at strong accuracy `0.5`, weak accuracy `1.0`, parse-fail rate `0.0`, polarity/predicate match rate `1.0`. |
| Sparse-probe bundle steering | Complete at L45 and L53 | `docs/qwen35_scope_sparse_bundle_steering_27b_l45_l0_100_property_smoke.json` and `docs/qwen35_scope_sparse_bundle_steering_27b_l53_l0_100_property_smoke.json`; `scripts/stage2_qwen_steer_sparse_probe_bundle.py` ports Gemma sparse-probe decoder-bundle steering to Qwen HF hooks using Qwen Scope W80K L0_100 decoder columns, and `scripts/stage2_qwen35_27b_sparse_bundle_property_smoke.sbatch` is layer/task-aware. Slurm job `456263` trained the L45 sparse probe with test AUC `0.8019`; job `456267` trained the stronger L53 sparse probe with test AUC `0.8687`. Both tested 8 balanced h3/h4 rows. Baseline, bundle +/-0.5sd, shuffled, random, and orthogonal controls all stayed at strong accuracy `0.5`, weak accuracy `1.0`, parse-fail rate `0.0`, with zero output correctness changes versus baseline. |
| Hard-foil forced choice | Complete | `docs/qwen35_27b_infer_subtype_h4_hardfoil_forced_choice.json`; Qwen recovered 43/64 selected h4 subtype free-form failures under forced choice. |
| Full-residual clean-to-corrupt patching | Complete pilot | `docs/qwen35_27b_infer_subtype_clean_to_corrupt_patching_margin_pilot.json`; weak/null single-site effect. |
| Full-residual corrupt-to-clean patching | Complete pilot | `docs/qwen35_27b_infer_subtype_corrupt_to_clean_patching_margin_pilot.json`; weak/null single-site effect. |

## Next Decision Points

1. Treat the completed Qwen residual-SAE causal mirror as matching the Gemma
   predictive-versus-causal pattern: L53 improves readout strength but not
   steering control.
2. The direct Gemma MLP-output SAE, transcoder, and crosscoder mirrors need
   additional Qwen artifact support or local Qwen dictionary training before
   they are one-for-one comparable.
