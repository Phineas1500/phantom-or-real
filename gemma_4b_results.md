# Gemma 3 4B Results

This file backtracks the Gemma 3 4B Phase A, Phase B, and Phase C artifacts currently present in the repo and records the main accuracy/AUC numbers plus the files that contain them.

## Source Index

- `results/full/gemma3_4b_infer_property_runmeta.json`: Stage 1 / Phase A property generation accuracy by height.
- `results/full/gemma3_4b_infer_subtype_runmeta.json`: Stage 1 / Phase A subtype generation accuracy by height.
- `results/stage2/phase_a_verification.json`: Phase A activation artifact verification for L11/L20/L22.
- `results/stage2/baselines/b0_summary_gemma3_4b.json`: Phase B B0 pre-output baselines by task/split/height.
- `results/stage2/probe_comparison.json`: Phase B raw residual probe summary and selected layer comparison.
- `results/stage2/probe_auc.json`: Full Phase B raw residual probe results by layer/method/split.
- `docs/stage2_invariants.4b.semani.json`: Phase C pinned Gemma Scope 2 4B SAE/transcoder releases.
- `docs/sae_reconstruction_probe_4b_l22_s1.json`: Phase C residual SAE reconstruction/error diagnostic on S1.
- `docs/sae_reconstruction_probe_4b_l22_s3_target_symbol.json`: Phase C residual SAE reconstruction/error diagnostic on S3.
- `results/stage2/sae_features/`: Phase C sparse SAE/transcoder feature caches.
- `results/stage2/activations/`: Phase A residual activations and Phase C exact MLP hook activations.

## Phase A: Behavioral Accuracy And Activation Extraction

### Generation Accuracy By Height

| Task | Height | n | Strong accuracy | Parse fail rate | Source |
|---|---:|---:|---:|---:|---|
| infer_property | h1 | 1000 | 0.7400 | 0.0450 | `results/full/gemma3_4b_infer_property_runmeta.json` |
| infer_property | h2 | 2000 | 0.2950 | 0.0345 | `results/full/gemma3_4b_infer_property_runmeta.json` |
| infer_property | h3 | 3000 | 0.2040 | 0.0197 | `results/full/gemma3_4b_infer_property_runmeta.json` |
| infer_property | h4 | 5000 | 0.1640 | 0.0226 | `results/full/gemma3_4b_infer_property_runmeta.json` |
| infer_subtype | h1 | 1000 | 0.7690 | 0.0170 | `results/full/gemma3_4b_infer_subtype_runmeta.json` |
| infer_subtype | h2 | 2000 | 0.1165 | 0.0075 | `results/full/gemma3_4b_infer_subtype_runmeta.json` |
| infer_subtype | h3 | 3000 | 0.0133 | 0.0013 | `results/full/gemma3_4b_infer_subtype_runmeta.json` |
| infer_subtype | h4 | 5000 | 0.0090 | 0.0016 | `results/full/gemma3_4b_infer_subtype_runmeta.json` |

### Activation Artifact Verification

Verification file: `results/stage2/phase_a_verification.json`. Overall status is `failed`; equivalence gate pass is `True`. The file reports spot-check mismatches at L22, but all hard artifact checks below pass and tensors are present with row/order invariants.

| Task | Layer | Rows | Shape | dtype | Example-id order match | Artifact ok |
|---|---:|---:|---|---|---|---|
| infer_property | L11 | 11000 | [11000, 2560] | bfloat16 | True | True |
| infer_property | L20 | 11000 | [11000, 2560] | bfloat16 | True | True |
| infer_property | L22 | 11000 | [11000, 2560] | bfloat16 | True | True |
| infer_subtype | L11 | 11000 | [11000, 2560] | bfloat16 | True | True |
| infer_subtype | L20 | 11000 | [11000, 2560] | bfloat16 | True | True |
| infer_subtype | L22 | 11000 | [11000, 2560] | bfloat16 | True | True |

## Phase B: Baselines And Raw Residual Probes

### B0 Baselines

Source: `results/stage2/baselines/b0_summary_gemma3_4b.json`. These are the strongest pre-output B0 baselines per task/split/height.

| Task | Split | Height | Best variant | AUC |
|---|---|---:|---|---:|
| infer_property | s1 | 1 | b0_namefreq | 0.7022 |
| infer_property | s1 | 2 | b0_namefreq | 0.5718 |
| infer_property | s1 | 3 | b0_namefreq | 0.5993 |
| infer_property | s1 | 4 | b0_namefreq | 0.6478 |
| infer_property | s2 | 3 | b0_namefreq | 0.5823 |
| infer_subtype | s1 | 1 | b0_namefreq | 0.7118 |
| infer_subtype | s1 | 2 | b0_prompt | 0.6157 |
| infer_subtype | s1 | 3 | b0_height | 0.5000 |
| infer_subtype | s1 | 4 | b0_prompt | 0.5994 |
| infer_subtype | s2 | 3 | b0_height | 0.5000 |

### Best Raw Residual Probe Summary

Source: `results/stage2/probe_comparison.json`. These are logistic raw residual probes. Global selected layer is `L22` at depth fraction `0.6471`.

| Split | Task | Best layer | Test AUC | B0 AUC | Delta over B0 |
|---|---|---|---:|---:|---:|
| s1 | infer_property | L22 | 0.9035 | 0.6941 | 0.2093 |
| s1 | infer_subtype | L22 | 0.9741 | 0.9244 | 0.0497 |
| s3 | infer_property | L22 | 0.9060 | 0.6767 | 0.2293 |
| s3 | infer_subtype | L22 | 0.9720 | 0.9453 | 0.0268 |

### Raw Residual Logistic AUC By Layer

Source: `results/stage2/probe_auc.json`. S2 entries are skipped in the source file, so this table records S1 and S3.

| Task | Layer | S1 test AUC | S1 delta vs B0 | S3 test AUC | S3 delta vs B0 |
|---|---|---:|---:|---:|---:|
| infer_property | L11 | 0.7974 | 0.1033 | 0.7894 | 0.1127 |
| infer_property | L20 | 0.8730 | 0.1789 | 0.8774 | 0.2007 |
| infer_property | L22 | 0.9035 | 0.2093 | 0.9060 | 0.2293 |
| infer_subtype | L11 | 0.9310 | 0.0066 | 0.9423 | -0.0030 |
| infer_subtype | L20 | 0.9708 | 0.0465 | 0.9689 | 0.0237 |
| infer_subtype | L22 | 0.9741 | 0.0497 | 0.9720 | 0.0268 |

### Raw Residual Probe Per-Height AUC At Selected Layer L22

| Split | Task | Height | AUC | n | positives | negatives |
|---|---|---:|---:|---:|---:|---:|
| s1 | infer_property | h1 | 0.8416 | 140 | 111 | 29 |
| s1 | infer_property | h2 | 0.8552 | 292 | 89 | 203 |
| s1 | infer_property | h3 | 0.8811 | 442 | 92 | 350 |
| s1 | infer_property | h4 | 0.8957 | 737 | 123 | 614 |
| s1 | infer_subtype | h1 | 0.8430 | 148 | 115 | 33 |
| s1 | infer_subtype | h2 | 0.8900 | 298 | 35 | 263 |
| s1 | infer_subtype | h3 | 0.9774 | 449 | 6 | 443 |
| s1 | infer_subtype | h4 | 0.8310 | 750 | 7 | 743 |
| s3 | infer_property | h1 | 0.7029 | 144 | 108 | 36 |
| s3 | infer_property | h2 | 0.8815 | 283 | 89 | 194 |
| s3 | infer_property | h3 | 0.9032 | 465 | 102 | 363 |
| s3 | infer_property | h4 | 0.9036 | 715 | 137 | 578 |
| s3 | infer_subtype | h1 | 0.5896 | 160 | 127 | 33 |
| s3 | infer_subtype | h2 | 0.8791 | 291 | 34 | 257 |
| s3 | infer_subtype | h3 | 0.9333 | 488 | 5 | 483 |
| s3 | infer_subtype | h4 | 0.9207 | 748 | 4 | 744 |

## Phase C: Gemma Scope 2 4B Sparse Features

### Pinned SAE/Transcoder Releases

Source: `docs/stage2_invariants.4b.semani.json`. Status: `pinned_4b_l20_l22_residual_l22_mlp_skip_transcoder_artifacts`.

| Release entry | d_in | d_sae | d_out | HF subfolder | Snapshot |
|---|---:|---:|---:|---|---|
| `gemma-scope-2-4b-it-mlp-all/layer_22_width_16k_l0_small` | 2560 | 16384 |  | `mlp_out_all` | `3e94b68be95290aada5b7525cf431d3040f81bb1` |
| `gemma-scope-2-4b-it-res-all/layer_20_width_16k_l0_small` | 2560 | 16384 |  | `resid_post_all` | `3e94b68be95290aada5b7525cf431d3040f81bb1` |
| `gemma-scope-2-4b-it-res-all/layer_20_width_262k_l0_small` | 2560 | 262144 |  | `resid_post_all` | `3e94b68be95290aada5b7525cf431d3040f81bb1` |
| `gemma-scope-2-4b-it-res-all/layer_22_width_16k_l0_small` | 2560 | 16384 |  | `resid_post_all` | `3e94b68be95290aada5b7525cf431d3040f81bb1` |
| `gemma-scope-2-4b-it-res-all/layer_22_width_262k_l0_small` | 2560 | 262144 |  | `resid_post_all` | `3e94b68be95290aada5b7525cf431d3040f81bb1` |
| `gemma-scope-2-4b-it-transcoders-all/layer_22_width_16k_l0_small_affine` | 2560 | 16384 | 2560 | `transcoder_all` | `3e94b68be95290aada5b7525cf431d3040f81bb1` |
| `gemma-scope-2-4b-it-transcoders-all/layer_22_width_262k_l0_small_affine` | 2560 | 262144 | 2560 | `transcoder_all` | `3e94b68be95290aada5b7525cf431d3040f81bb1` |

### Feature Cache Metadata

Source directory: `results/stage2/sae_features/`.

| Cache meta file | Rows | top_k | d_in | d_sae | d_out | l0_mean |
|---|---:|---:|---:|---:|---:|---:|
| `results/stage2/sae_features/gemma3_4b_infer_property_L20_layer_20_width_16k_l0_small_top128.meta.json` | 11000 | 128 | 2560 | 16384 |  | 14.2952 |
| `results/stage2/sae_features/gemma3_4b_infer_property_L20_layer_20_width_262k_l0_small_top128.meta.json` | 11000 | 128 | 2560 | 262144 |  | 15.1965 |
| `results/stage2/sae_features/gemma3_4b_infer_property_L22_layer_22_width_16k_l0_small_top128.meta.json` | 11000 | 128 | 2560 | 16384 |  | 20.8171 |
| `results/stage2/sae_features/gemma3_4b_infer_property_L22_layer_22_width_262k_l0_small_top128.meta.json` | 11000 | 128 | 2560 | 262144 |  | 15.6955 |
| `results/stage2/sae_features/gemma3_4b_infer_property_L22_mlp_in_weighted_layer_22_width_16k_l0_small_affine_top128.meta.json` | 11000 | 128 | 2560 | 16384 | 2560 | 15.8360 |
| `results/stage2/sae_features/gemma3_4b_infer_property_L22_mlp_in_weighted_layer_22_width_262k_l0_small_affine_top128.meta.json` | 11000 | 128 | 2560 | 262144 | 2560 | 20.3779 |
| `results/stage2/sae_features/gemma3_4b_infer_property_L22_mlp_out_hook_layer_22_width_16k_l0_small_top128.meta.json` | 11000 | 128 | 2560 | 16384 |  | 44.9506 |
| `results/stage2/sae_features/gemma3_4b_infer_subtype_L20_layer_20_width_16k_l0_small_top128.meta.json` | 11000 | 128 | 2560 | 16384 |  | 13.6453 |
| `results/stage2/sae_features/gemma3_4b_infer_subtype_L20_layer_20_width_262k_l0_small_top128.meta.json` | 11000 | 128 | 2560 | 262144 |  | 14.5786 |
| `results/stage2/sae_features/gemma3_4b_infer_subtype_L22_layer_22_width_16k_l0_small_top128.meta.json` | 11000 | 128 | 2560 | 16384 |  | 20.9625 |
| `results/stage2/sae_features/gemma3_4b_infer_subtype_L22_layer_22_width_262k_l0_small_top128.meta.json` | 11000 | 128 | 2560 | 262144 |  | 17.9848 |
| `results/stage2/sae_features/gemma3_4b_infer_subtype_L22_mlp_in_weighted_layer_22_width_16k_l0_small_affine_top128.meta.json` | 11000 | 128 | 2560 | 16384 | 2560 | 15.9644 |
| `results/stage2/sae_features/gemma3_4b_infer_subtype_L22_mlp_in_weighted_layer_22_width_262k_l0_small_affine_top128.meta.json` | 11000 | 128 | 2560 | 262144 | 2560 | 19.1801 |
| `results/stage2/sae_features/gemma3_4b_infer_subtype_L22_mlp_out_hook_layer_22_width_16k_l0_small_top128.meta.json` | 11000 | 128 | 2560 | 16384 |  | 42.8449 |

### Reconstruction/Error Diagnostics

Sources: `docs/sae_reconstruction_probe_4b_l22_s1.json` and `docs/sae_reconstruction_probe_4b_l22_s3_target_symbol.json`.

| Split | SAE ID | Task | Energy explained | Reconstruction AUC | Error AUC |
|---|---|---|---:|---:|---:|
| s1 | `layer_22_width_16k_l0_small` | infer_property | 0.9802 | 0.8051 | 0.8978 |
| s1 | `layer_22_width_16k_l0_small` | infer_subtype | 0.9806 | 0.9543 | 0.9733 |
| s1 | `layer_22_width_262k_l0_small` | infer_property | 0.9823 | 0.8085 | 0.9005 |
| s1 | `layer_22_width_262k_l0_small` | infer_subtype | 0.9832 | 0.9615 | 0.9760 |
| s3 | `layer_22_width_16k_l0_small` | infer_property | 0.9802 | 0.8285 | 0.9075 |
| s3 | `layer_22_width_16k_l0_small` | infer_subtype | 0.9806 | 0.9634 | 0.9703 |
| s3 | `layer_22_width_262k_l0_small` | infer_property | 0.9823 | 0.8225 | 0.9062 |
| s3 | `layer_22_width_262k_l0_small` | infer_subtype | 0.9832 | 0.9583 | 0.9726 |

## Current Phase C Artifact Checklist

- Residual SAE feature caches: 8 full caches under `results/stage2/sae_features/` for L20/L22 x 16K/262K x property/subtype.
- Exact-hook activations: `results/stage2/activations/gemma3_4b_infer_{property,subtype}_L22_mlp_in_weighted.*` and `..._mlp_out_hook.*`.
- MLP-out SAE features: 2 caches for L22 16K under `results/stage2/sae_features/`.
- Transcoder features: 4 caches for L22 affine 16K/262K under `results/stage2/sae_features/`.
- Reconstruction diagnostics: S1 and S3 JSONs under `docs/`.

## Phase D Run Plan And Result Slots

Phase D sbatches were added on 2026-04-30. Fill in the result summaries below as the jobs finish.

| Phase D step | Sbatch | Expected outputs | Status |
|---|---|---|---|
| D.1 SAE probes | `scripts/stage2_phase_d_4b_d1_sae_probes.sbatch` | `docs/sae_probe_4b_l{20,22}_{16k,262k}_{s1,s3_target_symbol}.json`, `docs/sae_probe_4b_l22_mlp_out_hook_16k_{s1,s3_target_symbol}.json` | complete |
| D.2 transcoder probes | `scripts/stage2_phase_d_4b_d2_transcoder_probes.sbatch` | `docs/transcoder_probe_4b_l22_{16k,262k}_affine_exact_{s1,s3_target_symbol}.json` | complete |
| D.3 transcoder components | `scripts/stage2_phase_d_4b_d3_transcoder_components.sbatch` | `docs/transcoder_component_probe_4b_l22_{16k,262k}_affine_exact_{s1,s3_target_symbol}.json` | complete |
| D.4 sparse concat | `scripts/stage2_phase_d_4b_d4_sparse_concat.sbatch` | `docs/sparse_concat_probe_4b_l22_all_sparse_*`, `docs/sparse_concat_probe_4b_l20_l22_resid_*`, `docs/sparse_concat_probe_4b_l20_l22_all_sparse_*` | complete |
| D.5 dense-active concat | `scripts/stage2_phase_d_4b_d5_dense_active_concat.sbatch` | `docs/dense_active_sparse_concat_probe_4b_l20_l22_all_sparse_{s1,s3_target_symbol}.json` | complete |
| D.6 sparse ablation | `scripts/stage2_phase_d_4b_d6_sparse_ablation.sbatch` | `docs/sparse_concat_ablation_4b_l20_l22_all_sparse_summary.json` plus `docs/ablation_4b_l20_l22_all_sparse/*.json` | complete |
| D.7 feature stability | `scripts/stage2_phase_d_4b_d7_feature_stability.sbatch` | `docs/sae_feature_stability_4b_l22_s1.json` | complete |

Canonical Phase B raw-probe docs for Phase D/report assembly:

- `docs/raw_probe_4b_s1.json`
- `docs/raw_probe_4b_s3_target_symbol.json`

### Phase D Result Summary

D.1-D.7 completed and validated on 2026-04-30. All expected JSON outputs exist; component probes report `status=ok`; `docs/sae_feature_stability_4b_l22_s1.json` has non-empty stable same-width candidate lists for both L22 residual widths.

#### D.1-D.2 Per-Block Sparse Probe AUCs

| Feature family | Split | Property AUC | Subtype AUC | Source |
|---|---|---:|---:|---|
| Resid SAE L20 16K | s1 | 0.7727 | 0.9536 | `docs/sae_probe_4b_l20_16k_s1.json` |
| Resid SAE L20 16K | s3 | 0.7790 | 0.9657 | `docs/sae_probe_4b_l20_16k_s3_target_symbol.json` |
| Resid SAE L20 262K | s1 | 0.7908 | 0.9436 | `docs/sae_probe_4b_l20_262k_s1.json` |
| Resid SAE L20 262K | s3 | 0.7822 | 0.9642 | `docs/sae_probe_4b_l20_262k_s3_target_symbol.json` |
| Resid SAE L22 16K | s1 | 0.8079 | 0.9556 | `docs/sae_probe_4b_l22_16k_s1.json` |
| Resid SAE L22 16K | s3 | 0.8199 | 0.9654 | `docs/sae_probe_4b_l22_16k_s3_target_symbol.json` |
| Resid SAE L22 262K | s1 | 0.8076 | 0.9636 | `docs/sae_probe_4b_l22_262k_s1.json` |
| Resid SAE L22 262K | s3 | 0.8194 | 0.9708 | `docs/sae_probe_4b_l22_262k_s3_target_symbol.json` |
| MLP-out SAE L22 16K | s1 | 0.8122 | 0.9607 | `docs/sae_probe_4b_l22_mlp_out_hook_16k_s1.json` |
| MLP-out SAE L22 16K | s3 | 0.8241 | 0.9693 | `docs/sae_probe_4b_l22_mlp_out_hook_16k_s3_target_symbol.json` |
| Transcoder L22 16K affine | s1 | 0.8046 | 0.9567 | `docs/transcoder_probe_4b_l22_16k_affine_exact_s1.json` |
| Transcoder L22 16K affine | s3 | 0.8163 | 0.9695 | `docs/transcoder_probe_4b_l22_16k_affine_exact_s3_target_symbol.json` |
| Transcoder L22 262K affine | s1 | 0.8037 | 0.9635 | `docs/transcoder_probe_4b_l22_262k_affine_exact_s1.json` |
| Transcoder L22 262K affine | s3 | 0.8225 | 0.9688 | `docs/transcoder_probe_4b_l22_262k_affine_exact_s3_target_symbol.json` |

#### D.3 Transcoder Component AUCs

| Width | Split | Task | Latent | Skip | Full | Error | Source |
|---|---|---|---:|---:|---:|---:|---|
| 16k | s1 | infer_property | 0.8043 | 0.8697 | 0.8713 | 0.8548 | `docs/transcoder_component_probe_4b_l22_16k_affine_exact_s1.json` |
| 16k | s1 | infer_subtype | 0.9546 | 0.9602 | 0.9593 | 0.9512 | `docs/transcoder_component_probe_4b_l22_16k_affine_exact_s1.json` |
| 16k | s3 | infer_property | 0.8182 | 0.8723 | 0.8687 | 0.8610 | `docs/transcoder_component_probe_4b_l22_16k_affine_exact_s3_target_symbol.json` |
| 16k | s3 | infer_subtype | 0.9660 | 0.9437 | 0.9511 | 0.9485 | `docs/transcoder_component_probe_4b_l22_16k_affine_exact_s3_target_symbol.json` |
| 262k | s1 | infer_property | 0.7904 | 0.8688 | 0.8703 | 0.8604 | `docs/transcoder_component_probe_4b_l22_262k_affine_exact_s1.json` |
| 262k | s1 | infer_subtype | 0.9536 | 0.9594 | 0.9555 | 0.9547 | `docs/transcoder_component_probe_4b_l22_262k_affine_exact_s1.json` |
| 262k | s3 | infer_property | 0.8127 | 0.8719 | 0.8711 | 0.8688 | `docs/transcoder_component_probe_4b_l22_262k_affine_exact_s3_target_symbol.json` |
| 262k | s3 | infer_subtype | 0.9487 | 0.9473 | 0.9489 | 0.9459 | `docs/transcoder_component_probe_4b_l22_262k_affine_exact_s3_target_symbol.json` |

#### D.4-D.5 Sparse Concat And Dense-Active AUCs

| Combo | Split | Property AUC | Subtype AUC | Source |
|---|---|---:|---:|---|
| L22 all sparse | s1 | 0.8402 | 0.9675 | `docs/sparse_concat_probe_4b_l22_all_sparse_s1.json` |
| L22 all sparse | s3 | 0.8489 | 0.9754 | `docs/sparse_concat_probe_4b_l22_all_sparse_s3_target_symbol.json` |
| L20+L22 residual only | s1 | 0.8253 | 0.9681 | `docs/sparse_concat_probe_4b_l20_l22_resid_s1.json` |
| L20+L22 residual only | s3 | 0.8400 | 0.9710 | `docs/sparse_concat_probe_4b_l20_l22_resid_s3_target_symbol.json` |
| L20+L22 all sparse | s1 | 0.8421 | 0.9693 | `docs/sparse_concat_probe_4b_l20_l22_all_sparse_s1.json` |
| L20+L22 all sparse | s3 | 0.8500 | 0.9760 | `docs/sparse_concat_probe_4b_l20_l22_all_sparse_s3_target_symbol.json` |
| Dense-active L20+L22 all sparse | s1 | 0.8421 | 0.9672 | `docs/dense_active_sparse_concat_probe_4b_l20_l22_all_sparse_s1.json` |
| Dense-active L20+L22 all sparse | s3 | 0.8517 | 0.9758 | `docs/dense_active_sparse_concat_probe_4b_l20_l22_all_sparse_s3_target_symbol.json` |

#### D.6 Leave-One-Block-Out AUCs

| Split | Ablation | Property AUC | Subtype AUC |
|---|---|---:|---:|
| s1 | drop_l20_resid16k | 0.8419 | 0.9692 |
| s1 | drop_l20_resid262k | 0.8415 | 0.9671 |
| s1 | drop_l22_mlpout16k | 0.8362 | 0.9683 |
| s1 | drop_l22_resid16k | 0.8393 | 0.9692 |
| s1 | drop_l22_resid262k | 0.8391 | 0.9686 |
| s1 | drop_l22_tc16k | 0.8411 | 0.9689 |
| s1 | drop_l22_tc262k | 0.8403 | 0.9679 |
| s1 | full | 0.8421 | 0.9693 |
| s3 | drop_l20_resid16k | 0.8491 | 0.9757 |
| s3 | drop_l20_resid262k | 0.8497 | 0.9758 |
| s3 | drop_l22_mlpout16k | 0.8459 | 0.9755 |
| s3 | drop_l22_resid16k | 0.8515 | 0.9756 |
| s3 | drop_l22_resid262k | 0.8525 | 0.9761 |
| s3 | drop_l22_tc16k | 0.8548 | 0.9753 |
| s3 | drop_l22_tc262k | 0.8535 | 0.9744 |
| s3 | full | 0.8500 | 0.9760 |

#### D.7 Feature Stability

Source: `docs/sae_feature_stability_4b_l22_s1.json`. Stable same-width candidate counts: `{'layer_22_width_16k_l0_small': 32, 'layer_22_width_262k_l0_small': 29}`.

Interpretation: the strongest sparse-only 4B concat (`L20+L22 all sparse`) reaches 0.8421/0.9693 on S1 and 0.8500/0.9760 on S3 for property/subtype. This remains below raw L22 for property but roughly matches or slightly exceeds raw L22 on subtype, where class imbalance and tiny h3/h4 positive counts remain an important caveat.


## Phase D.8-D.9 Results

Big-affine artifact inventory: `docs/gemma_scope_4b_l22_big_affine_inventory.md` found `transcoder_all/layer_22_width_262k_l0_big_affine` in `google/gemma-scope-2-4b-it`.

Big-affine top-512 feature cache metadata:

| Task | Rows | Top K | L0 Mean | d_in | d_sae | d_out | Metadata |
|---|---:|---:|---:|---:|---:|---:|---|
| property | 11000 | 512 | 171.4042 | 2560 | 262144 | 2560 | `results/stage2/sae_features/gemma3_4b_infer_property_L22_mlp_in_weighted_layer_22_width_262k_l0_big_affine_top512.meta.json` |
| subtype | 11000 | 512 | 159.5584 | 2560 | 262144 | 2560 | `results/stage2/sae_features/gemma3_4b_infer_subtype_L22_mlp_in_weighted_layer_22_width_262k_l0_big_affine_top512.meta.json` |

Direct sparse transcoder probe AUCs:

| Method | Split | Property AUC | Subtype AUC | Source |
|---|---|---:|---:|---|
| Transcoder L22 262K big-affine top512 | s1 | 0.8552 | 0.9693 | `docs/transcoder_probe_4b_l22_262k_big_affine_exact_top512_s1.json` |
| Transcoder L22 262K big-affine top512 | s3 | 0.8745 | 0.9765 | `docs/transcoder_probe_4b_l22_262k_big_affine_exact_top512_s3_target_symbol.json` |

Component diagnostics:

| Split | Task | Full Energy | Latent Energy | Skip Energy | Error AUC | Full AUC | Latent AUC | Skip AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| s1 | property | 0.9284 | 0.2542 | 0.8153 | 0.8527 | 0.8573 | 0.8392 | 0.8668 |
| s1 | subtype | 0.9302 | 0.2927 | 0.8225 | 0.9434 | 0.9442 | 0.9354 | 0.9570 |
| s3 | property | 0.9284 | 0.2542 | 0.8153 | 0.8583 | 0.8673 | 0.8643 | 0.8753 |
| s3 | subtype | 0.9302 | 0.2927 | 0.8225 | 0.9311 | 0.9227 | 0.9089 | 0.9436 |

D.9 deliverables:

- D.9 4B comparison section appended to `docs/stage2_results_pack.md`.
- D.9 Phase E handoff shortlist written to `docs/feature_candidate_shortlist_4b.md`.

## Notes

- S2 is omitted/skipped in the available comparison summaries because the current S2 split is non-evaluable in this dataset snapshot.
- Phase A verification reports L22 spot-check mismatches, but the hard artifact checks for row count, shape, dtype, and example-id order pass for all selected 4B activation tensors.
- Subtype h3/h4 per-height AUCs should be interpreted carefully because the holdout positive counts are very small in several splits.
