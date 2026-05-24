# Qwen 27B Completion Audit

This audit records the current scope decision for the persistent goal to mirror
the Gemma Scope experiments on Qwen Scope. As of 2026-05-24, the active target is
`Qwen/Qwen3.5-27B`; smaller Qwen variants such as 2B/9B are not required for the
current replication because the user confirmed that 27B is good enough.

## Verification Snapshot

Commands run from `/scratch/scholar/skiron/phantom-or-real`:

- Parsed all current `docs/qwen*.json` artifacts: `83` parsed, `0` failed.
- Checked inference row counts: `11000` rows for
  `qwen35_27b_infer_property.jsonl` and `11000` rows for
  `qwen35_27b_infer_subtype.jsonl`.
- Checked the newest single-feature steering JSONL: `104` rows, matching 8
  selected examples x 13 conditions.
- Checked representative artifacts across 26 Qwen 27B experiment families:
  `0` missing.

## Requirement Audit

| Gemma 27B experiment family | Qwen 27B evidence | Status |
|---|---|---|
| Behavioral inference, validation, and splits | `results/full/with_errortype/qwen35_27b_infer_{property,subtype}.jsonl`, `docs/qwen35_27b_inference_merge_summary.json`, `results/stage2/splits.jsonl` | Complete |
| Metadata baselines | `docs/qwen35_27b_b0_summary.json` | Complete |
| Raw residual probes and layer scan | `docs/qwen_scope_raw_probe_27b_l45_{s1,s3}.json`, `docs/qwen_scope_raw_probe_27b_layers_16_31_40_53_{s1,s3}.json` | Complete |
| Label shuffle, cross-task transfer, and metadata residualization | `docs/qwen_scope_raw_probe_27b_l45_s1_label_shuffle.json`, `docs/qwen_scope_raw_probe_transfer_27b_l45_{s1,s3}.json`, `docs/qwen_scope_raw_probe_metadata_residualization_27b_l45.json` | Complete |
| Qwen Scope residual SAE probes | `docs/qwen_scope_sae_probe_27b_l45_w80k_l0_{50,100}_{s1,s3}.json`, L53 reports in `docs/qwen_scope_sae_probe_27b_l53_w80k_l0_{50,100}_{s1,s3}.json` | Complete |
| Residual SAE reconstruction/error diagnostics | `docs/qwen_scope_reconstruction_probe_27b_l45_w80k_l0_{50,100}_{s1,s3}.json`, L53 reconstruction reports | Complete |
| Raw MLP-site probes | `docs/qwen_scope_raw_probe_27b_l45_53_mlp_in_weighted_{s1,s3}.json`, `docs/qwen_scope_raw_probe_27b_l45_53_mlp_out_{s1,s3}.json` | Complete |
| MLP-output SAE analogue | Local split-trained Qwen L53 `mlp_out` dictionary reports and reconstruction diagnostics | Complete as local stand-in; no first-party Qwen Scope MLP SAE artifact found |
| Skip/affine transcoder analogue | Local split-trained Qwen L53 `mlp_in_weighted -> mlp_out` transcoder reports and component diagnostics | Complete as local stand-in; no first-party Qwen Scope transcoder artifact found |
| Big-L0/component diagnostics analogue | Local dictionary/transcoder component diagnostics and Qwen Scope residual L0_100 runs | Complete as available analogue; exact Gemma big-L0 transcoder artifact has no Qwen counterpart |
| Sparse feature-family concat and ablation | Residual-only concat, residual+local MLP/transcoder concat, leave-one-block-out ablation | Complete |
| Crosscoder analogue | Local multi-layer top-k crosscoder and raw-concat baseline over Qwen layers `16,31,40,53` | Complete as local stand-in; no first-party Qwen Scope crosscoder artifact found |
| Dense-active sparse scaling controls | Qwen residual SAE dense-active controls plus local learned-sparse dense-active controls | Complete |
| Dtype sanity and feature dashboards | `docs/qwen_scope_sparse_dtype_sanity_27b.json`, L45/L53 feature-stability and mini-dashboard reports | Complete |
| Free-form raw steering | L45/L53 raw-direction smokes in `docs/qwen35_raw_steering_27b_l{45,53}_property_pilot.json` | Complete for the Qwen causal mirror |
| Answer-property steering | `docs/qwen35_answer_property_steering_27b_l45_polarity_smoke.json` | Complete |
| Sparse-probe bundle steering | `docs/qwen35_scope_sparse_bundle_steering_27b_l45_l0_100_property_smoke.json`, `docs/qwen35_scope_sparse_bundle_steering_27b_l53_l0_100_property_smoke.json` | Complete |
| Single-feature steering | `docs/qwen35_scope_single_feature_steering_27b_l53_l0_100_7169_23296_4212_property_smoke.json` | Complete |
| Hard-foil forced choice | `docs/qwen35_27b_infer_subtype_h4_hardfoil_forced_choice.json` | Complete |
| Clean-to-corrupt and corrupt-to-clean patching | `docs/qwen35_27b_infer_subtype_clean_to_corrupt_patching_margin_pilot.json`, `docs/qwen35_27b_infer_subtype_corrupt_to_clean_patching_margin_pilot.json` | Complete |
| 4B comparison branch | Not run for Qwen; user scoped the remaining replication to 27B | Not required for current goal |

## Result

For the user-scoped 27B target, every Gemma 27B experiment family has a Qwen 27B
artifact, local stand-in, or documented first-party-artifact limitation. The
remaining caveat is provenance, not unrun local coverage: Qwen Scope currently
publishes residual SAE families for the 27B target, while Gemma's MLP-output
SAE, transcoder, and crosscoder artifact families do not have direct Qwen Scope
counterparts in the current local replication.
