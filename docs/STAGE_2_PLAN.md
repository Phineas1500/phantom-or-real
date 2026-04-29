# Stage 2 Plan: Operational Status

This file is the short working plan for Stage 2. Detailed execution notes live
in `docs/REPORT_NOTES.md`; compact report-ready numbers live in
`docs/stage2_results_pack.md`; raw JSON outputs remain the source of truth for
metrics.

## Current Status

As of 2026-04-28, the active workspace scope is Gemma 3 27B. Gemma 3 4B results
will be merged later from the teammate run.

Completed:

- Scholar `phantom` environment recreated from the project YAMLs:
  `transformer-lens==3.0.0`, `sae-lens==6.39.0`,
  `scikit-learn==1.8.0`.
- Stage 1 JSONL hashes, model/tokenizer revisions, and chat-template checksums
  pinned in `docs/stage2_invariants.json`.
- 27B prompt reconstruction and activation artifact validation implemented in
  `scripts/validate_activations.py`.
- S1 random splits and S3 target-symbol-heldout splits generated for 27B.
  Planned topology-heldout S2 is recorded but not evaluable on the shipped
  data because canonical topology leaves too few groups.
- Metadata-only B0 baselines generated for S1 and S3.
- Full 27B raw residual extraction completed for layers 15, 30, and 45 across
  both tasks. Each activation file has shape `[11000, 5376]`, dtype bf16, and
  matching sidecars.
- Raw residual probes show a strong pre-generation correctness signal. Best
  layer is L45 for both tasks.
- Label-shuffle, cross-task transfer, and S3 heldout-target controls completed.
- L45 residual SAE probes completed for Gemma Scope 2 widths 16K and 262K.
- Top-512 diagnostic completed; top-128 already captured all active residual
  SAE features in the tested files.
- Feature-stability and reconstruction/error diagnostics completed.
- L45 MLP-output SAE and L45 affine skip-transcoder pilots completed.
- First steering pilot completed; result is null/inconclusive, not causal
  evidence.
- A bounded 27B crosscoder pilot completed as Scholar job `451181`.

Current scientific story:

- Raw pre-generation activations robustly predict ontology reasoning
  success/failure beyond metadata baselines.
- Tested Gemma Scope residual SAE features retain some signal, but trail raw
  residual probes.
- Reconstruction/error diagnostics are the main pivot: residual SAEs reconstruct
  about 95% of activation energy, yet the raw-minus-reconstruction error
  recovers nearly the full raw probe signal.
- MLP-output and skip-transcoder pilots do not rescue sparse-feature
  localization.
- The multi-layer crosscoder pilot also does not rescue sparse-feature
  localization: raw concat over the crosscoder layers nearly matches raw L45,
  but crosscoder features trail raw concat.
- Dense active-feature probes rule out sparse CSR scaling/centering as the main
  cause of the raw-vs-sparse gap.
- Steering has not established a causal feature or direction.

## Active Scope

Keep the remaining Stage 2 work narrow:

- Model: Gemma 3 27B.
- Tasks: `infer_property` and `infer_subtype`.
- Label: `is_correct_strong`, with `parse_failed=True` filtered for main
  probe training.
- Main feature sources: raw residual L45, residual SAE L45 16K/262K,
  reconstruction/error components, MLP-output pilot, skip-transcoder pilot, and
  the bounded crosscoder pilot.
- Main splits: S1 random and S3 target-symbol heldout.
- Main report claim: raw activations contain a robust correctness signal, but
  the tested sparse dictionaries do not cleanly localize it.

Out of scope unless project ownership changes:

- Running 4B in this workspace.
- `infer_membership_relation` or multi-hypothesis examples.
- Training new SAEs.
- Cross-model-family comparisons.
- Steering on 4B or `infer_subtype`.
- Broad Gemma Scope sweeps.
- Probing generated CoT/intermediate tokens.

## Invariants

Do not interpret a result unless these hold:

1. Prompts are reconstructed through `src.messages.build_messages`.
2. The tokenizer chat template uses `add_generation_prompt=True`.
3. Activations are taken at the last pre-generation token position.
4. Stage 1 JSONLs match hashes in `docs/stage2_invariants.json`.
5. Model/tokenizer revisions match `docs/stage2_invariants.json`.
6. Raw residual hooks are `blocks.{L}.hook_resid_post`.
7. Any non-residual hook or sparse dictionary is recorded with its exact site,
   release ID, config hash, and parameter hash before being treated as final.
8. Any GPT judge calls use a dated model snapshot, not a moving alias.

## Key Results To Report

### Behavioral Context

Stage 1 generated 11,000 labeled 27B rows per task. Strong accuracy collapses
with height:

| Task | h1 | h2 | h3 | h4 |
| --- | ---: | ---: | ---: | ---: |
| `infer_property` | 0.960 | 0.577 | 0.392 | 0.264 |
| `infer_subtype` | 0.973 | 0.325 | 0.114 | 0.055 |

### Metadata Baselines

Use deltas over the strongest matching B0 baseline:

| Split | Task | Best B0 | Test AUC |
| --- | --- | --- | ---: |
| S1 | `infer_property` | `b0_prompt` | 0.743 |
| S1 | `infer_subtype` | `b0_height` | 0.841 |
| S3 | `infer_property` | `b0_namefreq` | 0.711 |
| S3 | `infer_subtype` | `b0_prompt` | 0.859 |

### Raw Residual Probes

| Split | Task | Raw L45 AUC | 95% CI | Delta vs B0 |
| --- | --- | ---: | --- | ---: |
| S1 | `infer_property` | 0.897 | [0.881, 0.912] | +0.153 |
| S1 | `infer_subtype` | 0.914 | [0.896, 0.932] | +0.073 |
| S3 | `infer_property` | 0.884 | [0.868, 0.901] | +0.173 |
| S3 | `infer_subtype` | 0.917 | [0.898, 0.934] | +0.058 |

Label-shuffle S1 controls stayed near chance: property 0.493, subtype 0.481.

### Residual SAE Probes

| Split | Task | SAE 16K AUC | SAE 262K AUC | Raw L45 AUC |
| --- | --- | ---: | ---: | ---: |
| S1 | `infer_property` | 0.786 | 0.806 | 0.897 |
| S1 | `infer_subtype` | 0.876 | 0.870 | 0.914 |
| S3 | `infer_property` | 0.799 | 0.779 | 0.884 |
| S3 | `infer_subtype` | 0.865 | 0.867 | 0.917 |

### Reconstruction/Error Diagnostic

| Split | Task | SAE width | Energy explained | Recon AUC | Error AUC | Raw AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| S1 | `infer_property` | 16K | 0.948 | 0.786 | 0.894 | 0.897 |
| S1 | `infer_property` | 262K | 0.955 | 0.806 | 0.897 | 0.897 |
| S1 | `infer_subtype` | 16K | 0.948 | 0.877 | 0.916 | 0.914 |
| S1 | `infer_subtype` | 262K | 0.954 | 0.870 | 0.915 | 0.914 |
| S3 | `infer_property` | 16K | 0.948 | 0.799 | 0.881 | 0.884 |
| S3 | `infer_property` | 262K | 0.955 | 0.788 | 0.886 | 0.884 |
| S3 | `infer_subtype` | 16K | 0.948 | 0.865 | 0.916 | 0.917 |
| S3 | `infer_subtype` | 262K | 0.954 | 0.867 | 0.914 | 0.917 |

### Site And Sparse-Artifact Pilots

| Split | Task | Raw `mlp_out` | MLP-out SAE 16K | Raw `mlp_in` | Skip-transcoder 16K |
| --- | --- | ---: | ---: | ---: | ---: |
| S1 | `infer_property` | 0.895 | 0.577 | 0.897 | 0.722 |
| S1 | `infer_subtype` | 0.916 | 0.674 | 0.915 | 0.821 |
| S3 | `infer_property` | 0.892 | 0.550 | 0.885 | 0.722 |
| S3 | `infer_subtype` | 0.915 | 0.702 | 0.914 | 0.841 |

Interpretation: raw same-site activations carry the signal, while tested sparse
features only partially expose it.

### Crosscoder Pilot

Job `451181` tested the smallest available 27B IT crosscoder:

- HF repo: `google/gemma-scope-2-27b-it`.
- Crosscoder: `crosscoder/layer_16_31_40_53_width_65k_l0_medium`.
- Layers: `{16,31,40,53}`.
- Required baseline: raw-concat probe over the same four residual layers.

Report the crosscoder as a bounded multi-layer check. Never compare it only to
raw L45; the fair comparison is crosscoder features vs raw concat over the same
layer set.

| Split | Task | Raw concat AUC | Crosscoder 65K AUC | Raw L45 AUC |
| --- | --- | ---: | ---: | ---: |
| S1 | `infer_property` | 0.893 | 0.787 | 0.897 |
| S1 | `infer_subtype` | 0.904 | 0.868 | 0.914 |
| S3 | `infer_property` | 0.883 | 0.724 | 0.884 |
| S3 | `infer_subtype` | 0.903 | 0.853 | 0.917 |

Interpretation: the crosscoder features trail raw concat on every task/split.
This strengthens the conclusion that the tested Gemma Scope sparse dictionaries
only partially expose the correctness signal.

### Steering

The first 27B property steering pilot used 8 balanced S1 test rows, prompt-only
L45 raw-direction interventions at +/-2 SD, and orthogonal controls. It caused
zero strong-correctness flips; the one output change also appeared under
orthogonal control. Treat this as a plumbing/null result.

Do not spend more project time on steering unless the report specifically needs
a stronger null. If needed, the next design should fix the generation length
protocol and test all-token or decode-step steering with a small strength sweep.

## Remaining Work

Immediate:

- Keep `docs/REPORT_NOTES.md`, `docs/stage2_results_pack.md`, and
  `docs/report_outline.md` aligned with any new result interpretation.
- Run the full test suite after doc/code updates.

Report-critical:

- Keep the final narrative centered on raw-probe signal plus sparse-feature
  localization failure.
- Add teammate 4B results only as a comparison once available.
- Include S3 as a heldout-target diagnostic, not a full name-scramble result.
- State explicitly that S2 topology-heldout was not evaluable on the shipped
  dataset.
- Present steering as inconclusive.

Optional only if time remains:

- Stronger prompt-length/name-frequency residualization.
- Float32 encoding or hook/L0 sanity checks for the weakest non-residual
  sparse artifacts.
- Name-scramble regeneration.
- A better steering null with all-token/decode-step intervention.

## Key Files

- `docs/REPORT_NOTES.md`: chronological report notes.
- `docs/stage2_results_pack.md`: compact report-ready result tables.
- `docs/report_outline.md`: current final report outline.
- `docs/stage2_invariants.json`: hashes and model/dictionary invariants.
- `results/stage2/splits.jsonl`: S1/S2/S3 split assignments.
- `scripts/stage2_probe_raw.py`: raw activation probes.
- `scripts/stage2_probe_raw_concat.py`: raw-concat crosscoder baseline.
- `scripts/stage2_extract_crosscoder_features.py`: crosscoder sparse feature
  extraction.
- `scripts/stage2_probe_crosscoder.py`: crosscoder feature probes.
- `scripts/stage2_probe_dense_active_sparse.py`: dense active-column scaling
  check for sparse artifacts.
- `scripts/stage2_crosscoder_27b_layers_16_31_40_53_65k.sbatch`: completed
  crosscoder pilot job script.

## Checklist

- [x] Stage 1 data and hashes pinned.
- [x] 27B prompt/activation validation.
- [x] S1 and S3 splits.
- [x] B0 metadata baselines.
- [x] Full 27B raw residual extraction.
- [x] Raw residual probes with bootstrap CIs.
- [x] Label-shuffle controls.
- [x] Cross-task transfer.
- [x] Residual SAE 16K/262K extraction and probes.
- [x] Top-k truncation diagnostic.
- [x] SAE feature-stability diagnostic.
- [x] Reconstruction/error diagnostic.
- [x] MLP-output SAE pilot.
- [x] Skip-transcoder pilot.
- [x] Steering pilot and orthogonal controls.
- [x] Crosscoder width-65K pilot over layers 16/31/40/53.
- [x] Crosscoder invariants pinned.
- [x] Dense active-feature scaling sanity check.
- [ ] Teammate 4B comparison tables.
- [ ] Final report figures/tables assembled.
