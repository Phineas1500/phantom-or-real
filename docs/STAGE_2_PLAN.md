# Stage 2 Plan: Operational Status

This file is the short working plan for Stage 2. Detailed execution notes live
in `docs/REPORT_NOTES.md`; compact report-ready numbers live in
`docs/stage2_results_pack.md`; raw JSON outputs remain the source of truth for
metrics.

## Current Status

As of 2026-04-29, the active workspace scope is Gemma 3 27B. Gemma 3 4B results
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
- Prompt/name metadata residualization diagnostic completed. Adding the raw L45
  probe score to the rich `b0_namefreq` metadata baseline improves test AUC by
  `+0.155/+0.078` on S1 property/subtype and `+0.173/+0.071` on S3
  property/subtype, nearly recovering raw-probe performance.
- L45 residual SAE probes completed for Gemma Scope 2 widths 16K and 262K.
- Top-512 diagnostic completed; top-128 already captured all active residual
  SAE features in the tested files.
- Feature-stability and reconstruction/error diagnostics completed.
- L45 MLP-output SAE and L45 16K affine skip-transcoder pilots completed; both
  old bare-normalized runs are now treated as exploratory after the exact-hook
  audit. Corrected exact-hook reruns have been completed for MLP-output SAE,
  16K skip-transcoder, and 262K transcoder.
- First steering pilot completed; result is null/inconclusive, not causal
  evidence.
- A bounded 27B crosscoder pilot completed as Scholar job `451181`.
- Dense active-feature scaling and bf16-vs-fp32 sparse encoding sanity checks
  completed.
- Neuronpedia availability audited for the current top features. It is not a
  direct audit path for the L45 residual SAE features because public
  `gemma-3-27b-it` residual dashboards currently cover layers 16/31/40/53, not
  45. Neuronpedia does have all-layer `gemmascope-2-transcoder-262k`
  dashboards, including layer 45.
- L45 262K affine transcoder exact-hook audit completed as Scholar job
  `451225`; it found the old 262K input/target were missing learned RMSNorm
  weights and that the raw Gemma Scope affine skip tensor matches `x @ W_skip`
  for this artifact.
- Corrected L45 262K affine transcoder rerun completed as Scholar job
  `451226`, including exact weighted-input extraction, exact raw input/output
  probes, exact sparse probes, exact component diagnostics, and refreshed
  Neuronpedia audit.
- L30 residual SAE 16K/262K extraction and probes completed as Scholar jobs
  `451569` and `451571`, including L30-only, L45 five-block, L30+L45
  residual-only, and L30+L45 all-sparse concat probes.

Current scientific story:

- Raw pre-generation activations robustly predict ontology reasoning
  success/failure beyond metadata baselines.
- The metadata residualization diagnostic supports this as a conditional
  activation signal: raw L45 scores add `+0.06` to `+0.18` AUC over prompt/name
  metadata across the two tasks and S1/S3 splits.
- Tested Gemma Scope residual SAE features retain some signal, but trail raw
  residual probes.
- Reconstruction/error diagnostics are the main pivot: residual SAEs reconstruct
  about 95% of activation energy, yet the raw-minus-reconstruction error
  recovers nearly the full raw probe signal.
- The corrected Neuronpedia-facing L45 262K transcoder is no longer weak:
  exact S1 property/subtype AUCs are `0.795/0.873`, and exact S3 AUCs are
  `0.802/0.885`. This is residual-SAE-like but still below raw same-site
  activations.
- The exact-hook 16K skip-transcoder rerun also fixes the old weak pilot:
  exact 16K AUCs are S1 `0.787/0.868` and S3 `0.785/0.880`, versus old
  bare-normalized 16K AUCs S1 `0.722/0.821` and S3 `0.722/0.841`. Exact 262K
  remains slightly stronger on most sparse-latent comparisons.
- Corrected 262K component diagnostics are now interpretable: full output
  explains `0.672/0.661` energy for property/subtype, with global cosine
  `0.821/0.814`. Dense full/error components still trail raw exact
  activations, so this is partial sparse localization rather than a clean
  sparse mechanism.
- Exact-hook MLP-output SAE rerun fixes another old hook/scale issue. The old
  bare-normalized MLP-output SAE pilot was weak, but exact `hook_mlp_out`
  features reach S1 `0.811/0.878` and S3 `0.807/0.879` for property/subtype.
  They are meaningful, but still below raw exact `hook_mlp_out`.
- Sparse feature-family concatenation helps but does not close the gap. The
  best sparse-only result so far is the L30+L45 all-sparse concat, which
  reaches S1 `0.839/0.887` and S3 `0.834/0.892` for property/subtype, still
  below raw exact activations.
- Refreshed exact Neuronpedia audit still shows mostly generic/lexical/code
  feature explanations rather than clean ontology-reasoning mechanisms.
- The multi-layer crosscoder pilot also does not rescue sparse-feature
  localization: raw concat over the crosscoder layers nearly matches raw L45,
  but crosscoder features trail raw concat.
- Dense active-feature probes rule out sparse CSR scaling/centering as the main
  cause of the raw-vs-sparse gap.
- Float32 re-encoding shows nearly identical sparse active sets, so bf16
  encoding instability is also not the explanation.
- Steering has not established a causal feature or direction.

## Active Scope

Keep the remaining Stage 2 work narrow:

- Model: Gemma 3 27B.
- Tasks: `infer_property` and `infer_subtype`.
- Label: `is_correct_strong`, with `parse_failed=True` filtered for main
  probe training.
- Main feature sources: raw residual L45, residual SAE L30/L45 16K/262K,
  residual-SAE reconstruction/error components, corrected exact-hook L45
  16K/262K transcoders, exact-hook MLP-output SAE, sparse feature-family
  concat, and the bounded crosscoder pilot.
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
8. For Gemma Scope 2 non-residual dictionaries, exact hook/scale alignment must
   be checked before interpreting weak sparse-feature results. For the L45 262K
   transcoder, the reportable input is `ln2.hook_normalized * ln2.w`, the
   target is `blocks.45.hook_mlp_out`, and the affine skip path uses
   `x @ W_skip`.
9. Any GPT judge calls use a dated model snapshot, not a moving alias.

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

| Split | Task | Raw exact `mlp_out` | Exact MLP-out SAE 16K | Old MLP-out SAE pilot | Raw exact `mlp_in` | Old 16K TC pilot | Exact 16K TC | Exact 262K TC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | `infer_property` | 0.896 | 0.811 | 0.577 | 0.897 | 0.722 | 0.787 | 0.795 |
| S1 | `infer_subtype` | 0.916 | 0.878 | 0.674 | 0.916 | 0.821 | 0.868 | 0.873 |
| S3 | `infer_property` | 0.892 | 0.807 | 0.550 | 0.885 | 0.722 | 0.785 | 0.802 |
| S3 | `infer_subtype` | 0.915 | 0.879 | 0.702 | 0.914 | 0.841 | 0.880 | 0.885 |

Interpretation: raw same-site activations carry the signal, while tested sparse
features only partially expose it. Exact-hook MLP-output SAE and exact-hook
16K/262K transcoder features are much stronger than the old bare-normalized
pilots, confirming the hook/scale issue, but they still trail raw exact
activations.

### 262K Transcoder Component Diagnostic

Quick diagnostic settings: `C=1.0`, `liblinear`, no bootstrap resampling. The
old cached-target diagnostic is superseded by the exact-hook rerun below.

| Split | Task | Latent | Affine skip | Full | Error | Full energy | Raw exact `mlp_in` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | `infer_property` | 0.791 | 0.856 | 0.862 | 0.864 | 0.672 | 0.897 |
| S1 | `infer_subtype` | 0.867 | 0.890 | 0.897 | 0.888 | 0.661 | 0.916 |
| S3 | `infer_property` | 0.796 | 0.844 | 0.851 | 0.861 | 0.672 | 0.885 |
| S3 | `infer_subtype` | 0.883 | 0.882 | 0.889 | 0.886 | 0.661 | 0.914 |

Interpretation: the exact 262K transcoder reconstructs a substantial part of
the MLP output and exposes moderate-to-strong correctness signal, but the full
and error components still trail raw exact activations.

### Sparse Feature-Family Concat

The best remaining cheap test was whether residual sparse features and exact
transcoder sparse features are complementary. `scripts/stage2_probe_sparse_concat.py`
keeps top-k matrices sparse, aligns sidecars, and probes horizontal
concatenations with the same S1/S3 split protocol.

| Split | Task | Residual 262K + exact TC 262K | Residual 16K + residual 262K + exact TC 262K | Raw exact `mlp_in` |
| --- | --- | ---: | ---: | ---: |
| S1 | `infer_property` | 0.815 | 0.822 | 0.897 |
| S1 | `infer_subtype` | 0.870 | 0.884 | 0.916 |
| S3 | `infer_property` | 0.800 | 0.814 | 0.885 |
| S3 | `infer_subtype` | 0.881 | 0.885 | 0.914 |

Interpretation: sparse feature families are complementary, but the combined
features still do not bridge the raw activation gap. This is a useful positive
partial-localization result, not a complete sparse-mechanism result.

Adding the exact-hook MLP-output SAE 16K block gives the current strongest
sparse-only L45 property result:

| Split | Task | Previous all-L45 concat | + exact MLP-output SAE 16K | Low-C tuned | Raw exact `mlp_out` |
| --- | --- | ---: | ---: | ---: | ---: |
| S1 | `infer_property` | 0.822 | 0.828 | 0.830 | 0.896 |
| S1 | `infer_subtype` | 0.884 | 0.883 | 0.888 | 0.916 |
| S3 | `infer_property` | 0.814 | 0.823 | 0.828 | 0.892 |
| S3 | `infer_subtype` | 0.885 | 0.885 | 0.888 | 0.915 |

Interpretation: the exact MLP-output SAE contributes complementary property
signal, especially under S3. Expanding the regularization grid below `C=0.01`
improves all four sparse-concat AUCs modestly, but still leaves a large gap to
raw exact activations.

L30 residual sparse features are weak standalone property probes but add
complementary signal when combined with the corrected L45 sparse family:

| Split | Task | L30 16K | L30 262K | L30 concat | L45 five-block | L30+L45 all sparse | Raw L45 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | `infer_property` | 0.752 | 0.770 | 0.786 | 0.832 | 0.839 | 0.897 |
| S1 | `infer_subtype` | 0.860 | 0.867 | 0.872 | 0.883 | 0.887 | 0.914 |
| S3 | `infer_property` | 0.748 | 0.771 | 0.788 | 0.829 | 0.834 | 0.884 |
| S3 | `infer_subtype` | 0.860 | 0.866 | 0.864 | 0.889 | 0.892 | 0.917 |

Interpretation: L30 is not stronger than L45 by itself. Its value is
complementarity in the all-sparse concat, where it gives the current strongest
sparse-only property result and the strongest S3 subtype sparse result. The
raw-vs-sparse gap remains.

Leave-one-block-out validation of the L30+L45 all-sparse concat supports this
as a distributed complementarity result. With the full concat's selected C
values and no bootstrap CIs, removing any single block changes AUC by at most
about `0.007`. The biggest property drops come from removing L45 residual 262K
on S1, L30 residual 262K on both splits, exact MLP-output 16K on both splits,
and L45 residual 16K on S3. The aggregate report is
`docs/sparse_concat_ablation_27b_l30_l45_all_sparse_summary.json`.

A dense-active/centered rerun of the same L30+L45 all-sparse concat is
effectively unchanged: S1 `0.839/0.888` and S3 `0.834/0.893`. This reinforces
the earlier L45-only conclusion that sparse-column scaling/centering is not
the missing bridge to raw activations.

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

Most relevant to the raw-vs-sparse gap:

- Finish job `451606`, which probes cached raw residual activations at
  `{16,31,40,53}` individually. Use it to decide whether L40, L53, or neither
  deserves another residual SAE extraction.
- If L40 or L53 looks strong, run residual SAE 16K/262K extraction for that
  layer and add it to the sparse concat. This is the cleanest next attempt to
  test whether the missing signal is distributed across layers.
- Consider a multi-layer exact sparse concat near L40/L45/L53 after the raw
  layer decision check.
- Try a higher-L0 or denser 262K transcoder variant only if the artifact exists
  cleanly and exact hook/scale alignment can be verified.

Useful for report defensibility, but less likely to explain the sparse gap:

- Optional Neuronpedia-facing layer-40 or layer-53 residual SAE probe if the
  final report needs a residual feature-dashboard audit.

Low priority unless the final report specifically needs them:

- Crosscoder `l0_big` variant over layers 16/31/40/53.
- Name-scramble regeneration.
- Paraphrase-preserving B.3 test.
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
- `scripts/stage2_probe_metadata_residualization.py`: raw-score conditional
  metadata and residualization diagnostic.
- `scripts/stage2_probe_sparse_concat.py`: sparse hstack probe for combining
  multiple top-k feature artifacts.
- `scripts/stage2_mlp_out_exact_27b_L45_16k.sbatch`: exact `hook_mlp_out`
  MLP-output SAE rerun script.
- `scripts/stage2_sparse_dtype_sanity_report.py`: bf16-vs-fp32 sparse encoding
  comparison report.
- `scripts/stage2_sparse_dtype_sanity_27b.sbatch`: completed dtype sanity job
  script.
- `scripts/stage2_transcoder_27b_L45_262k_affine.sbatch`: completed
  Neuronpedia-facing 262K transcoder job script.
- `scripts/stage2_transcoder_component_diagnostics.py`: splits an affine
  transcoder into latent, skip, full, and error component activation files and
  probes each component.
- `scripts/stage2_transcoder_components_27b_L45_262k_affine.sbatch`:
  superseded bare-normalized L45 262K component diagnostic job script.
- `scripts/stage2_transcoder_hook_audit.py`: audits exact Gemma Scope
  transcoder input/output hook alignment.
- `scripts/stage2_extract_exact_transcoder_hooks.py`: extracts weighted
  transcoder input and exact `hook_mlp_out` target activations.
- `scripts/stage2_transcoder_exact_27b_L45_262k_affine.sbatch`: corrected
  exact-hook 262K transcoder rerun script.
- `scripts/stage2_transcoder_exact_27b_L45_16k_affine.sbatch`: corrected
  exact-hook 16K skip-transcoder rerun script.
- `scripts/stage2_sae_extract_27b_L30_resid.sbatch`: completed L30 residual
  SAE 16K/262K extraction script.
- `scripts/stage2_probe_27b_L30_resid_concat.sbatch`: completed L30 residual
  SAE and L30+L45 sparse concat probe script.
- `scripts/stage2_neuronpedia_feature_audit.py`: Neuronpedia API audit for top
  sparse probe features.
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
- [x] Prompt/name metadata residualization diagnostic.
- [x] Residual SAE 16K/262K extraction and probes.
- [x] Top-k truncation diagnostic.
- [x] SAE feature-stability diagnostic.
- [x] Reconstruction/error diagnostic.
- [x] MLP-output SAE pilot, superseded by exact-hook rerun.
- [x] Exact-hook MLP-output SAE rerun.
- [x] Skip-transcoder 16K pilot, now exploratory because of exact-hook caveat.
- [x] Steering pilot and orthogonal controls.
- [x] Crosscoder width-65K pilot over layers 16/31/40/53.
- [x] Crosscoder invariants pinned.
- [x] Dense active-feature scaling sanity check.
- [x] bf16-vs-fp32 sparse encoding sanity check.
- [x] Neuronpedia availability audit for current L45 features.
- [x] L45 262K transcoder initial bare-normalized probe.
- [x] L45 262K transcoder hook audit.
- [x] L45 262K exact-hook transcoder probe.
- [x] L45 262K exact-hook component diagnostic.
- [x] L45 262K exact-hook Neuronpedia audit.
- [x] L45 16K exact-hook skip-transcoder rerun and component diagnostic.
- [x] L45 sparse feature-family concat probe.
- [x] L30 residual SAE 16K/262K extraction and L30+L45 sparse concat probe.
- [x] Corrected exact dense-active sparse scaling check.
- [x] 262K transcoder invariants pinned.
- [ ] Teammate 4B comparison tables.
- [ ] Final report figures/tables assembled.
