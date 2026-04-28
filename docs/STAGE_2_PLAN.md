# Stage 2 Plan: Probes, SAE Features, and Steering

This is the operational plan for Stage 2. It keeps only the decisions,
commands, invariants, and deliverables needed to run the work. Longer
reasoning, dead alternatives, and budget arithmetic were moved out of the main
plan so this file can stay useful during execution.

## Current Status

As of 2026-04-27:

- The Scholar `phantom` environment is the active environment for this project.
  It includes `transformer-lens==3.0.0`, `sae-lens==6.39.0`, and
  `scikit-learn==1.8.0`.
- `docs/stage2_invariants.json` exists and pins all four shipped Stage 1 JSONL
  SHA-256 hashes, Gemma 3 4B revision
  `093f9f388b31de276ce2de164bdc2081324b9767`, Gemma 3 27B revision
  `005ad3404e59d6023443cb575daa05336842228a`, and the shared Gemma
  chat-template checksums.
- `src.messages.build_messages` is the single source of truth for Gemma's
  no-system-role prompt concatenation. `src.inference.build_messages` re-exports
  it for Stage 1 compatibility.
- `src.activations` and `scripts/stage2_extract.py` implement raw residual
  extraction at `blocks.{L}.hook_resid_post`.
- The 2x A40 Scholar J-node path has been measured for Gemma 3 27B:
  - TransformerLens 3.0 loads 27B bf16 with `n_devices=2`;
  - use `HookedTransformer.from_pretrained_no_processing`;
  - request high CPU RAM, currently `--mem=180G`;
  - apply deterministic whole-block remapping for multi-GPU forward;
  - raw extraction default is batch size 32;
  - batch size 64 worked for 64 height-4 prompts of 202-217 tokens, but keep
    batch 32 as the conservative default;
  - load and encode one SAE per layer at a time from cached residuals;
  - SAE chunk size 512 worked for both width-16K and width-262K layer-30 SAEs.
- Current execution ownership: this workspace run is focused on Gemma 3 27B.
  The scripts remain model-filterable so a teammate can run the same controls
  for Gemma 3 4B, but current generated Phase 0 artifacts are 27B-only.
- `docs/stage2_inventory.json` and `docs/stage2_splits_summary.json` have been
  generated for Gemma 3 27B. S1 and the replacement target-symbol-heldout S3
  are evaluable for both tasks and all heights. The planned topology-heldout S2
  is not evaluable on this shipped dataset because canonical topology leaves
  only one or two groups per task/height.
- `docs/stage2_b0_summary_27b_s1.json` records 27B S1 metadata baselines.
  Strongest pre-output baselines are AUC 0.743 for `infer_property`
  (`b0_prompt`) and AUC 0.841 for `infer_subtype` (`b0_height`).
- `docs/stage2_b0_summary_27b_s3.json` records 27B S3 target-symbol-heldout
  metadata baselines. Strongest pre-output baselines are AUC 0.711 for
  `infer_property` (`b0_namefreq`) and AUC 0.859 for `infer_subtype`
  (`b0_prompt`).
- `scripts/validate_activations.py` validates Stage 2 extraction inputs and
  optional written artifacts. The 27B property and subtype input reports are
  `ok` in `docs/stage2_equivalence_27b_infer_property.json` and
  `docs/stage2_equivalence_27b_infer_subtype.json`.
- Pilot job `449831` extracted layer 30 for 16 height-4 rows from each 27B
  task. Both artifact validation reports are `ok`; outputs have shape
  `[16, 5376]`, dtype bf16, and zero sidecar mismatches.
- Layer-selection job `449832` extracted layers 15, 30, and 45 for 512
  non-parse height-4 rows per 27B task. Artifact validation passed for all six
  files. `docs/layer_selection.json` keeps `[15, 30, 45]` for full extraction
  because the pilot was h4-only and subtype holdouts had only four positives.
- Full extraction job `449835` completed on 2026-04-27. It wrote six 27B raw
  residual files under `results/stage2/activations/`; both full validation
  reports are `ok`. Each file has shape `[11000, 5376]` and dtype bf16.
- `docs/raw_probe_27b_s1.json` records 27B S1 raw-residual logistic probes for
  layers 15, 30, and 45. Best layer is L45 for both tasks: property test AUC
  0.897, subtype test AUC 0.914. Both beat the matching B0 baselines.
- `docs/raw_probe_27b_s1_label_shuffle.json` records a bounded shuffled-label
  sanity check. Best selected-layer test AUCs are near chance: property 0.493,
  subtype 0.481.
- `docs/raw_probe_transfer_27b_s1.json` records cross-task raw-probe transfer.
  L45 transfers property-to-subtype at target test AUC 0.862 and
  subtype-to-property at target test AUC 0.786, so there is shared signal but
  not full task invariance.
- `docs/stage2_invariants.json` now pins the Gemma Scope 2 27B residual L45
  width-16K and width-262K SAEs at HF snapshot
  `5c58dd4cddd52cef653059d85e12a86bf6222a28`.
- SAE pilot job `449999` encoded 512 L45 rows for each 27B task with the
  width-16K SAE. Outputs are sparse top-128 feature files under
  `results/stage2/sae_features/pilots/`.
- Full SAE extraction job `450004` encoded all 11,000 L45 rows for each 27B
  task with `layer_45_width_16k_l0_small`. `docs/sae_probe_27b_l45_16k_s1.json`
  records first SAE probes: property test AUC 0.786, subtype test AUC 0.876.
- Full SAE extraction job `450029` encoded all 11,000 L45 rows for each 27B
  task with `layer_45_width_262k_l0_small`. `docs/sae_probe_27b_l45_262k_s1.json`
  records probes: property test AUC 0.806, subtype test AUC 0.870.
- Top-512 diagnostic job `450038` encoded all 11,000 L45 rows for each 27B
  task with the width-16K SAE. `docs/sae_probe_27b_l45_16k_top512_s1.json`
  exactly matches the top-128 probe metrics because all ranks after 128 are
  zero; observed max L0 was 24 for property and 23 for subtype.
- `docs/sae_feature_stability_27b_l45_s1.json` records the first L45 SAE
  top-weight stability analysis. It refits the saved-best-C SAE probes on
  train-active features, reproduces the saved metrics, and finds same-sign
  top-10 task overlap of 4 features for both width-16K and width-262K.
- `docs/sae_reconstruction_probe_27b_l45_s1.json` records the L45
  raw-vs-SAE reconstruction diagnostic. SAE reconstructions explain about
  94.8-95.5% of raw residual energy, but reconstruction probes match sparse
  SAE-feature probes while raw-minus-reconstruction error probes recover the
  full raw probe signal.
- `docs/sae_probe_27b_l45_16k_s3_target_symbol.json` and
  `docs/sae_probe_27b_l45_262k_s3_target_symbol.json` record 27B S3
  target-symbol-heldout SAE probes. Property still beats B0, but subtype only
  barely clears its stronger S3 B0 threshold.
- `docs/sae_reconstruction_probe_27b_l45_s3_target_symbol.json` records S3
  reconstruction/error probes from the cached reconstruction artifacts. The S3
  result matches S1: reconstruction probes track SAE-feature probes, while
  raw-minus-reconstruction error probes recover the raw L45 signal.
- `docs/raw_steering_pilot_27b_l45_property.json` records the first 27B
  `infer_property` raw-direction steering pilot. Prompt-only L45 raw +/-2 SD
  interventions on 8 balanced S1 test rows caused zero strong-correctness flips;
  the one changed output also appeared for matched orthogonal controls.
- `docs/stage2_results_pack.md` and `docs/report_outline.md` summarize the
  current 27B report story while Gemma 3 4B teammate results remain pending.
- The next scoped Gemma Scope 2 branch is a single L45 MLP-output site pilot,
  not a broad artifact sweep. It uses `mlp_out_all/layer_45_width_16k_l0_small`
  and extracts `blocks.{layer}.ln2_post.hook_normalized` into
  site-suffixed activation files.
- Scholar job `451090` completed the L45 MLP-output site pilot. Raw same-site
  probes match raw residual strength, but the L45 MLP-output width-16K SAE is
  much weaker than the residual SAEs.
- GORMAN access is configured through `queue.cs.purdue.edu`. A scratch venv
  with CUDA 12.6 PyTorch and TransformerLens imports on a V100, so GORMAN is a
  plausible fallback for 27B fp16 extraction. It is not needed for the current
  MLP-site result because Scholar completed job `451090`.

Measured jobs:

- `449067`: successful 27B TL pilot on 2x A40, layer 30, 10 rows, output shape
  `[10, 5376]`.
- `449081`: J-node measurement run, batch sweep and SAE headroom. Artifact:
  `results/stage2/pilots/j2_stage2_measure_L30_h4_20260425T040105Z.json`.

## Research Scope

Primary question: do raw residual or SAE-feature probes reveal a signal for
whether Gemma solves the InAbHyD reasoning task, beyond metadata baselines such
as height and prompt length?

Current 27B scope:

- Model: Gemma 3 27B. Gemma 3 4B can be merged later from the teammate run.
- Tasks: `infer_property` and `infer_subtype`.
- Labels: Stage 1 `is_correct_strong`, with `parse_failed=True` filtered for
  the main probe training.
- Feature sources: raw residual stream, Gemma Scope 2 residual SAEs
  width-16K and width-262K, and skip-transcoder features only if the catalog and
  compute path are clean.
- Probe families: difference-of-means and logistic regression.
- Cross-task transfer within each model.
- Steering validation on 27B `infer_property` only, after stable SAE features
  have been selected.

Out of scope:

- `infer_membership_relation`.
- Multi-hypothesis examples.
- Training new SAEs.
- Running 4B in this workspace unless ownership changes.
- Steering on 4B.
- Steering on `infer_subtype`.
- Cross-model-family comparisons.
- Probing intermediate CoT tokens.

## Non-Negotiable Invariants

These must hold before interpreting any probe result:

1. Prompt reconstruction uses `src.messages.build_messages` and the model
   tokenizer chat template with `add_generation_prompt=True`.
2. Activations are taken at the last pre-generation token position after chat
   templating.
3. Stage 1 JSONLs match the SHA-256 hashes in `docs/stage2_invariants.json`.
4. Model and tokenizer revisions match `docs/stage2_invariants.json`.
5. Residual hooks are exactly `blocks.{L}.hook_resid_post` unless a verified
   TransformerLens hook-name change is recorded in the invariants file.
6. SAE release IDs and checkpoint hashes are pinned before full SAE feature
   extraction starts.
7. Any GPT judge calls use a dated model snapshot, not a moving alias.

Regenerate invariants after intentional artifact changes:

```bash
python scripts/stage2_write_invariants.py \
  --output docs/stage2_invariants.json \
  --jsonl-dir results/full/with_errortype \
  --hf-cache /scratch/scholar/$USER/hf-cache \
  --local-files-only
```

## Key Files

- `docs/stage2_invariants.json`: model/tokenizer revisions, chat-template
  checksums, and Stage 1 JSONL hashes.
- `docs/REPORT_NOTES.md`: running notes for final report writing. Add a dated
  bullet when we discover an important result, caveat, compute constraint, or
  design decision.
- `src/messages.py`: prompt construction.
- `src/activations.py`: residual extraction library.
- `scripts/stage2_extract.py`: raw residual extraction CLI.
- `scripts/validate_activations.py`: input/artifact validation gate for
  extraction.
- `scripts/stage2_extract_27b_pilot.sbatch`: small 27B layer-30 extraction
  pilot for both tasks on 2x A40.
- `scripts/stage2_layerpilot_27b_h4.sbatch`: layer-selection pilot for 27B
  height-4 rows across layers 15, 30, and 45.
- `scripts/stage2_extract_27b_full.sbatch`: full 27B raw residual extraction
  for both tasks and selected layers.
- `scripts/stage2_probe_raw.py`: quick logistic probe runner for raw residual
  safetensors.
- `scripts/stage2_write_invariants.py`: invariant writer.
- `scripts/stage2_j2_measure.py`: measurement-only J-node utility.
- `docs/STAGE_2_PLAN_ARCHIVE.md`: old long-form plan kept for reference.

## Phase 0: Data Controls

Do this before full extraction.

### 0.1 Invariants

Done for model/tokenizer revisions and Stage 1 JSONL hashes. Still pending:

- SAE release IDs and checkpoint hashes after layer selection.
- GPT judge snapshot ID before paraphrase or CoT-classification calls.

### 0.2 Inventory

Create `docs/stage2_inventory.json` with counts by:

- `(model, task, height, parse_failed, is_correct_strong)`;
- `(model, task, height)` totals and class balance after
  `parse_failed=False` filtering.

Known 27B cells to watch:

- `infer_property`, h=1: only 37 non-parse negative rows.
- `infer_subtype`, h=1: only 27 non-parse negative rows.
- `infer_property`, h=2-4: parse-failure rates are 9.55%, 8.77%, and 10.36%.
- `infer_subtype`, h=2: parse-failure rate is 6.75%.

### 0.3 Splits

Write `results/stage2/splits.jsonl`.

Required split families:

- S1: height-stratified random split, 70/15/15.
- S2: ontology-topology-held-out split. Current implementation records this
  split but marks it non-evaluable for 27B because canonical topology has too
  few groups to form train/val/test holdouts.
- S3: target-symbol-heldout split. The heldout group is the hypothesis subject
  (`target_concept` for property and target subtype for subtype), assigned
  globally per `(model, task)` so train/val/test do not share target symbols.
  This is an existing-data replacement for the non-evaluable S2, not a full
  name-scrambled regeneration.

S2 grouping must canonicalize ontology topology rather than raw concept names.
Record residual class or height imbalance when exact stratification is not
possible.

Do not report S2 probe metrics unless a richer topology generator produces
enough groups. Use S3 for the current heldout-symbol generalization diagnostic.

### 0.4 Metadata Baselines

Train B0 baselines before claiming activation probes are meaningful:

- B0-height: height only.
- B0-prompt: height, prompt token count, theory/observation counts, direct-path
  metadata, parent salience.
- B0-namefreq: B0-prompt plus train-set name-frequency features.

Post-output diagnostics are useful but not fair baselines:

- D0-strategy: output strategy labels.
- D0-parsefail: parse failure labels.

Every activation or SAE probe should report delta over the strongest matching
pre-output B0 baseline, not only absolute AUC.

Current 27B S1 thresholds:

- `infer_property`: beat `b0_prompt`, test AUC 0.743.
- `infer_subtype`: beat `b0_height`, test AUC 0.841.

### 0.5 Label-Shuffle Check

Train one raw-residual probe with shuffled labels on S1. Expected AUC is about
0.50. Add S2 only after there is an evaluable heldout design.

Current 27B status: done in `docs/raw_probe_27b_s1_label_shuffle.json`.
This control used the same S1 splits and layers as the main raw probe, with
`C=1.0` and `max_iter=300` to keep noisy-label optimization bounded. Some
fits reached the iteration cap, but selected-layer test AUCs stayed near
chance: 0.493 for property and 0.481 for subtype.

## Phase A: Activation Extraction

### A.1 Extraction Contract

For each `(model, task, layer)`:

- input is one shipped JSONL under `results/full/with_errortype/`;
- output is one safetensors file:
  `results/stage2/activations/{model}_{task}_L{layer}.safetensors`;
- tensor key is `activations`;
- shape is `[N, D]`;
- dtype is bf16;
- sidecar is `{model}_{task}_L{layer}.example_ids.jsonl`;
- sidecar preserves Stage 1 row order and includes row index, example ID,
  height, label, parse-failed flag, token count, and hook name.

Example 4B command for teammate use:

```bash
python scripts/stage2_extract.py \
  --jsonl results/full/with_errortype/gemma3_4b_infer_property.jsonl \
  --model google/gemma-3-4b-it \
  --model-key gemma3_4b \
  --layers 12,22,32 \
  --batch-size 32 \
  --n-devices 1 \
  --out-dir results/stage2/activations
```

Example 27B J-node command inside a 2x A40 allocation:

```bash
python scripts/stage2_extract.py \
  --jsonl results/full/with_errortype/gemma3_27b_infer_property.jsonl \
  --model google/gemma-3-27b-it \
  --model-key gemma3_27b \
  --layers 15,30,45 \
  --batch-size 32 \
  --n-devices 2 \
  --n-ctx 4096 \
  --out-dir results/stage2/activations
```

### A.2 Equivalence Check

Before full extraction, implement and run `scripts/validate_activations.py`.

Required tiers:

1. Prompt token IDs match Stage 1 reconstruction.
2. Model/tokenizer/chat-template revisions match invariants.
3. First-token top-5 logits agree with the Stage 1 serving stack where
   possible.
4. Greedy output byte match on 50 rows, if possible.
5. Label agreement on 200 regenerated rows is at least 98% if byte match fails.

Proceed only if tiers 1-3 pass and tier 5 passes when needed.

Current validator status:

- Local tiers 1-2 pass for both 27B task JSONLs.
- Property prompt token counts: min 134, mean 186.7, max 223.
- Subtype prompt token counts: min 140, mean 192.1, max 230.
- Tiers requiring the original Stage 1 serving stack are recorded as skipped.
- After the pilot extraction, rerun validator with `--activation-dir` and
  `--layers` to check safetensors shape, metadata, sidecar row order, hook
  names, and token counts.

Pilot artifact status:

- Job `449831` ran on `scholar-j002` from 03:14:18 to 03:25:52 EDT on
  2026-04-27.
- Property L30 pilot: 16 rows, shape `[16, 5376]`, bf16, 2.03 rows/s.
- Subtype L30 pilot: 16 rows, shape `[16, 5376]`, bf16, 5.52 rows/s.
- Whole-block device map was as expected: blocks 0, 1, and 30 on `cuda:0`;
  blocks 31, 60, and 61 plus final/unembed on `cuda:1`.

### A.3 Layer Selection Pilot

Run a full-layer sweep on a subset before committing to full extraction:

- For the current 27B run, start with 512 non-parse height-4 rows per task.
- Filter `parse_failed=True` for the main pilot.
- Train quick logistic probes per layer with `scripts/stage2_probe_raw.py`.
- Pick three layers per model based on probe performance and stability.
- Early/mid/late is acceptable only if the pilot is inconclusive.

Write selected layers to `docs/layer_selection.json`.

Current pilot job script: `scripts/stage2_layerpilot_27b_h4.sbatch` extracts
layers 15, 30, and 45 for 27B property/subtype, validates artifacts, and writes
`docs/layer_selection_pilot_27b_h4.json`.

Pilot result: `docs/layer_selection.json` selects layers 15, 30, and 45 for
full 27B extraction. Do not overinterpret the pilot probe AUCs; subtype h4
holdout splits had only four positives each.

### A.4 Full Extraction

After layer selection:

- extract Gemma 3 27B for this workspace run;
- extract both tasks;
- extract all three selected layers;
- verify row counts and shapes immediately after each file.

Expected full raw residual files for this 27B run: 6.

Current full extraction script: `scripts/stage2_extract_27b_full.sbatch`. It
extracts all rows for property and subtype at layers 15, 30, and 45, then writes
full artifact validation reports under `docs/`.

Current full extraction status: done for 27B. Property ran at 6.34 rows/s and
subtype ran at 6.19 rows/s while capturing three layers.

## Phase B: Raw Residual Probes

Train probes on raw residuals for each `(model, task, layer)`.

Probe requirements:

- train on S1 for the main random-split result and S3 for target-symbol
  heldout generalization;
- filter `parse_failed=True` for main analyses;
- standardize features using train statistics only;
- report aggregate AUC and per-height AUC;
- report delta over B0 baselines;
- include confidence intervals by bootstrap over examples.

Current 27B S1 point metrics are in `docs/raw_probe_27b_s1.json`:

| Task | B0 threshold | Best raw layer | Val AUC | Test AUC | 95% bootstrap CI | Delta vs B0 |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| `infer_property` | 0.743 | L45 | 0.881 | 0.897 | [0.881, 0.912] | +0.153 |
| `infer_subtype` | 0.841 | L45 | 0.917 | 0.914 | [0.896, 0.932] | +0.073 |

All selected raw layers beat the task-matched B0 thresholds on S1. Per-height
AUCs are useful diagnostics, but h1 values are unstable because the S1 heldout
has only 4-5 negatives for h1.

Cross-task transfer uses source-task validation to select the layer/C and
evaluates only once on the target-task test split:

| Direction | Selected layer | Source test AUC | Target test AUC | Target 95% bootstrap CI |
| --- | --- | ---: | ---: | ---: |
| `infer_property` -> `infer_subtype` | L45 | 0.897 | 0.862 | [0.837, 0.884] |
| `infer_subtype` -> `infer_property` | L45 | 0.914 | 0.786 | [0.763, 0.809] |

Current 27B S3 target-symbol-heldout metrics:

| Task | B0 threshold | Best raw layer | Val AUC | Test AUC | 95% bootstrap CI | Delta vs B0 |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| `infer_property` | 0.711 | L45 | 0.875 | 0.884 | [0.868, 0.901] | +0.173 |
| `infer_subtype` | 0.859 | L45 | 0.909 | 0.917 | [0.898, 0.934] | +0.058 |

S3 cross-task transfer:

| Direction | Selected layer | Source test AUC | Target test AUC | Target 95% bootstrap CI |
| --- | --- | ---: | ---: | ---: |
| `infer_property` -> `infer_subtype` | L45 | 0.884 | 0.846 | [0.823, 0.872] |
| `infer_subtype` -> `infer_property` | L45 | 0.917 | 0.788 | [0.766, 0.810] |

Interpretation: the raw L45 success/failure signal survives heldout target
symbols and still beats task-matched metadata baselines. S3 does not prove full
name-scramble invariance, but it reduces the risk that S1 was driven only by
repeated target lexical items.

Transfer and comparison:

- Cross-task transfer within the same model:
  - train on `infer_property`, test on `infer_subtype`;
  - train on `infer_subtype`, test on `infer_property`.
- Cross-model comparison:
  - compare metrics tables across 4B and 27B;
  - do not transfer raw linear directions across model widths.

Diagnostics:

- prompt length regression;
- name-scramble re-extraction;
- parse-failed 3-way analysis only as interpretation, not the main result.

Outputs:

- `results/stage2/probes/`;
- `results/stage2/probe_auc.json`;
- `results/stage2/probe_transfer.json`;
- `results/stage2/probe_comparison.json`;
- `results/stage2/probe_diagnostics.json`.

## Phase C: SAE Feature Extraction

Use cached raw residuals as the input. Do not keep the 27B model resident while
running SAE feature extraction unless the extraction path truly needs it.

J-node defaults from measurement:

- one SAE per layer at a time;
- chunk size 512;
- width-16K and width-262K both fit for layer 30 on 2x A40 with resident 27B,
  so they should fit comfortably when run from cached residuals.

Before starting:

- pin exact SAE release IDs and checkpoint hashes in
  `docs/stage2_invariants.json`;
- record actual SAE IDs exactly as SAE Lens reports them.

Current pilot status:

- `scripts/stage2_extract_sae_features.py` encodes cached residual files into
  sparse top-k SAE feature files with tensors `top_values`, `top_indices`, and
  `l0`.
- `scripts/stage2_sae_pilot_27b_L45_16k.sbatch` ran as job `449999` on
  `scholar-j001`. It encoded 512 rows for each 27B task at L45 using
  `gemma-scope-2-27b-it-res-all/layer_45_width_16k_l0_small`.
- Pilot tensors have `top_values`/`top_indices` shape `[512, 128]` and `l0`
  shape `[512]`. Mean L0 was 17.57 for property and 17.96 for subtype.
- The L45 width-16K and width-262K SAEs are pinned in
  `docs/stage2_invariants.json`; any additional layers still need pinning
  before extraction.
- Full L45 width-16K extraction job `450004` wrote two full feature files under
  `results/stage2/sae_features/`. Each has `top_values`/`top_indices` shape
  `[11000, 128]` and `l0` shape `[11000]`.
- Full L45 width-262K extraction job `450029` wrote two full feature files
  under `results/stage2/sae_features/`. Each has `top_values`/`top_indices`
  shape `[11000, 128]` and `l0` shape `[11000]`.
- L45 width-16K top-512 diagnostic job `450038` wrote two full feature files.
  Each has `top_values`/`top_indices` shape `[11000, 512]`; ranks after 128 are
  all zero because the largest observed L0 was 24 for property and 23 for
  subtype.

Outputs:

- sparse SAE feature files under `results/stage2/sae_features/`;
- metadata recording release, SAE ID, layer, width, chunk size, and source
  residual file.

## Phase D: SAE Feature Probes

Train the same probe families on SAE features.

Required comparisons:

- raw residual probe vs width-16K SAE probe;
- raw residual probe vs width-262K SAE probe;
- dense SAE features vs sparse top-k/logistic features;
- stability of top features across splits, seeds, widths, and adjacent layers.

Only features that are stable and beat B0 baselines should be considered for
causal validation.

Current L45 S1 metrics:

| Task | B0 threshold | SAE width | Top-k | SAE test AUC | 95% bootstrap CI | Raw L45 test AUC |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| `infer_property` | 0.743 | 16K | 128 | 0.786 | [0.763, 0.808] | 0.897 |
| `infer_property` | 0.743 | 16K | 512 | 0.786 | [0.763, 0.808] | 0.897 |
| `infer_property` | 0.743 | 262K | 128 | 0.806 | [0.784, 0.828] | 0.897 |
| `infer_subtype` | 0.841 | 16K | 128 | 0.876 | [0.852, 0.899] | 0.914 |
| `infer_subtype` | 0.841 | 16K | 512 | 0.876 | [0.852, 0.899] | 0.914 |
| `infer_subtype` | 0.841 | 262K | 128 | 0.870 | [0.845, 0.895] | 0.914 |

Both SAE widths beat B0 but do not recover the full raw-residual signal. Width
262K modestly improves property but not subtype. Top-512 on width-16K exactly
matches top-128 because top-128 already retains every nonzero active feature.
Observed top-128 max L0 was 24/23 for width-16K property/subtype and 22/22 for
width-262K property/subtype, so a top-512 width-262K rerun is not useful. Next
checks should focus on feature stability, adjacent layers, and raw-vs-SAE
reconstruction/residual diagnostics before any steering decision.

Current L45 S3 target-symbol-heldout SAE metrics:

| Task | B0 threshold | SAE width | Top-k | SAE test AUC | 95% bootstrap CI | Raw L45 test AUC |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| `infer_property` | 0.711 | 16K | 128 | 0.799 | [0.777, 0.820] | 0.884 |
| `infer_property` | 0.711 | 262K | 128 | 0.779 | [0.757, 0.802] | 0.884 |
| `infer_subtype` | 0.859 | 16K | 128 | 0.865 | [0.838, 0.889] | 0.917 |
| `infer_subtype` | 0.859 | 262K | 128 | 0.867 | [0.839, 0.892] | 0.917 |

Interpretation: the SAE-heldout result is weaker than raw S3. Property remains
comfortably above B0, but subtype is only marginally above the metadata
baseline, so S3 strengthens the conclusion that the raw residual signal is more
robust than the tested Gemma Scope SAE feature representations.

Current L45 top-feature stability notes:

- `docs/sae_feature_stability_27b_l45_s1.json` refits the saved-best-C probes
  on train-active SAE columns, then records standardized logistic coefficients,
  feature activation densities, same-width task overlap, and cross-width
  activation-pattern correlations.
- The effective train-active feature support is small: 62/46 distinct width-16K
  features for property/subtype and 82/82 distinct width-262K features, despite
  nominal SAE widths of 16,384 and 262,144.
- Same-width task overlap is meaningful but not complete. Width-16K has 4/10
  same-sign top-10 overlap features (`1096`, `19`, `180`, `4329`) and 14/15
  same-sign top-25 overlaps. Width-262K has 4/10 same-sign top-10 overlap
  features (`368`, `160112`, `64600`, `9994`) and 6/7 same-sign top-25
  overlaps.
- Cross-width top-50 activation-pattern matches are partial. For property,
  16/50 width-16K top features have a best width-262K match at abs correlation
  at least 0.5; for subtype, 17/46 do. These are useful candidates, but feature
  IDs are not width-comparable and several high-weight features are very dense,
  so this is not yet a clean localized mechanism.

Current L45 reconstruction/error diagnostic:

| Task | SAE width | Energy explained | Reconstruction test AUC | Error test AUC | Raw L45 test AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| `infer_property` | 16K | 0.948 | 0.786 | 0.894 | 0.897 |
| `infer_property` | 262K | 0.955 | 0.806 | 0.897 | 0.897 |
| `infer_subtype` | 16K | 0.948 | 0.877 | 0.916 | 0.914 |
| `infer_subtype` | 262K | 0.954 | 0.870 | 0.915 | 0.914 |

Interpretation: the Gemma Scope L45 residual SAEs reconstruct most activation
energy, but the correctness-predictive raw direction is largely in the small
reconstruction error rather than in the decoded SAE subspace. This explains why
sparse SAE-feature probes trail raw residual probes even when top-k truncation
is absent. It also argues against moving directly to SAE-feature steering as the
main causal test; a raw-direction or reconstruction-error diagnostic should be
used before selecting a steering target.

Current L45 S3 target-symbol-heldout reconstruction/error diagnostic:

| Task | SAE width | Energy explained | Reconstruction test AUC | Error test AUC | Raw L45 test AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| `infer_property` | 16K | 0.948 | 0.799 | 0.881 | 0.884 |
| `infer_property` | 262K | 0.955 | 0.788 | 0.886 | 0.884 |
| `infer_subtype` | 16K | 0.948 | 0.865 | 0.916 | 0.917 |
| `infer_subtype` | 262K | 0.954 | 0.867 | 0.914 | 0.917 |

The S3 component probe result repeats the S1 pattern under heldout target
symbols. This reduces the chance that the raw-SAE gap is a random-split lexical
artifact.

Targeted Gemma Scope site pilot:

- `scripts/stage2_mlp_site_27b_L45_16k.sbatch` is the first non-residual site
  pilot. It extracts L45 `mlp_out` activations for both 27B tasks, encodes
  `gemma-scope-2-27b-it-mlp-all/layer_45_width_16k_l0_small`, and writes raw
  same-site plus sparse SAE S1/S3 probe summaries.
- Result: the MLP site itself carries the signal, but the MLP-output SAE does
  not. Raw L45 `mlp_out` test AUCs were S1 property/subtype 0.895/0.916 and S3
  property/subtype 0.892/0.915. The width-16K MLP-output SAE reached only
  0.577/0.674 on S1 and 0.550/0.702 on S3.
- Interpretation: non-residual site features do not rescue the SAE story. Do
  not spend project time on a crosscoder pilot unless the report needs it as a
  clearly labeled future-work check; any such pilot would need a raw-concat
  baseline over the same layer set.

Outputs:

- `results/stage2/sae_probes/`;
- `results/stage2/sae_probe_auc.json`;
- `docs/sae_feature_stability_27b_l45_s1.json`;
- `docs/sae_reconstruction_probe_27b_l45_s1.json`;
- `docs/sae_probe_27b_l45_16k_s3_target_symbol.json`;
- `docs/sae_probe_27b_l45_262k_s3_target_symbol.json`;
- `docs/sae_reconstruction_probe_27b_l45_s3_target_symbol.json`;
- `docs/raw_probe_27b_l45_mlp_out_s1.json` and
  `docs/raw_probe_27b_l45_mlp_out_s3_target_symbol.json`;
- `docs/sae_probe_27b_l45_mlp_out_16k_s1.json` and
  `docs/sae_probe_27b_l45_mlp_out_16k_s3_target_symbol.json`;
- `results/stage2/stable_features.json` if a later pass selects a steering set.

## Phase E: Steering Validation

Scope: Gemma 3 27B on `infer_property` only.

Current pilot status:

- `scripts/stage2_steer_raw_direction.py` refits the saved S1 raw L45 logistic
  probe, recovers the raw residual-space direction, samples a matched
  orthogonal control direction, and runs deterministic local TL generation.
- `scripts/stage2_steer_raw_27b_L45_property_pilot.sbatch` is the bounded
  Scholar job. Current defaults are 8 balanced S1 test rows, prompt-only
  interventions, raw +/-2 SD, orthogonal +/-2 SD, and `max_new_tokens=64`.
- Job `450140` completed on `scholar-j001` in 1,997 seconds. The pilot found no
  strong-correctness flips for either raw direction or orthogonal controls.
  Strong accuracy was 3/8 for every condition; parse failures were 0/8 baseline
  and 1/8 for every steered condition. All generations reached the 64-token
  cap, so future runs should improve the stop/length protocol before scaling.
- Interpretation: prompt-only steering at this strength is not enough evidence
  for causal control. If steering remains a priority, test a different
  intervention design next, such as all-token L45 steering or a strength sweep,
  before scaling up sample size.

Run a staged steering experiment:

1. Small pilot on a balanced subset to confirm intervention plumbing and output
   validity.
2. Main run only for stable SAE features or raw directions that survived Phase D.
3. Orthogonal/random-direction controls.

Interventions:

- steer top correctness-associated SAE features up and down;
- steer raw probe direction up and down;
- compare to random orthogonal directions with matched norm;
- optionally test pre-CoT-only vs all-token intervention if the pilot suggests
  strong sensitivity.

Metrics:

- strong and weak correctness;
- parse-failure rate;
- output length;
- label flips by original correctness;
- qualitative samples for report figures.

Do not run expensive steering sweeps until raw and SAE probes clearly beat B0
baselines.

## Phase F: Write-Up

Keep `docs/REPORT_NOTES.md` current as work proceeds. Add short dated notes for:

- unexpected compute constraints;
- measured throughput or memory limits;
- invariant or equivalence failures;
- probe results that change the scientific interpretation;
- steering effects or null results;
- caveats that should appear in the final report.

The final report should emphasize:

- Stage 1 behavioral label construction and dataset scope;
- why metadata baselines are necessary;
- whether activations add predictive signal beyond those baselines;
- whether SAE features localize a stable mechanism;
- whether steering causally changes behavior or only disrupts generation.

## Deliverables Checklist

Phase 0:

- [x] `docs/stage2_invariants.json` with model/tokenizer revisions and Stage 1
  JSONL SHA-256s.
- [x] L45 width-16K and width-262K SAE releases pinned in
  `docs/stage2_invariants.json`.
- [ ] Additional SAE release IDs pinned before additional SAE extraction.
- [ ] GPT judge snapshot pinned if judge calls are used.
- [x] `docs/stage2_inventory.json` for Gemma 3 27B.
- [x] `results/stage2/splits.jsonl` for Gemma 3 27B; S1 evaluable, S2 recorded
  but non-evaluable, S3 target-symbol heldout evaluable.
- [x] B0 metadata baselines for Gemma 3 27B S1.
- [x] B0 metadata baselines for Gemma 3 27B S3.
- [x] Replace S2 for current reporting with S3 target-symbol heldout.
- [x] Label-shuffle sanity check.

Phase A:

- [x] `scripts/validate_activations.py`.
- [x] 27B input validation reports for both tasks.
- [x] Pilot artifact validation reports for both tasks.
- [x] `docs/layer_selection.json` for Gemma 3 27B.
- [x] Full artifact validation reports for both 27B tasks.
- [x] 6 raw residual activation files for Gemma 3 27B.
- [x] 6 `.example_ids.jsonl` sidecars for Gemma 3 27B.

Phase B:

- [x] Raw residual probes for Gemma 3 27B, both tasks, and selected layers.
- [x] S1 point metrics.
- [x] Bootstrap confidence intervals for raw residual probes.
- [x] S3 target-symbol-heldout raw residual probes.
- [x] Cross-task transfer tables.
- [ ] Cross-model comparison tables after teammate 4B results are available.
- [ ] Diagnostics against prompt length and name scrambling.

Phase C/D:

- [x] L45 width-16K SAE pilot extraction complete.
- [x] L45 width-16K full top-128 SAE extraction complete.
- [x] L45 width-16K top-128 SAE probe metrics.
- [x] L45 width-262K full top-128 SAE extraction complete.
- [x] L45 width-262K top-128 SAE probe metrics.
- [x] L45 width-16K top-512 truncation diagnostic.
- [x] L45 width-16K/262K top-feature stability diagnostic.
- [x] L45 width-16K/262K reconstruction/error probe diagnostic.
- [x] L45 width-16K/262K S3 target-symbol SAE probe metrics.
- [x] L45 width-16K/262K S3 reconstruction/error probe diagnostic.
- [x] L45 MLP-output width-16K site pilot.
- [x] Additional SAE release IDs pinned.
- [ ] SAE feature extraction complete.
- [ ] Broader SAE probe metrics.
- [ ] Stable feature set selected.

Phase E:

- [x] Steering pilot.
- [ ] Main steering run if justified by Phase D.
- [x] Orthogonal-direction baseline for the pilot.
- [x] Pilot qualitative examples and quantitative steering metrics.
