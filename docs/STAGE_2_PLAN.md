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
  generated for Gemma 3 27B. S1 is evaluable for both tasks and all heights.
  The planned topology-heldout S2 is not evaluable on this shipped dataset
  because canonical topology leaves only one or two groups per task/height.
- `docs/stage2_b0_summary_27b_s1.json` records 27B S1 metadata baselines.
  Strongest pre-output baselines are AUC 0.743 for `infer_property`
  (`b0_prompt`) and AUC 0.841 for `infer_subtype` (`b0_height`).
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
- S3: name-scrambled generalization set, generated later for diagnostics.

S2 grouping must canonicalize ontology topology rather than raw concept names.
Record residual class or height imbalance when exact stratification is not
possible.

Do not report S2 probe metrics until an alternative heldout definition is
chosen or a richer topology generator produces enough groups.

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

## Phase B: Raw Residual Probes

Train probes on raw residuals for each `(model, task, layer)`.

Probe requirements:

- train on S1 now; add S2 only after the heldout definition is redesigned;
- filter `parse_failed=True` for main analyses;
- standardize features using train statistics only;
- report aggregate AUC and per-height AUC;
- report delta over B0 baselines;
- include confidence intervals by bootstrap over examples.

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

Outputs:

- `results/stage2/sae_probes/`;
- `results/stage2/sae_probe_auc.json`;
- `results/stage2/stable_features.json`.

## Phase E: Steering Validation

Scope: Gemma 3 27B on `infer_property` only.

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
- [ ] SAE release IDs pinned in `docs/stage2_invariants.json`.
- [ ] GPT judge snapshot pinned if judge calls are used.
- [x] `docs/stage2_inventory.json` for Gemma 3 27B.
- [x] `results/stage2/splits.jsonl` for Gemma 3 27B; S1 evaluable, S2 recorded
  but non-evaluable.
- [x] B0 metadata baselines for Gemma 3 27B S1.
- [ ] Redesign or replace S2 before reporting heldout-topology results.
- [ ] Label-shuffle sanity check.

Phase A:

- [x] `scripts/validate_activations.py`.
- [x] 27B input validation reports for both tasks.
- [x] Pilot artifact validation reports for both tasks.
- [x] `docs/layer_selection.json` for Gemma 3 27B.
- [ ] `results/stage2/equivalence_report.json` or equivalent merged report.
- [ ] 6 raw residual activation files for Gemma 3 27B.
- [ ] 6 `.example_ids.jsonl` sidecars for Gemma 3 27B.

Phase B:

- [ ] Raw residual probes for Gemma 3 27B, both tasks, and selected layers.
- [ ] S1 metrics.
- [ ] S2 metrics only after redesign.
- [ ] Cross-task transfer tables.
- [ ] Cross-model comparison tables after teammate 4B results are available.
- [ ] Diagnostics against prompt length and name scrambling.

Phase C/D:

- [ ] SAE release IDs pinned.
- [ ] SAE feature extraction complete.
- [ ] SAE probe metrics.
- [ ] Stable feature set selected.

Phase E:

- [ ] Steering pilot.
- [ ] Main steering run if justified by Phase D.
- [ ] Orthogonal-direction baseline.
- [ ] Qualitative examples and quantitative steering metrics.
