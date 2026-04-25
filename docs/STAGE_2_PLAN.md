# Stage 2 Plan: Probes, SAE Features, and Steering

This is the operational plan for Stage 2. It keeps only the decisions,
commands, invariants, and deliverables needed to run the work. Longer
reasoning, dead alternatives, and budget arithmetic were moved out of the main
plan so this file can stay useful during execution.

## Current Status

As of 2026-04-25:

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

Measured jobs:

- `449067`: successful 27B TL pilot on 2x A40, layer 30, 10 rows, output shape
  `[10, 5376]`.
- `449081`: J-node measurement run, batch sweep and SAE headroom. Artifact:
  `results/stage2/pilots/j2_stage2_measure_L30_h4_20260425T040105Z.json`.

## Research Scope

Primary question: do raw residual or SAE-feature probes reveal a signal for
whether Gemma solves the InAbHyD reasoning task, beyond metadata baselines such
as height and prompt length?

In scope:

- Models: Gemma 3 4B and Gemma 3 27B.
- Tasks: `infer_property` and `infer_subtype`.
- Labels: Stage 1 `is_correct_strong`, with `parse_failed=True` filtered for
  the main probe training.
- Feature sources: raw residual stream, Gemma Scope 2 residual SAEs
  width-16K and width-262K, and skip-transcoder features only if the catalog and
  compute path are clean.
- Probe families: difference-of-means and logistic regression.
- Cross-task transfer within each model.
- Cross-model comparison as tables, not raw probe transfer.
- Steering validation on 27B `infer_property` only, after stable SAE features
  have been selected.

Out of scope:

- `infer_membership_relation`.
- Multi-hypothesis examples.
- Training new SAEs.
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

Known cells to watch:

- 4B `infer_subtype`, h=3: low positive count.
- 4B `infer_subtype`, h=4: low positive count.

### 0.3 Splits

Write `results/stage2/splits.jsonl`.

Required split families:

- S1: height-stratified random split, 70/15/15.
- S2: ontology-topology-held-out split, also 70/15/15 where possible.
- S3: name-scrambled generalization set, generated later for diagnostics.

S2 grouping must canonicalize ontology topology rather than raw concept names.
Record residual class or height imbalance when exact stratification is not
possible.

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

### 0.5 Label-Shuffle Check

Train one raw-residual probe with shuffled labels on S1 and S2. Expected AUC is
about 0.50. Anything materially higher suggests leakage.

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

Example 4B command:

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

### A.3 Layer Selection Pilot

Run a full-layer sweep on a subset before committing to full extraction:

- Start with 500 rows per model/task if compute allows.
- Filter `parse_failed=True` for the main pilot.
- Train quick probes per layer.
- Pick three layers per model based on probe performance and stability.
- Early/mid/late is acceptable only if the pilot is inconclusive.

Write selected layers to `docs/layer_selection.json`.

### A.4 Full Extraction

After layer selection:

- extract both models;
- extract both tasks;
- extract all three selected layers;
- verify row counts and shapes immediately after each file.

Expected full raw residual files: 12.

## Phase B: Raw Residual Probes

Train probes on raw residuals for each `(model, task, layer)`.

Probe requirements:

- train on S1 and S2;
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
- [ ] `docs/stage2_inventory.json`.
- [ ] `results/stage2/splits.jsonl`.
- [ ] B0 metadata baselines.
- [ ] Label-shuffle sanity check.

Phase A:

- [ ] `scripts/validate_activations.py`.
- [ ] `results/stage2/equivalence_report.json`.
- [ ] `docs/layer_selection.json`.
- [ ] 12 raw residual activation files.
- [ ] 12 `.example_ids.jsonl` sidecars.

Phase B:

- [ ] Raw residual probes for both models, both tasks, and selected layers.
- [ ] S1 and S2 metrics.
- [ ] Cross-task transfer tables.
- [ ] Cross-model comparison tables.
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
