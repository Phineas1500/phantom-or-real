# Report Notes

Use this file as a running memory for final-report-relevant facts,
measurements, caveats, and decisions. Add short dated notes as we work so the
final report is easier to assemble.

## Final Project Report Guidelines

* **(a) Format & Length:** The final report must contain between **4 and 6 pages** in [ICLR 2026 format](https://github.com/ICLR/Master-Template/raw/master/iclr2026.zip).
* **(b) Section "Introduction" (~1 page):** Describe the objective and clearly define the task statistically (e.g., learn $p(x)$, learn $p(y|x)$ where train and test data have the same distribution, etc.).
* **(c) Section "Dataset" (~1/4 page):** Describe the dataset used and explain why it aligns with your task.
* **(d) Section "Proposed Approach" (2+ pages):** Describe the deep learning methods that will be used (or tried). Formally write down the objective function and describe why this objective aligns with the one outlined in the introduction.
* **(e) Section "Related Work" (~1/2 page):** Describe related literature and explain how it connects to your proposed work.
* **(f) Section "Results" (1+ page):** Describe your experimental results and the metrics used to evaluate success.

## Running Notes

### 2026-04-25

#### Stage 1 Retrospective

- Generated two Stage 1 prompt datasets from InAbHyD single-hypothesis tasks:
  `data/pilot/` has 400 prompt examples (50 per height x 4 heights x 2 tasks),
  and `data/full/` has 22,000 prompt examples (1,000/2,000/3,000/5,000 at
  heights 1-4 x 2 tasks). Running both Gemma 3 4B and Gemma 3 27B over the full
  prompts produced 44,000 labeled model rows.
- Shipped Stage 1 rows are the four JSONLs under
  `results/full/with_errortype/`, each with 11,000 rows:
  `gemma3_27b_infer_property`, `gemma3_27b_infer_subtype`,
  `gemma3_4b_infer_property`, and `gemma3_4b_infer_subtype`.
- Full-data seeds are pinned in `src.config.SHIPPED_SEEDS`. The `data/full/`
  pickles are the authoritative prompt source because upstream ontology
  construction still has some set-iteration nondeterminism even with fixed RNG
  seeds; regeneration is statistically equivalent but not guaranteed
  byte-identical.
- Inference used Gemma 3 4B-IT and 27B-IT through OpenAI-compatible vLLM-style
  endpoints at `temperature=0`. For Gemma, Stage 1 concatenated system and user
  content into one user message because the Gemma chat template does not accept
  a `system` role; Stage 2 must preserve this exactly.
- Scoring used the upstream InAbHyD v2 strong/weak accuracy pipeline with the
  first parsed hypothesis as the main strong label. We extended the Gemma parser
  for common rephrasings such as "Being X implies being Y" and treated hedged
  disjunctions like "X is Y or X is not Y" as parse failures rather than
  arbitrary guesses.
- Headline strong accuracy falls monotonically with height. 27B property:
  96.0%, 57.7%, 39.2%, 26.4%. 27B subtype: 97.3%, 32.5%, 11.4%, 5.5%.
  4B property: 74.0%, 29.6%, 20.5%, 16.4%. 4B subtype: 76.9%, 11.6%,
  1.3%, 0.9%.
- Positive counts are adequate for most Stage 2 probe cells, but 4B subtype is
  sparse at h=3 and h=4: only 40 and 45 strong-correct rows respectively.
  Those per-height probe results need wide intervals or aggregation.
- Structural annotation found `has_direct_member=True` for every shipped row.
  This happened after fixing an upstream `normalize_to_singular` bug affecting
  Thomas, Charles, James, and Nicholas. Consequence for the report: the planned
  `has_direct_member=True` vs `False` shortcut-availability slice is vacuous on
  this shipped dataset.
- 4B and 27B diverge in output strategy at depth. 4B falls back to entity-level
  enumeration heavily on subtype h>=3, while 27B remains mostly concept-level.
  This is a major caveat for interpreting 4B probes: a correctness probe may be
  detecting strategy choice rather than reasoning success.
- Error-type labeling covered 33,294 incorrect rows. The 200-row nano-vs-mini
  agreement check was only 57.5%, so the full shipped error labels used
  `gpt-5.4-mini`. Dominant qualitative modes: 27B property is often
  `unnecessary`; subtype failures are mostly `wrong_direction`.
- Stage 1 checks run and/or recorded:
  `tests/test_pipeline_smoke.py` mocks OpenAI calls and verifies row schema,
  scoring, and structural annotations; `tests/test_analysis_smoke.py` exercises
  `scripts/sanity_check.py` and `scripts/make_plots.py` on synthetic data;
  `scripts/validate_annotations.py` passed with 100% `has_direct_member` on the
  validation samples; `scripts/sanity_check.py` on the shipped full JSONLs found
  no integrity issues, only parse-failure warnings for 27B property h=2-4 and
  27B subtype h=2.
- Stage 1 report artifacts already exist:
  `docs/behavioral_results_draft.md`, `results/full/summary_accuracy.json`,
  `results/full/summary_by_structure.json`, `results/full/sanity.json`, and
  figures under `docs/figures/full_with_errortype/`.

#### Stage 2 Setup And Measurement

- Recreated the Scholar environment from the project `phantom` environment files
  rather than the older CS373 environment. This matters for reproducibility:
  Stage 2 uses `transformer-lens==3.0.0`, `sae-lens==6.39.0`, and
  `scikit-learn==1.8.0`.
- Verified Gemma 3 27B activation extraction on a Scholar J node with 2x A40
  GPUs. TransformerLens 3.0 multi-GPU works, but needs deterministic whole-block
  device remapping after load. Use
  `HookedTransformer.from_pretrained_no_processing`, `n_devices=2`, bf16, and
  high CPU RAM (`--mem=180G` was sufficient).
- J-node measurement job `449081` found raw residual extraction batch size 32 is
  a conservative default for 27B on 2x A40; batch size 64 also worked for 64
  height-4 prompts with 202-217 tokens but has less margin.
- The same J-node measurement loaded Gemma Scope 2 layer-30 residual SAEs
  width-16K and width-262K alongside the resident 27B model. For production,
  cache residuals first and run one SAE/layer at a time with chunk size 512.
- Generated `docs/stage2_invariants.json`. It pins all four shipped Stage 1
  JSONL SHA-256 hashes and the Gemma 3 4B/27B model-tokenizer revisions used for
  Stage 2 prompt reconstruction.
- Shortened `docs/STAGE_2_PLAN.md` from a long planning memo into an operational
  checklist. Removed budget arithmetic, stale alternatives, and verbose
  justification while keeping invariants, measured compute defaults, extraction
  contracts, and deliverables.

### 2026-04-27

#### Stage 2 Phase 0 Controls

- Current execution scope is Gemma 3 27B only; the teammate is handling Gemma 3
  4B. The new Stage 2 Phase 0 scripts keep `--models`/`--tasks` filters so the
  same controls can be rerun for 4B later without changing code.
- Generated `docs/stage2_inventory.json` for 27B only: 22,000 rows across
  `infer_property` and `infer_subtype`. Warnings: low non-parse negative counts
  at height 1 for property (37) and subtype (27); high parse-failure rates for
  property h=2-4 (9.55%, 8.77%, 10.36%) and subtype h=2 (6.75%).
- Generated `results/stage2/splits.jsonl` and
  `docs/stage2_splits_summary.json` for 27B only. S1 is evaluable for both
  tasks and all heights after filtering parse failures.
- Important caveat: the planned canonical ontology-topology-heldout S2 is not
  evaluable on the shipped dataset. After anonymizing names, each task/height
  cell has only one or two topology groups, so S2 cannot form train/val/test
  heldouts. Do not report S2 probe metrics until the heldout design changes.
- Ran metadata-only B0 baselines on 27B S1 using local Hugging Face tokenizer
  prompt lengths. Strongest pre-output baselines: `infer_property` uses
  `b0_prompt` with test AUC 0.743; `infer_subtype` uses `b0_height` with test
  AUC 0.841. Activation probes should be interpreted as useful only if they
  beat these thresholds on the matching task/split.
- Added focused tests for Stage 2 Phase 0 helpers. The tests cover topology
  hashes ignoring symbol names, inventory warnings, split assignment coverage,
  and explicit non-evaluability warnings for degenerate S2 splits.

#### Stage 2 Activation Validation

- Implemented `scripts/validate_activations.py`, which checks Stage 1 JSONL
  hashes, local model/tokenizer/chat-template invariants, Gemma prompt
  reconstruction, prompt token counts, and optional activation artifacts
  (`.safetensors`, `.meta.json`, and `.example_ids.jsonl`).
- Ran the validator on both 27B Stage 1 JSONLs with local Hugging Face cache and
  `n_ctx=4096`. Both reports are `ok`: property prompt lengths are 134-223
  tokens with mean 186.7; subtype prompt lengths are 140-230 tokens with mean
  192.1.
- The validator records Stage 1 serving-stack top-5-logit and greedy-output
  byte-match checks as skipped because they require the original serving stack
  or a comparable live endpoint. Local extraction should still proceed because
  hashes, tokenizer revision, chat template, and prompt tokenization pass.
- Added `scripts/stage2_extract_27b_pilot.sbatch`, a 2x A40 pilot job that
  extracts layer 30 for 16 height-4 rows from each 27B task, then validates the
  written activation artifacts. This is the next gate before layer-selection
  extraction.
- Pilot job `449831` succeeded on `scholar-j002` from 03:14:18 to 03:25:52 EDT.
  It extracted layer-30 residuals for 16 height-4 rows from each 27B task and
  validated both artifact sets. Each safetensors file has shape `[16, 5376]`
  and dtype bf16; sidecar validation found zero row-order or token-count
  mismatches.
- The pilot confirmed the TransformerLens whole-block remapping is working:
  blocks 0, 1, and 30 were on `cuda:0`, while blocks 31, 60, 61, `ln_final`,
  and `unembed` were on `cuda:1`. Recorded extraction rates were 2.03 rows/s
  for property and 5.52 rows/s for subtype on this tiny run.
- Added `scripts/stage2_probe_raw.py` and
  `scripts/stage2_layerpilot_27b_h4.sbatch` for the next layer-selection gate.
  The layer pilot extracts layers 15, 30, and 45 on 512 non-parse height-4 rows
  per 27B task, validates artifacts, and trains quick logistic probes to write
  `docs/layer_selection_pilot_27b_h4.json`.
- Layer-selection job `449832` succeeded on `scholar-j001` from 03:30:37 to
  03:43:48 EDT. It extracted six pilot files: property/subtype x layers 15, 30,
  and 45, each with shape `[512, 5376]`, and validation passed for all artifacts.
  Extraction throughput was about 5.70 rows/s for property and 5.59 rows/s for
  subtype when capturing three layers.
- Quick h4-only probe results were inconclusive for choosing one best layer.
  Property validation favored L45 (val AUC 0.791, test AUC 0.683), while subtype
  validation also favored L45 but test AUC dropped to 0.5625; subtype holdouts
  had only four positives each. Decision recorded in `docs/layer_selection.json`:
  keep the early/mid/late set `[15, 30, 45]` for full 27B extraction.
- Added `scripts/stage2_extract_27b_full.sbatch` for full 27B raw residual
  extraction: both tasks, all rows, selected layers 15/30/45, batch size 32,
  2x A40, followed by full artifact validation reports.
- Full extraction job `449835` succeeded on `scholar-j002` from 03:47:45 to
  04:56:55 EDT. It wrote all six 27B raw residual files under
  `results/stage2/activations/`: property/subtype x layers 15, 30, and 45.
  Each file has shape `[11000, 5376]`, dtype bf16, and matching sidecars.
- Full artifact validation passed for both tasks. Property prompt lengths were
  134-223 tokens, mean 186.7; subtype prompt lengths were 140-230 tokens, mean
  192.1. Sidecar validation found zero sampled row-order/token-count
  mismatches. Extraction rates while capturing three layers were 6.34 rows/s for
  property and 6.19 rows/s for subtype.

#### Stage 2 Raw Residual Probes

- Added split-aware raw probe support so `scripts/stage2_probe_raw.py` can use
  the precomputed `results/stage2/splits.jsonl` S1 assignments rather than
  making a fresh random split.
- Full 27B S1 raw residual probes used layers 15, 30, and 45 with
  `parse_failed=True` filtered. All selected layers beat the strongest B0
  metadata baselines on test AUC.
- Best raw residual layer for both tasks was L45. `infer_property` reached
  val/test AUC 0.881/0.897, beating the B0 prompt baseline test AUC 0.743 by
  about +0.153. `infer_subtype` reached val/test AUC 0.917/0.914, beating the
  B0 height baseline test AUC 0.841 by about +0.073.
- Per-layer test AUCs increased with depth: property L15/L30/L45 =
  0.786/0.856/0.897; subtype L15/L30/L45 = 0.854/0.909/0.914.
- Shuffled-label sanity check stayed near chance. The bounded control used the
  same layers and S1 split, with `C=1.0` and `max_iter=300`; noisy-label fits
  hit the iteration cap, but best selected-layer test AUCs were 0.493 for
  property and 0.481 for subtype.
- Interpretation for the report: 27B raw residuals contain predictive signal
  about Stage 1 success beyond height/prompt metadata. This justifies moving to
  SAE feature extraction/probing, while keeping bootstrap CIs and transfer
  diagnostics as pending Phase B work.
- Reran the full 27B S1 raw probes with 1,000 bootstrap samples over heldout
  examples. L45 property test AUC 0.897 has 95% CI [0.881, 0.912]; L45 subtype
  test AUC 0.914 has 95% CI [0.896, 0.932]. These intervals remain clearly
  above the matching B0 thresholds.
- Per-height diagnostic caveat: h1 heldout cells have only 5 negatives for
  property and 4 negatives for subtype, so h1 AUCs are unstable and should not
  be overinterpreted. The deeper-height cells provide more reliable per-height
  checks.
- Cross-task transfer is positive but weaker than within-task probing.
  Property-trained L45 transfers to subtype at target test AUC 0.862, 95% CI
  [0.837, 0.884]. Subtype-trained L45 transfers to property at target test AUC
  0.786, 95% CI [0.763, 0.809]. This suggests a shared success/failure signal
  plus task-specific components.
- Added an evaluable replacement for the failed topology-heldout S2:
  target-symbol-heldout S3. The group key is the hypothesis subject
  (`target_concept` for property, target subtype for subtype), assigned
  globally per 27B task so train/val/test do not share target symbols across
  heights. This is not a full name-scrambled regeneration, but it tests whether
  probes survive unseen target lexical items using existing Stage 1 labels.
- Regenerated `results/stage2/splits.jsonl` and
  `docs/stage2_splits_summary.json` with S3. S3 is evaluable for both 27B tasks
  and all heights; S2 remains non-evaluable and should not be reported as a
  topology-heldout result.
- S3 metadata baselines are in `docs/stage2_b0_summary_27b_s3.json`.
  Strongest pre-output B0 baselines are property `b0_namefreq` test AUC 0.711
  and subtype `b0_prompt` test AUC 0.859.
- S3 raw residual probes are in `docs/raw_probe_27b_s3_target_symbol.json`.
  L45 remains best for both tasks. Property reaches val/test AUC 0.875/0.884,
  95% CI [0.868, 0.901], beating B0 by +0.173. Subtype reaches val/test AUC
  0.909/0.917, 95% CI [0.898, 0.934], beating B0 by +0.058.
- S3 cross-task transfer is positive but weaker than within-task, matching the
  S1 pattern. Property-trained L45 transfers to subtype at target test AUC
  0.846, 95% CI [0.823, 0.872]. Subtype-trained L45 transfers to property at
  target test AUC 0.788, 95% CI [0.766, 0.810].
- Interpretation update: the main raw-probe claim is stronger after S3. The
  27B raw L45 correctness signal is not only memorizing repeated target symbols
  in the random S1 split, though full name-scramble/model-regeneration
  invariance remains untested.

#### Stage 2 SAE Feature Extraction

- Added cached-residual SAE extraction tooling. The extractor reads a raw
  residual `.safetensors` file, loads a Gemma Scope SAE through SAE Lens, and
  writes sparse top-k features (`top_values`, `top_indices`, `l0`) plus a
  copied sidecar and metadata/checksums.
- Verified remotely that Gemma Scope 2 has L45 residual SAEs for 27B at both
  width-16K and width-262K. The local cache initially only had layer-30 SAEs
  from the earlier measurement job.
- Pinned the L45 width-16K SAE in `docs/stage2_invariants.json`:
  `gemma-scope-2-27b-it-res-all/layer_45_width_16k_l0_small`, HF snapshot
  `5c58dd4cddd52cef653059d85e12a86bf6222a28`, config SHA-256
  `847532cdf078e129e0dce8efce2bb417b6b29d6afe7762efbf060d5a36caf94a`,
  params SHA-256
  `e6aeef8fa7cdf7d7fa9f345a6ded25b55725be54929934910d792acb2bd9a9c4`.
- SAE pilot job `449999` ran on `scholar-j001` and encoded 512 L45 rows for
  each 27B task with the width-16K SAE. Both outputs have `top_values` and
  `top_indices` shape `[512, 128]`, `l0` shape `[512]`, bf16 top values, and
  int64 top indices.
- Pilot mean L0 was 17.57 for property and 17.96 for subtype. The property
  encode loop ran at 285 rows/s, while subtype ran at 2,088 rows/s after the
  SAE/model path was warm in cache. This supports moving to full L45 width-16K
  feature extraction before trying width-262K.
- Full L45 width-16K SAE extraction job `450004` completed for both 27B tasks.
  Each full feature file has 11,000 rows with `top_values`/`top_indices` shape
  `[11000, 128]` and `l0` shape `[11000]`. Full-run mean L0 was 14.34 for
  property and 14.08 for subtype.
- First SAE probes on L45 width-16K top-128 features beat B0 but are weaker
  than raw residual probes. Property test AUC was 0.786, 95% CI [0.763, 0.808],
  compared with B0 0.743 and raw L45 0.897. Subtype test AUC was 0.876, 95% CI
  [0.852, 0.899], compared with B0 0.841 and raw L45 0.914.
- Interpretation for the report: sparse Gemma Scope features retain some
  correctness signal, but top-128 width-16K features do not yet localize the
  full raw-residual signal. Width-262K and/or more retained features should be
  tested before selecting features for steering.
- Pinned the L45 width-262K SAE in `docs/stage2_invariants.json`:
  `gemma-scope-2-27b-it-res-all/layer_45_width_262k_l0_small`, HF snapshot
  `5c58dd4cddd52cef653059d85e12a86bf6222a28`, config SHA-256
  `430435aaaed94f11bad0bab89c6ff4b7ae1ace122df6e9ca4f36e9d5e022667e`,
  params SHA-256
  `c2153afab970b0d63c76cc6f40e2dbeb86db8a3604b0bc54aee457b1e01dc757`.
- Full L45 width-262K SAE extraction initially queued with 120 GB RAM but was
  resubmitted at 80 GB because an A40 had enough GPU capacity but not 120 GB
  allocatable node memory. Job `450029` completed on `scholar-j001`.
- Width-262K full feature files have 11,000 rows with `top_values`/`top_indices`
  shape `[11000, 128]` and `l0` shape `[11000]`. Mean L0 was 14.54 for property
  and 14.66 for subtype. The cached params blob is about 11 GB.
- Width-262K top-128 SAE probes slightly improved property but not subtype:
  property test AUC 0.806, 95% CI [0.784, 0.828], vs width-16K 0.786 and raw
  L45 0.897; subtype test AUC 0.870, 95% CI [0.845, 0.895], vs width-16K 0.876
  and raw L45 0.914.
- L45 width-16K top-512 diagnostic job `450038` completed on `scholar-j001`.
  The extracted feature files have `top_values`/`top_indices` shape
  `[11000, 512]`, but ranks after 128 are all zero. Observed max L0 was 24 for
  property and 23 for subtype.
- Top-512 width-16K probes exactly matched top-128: property test AUC 0.786,
  95% CI [0.763, 0.808]; subtype test AUC 0.876, 95% CI [0.852, 0.899].
  Existing top-128 files also already cover all nonzero activations for the
  width-262K SAE, where max L0 was 22 for both tasks.
- Interpretation update: increasing SAE width alone does not close the raw-SAE
  gap, and top-k truncation is not the explanation because top-128 already
  keeps every nonzero active feature for the tested L45 SAEs. Do not spend time
  on a top-512 width-262K rerun. Next useful checks are feature stability,
  adjacent-layer probes, and raw-vs-SAE reconstruction/residual diagnostics.
- Added `scripts/stage2_analyze_sae_features.py` and
  `src/stage2_feature_stability.py` to inspect L45 SAE top-weight stability.
  The script refits the saved-best-C sparse logistic probes, ranks standardized
  coefficients, records feature activation densities, measures same-width
  property/subtype overlap, and compares cross-width top-feature activation
  patterns by correlation.
- `docs/sae_feature_stability_27b_l45_s1.json` reproduces the saved SAE probe
  metrics after reducing each fit to train-active SAE features only. Effective
  train-active support is small: 62/46 distinct width-16K features for
  property/subtype and 82/82 distinct width-262K features, out of nominal
  widths 16,384 and 262,144.
- Same-width task overlap is nontrivial. Width-16K has four same-sign top-10
  overlap features across property/subtype (`1096`, `19`, `180`, `4329`) and
  14 same-sign overlaps in the top-25 set. Width-262K also has four same-sign
  top-10 overlaps (`368`, `160112`, `64600`, `9994`) and six same-sign overlaps
  in the top-25 set.
- Cross-width feature correspondence is partial rather than clean. Among top
  width-16K features, 16/50 for property and 17/46 for subtype have a best
  width-262K activation-pattern match with abs correlation at least 0.5. This
  gives candidate feature pairs but does not yet establish a localized
  mechanism, especially because several high-weight features are dense across
  examples.
- Added `scripts/stage2_sae_reconstruction_diagnostics.py`,
  `src/stage2_reconstruction.py`, and
  `scripts/stage2_sae_reconstruct_27b_L45.sbatch` for raw-vs-SAE
  reconstruction diagnostics. The script reconstructs cached L45 residuals from
  stored sparse top-k SAE activations, writes reconstruction and
  raw-minus-reconstruction error activation files, then trains the same dense
  S1 probes on both components.
- Reconstruction job `450072` completed on `scholar-j001`. It wrote the summary
  artifact `docs/sae_reconstruction_probe_27b_l45_s1.json` plus ignored
  reconstruction/error activation files under
  `results/stage2/sae_reconstructions/`. Three dense lbfgs fits emitted
  max-iteration warnings, consistent with earlier dense raw-probe behavior.
- L45 SAE reconstructions explain most raw residual energy: 0.948/0.948 for
  width-16K property/subtype and 0.955/0.954 for width-262K property/subtype.
  Mean row cosine is about 0.978-0.980.
- Reconstruction probes track the sparse SAE-feature probes, not the raw probe:
  width-16K property/subtype reconstruction test AUCs are 0.786/0.877, and
  width-262K reconstruction test AUCs are 0.806/0.870.
- Raw-minus-reconstruction error probes recover essentially the full raw L45
  signal despite the error being only about 4-5% of activation energy. Error
  test AUCs are 0.894/0.916 for width-16K property/subtype and 0.897/0.915 for
  width-262K property/subtype, compared with raw L45 test AUCs 0.897/0.914.
- Interpretation update: the raw-SAE gap is not caused by top-k truncation or
  by too little reconstructed activation energy. The correctness-predictive
  direction appears to live mainly in the SAE reconstruction error, so SAE
  feature steering is not yet the right main causal test without an additional
  raw-direction or reconstruction-error diagnostic.

#### Stage 2 Steering Pilot

- Added a raw L45 direction steering pilot for Gemma 3 27B `infer_property`.
  The script refits the same S1 standardized logistic probe used in
  `docs/raw_probe_27b_s1.json`, recovers the raw residual-space direction, and
  applies prompt-only interventions at the final pre-generation L45 residual.
- TransformerLens 3.0 generation needed two implementation fixes on Scholar:
  set `BD_PATH=/scratch/scholar/$USER/beyond-deduction` in the Slurm job so
  generated outputs can be scored on compute nodes, and set PyTorch's default
  dtype to the model dtype during `model.generate()` so the TL KV cache is bf16
  rather than float32.
- The bounded pilot job `450140` ran on `scholar-j001` and wrote
  `docs/raw_steering_pilot_27b_l45_property.json` plus ignored row-level outputs
  under `results/stage2/steering/`. It used 8 balanced S1 test rows
  (h3/h4 x original correct/incorrect), conditions baseline, raw +/-2 SD, and
  orthogonal-control +/-2 SD, with `max_new_tokens=64`.
- Pilot result: prompt-only raw L45 steering caused zero strong-correctness
  flips. Strong accuracy was 3/8 for baseline and 3/8 for every raw and
  orthogonal condition. The only output change occurred on one h4 row and
  appeared identically for raw and orthogonal controls, producing one parse
  failure in every steered condition. Seven of eight rows had byte-identical
  outputs across all conditions. All generations hit the 64-token cap, so the
  pilot should be interpreted as a bounded causal/plumbing check rather than a
  polished generation protocol.
- Interpretation update: the raw L45 probe direction is predictive but this
  small prompt-only intervention did not produce direction-specific causal
  control. The next steering test, if pursued, should change the intervention
  design rather than simply scaling this exact pilot: likely all-token or
  later-token steering, a larger strength sweep, and a smaller/faster row set
  for iteration.
