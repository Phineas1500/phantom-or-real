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
- Ran S3 target-symbol-heldout probes on the cached L45 SAE features. Width-16K
  test AUCs were 0.799 for property, 95% CI [0.777, 0.820], and 0.865 for
  subtype, 95% CI [0.838, 0.889]. Width-262K test AUCs were 0.779 for property,
  95% CI [0.757, 0.802], and 0.867 for subtype, 95% CI [0.839, 0.892].
  Property stays above the S3 B0 baseline (0.711), but subtype only barely
  exceeds its stronger S3 B0 baseline (0.859).
- Added a `--reuse-existing` mode to
  `scripts/stage2_sae_reconstruction_diagnostics.py` so existing
  reconstruction/error activation files can be probed on a new split family
  without re-decoding the SAE features.
- S3 reconstruction/error probes repeat the S1 conclusion. Reconstruction AUCs
  track the SAE probes: 16K property/subtype 0.799/0.865 and 262K
  property/subtype 0.788/0.867. Error probes recover the raw-level S3 signal:
  16K property/subtype 0.881/0.916 and 262K property/subtype 0.886/0.914,
  compared with raw L45 S3 property/subtype 0.884/0.917. Several dense lbfgs
  fits again reached the 2,000-iteration cap, matching the earlier S1 dense
  diagnostic behavior.

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

### 2026-04-28

#### Report Scaffold And Site-Pilot Plan

- Added `docs/stage2_results_pack.md` and `docs/report_outline.md` as compact
  report scaffolds for the current 27B story. They leave teammate Gemma 3 4B
  results as a later comparison rather than blocking the 27B write-up.
- Decided not to launch a broad Gemma Scope 2 sweep. The next optional
  mechanistic branch is a single 27B L45 MLP-output site pilot, because it
  directly tests whether the residual-SAE gap is specific to residual
  dictionaries or whether MLP-site sparse features expose more of the raw
  correctness signal.
- SAE Lens metadata confirms Gemma Scope 2 has 27B-IT L45 `mlp_out_all` and
  `transcoder_all` artifacts, including `layer_45_width_16k_l0_small`.
  These artifacts are not currently in the local HF cache, so the first pilot
  job will need to download them.
- Added site-aware extraction/probe plumbing so Stage 2 activation artifacts can
  be named with non-residual sites such as
  `gemma3_27b_infer_property_L45_mlp_out` while preserving existing residual
  filenames. The first planned hook is
  `blocks.{layer}.ln2_post.hook_normalized`. Later exact-hook auditing showed
  this bare normalized hook is missing the learned RMSNorm weight relative to
  `blocks.{layer}.hook_mlp_out`, so treat the MLP-output SAE pilot below as
  exploratory rather than report-central.
- Scholar job `451090` completed the L45 MLP-output site pilot on
  `scholar-j000`. Raw L45 `mlp_out` activations carry essentially the same
  correctness signal as raw residuals: S1 property/subtype test AUCs
  0.895/0.916 and S3 property/subtype test AUCs 0.892/0.915.
- The Gemma Scope 2 L45 `mlp_out_all` width-16K SAE does not recover that
  signal. MLP-out SAE test AUCs were S1 property/subtype 0.577/0.674 and S3
  property/subtype 0.550/0.702. This is weaker than the residual SAE result,
  not stronger.
- Interpretation update: the raw correctness signal is present at the
  post-MLP site, but the tested MLP-output sparse dictionary exposes little of
  it under the bare-normalized pilot extraction. After the 262K exact-hook
  audit, do not use this as a primary negative claim about MLP-output SAEs
  without an exact `hook_mlp_out` rerun; use it only as a motivation for the
  later exact 262K transcoder check.
- GORMAN access works through `queue.cs.purdue.edu` with the dedicated SSH key.
  The cluster has two `gorman-gpu` DGX-1 nodes, each with 8x Tesla V100 SXM2
  32GB and 512 GB RAM. A tiny diagnostic job `9883` verified `nvidia-smi` on
  `gorman2`; job `9884` verified a scratch Python venv with
  `torch==2.11.0+cu126` and `transformer-lens==3.0.0` can see a V100. A
  4-GPU 27B fp16 load pilot was submitted as `9885` but canceled because it was
  scheduled for May 1 and Scholar had already completed the MLP-site run.
- Ran one additional computation-oriented sparse-artifact check before stopping
  the Gemma Scope 2 branch: the L45 width-16K affine skip-transcoder. The first
  submission, Scholar job `451126`, failed because the metadata hook
  `blocks.45.hook_mlp_in` did not fire under the local TransformerLens Gemma 3
  extraction path. The corrected job `451137` used
  `blocks.45.ln2.hook_normalized` for the pre-MLP normalized site and completed
  on `scholar-j001` in 1:14:18.
- Raw L45 `mlp_in` activations match the raw residual/MLP-output story:
  S1 property/subtype test AUCs 0.897/0.915 and S3 property/subtype test AUCs
  0.885/0.914.
- The affine skip-transcoder features expose some signal but do not close the
  raw gap. Skip-transcoder test AUCs were S1 property/subtype 0.722/0.821 and
  S3 property/subtype 0.722/0.841. This is better than the weak MLP-output SAE
  but still far below raw activations, and weaker than the residual SAEs for
  property.
- Later 262K exact-hook auditing showed this 16K skip-transcoder pilot also
  used the bare pre-MLP normalized input rather than learned-weighted
  `ln2.hook_normalized * ln2.w`. Treat the 16K result as a preliminary pilot,
  not the final transcoder comparison.
- Crosscoders are feasible but heavier: the 27B IT repo has weakly causal
  crosscoders over residual layers `{16,31,40,53}`, but the smallest width is
  65K and uses four parameter shards of about 1.76 GB each. Given the
  skip-transcoder result, keep crosscoders as optional future work rather than
  the next main experiment; any such pilot must compare against a raw-concat
  baseline over the same layers.
- Submitted Scholar job `451181` for a bounded 27B crosscoder pilot while
  report cleanup proceeds. The job extracts residual layers `{16,31,40,53}`,
  probes a raw concatenated baseline over those same layers, encodes
  `crosscoder/layer_16_31_40_53_width_65k_l0_medium`, and probes the
  crosscoder features on S1 and S3. The first submission attempt requested
  220 GB and was rejected by Slurm, so the running job uses the known-good
  180 GB J-node memory request.
- Trimmed `docs/STAGE_2_PLAN.md` from 829 lines to a short operational status
  plan. The report-critical tables are now concentrated in
  `docs/stage2_results_pack.md`, while chronological details remain in this
  file and metric JSONs.
- Crosscoder job `451181` completed successfully in 1:47:54 with max RSS about
  108 GB. It wrote raw residual layers `{16,31,40,53}` for both 27B tasks,
  encoded both tasks with
  `crosscoder/layer_16_31_40_53_width_65k_l0_medium`, and wrote S1/S3
  raw-concat and crosscoder probe JSONs.
- Crosscoder result: raw concat over layers `{16,31,40,53}` nearly matches raw
  L45, but crosscoder features trail raw concat on every split. S1
  property/subtype AUCs: raw concat 0.893/0.904, crosscoder 0.787/0.868. S3
  property/subtype AUCs: raw concat 0.883/0.903, crosscoder 0.724/0.853.
  Interpretation: the multi-layer crosscoder pilot strengthens the sparse
  dictionary cautionary story rather than rescuing feature localization.
- Pinned the completed crosscoder artifact in `docs/stage2_invariants.json`:
  HF snapshot `5c58dd4cddd52cef653059d85e12a86bf6222a28`, config SHA-256
  `80f946678ba690be490d79a8e3bed3b1c34266f650c6226b10b63f508d007628`, and
  four parameter-shard hashes recorded under the crosscoder entry.
- Added a cross-method comparison to `docs/stage2_results_pack.md`. Main
  report insight: crosscoders are middle-tier. They beat the weak MLP-output
  SAE and usually beat the skip-transcoder, but they do not improve on
  residual SAEs, trail the fair raw-concat baseline substantially, and do not
  robustly beat metadata baselines on S3.

### 2026-04-29

#### Dense Active-Feature Sanity Check

- Added `scripts/stage2_probe_dense_active_sparse.py` and
  `src/stage2_dense_active.py` to test whether sparse CSR scaling/centering is
  causing the raw-vs-sparse probe gap. The script selects train-active feature
  columns from SAE/transcoder/crosscoder top-k artifacts, materializes those
  columns as dense arrays, and reruns the same split-aware logistic probe with
  ordinary centered scaling.
- Dense-active probes do not remedy the gap. Residual SAE, skip-transcoder, and
  crosscoder AUCs are essentially unchanged from the standard sparse probes.
  Crosscoder dense-active AUCs were S1 property/subtype 0.786/0.868 and S3
  property/subtype 0.725/0.854, matching the standard crosscoder result
  0.787/0.868 and 0.724/0.853.
- MLP-out SAE improves under dense-active centering but remains weak:
  S1 property/subtype 0.617/0.740 and S3 property/subtype 0.611/0.763. This
  means sparse scaling contributed somewhat to the weakest MLP-out numbers, but
  it does not explain the main raw-vs-sparse disparity.

#### Dtype/Neuronpedia Follow-Ups

- Scholar job `451218` compared existing bfloat16 sparse encodings against
  float32 re-encodings of the first 512 rows for the L45 residual SAE, L45
  MLP-output SAE, L45 affine skip-transcoder, and `{16,31,40,53}` crosscoder
  artifacts. It completed on `scholar-j001` in 4:43 and wrote
  `docs/sparse_dtype_sanity_27b.json`.
- The bf16-vs-fp32 sanity check does not explain the raw-vs-sparse gap. Mean
  active-feature Jaccard was 0.995/0.995 for residual SAE property/subtype,
  1.000/1.000 for MLP-output SAE, 0.991/0.990 for skip-transcoder, and
  0.994/0.992 for crosscoder. The weak non-residual L0 values persist under
  fp32: MLP-output stays exactly L0=2 and skip-transcoder stays around L0=5-6.
- Neuronpedia is not a direct audit path for our current top L45 residual SAE
  features. Its public `gemma-3-27b-it` residual dashboards currently cover
  layers `16`, `31`, `40`, and `53`, while our main residual SAE feature
  probes are layer `45`. Querying the API for
  `45-gemmascope-2-res-16k/180` returned "not available." See
  `docs/neuronpedia_feature_audit_27b.md`.
- Correction/clarification: Neuronpedia does expose all-layer
  `gemmascope-2-transcoder-262k` sources, including layer `45`. That gives a
  possible L45 Neuronpedia audit route, but it would require a same-source
  L45 262K transcoder probe. It does not directly audit the completed L45
  residual SAE probes or the L45 16K affine skip-transcoder pilot.
- The local L45 feature-stability output has an important cautionary pattern:
  several top probe features are nearly always active, for example 16K feature
  `1096` and 262K feature `160112`. These are predictive but not obviously
  satisfying sparse mechanistic candidates. If we want to use Neuronpedia in
  the report, the clean route is an auxiliary residual SAE probe at layer `40`
  or `53`, not retrofitting links onto the current L45 result.

#### L45 262K Transcoder And Neuronpedia Audit (Superseded Bare-Normalized Run)

- Ran the Neuronpedia-facing L45 262K affine transcoder as Scholar job
  `451219`. It reused existing L45 `mlp_in` activations, downloaded/cached the
  11 GB HF artifact
  `transcoder_all/layer_45_width_262k_l0_small_affine`, encoded both 27B
  tasks, ran S1/S3 probes, exported top S1 features, and queried Neuronpedia
  source `45-gemmascope-2-transcoder-262k`. Job completed on `scholar-j003` in
  4:28 with max RSS about 7.5 GB.
- The 262K transcoder does not rescue the computation-feature story. S1
  property/subtype AUCs were 0.654/0.811; S3 property/subtype AUCs were
  0.653/0.831. This is worse than the L45 16K affine skip-transcoder on
  property and slightly weaker/comparable on subtype.
- Despite being width 262K, the artifact produced only 23 distinct active
  top-k features across each full task dataset. Mean L0 was 6.73 for property
  and 5.97 for subtype, with max L0 12/11. Dense-active centered probes were
  effectively identical to the standard sparse probes, so sparse scaling is not
  the explanation.
- Neuronpedia audit result: top L45 262K transcoder features mostly look like
  generic lexical/style/code features, not ontology-reasoning mechanisms.
  Property top explanations include "accessories", "titles and labels", "code
  syntax", "numbers", and "code and symbols"; subtype includes "open", "to
  be", "titles and labels", "code elements", "code syntax", "quotes", and
  "months". See `docs/neuronpedia_transcoder_audit_27b_l45_262k.md`.
- Pinned the 262K transcoder in `docs/stage2_invariants.json`: HF snapshot
  `5c58dd4cddd52cef653059d85e12a86bf6222a28`, config SHA-256
  `7b008b0ec1a40fb50a5d95d5f0d74177ac2c50409fb115821304ef44ae915bdb`, and
  params SHA-256
  `0957750683ed72117c903f9f808ee536f38fe7ded44f54dd3de62e4d06c892e5`.
- Follow-up probe sanity checks do not suggest an easy post-hoc rescue. An
  expanded C grid `0.001,0.01,0.1,1,10,100,1000` left S1 AUCs effectively
  unchanged: property 0.654 and subtype 0.811. Binary active-feature probes
  were worse, with S1 property/subtype AUCs 0.648/0.732. This points away from
  regularization range or activation-magnitude scaling as the cause of the weak
  262K result.

#### L45 262K Transcoder Component Diagnostic (Superseded Cached-Target Run)

- Added `scripts/stage2_transcoder_component_diagnostics.py` and
  `scripts/stage2_transcoder_components_27b_L45_262k_affine.sbatch` to split
  the L45 262K affine transcoder into four dense probe targets: decoded sparse
  latent output, affine skip output, full latent+skip output, and
  `target - full` error against our cached `mlp_out` target. The completed quick
  run is Scholar job `451224`; outputs are
  `docs/transcoder_component_probe_27b_l45_262k_affine_s1.json` and
  `docs/transcoder_component_probe_27b_l45_262k_affine_s3_target_symbol.json`.
- Component AUCs show that the sparse latents are the weak part, not the dense
  affine path. S1 property latent/skip/full/error AUCs were
  0.646/0.859/0.864/0.861; S1 subtype was 0.806/0.893/0.898/0.897. S3 property
  was 0.647/0.837/0.846/0.845; S3 subtype was 0.822/0.879/0.889/0.887.
- Interpretation: the 262K transcoder's affine skip/full dense components
  recover much more correctness signal than the sparse latent features, with
  subtype nearly raw-level and property still below raw but far above the
  latent probe. This supports the broader conclusion that the relevant
  correctness signal is present in dense activation geometry but is not cleanly
  localized by the tested sparse feature latents.
- Important caveat: reconstruction-quality numbers from this diagnostic should
  not be interpreted like the residual-SAE reconstruction/error result. The
  full component has huge negative energy explained against our cached
  `blocks.45.ln2_post.hook_normalized` target (about `-1.57e5`) and mean row
  cosine about `-0.026`, indicating a target-site or scale mismatch with the
  Gemma Scope 2 transcoder training target. Treat the component AUCs as a
  predictive decomposition of available transformations, not as faithful
  transcoder reconstruction evidence.

#### L45 262K Transcoder Hook Audit

- Ran a 16-row exact-hook audit as Scholar job `451225`; output is
  `docs/transcoder_hook_audit_27b_l45_262k.json`. The audit compared
  `hook_mlp_in`, `ln2.hook_normalized`, learned-weighted `ln2`, `hook_mlp_out`,
  `ln2_post.hook_normalized`, and learned-weighted `ln2_post`.
- The cached `mlp_out` target was missing the learned post-MLP RMSNorm weight:
  `hook_mlp_out` and weighted `ln2_post` are effectively identical
  (global cosine `0.999997`, mean L2 about `5007`), while
  `ln2_post.hook_normalized` has mean L2 about `73` and only cosine `0.542`
  with `hook_mlp_out`.
- The best input is learned-weighted `ln2`, not bare `ln2.hook_normalized` or
  `hook_mlp_in`. On the audit sample, exact weighted input gives feature L0
  about `34.0`, while the old bare-normalized input gives L0 about `7.25`.
  This likely explains the unusually tiny L0 and weak 262K latent probe we saw
  before.
- The Gemma Scope `affine_skip_connection` tensor appears to need the
  untransposed multiplication `x @ W_skip` for this square L45 artifact. With
  weighted `ln2` input and `hook_mlp_out` target, full untransposed
  reconstruction explains about `0.687` of target energy with global cosine
  `0.831`; SAE Lens' generic `x @ W_skip.T` path explains only about `0.380`.
- Added `scripts/stage2_extract_exact_transcoder_hooks.py` and patched
  `scripts/stage2_transcoder_component_diagnostics.py` with
  `--skip-orientation untransposed` so the full exact-hook rerun can produce
  interpretable reconstruction stats.

#### L45 262K Exact-Hook Transcoder Rerun

- Ran the corrected exact-hook 262K affine transcoder rerun as Scholar job
  `451226`; job completed on `scholar-j003` at 2026-04-29 05:02:39 EDT. The
  rerun fixed four old-design issues: the input uses learned-weighted
  `ln2.hook_normalized * ln2.w`, the target uses `blocks.45.hook_mlp_out`,
  component diagnostics use the audit-selected untransposed skip path
  `x @ W_skip`, and Neuronpedia/top-feature analysis was refreshed from the
  corrected exact feature files.
- Corrected exact sparse 262K probes improve substantially over the old
  bare-normalized run. Exact S1 property/subtype AUCs are `0.795/0.873`, and
  exact S3 property/subtype AUCs are `0.802/0.885`. The old S1/S3 AUCs were
  `0.654/0.811` and `0.653/0.831`, so the old result should be treated as a
  hook/scale-mismatch diagnostic rather than the final 262K transcoder number.
- Exact input feature density is no longer anomalously tiny but still safely
  below top-k: mean L0 is `19.51` for property and `17.93` for subtype with
  `top_k=128`.
- Same-site exact raw activations remain stronger than the corrected sparse
  262K features. Exact raw `mlp_in_weighted` S1 property/subtype AUCs are
  `0.897/0.916`; exact raw `mlp_out_hook` S1 is `0.896/0.916`. S3 exact raw
  `mlp_in_weighted` is `0.885/0.914`, and exact raw `mlp_out_hook` is
  `0.892/0.915`.
- Corrected component diagnostics now have interpretable reconstruction
  statistics. Full latent+skip output explains `0.672` target energy for
  property and `0.661` for subtype, with global cosine `0.821/0.814`; this
  replaces the old nonsensical negative-energy result against the wrong-scale
  cached target.
- Exact component AUCs remain below raw but show that dense decoded components
  carry more signal than sparse latents alone. S1 property latent/skip/full/error
  AUCs are `0.791/0.856/0.862/0.864`; S1 subtype is
  `0.867/0.890/0.897/0.888`. S3 property is
  `0.796/0.844/0.851/0.861`; S3 subtype is
  `0.883/0.882/0.889/0.886`.
- Refreshed the Neuronpedia audit using corrected exact top features:
  `docs/neuronpedia_transcoder_audit_27b_l45_262k_exact.json` and `.md`.
  Corrected top features differ from the old audit but still do not look like
  clean ontology-reasoning mechanisms. Examples include "exhibit", "for all",
  "okay/affirmation", "technical contexts", "code and structure",
  "pronouns and scope", and several very dense generic features. Treat the old
  Neuronpedia audit as superseded for report-critical claims.

#### Sparse-Feature Combination Follow-Up

- Promising remaining routes for narrowing the sparse-vs-raw gap, in priority
  order: combine sparse feature families; rerun exact-hook MLP-output SAE if a
  matching artifact path is clean; rerun exact-hook 16K skip-transcoder for a
  fairer width comparison; try higher-L0/denser 262K transcoder variants if
  available; and, only if compute/time remain, do a multi-layer exact
  transcoder concat. Nonlinear probes are lower priority because they weaken
  the clean linear-probe interpretability story.
- Added `scripts/stage2_probe_sparse_concat.py` to horizontally concatenate
  sparse top-k feature artifacts while preserving the existing S1/S3 split-aware
  logistic probe protocol. The script aligns sidecars, filters parse failures,
  keeps the matrix sparse, and reports per-block widths/L0s.
- Primary concat result, residual SAE 262K + exact 262K transcoder: S1
  property/subtype AUCs are `0.815/0.870`; S3 property/subtype AUCs are
  `0.800/0.881`. This gives only a small property improvement over standalone
  residual/transcoder features and does not bridge to raw activations.
- All-L45 sparse concat, residual SAE 16K + residual SAE 262K + exact 262K
  transcoder, is stronger: S1 property/subtype AUCs are `0.822/0.884`; S3
  property/subtype AUCs are `0.814/0.885`. Bootstrap 95% CIs are roughly
  property `0.802-0.843` and subtype `0.859-0.906` on S1, and property
  `0.793-0.836` and subtype `0.861-0.908` on S3.
- Interpretation: sparse feature families are complementary, especially for
  property, but even the combined sparse feature set remains below raw exact
  activations (`0.897/0.916` S1 exact `mlp_in`, `0.885/0.914` S3 exact
  `mlp_in`). This strengthens the "partial sparse localization, not complete
  localization" story rather than fully rescuing sparse features.

#### Exact-Hook MLP-Output SAE Rerun

- Ran the exact-hook L45 MLP-output SAE rerun as Scholar job `451338`; output
  reports are `docs/sae_probe_27b_l45_mlp_out_hook_16k_s1.json` and
  `docs/sae_probe_27b_l45_mlp_out_hook_16k_s3_target_symbol.json`. The job
  reused cached exact `blocks.45.hook_mlp_out` activations from the 262K
  transcoder rerun, so it did not reload Gemma 27B.
- This fixes the old bare-normalized MLP-output pilot. The Gemma Scope
  `mlp_out_all/layer_45_width_16k_l0_small` config now records
  `hook_name=blocks.45.hook_mlp_out`, with mean L0 `23.58` for property and
  `22.45` for subtype. The old bare `ln2_post.hook_normalized` encoding had
  mean L0 exactly `2.0`, another sign that the old site was wrong-scale for
  this artifact.
- Exact MLP-output SAE AUCs improved sharply over the old pilot. Old
  bare-normalized S1 property/subtype AUCs were `0.577/0.674`, and exact-hook
  S1 is `0.811/0.878`. Old S3 was `0.550/0.702`, and exact-hook S3 is
  `0.807/0.879`.
- Exact raw `hook_mlp_out` still remains stronger: S1 property/subtype
  `0.896/0.916`; S3 property/subtype `0.892/0.915`. Interpretation: exact
  MLP-output SAE features are meaningful and roughly residual-SAE-like, but
  still partial. This reinforces the hook/scale caution and weakens the old
  "MLP-output SAE is useless" read.

#### All-L45 Sparse Concat With Exact MLP-Output SAE

- Added the exact-hook MLP-output SAE 16K feature block to the previous
  all-L45 sparse concat. New reports:
  `docs/sparse_concat_probe_27b_l45_resid16k_resid262k_exact_tc262k_mlpout16k_s1.json`
  and
  `docs/sparse_concat_probe_27b_l45_resid16k_resid262k_exact_tc262k_mlpout16k_s3_target_symbol.json`.
- The four-block concat is residual SAE 16K + residual SAE 262K + exact 262K
  affine-transcoder latents + exact MLP-output SAE 16K. It raises mean active
  L0 to about `71.9` for property and `69.0` for subtype, with total sparse
  width `557056`.
- AUCs: S1 property/subtype `0.828/0.883`; S3 property/subtype `0.823/0.885`.
  Relative to the previous three-block concat (`0.822/0.884` S1,
  `0.814/0.885` S3), the exact MLP-output block gives a small property gain
  and no meaningful subtype gain.
- Interpretation: the fixed MLP-output SAE has complementary information for
  property, especially under S3, but this still does not bridge to raw exact
  activations (`0.896/0.916` S1 exact `hook_mlp_out`,
  `0.892/0.915` S3 exact `hook_mlp_out`). The current best sparse-only result
  remains a partial-localization result rather than a complete sparse
  substitute for raw activations.

#### Remaining Targeted Gemma Scope 2 Checks (Historical Queue)

These were the next checks identified before the low-C, dense-active,
exact-16K, and L30 runs below. Those later sections supersede this queue.

- Best next low-cost check: rerun the current four-block sparse concat with a
  lower-regularization grid. The current best reports choose `C=0.01`, the
  smallest value in the default grid, so the optimum may sit below the tested
  range. Use a grid like `0.0001,0.0003,0.001,0.003,0.01,0.03,0.1`.
- Next implementation-mismatch checks: rerun dense-active centered probes on
  corrected exact-hook artifacts, especially exact MLP-output SAE 16K and exact
  262K affine-transcoder latents. Earlier dense-active checks mostly predated
  the exact-hook fixes.
- Next Gemma Scope 2 artifact checks, if time remains: exact-hook 16K
  skip-transcoder rerun with learned-weighted `ln2` input; crosscoder
  `layer_16_31_40_53_width_65k_l0_big`; and L30 residual SAE 16K/262K,
  possibly concatenated with L45, since raw L30 remains strong.
- Lower priority: broad 512K/1M width sweeps. Local cache currently only shows
  16K/262K for the relevant L45 residual/transcoder/MLP-output artifacts, and
  width alone has not been the main source of improvement so far.

#### Low-C Four-Block Sparse Concat Rerun

- Reran the current four-block sparse concat with lower regularization values:
  `0.0001,0.0003,0.001,0.003,0.01,0.03,0.1`. New reports:
  `docs/sparse_concat_probe_27b_l45_resid16k_resid262k_exact_tc262k_mlpout16k_lowc_s1.json`
  and
  `docs/sparse_concat_probe_27b_l45_resid16k_resid262k_exact_tc262k_mlpout16k_lowc_s3_target_symbol.json`.
- The default-grid run had selected `C=0.01`, the smallest tested value. The
  low-C rerun selects `C=0.003` for S1 property/subtype, `C=0.001` for S3
  property, and `C=0.003` for S3 subtype.
- AUCs improve on every task/split: S1 property/subtype from `0.828/0.883` to
  `0.830/0.888`; S3 property/subtype from `0.823/0.885` to `0.828/0.888`.
  Bootstrap 95% CIs are S1 property/subtype `0.810-0.851`/`0.863-0.908` and
  S3 property/subtype `0.807-0.848`/`0.863-0.909`.
- Interpretation: part of the sparse-vs-raw gap was ordinary regularization
  tuning, not a Gemma Scope 2 artifact limitation. The improvement is modest
  but consistent. Raw exact activations still remain stronger, so the main
  partial-localization conclusion is unchanged.

#### Corrected Exact Dense-Active Scaling Check

- Added a `--dense-active` mode to `scripts/stage2_probe_sparse_concat.py`.
  This mode keeps the existing sidecar-aligned sparse hstack, then selects
  train-active concat columns, materializes them as dense features, and uses the
  centered-scaling logistic probe path.
- Reran dense-active centered probes on corrected exact-hook individual
  artifacts:
  `docs/dense_active_exact_sparse_probe_27b_l45_corrected_s1.json` and
  `docs/dense_active_exact_sparse_probe_27b_l45_corrected_s3_target_symbol.json`.
  Exact MLP-output SAE dense-active AUCs are S1 `0.814/0.880` and S3
  `0.805/0.879`; exact 262K affine-transcoder dense-active AUCs are S1
  `0.800/0.878` and S3 `0.805/0.883`.
- Reran dense-active centered four-block concat:
  `docs/dense_active_sparse_concat_probe_27b_l45_resid16k_resid262k_exact_tc262k_mlpout16k_lowc_s1.json`
  and
  `docs/dense_active_sparse_concat_probe_27b_l45_resid16k_resid262k_exact_tc262k_mlpout16k_lowc_s3_target_symbol.json`.
  Active concat columns are about `578/471` on S1 property/subtype and
  `580/472` on S3.
- Dense-active four-block concat AUCs are S1 property/subtype `0.831/0.888`
  and S3 property/subtype `0.828/0.887`. Compared with the sparse low-C concat
  (`0.830/0.888` S1, `0.828/0.888` S3), this is effectively unchanged.
- Interpretation: centered scaling over train-active corrected features is not
  the hidden reason sparse probes trail raw activations. It gives at most tiny
  gains on S1 and slightly mixed S3 behavior, so the remaining raw-vs-sparse
  gap is not mainly a sparse-matrix standardization artifact.

#### L45 16K Exact-Hook Skip-Transcoder Rerun

- Ran the fair exact-hook 16K affine skip-transcoder comparison as Scholar job
  `451496`; job completed on `scholar-j003` at 2026-04-29 14:24 EDT. The run
  used the same corrected input/target convention as the 262K transcoder:
  `ln2.hook_normalized * ln2.w` for encoding and exact
  `blocks.45.hook_mlp_out` for component diagnostics.
- Corrected exact 16K sparse probes improve substantially over the old
  bare-normalized 16K pilot, especially for property. Exact 16K AUCs are S1
  property/subtype `0.787/0.868` and S3 property/subtype `0.785/0.880`; the
  old bare-normalized 16K pilot was S1 `0.722/0.821` and S3 `0.722/0.841`.
- Exact 262K still has a small edge over exact 16K on most sparse-latent
  comparisons: exact 262K reached S1 `0.795/0.873` and S3 `0.802/0.885`.
  Width therefore helps, but the difference is much smaller than the old
  bare-normalized comparison suggested.
- Exact 16K component diagnostics are interpretable but reconstruct less
  target energy than 262K. Full energy explained is `0.639` for property and
  `0.638` for subtype, versus `0.672/0.661` for exact 262K. Exact 16K
  latent/skip/full/error AUCs are S1 property
  `0.782/0.854/0.854/0.857`, S1 subtype `0.868/0.888/0.890/0.889`, S3
  property `0.781/0.841/0.848/0.855`, and S3 subtype
  `0.883/0.884/0.889/0.891`.
- Interpretation: hook/scale alignment was a real issue for 16K too. The fair
  rerun makes 16K skip-transcoder features roughly residual-SAE-like rather
  than weak, but it still does not bridge the raw activation gap. The best
  current story remains partial sparse localization, with 262K and sparse
  feature-family concat slightly stronger than the exact 16K standalone run.

#### L30 Residual SAE And Multi-Layer Sparse Concat

- Added separate L30 residual-SAE extraction and probe scripts after Scholar
  rejected a 6-hour combined job walltime:
  `scripts/stage2_sae_extract_27b_L30_resid.sbatch` and
  `scripts/stage2_probe_27b_L30_resid_concat.sbatch`. Extraction job `451569`
  completed successfully and wrote L30 residual SAE 16K/262K top-128 features
  for both 27B tasks. Probe job `451571` completed the individual, L30-only,
  L45 five-block, L30+L45 residual-only, and L30+L45 all-sparse concat probes.
- Standalone L30 residual SAE features are weaker than L45, especially for
  property. L30 16K AUCs are S1 `0.752/0.860` and S3 `0.748/0.860` for
  property/subtype. L30 262K improves modestly to S1 `0.770/0.867` and S3
  `0.771/0.866`.
- L30 residual 16K+262K concat improves over either L30 width alone: S1
  property/subtype `0.786/0.872`, S3 `0.788/0.864`. This confirms some
  within-layer width complementarity, but L30 alone is not competitive with
  the best L45 sparse blocks.
- Adding the corrected exact-16K transcoder to the L45 sparse concat gives only
  marginal gains. L45 five-block AUCs are S1 `0.832/0.883` and S3
  `0.829/0.889`, compared with the previous low-C four-block L45 concat at S1
  `0.830/0.888` and S3 `0.828/0.888`.
- L30+L45 residual-only concat is mixed: S1 `0.827/0.889`, S3 `0.816/0.874`.
  The extra L30 residual features help S1 subtype but do not robustly improve
  S3.
- The full L30+L45 all-sparse concat is the new strongest sparse-only result:
  S1 property/subtype `0.839/0.887`, S3 `0.834/0.892`. Bootstrap 95% CIs are
  S1 property/subtype `0.819-0.858`/`0.862-0.909` and S3
  `0.813-0.856`/`0.869-0.913`. Relative to the previous low-C four-block L45
  concat, this is `+0.009/-0.001` on S1 property/subtype and `+0.006/+0.004`
  on S3.
- Interpretation: L30 residual sparse features are not strong standalone
  property probes, but they do add complementary signal when combined with the
  corrected L45 sparse family. This is the best positive sparse-localization
  result so far, but it still trails raw L45 residual probes (`0.897/0.914` S1,
  `0.884/0.917` S3) and exact raw same-site probes. The main narrative remains
  narrowed partial localization, not a full sparse replacement for raw
  activations.
