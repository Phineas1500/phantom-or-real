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
