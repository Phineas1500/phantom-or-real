# Probe + Causal Validation Pipeline Implementation Plan (Stage 2)

## Context

**Project:** "Phantom or Real?" — Stage 2 of the project, covering the probe
training and causal validation work that depends on the Stage 1 behavioral
dataset. Full Stage 1 spec in `BEHAVIORAL_DATA_PLAN.md`; Stage 1 findings in
`docs/behavioral_results_draft.md`; shipped data in
`results/full/with_errortype/`.

The project's three components remain:
1. **Behavioral evaluation** (Stage 1, complete).
2. **Probe training** (this plan, Phases A–D): train classifiers on multiple
   feature sources (raw residual stream activations, Gemma Scope 2 SAE
   features at two widths, skip-transcoder features) to predict
   `is_correct_strong` from internal representations.
3. **Causal validation** (this plan, Phase E): steer the top features
   identified by probes and measure whether this causally flips model
   outcomes, compared to an orthogonal-direction baseline controlling for
   generic perturbation effects.

**What Stage 1 changed about this plan.** Two findings from Stage 1 directly
shape Stage 2:

- The "shortcut-availability" variant of the Ma et al. falsification
  concern (that contrastively selected SAE features track which
  shortcuts are *available* in the input rather than whether the model
  *used* them) is ruled out by construction on this dataset
  (`has_direct_member = 100%` uniformly, structural slicing vacuous).
  Stage 2 therefore focuses the phantom-vs-real test on the remaining
  interpretations: **shortcut usage** (does the probe detect that the
  model actually consulted the ontology chain?) and **surface-lexical
  features** (does the probe ride prompt length or task-specific lexical
  patterns rather than reasoning?).
- The two tasks share generation but diverge in ground-truth surface form.
  This makes cross-task probe transfer (`infer_property` → `infer_subtype`)
  a first-class experiment, not an afterthought: a probe that transfers
  cannot be riding task-specific lexical cues, which is a direct behavioral
  test of the phantom surface-feature concern.

**Reference papers.** Two source works anchor this plan:

- **Cox et al. (2026), "Decoding Answers Before Chain-of-Thought:
  Evidence from Pre-CoT Probes and Activation Steering."** Defines the
  difference-of-means probe on last-pre-CoT-token activations (§2.3), the
  activation steering with orthogonal-direction baseline (§2.4), and the
  four-cell classification of steered CoT traces — sound reasoning,
  non-entailment, confabulation, hallucination (§2.5, Table 1). Their
  "Future Work" section (§4.3) explicitly flags comparing the probe
  directions to SAE features as the next step; Stage 2 of this project
  is that follow-up.
- **Ma et al. (2026), "Falsifying Sparse Autoencoder Reasoning Features
  in Language Models"** (arXiv:2601.05679). Motivates the concern
  driving Stage 2. Ma et al. propose a **three-stage falsification
  framework** for evaluating SAE features claimed to encode reasoning:

  1. *Contrastive selection.* Rank SAE features by Cohen's d between
     reasoning and non-reasoning corpora; take top 100.
  2. *Causal token injection.* For each candidate, identify its top
     activating tokens/n-grams, inject them into non-reasoning text,
     and measure the activation increase. Features with large Cohen's d
     under injection are classified "token-driven," "partially token-
     driven," or "weakly token-driven."
  3. *LLM-guided falsification.* For the "context-dependent" features
     that survived token injection, use an LLM to generate false
     positives (non-reasoning text that activates the feature) and
     false negatives (meaning-preserving paraphrases of reasoning
     traces that suppress activation). A feature passes only if the
     LLM cannot reliably construct either.

  Their headline empirical result, across 22 configurations (Gemma-3-4B,
  Gemma-3-12B, Llama-3.1-8B, Gemma-2, DS-R1-Distill-Llama-8B, middle-to-
  late layers, Gemma Scope 2 / Llama Scope SAEs): **between 45% and 90%
  of contrastively selected features fail the token injection test** at
  any of three significance tiers, *and* **0 out of 248 remaining
  context-dependent features pass LLM-guided falsification** (Tables 3
  and 4). A supplementary steering experiment on the top 3 features of
  Gemma-3-12B at layer 22 reduced AIME accuracy from 26.7% to 10–26.7%
  and GPQA from 37.9% to 13.6–33.3% — i.e., steering either degraded or
  did not improve reasoning performance.

  Ma et al.'s own vocabulary is **cue-like structure**, **lexical
  cues**, **low-dimensional correlates**, and **falsification**; they
  do not use the term "phantom features." We adopt "phantom features"
  as project-internal shorthand (matching the project title "Phantom
  or Real?") for the broader phenomenon — probe-predictive SAE features
  that may be correlational rather than causal — and cite Ma et al. as
  the empirical motivation for testing it.

  **Key methodological distinction from our setup.** Ma et al.'s
  contrastive selection compares *reasoning text versus non-reasoning
  text* (s1K-1.1 math CoT versus Pile web text). Their falsification
  framework targets the specific failure mode where SAE features
  detect CoT-style lexical patterns (recurring discourse tokens like
  "Wait," "But," "Let," or "Let's break down") that separate the two
  corpora without capturing reasoning. Our Stage 2 contrastive
  selection compares *correct versus incorrect reasoning on the same
  task* — both sides produce full InAbHyD responses in the same
  format, so the CoT-lexical failure mode cannot apply directly. A
  different failure mode is plausible for us: features detecting
  surface cues that correlate with success (prompt length, height,
  concept name distribution). Phase B.3 targets this failure mode via
  prompt-length regression and name-scramble ablation; we extend it
  with an explicit paraphrase-preserving test inspired by Ma et al.'s
  false-negative construction (see Phase B.3).

  **Framing implication for Stage 2.** Given Ma et al.'s 0/248 result,
  there are two scientifically valuable outcomes: (a) we reproduce their
  conclusion on InAbHyD, demonstrating that the failure mode
  generalizes from text-style contrastive selection to success/failure
  contrastive selection within a task; or (b) we find SAE features that
  survive falsification-style tests, which would be a novel positive
  result. The plan should not implicitly frame "we found phantom
  features" as a failure — both directions produce a result.

Stage 2 is best framed as: *reproduce Cox et al.'s probing + steering
methodology on InAbHyD, then extend it to SAE features to test the
falsification concerns Ma et al. raise about contrastively selected
"reasoning" features.* The probing-via-difference-of-means and the
orthogonal-direction steering baseline are direct ports of Cox et al.
§2.3–§2.4. The novel contributions, in terms of what exists in the
literature as of April 2026, are:

- **Task novelty.** InAbHyD abductive reasoning with depth-controlled
  ontologies; neither Cox et al. nor Ma et al. tested anything like it.
  Their tasks are binary classification (Cox) or reasoning-text vs.
  non-reasoning-text (Ma).
- **Contrastive-selection novelty.** We select SAE features by
  correct-vs-incorrect outcome on the same task, not by reasoning-vs-
  non-reasoning text. This changes the failure mode the falsification
  framework is defending against.
- **Model-size novelty.** Ma et al. tested Gemma-3-4B and Gemma-3-12B
  but not 27B; Cox et al. stopped at 9B (Gemma 2) / 7B (Qwen 2.5).
  We test Gemma-3-27B, which is the largest model either methodology
  has been applied to.
- **Cross-task transfer as a falsification test.** Neither paper tests
  cross-task probe transfer as a surface-feature diagnostic, because
  their setups don't have two structurally identical tasks with
  different surface form. InAbHyD does (`infer_property` ↔
  `infer_subtype`), which enables a cleaner surface-vs-reasoning
  decomposition than either prior method.

## Repo integration

Stage 2 lives in the same `phantom-or-real` repo as Stage 1. The
existing layout (`src/` library code, `scripts/` orchestrators,
`tests/` smoke tests, `docs/` draft report + figures, `results/`
per-run outputs) is extended, not replicated. CLAUDE.md's coding
conventions carry over: no comments that merely describe what the code
does, docstrings on modules and non-trivial functions only, prefer
editing existing modules when responsibilities overlap, tests mock
network calls and never hit the wire.

### New modules to add under `src/`

- `src/activations.py` — TransformerLens-based residual-stream extraction
  at the last pre-CoT token position. Loads Gemma 3 via the official
  `transformer-lens` package (Gemma 3 support landed in mainline; see
  `transformer_lens.model_bridge.supported_architectures.gemma3`).
  Exposes `extract(example_ids, layers) → dict[example_id, tensor[L, D]]`.
  Imports `src.inference.build_messages` for the Gemma-3-no-system-role
  concatenation so the templated string is byte-identical to what
  Stage 1 inference received.
- `src/probes.py` — probe training (difference-of-means + logistic
  regression), train/val/test split, per-height breakdown,
  cross-task transfer evaluation, cross-model comparison (property
  table, not raw transfer — see B.2). Filters `parse_failed=True`
  rows before splitting.
- `src/sae_features.py` — Gemma Scope 2 SAE / skip-transcoder loading
  via SAE Lens, feature encoding of cached activations. Width-16K and
  Width-262K (Matryoshka-nested). Forks structure from Ma et al.'s
  `reasoning_features/features/collector.py`.
- `src/steering.py` — Phase E residual-stream steering intervention.
  Forks structure from Ma et al.'s `reasoning_features/steering/steerer.py`.
  Reuses `src.gemma3_parse.parse_hypotheses` for scoring steered outputs,
  which is essential: Stage 1's `is_correct_strong` labels depend on
  that parser, and scoring steered generations with anything else
  introduces a confound.

### New scripts under `scripts/`

- `scripts/stage2_extract.py` — orchestrator: sweeps (model, task, layer)
  combinations and writes `results/stage2/activations/` files. Analogous
  to `scripts/run_inference.py` for Stage 1.
- `scripts/stage2_train_probes.py` — probe training orchestrator, writes
  `results/stage2/probes/`.
- `scripts/stage2_j2_measure.py` — Scholar J-node measurement utility
  for Gemma 3 27B TransformerLens batch-size and SAE headroom checks.
  The paired `scripts/stage2_j2_measure.sbatch` wrapper is not part of
  the production pipeline, but records the measured J-node operating
  envelope used below.
- `scripts/validate_activations.py` — 5-tier equivalence check (see
  A.1) between TransformerLens greedy decode and Stage 1 vLLM outputs
  on 50 prompts, plus a 200-row label-agreement check and a
  determinism check. Writes `results/stage2/equivalence_report.json`
  and exits nonzero on hard-tier failures (tokenizer mismatch,
  revision-hash mismatch, label agreement <98%).
- `scripts/stage2_sanity_check.py` — pre-handoff checks for the Stage 2
  deliverables, analogous to `scripts/sanity_check.py`.

### Existing modules to reuse (do not duplicate)

- `src/bd_path.py` — the upstream-patching pattern
  (`_apply_normalize_singular_patch`) is the template if TransformerLens
  or SAE Lens needs similar monkey-patching for our setup.
- `src/env_loader.py::get_openai_gpt_credentials` — GPT judge auth for
  the paraphrase-preserving test in Phase B.3.
- `src/config.py::SYSTEM_PROMPT` and `SHIPPED_SEEDS` — reused for
  prompt reconstruction and any regenerated evaluation examples (name
  scramble set in B.3, paraphrases).
- `src/gemma3_parse.py::parse_hypotheses` — critical for Phase E
  scoring of steered outputs. Do not reimplement.
- `src/inference.py::build_messages` — Gemma-no-system-role
  concatenation. Do not reimplement.
- `src/error_classification.py::_is_gpt5_plus` and the retry/async
  pattern — template for any GPT-5.x judge call in Phase B.3.

### Environment additions

`environment.yml` and `environment.lock.yml` now include
`transformer-lens==3.0.0`, `sae-lens==6.39.0`, and
`scikit-learn==1.8.0` as resolved in the Scholar `phantom`
environment used for the J-node pilot.

### Test conventions

Mock OpenAI calls with `unittest.mock.patch` on
`openai.resources.chat.completions.AsyncCompletions.create`, as in
`tests/test_pipeline_smoke.py`. Stage 2's TransformerLens-dependent code
is harder to unit-test because TransformerLens has to load an actual
model — for those, keep tests offline with dummy tensors (verify probe
math on random inputs, verify SAE encoding/decoding round-trips), and
rely on `scripts/validate_activations.py` as the integration check.

## Objectives

1. Extract residual stream activations at the pre-CoT position for every
   example in the shipped JSONL dataset, at three empirically-chosen layers,
   for both Gemma 3 4B and Gemma 3 27B.
2. Train difference-of-means and linear probes to predict
   `is_correct_strong` from those activations, on four feature sources: raw
   residual stream, Gemma Scope 2 SAE Width-16K, Gemma Scope 2 SAE Width-262K
   (= 2^18 = "256K" in binary-prefix notation; upstream SAE Lens catalog
   uses 262K so we standardize on that),
   and skip-transcoder features.
3. Evaluate **cross-task transfer** within each model
   (`infer_property` ↔ `infer_subtype`) and **cross-model comparison**
   of independently trained 4B and 27B probes (see B.2 — raw cross-model
   probe transfer is not well-defined because `d_model` differs, 2560 vs.
   5376).
4. Decompose probe performance into reasoning-content vs. surface-feature
   contributions via prompt-length regression and name-scramble ablation.
5. Causally validate the top SAE features found by probes via activation
   steering, comparing against an orthogonal-direction baseline. Scope:
   Gemma 3 27B on `infer_property` only.
6. Produce a labeled deliverable for the final report's Probe Results and
   Causal Validation sections.

**Scope decisions made up front** (full justification in scope-decisions.md
or this document's "Scope and rationale" section if that file doesn't
exist):

- **Full probe + SAE + steering treatment on `infer_property`; cross-task
  transfer only for `infer_subtype`.** Preserves subtype as the held-out
  target for the phantom-surface-feature test.
- **Steering on Gemma 3 27B only.** 4B gets probe training +
  cross-model comparison (probe property table, not raw transfer — see
  B.2); no steering. 4B's output-strategy divergence at depth (falls
  back to entity-level enumeration ~50% on subtype h≥3) means it isn't
  attempting the same computation as 27B at depth, which makes steering
  results interpretation-ambiguous.
- **Three layers, empirically selected.** A pilot activation extraction at
  all 62 layers on a 500-example subset picks the three candidate reasoning
  layers by probe AUC before full extraction commits.

**Explicitly not in scope:**

- Multi-hypothesis examples (excluded from Stage 1; not revisited here).
- `infer_membership_relation` (has a built-in shortcut per paper footnote 5).
- Steering on Gemma 3 4B or on `infer_subtype`.
- Cross-model-family comparison (Llama, Gemma 2, Qwen).
- Probing intermediate CoT tokens. Following Cox et al. we probe only at
  the last pre-CoT position.
- Training new SAEs. We use published Gemma Scope 2 SAEs only.
- Measuring feature-interaction effects (e.g. steering two features
  simultaneously). Additive, not compositional.

## Compute platform and budget tiers

### Primary compute plan

- **Gemma 3 4B work: local 4090 (24 GB).** 4B in bf16 is ~8 GB, plus
  ~1.3 GB for a Width-262K SAE per layer, plus activation buffers —
  fits easily. Covers all Phase A extraction for 4B, all Phase B/C/D
  probe training and SAE feature extraction for 4B, and all Phase B.3
  diagnostics (name-scramble re-extraction, paraphrase tests). Probe
  training is GPU-trivial; most of this is bounded by SAE encoding
  throughput.
- **Gemma 3 27B work: Google Colab Pro pool, H100 (80 GB) preferred.**
  The primary assumption is that the project has access to multiple
  Colab Pro accounts (Ram + 1–2 teammates) with ~300 units each
  stacked over 3 months, plus ~100 new units per account per month.
  H100 holds Gemma 3 27B in bf16 (~54 GB) with ~25 GB headroom —
  enough for TransformerLens hook buffers, the Gemma Scope 2 SAE at
  Width-16K or Width-262K loaded alongside the model, and Phase E
  steering generation with batch > 1. Same weights in bf16 reduces
  the mismatch risk against Stage 1's vLLM run, but does not by itself
  make the "same model that produced the behavior" invariant hold —
  tokenizer revision, chat template, stop conditions, and prompt
  construction must also match. The invariant is accepted only after
  the 5-tier equivalence check in A.1 passes.
- **Scholar J-node fallback: 2x A40 (about 92 GB aggregate VRAM).**
  A 2026-04-24 pilot on `scholar-j001` (`sbatch`
  `scripts/stage2_tl_27b_j2_pilot.sbatch`, job `449067`) successfully
  loaded `google/gemma-3-27b-it` through TransformerLens 3.0 with
  `n_devices=2`, captured 10 last-position residual activations at
  `blocks.30.hook_resid_post`, and wrote a `[10, 5376]` bf16
  safetensors artifact. Two operational constraints matter: request
  high CPU RAM (`--mem=180G`; the default loader OOM-killed at 120G)
  and use `HookedTransformer.from_pretrained_no_processing` for bf16.
  TransformerLens 3.0's automatic multi-GPU mover can place neighboring
  blocks on alternating GPUs while `forward()` uses deterministic
  pipeline devices; the pilot works around this by re-placing whole
  blocks with `get_device_for_block_index()` before forwarding.

  Follow-up measurement on 2026-04-25 (`scripts/stage2_j2_measure.sbatch`,
  job `449081`, artifact
  `results/stage2/pilots/j2_stage2_measure_L30_h4_20260425T040105Z.json`)
  used 64 height-4 `infer_property` prompts (202-217 tokens after the
  Gemma chat template), peaked at 104,741,572K MaxRSS, and showed raw
  extraction batches 1, 4, 8, 16, 32, and 64 all succeed. Batch 32
  reached 5.54 rows/s with peak reserved memory 28.9 GiB per GPU;
  batch 64 reached 5.69 rows/s with peak reserved memory 30.2 GiB per
  GPU. Use **batch 32** as the conservative 27B J-node raw-extraction
  default and only raise to 64 after checking token lengths for the
  target shard. The same job loaded Gemma Scope 2 layer-30 residual SAEs
  alongside the resident model:
  `layer_30_width_16k_l0_small` loaded in 4.4 s with peak reserved
  memory 27.7 GiB on `cuda:0`; `layer_30_width_262k_l0_small` loaded
  in 13.0 s with peak reserved memory 37.5 GiB on `cuda:0` and encoded
  cached residual chunks up to 512 rows, peaking at 34.4 GiB reserved
  during encoding after cache cleanup. Therefore run SAE feature
  extraction from cached residuals, one SAE/layer at a time, with
  chunk size 512 as the initial J-node setting. Do not keep multiple
  large SAEs resident with the 27B model unless a separate measurement
  proves the extra headroom.
- **Colab Pro vs. Pro+ operational details.** Treat Pro sessions as
  interruptible and checkpoint aggressively. Colab's official FAQ
  states Pro+ supports up to 24 hours of continuous execution when
  compute units are available; Pro/Pro+ runtime behavior and GPU
  availability vary in practice and aren't contractually guaranteed.
  Don't rely on notebook liveness for long runs — use resumable jobs
  with checkpoint files. If truly uninterrupted execution is needed,
  spill to Modal or a dedicated VM rather than hacking around Colab
  limits. When H100 is unavailable, A100 80GB is a valid fallback for
  27B bf16 (still fits); build code so H100-vs-A100 is a config
  variable, not an assumption.
- **Modal fallback is behavioral, not for activations.** Stage 1's
  `gemma3-27b` Modal deployment (`src/config.py::MODAL_ENDPOINTS`) is
  a vLLM serving endpoint — it's the right tool for re-running
  generation (e.g. Phase E post-hoc scoring of steered outputs if
  batch throughput on Colab proves tight) but it does not expose
  residual stream activations. If Colab availability breaks down
  entirely, standing up a parallel Modal deployment based on
  `transformers.AutoModelForCausalLM` (following the pattern in
  `beyond-deduction/deployment/gemma3_27b_modal.py` but without vLLM)
  is the alternate path; ~$25 of H100 time at $2–4/hr.

### Invariants that must hold regardless of platform choice

1. **Activations come from the same model (same precision, same
   weights) as produced Stage 1 behavioral labels.** Stage 1 ran in
   bf16 via Modal/vLLM. H100 on Colab runs in bf16 via TransformerLens.
   These should match; run the 5-tier equivalence check in A.1
   (tokenizer identity → revision hashes → first-token top-5 logit
   agreement → greedy-output byte match → label agreement ≥98%) and
   proceed if tiers 3 and 5 pass even if byte-match fails.
2. **Session interruption must not lose progress.** Colab sessions
   are interruptible and exact runtime caps vary by tier and
   availability. Extraction on 27B takes ~3 hours per task; Phase E
   sweeps can take longer. Checkpoint every 500 examples
   (`.safetensors` fragment plus `.example_ids.jsonl` sidecar,
   resumable on restart) so any disconnect costs at most ~500 rows
   of re-work.
3. **Persistent storage is a shared Google Drive, not ephemeral
   `/content`.** Mount Drive at session start. Shared Drive matters
   for the pool arithmetic — teammate accounts write to the same
   Drive so activation files and results can be consumed by any
   account, not re-extracted.

### Compute budget arithmetic for the Pro pool

Assume H100 burns ~13 compute units per hour (estimate; see note
below). Per-account: 300 existing units + ~100 new units across a
3-week project window ≈ 375 units per account ≈ ~29 H100 hours.

| Pool size | Total units | H100 hours available |
| --- | --- | --- |
| Solo | ~375 | ~29 |
| 2 accounts | ~750 | ~58 |
| 3 accounts | ~1100 | ~85 |

Phase-level H100 demand (27B only — 4B runs on the 4090):

- Phase A equivalence + determinism checks: ~2 hours
- Phase A.2 layer-selection pilot: ~3 hours
- Phase A.3 full extraction (2 tasks × 3 layers × 11k rows): ~3 hours
- Phase B.3 name-scramble re-extraction (500 rows × 2 conditions ×
  3 layers): ~1 hour
- Phase B.3 paraphrase re-extraction (200 rows × 3 layers): ~0.5 hours
- Phase D SAE feature extraction (shared with A, trivial add): ~1 hour
- Phase E screening (19k gens): ~6–10 hours
- Phase E confirmation (57k gens after the revisions in E.3): ~15–25
  hours
- Phase E final (88k gens): ~25–40 hours

Totals by tier:

- **Minimum (screening only):** ~17–21 H100 hours. Solo-viable.
- **Target (screening + confirmation):** ~32–46 hours. 2-account
  pool fits. 3-account pool has margin.
- **Target + final:** ~57–86 hours. 2-account pool is tight; 3-account
  pool is the right size. Solo cannot do this.

### Verify H100 rate empirically before committing

The 13 units/hour figure is an estimate; the actual rate varies by
time, availability, and whether Colab routes you to H100-SXM5 vs.
H100-PCIe. **Before executing Phase A, run a 15-minute calibration
test**: load Gemma 3 27B in bf16 via TransformerLens, run 50 forward
passes, record compute-units-remaining delta, and compute actual
units-per-hour. If the rate is materially off (say ≥20 units/hour),
recompute the above table and reassess tier feasibility. This is a
cheap check with no downside.

### Work-partitioning strategy for the pool

The work partitions cleanly across accounts if the shared Google Drive
holds activation files and the stable-feature set:

- **Account 1 (Ram / primary MI):** Phase A 27B extraction, Phase D
  SAE feature extraction on 27B, Phase E screening + confirmation on
  the main feature sweep.
- **Account 2 (teammate):** Phase E baseline sweeps (random orthogonal
  + random same-SAE) on 27B — these are embarrassingly parallel with
  the feature sweeps, so running them on a separate account doubles
  screening throughput. Can also handle Phase A layer-selection pilot
  in parallel with Ram's extraction work.
- **Account 3 (if available):** Phase E confirmation/final runs on the
  top-ranked features in parallel with Ram's continued screening
  work. Also available as a general-purpose overflow account for
  whoever runs out of units first.

4B work on the 4090 is orthogonal to Colab and runs on Ram's local
machine regardless of pool size. B0 baselines, 4B probe training, and
4B diagnostic re-extractions cost zero Colab units.

### Compute monitoring checkpoints

Track compute units at each phase boundary on every account. Budget
reassessment triggers:

- After layer-selection pilot (Phase A.2): total pool spend should be
  <10% of available units.
- After Phase A full extraction: <20%.
- After Phase E screening: <50%. The remaining 50% is reserved for
  confirmation and final passes, which are by far the most expensive.
- If any checkpoint is exceeded, stop and reassess scope (cut
  confirmation features, cut α range, or drop final pass) before
  continuing.

### Tier sizing

Four tiers, from smallest to largest. The boundary between Target and
Extended is whether the Phase E final pass runs — separating it makes
the deliverables checklist unambiguous.

**Minimum (~3 weeks, solo-feasible).** Activation extraction on three
chosen layers only. Probes on raw activations + Gemma Scope 2
Width-262K SAE only. Phase E screening pass only (19k generations).
Produces probe AUC tables (with B0 comparison and bootstrap CIs),
cross-task transfer, cross-model comparison, screening-level steering
data with shared orthogonal and same-SAE baselines, decision-tree-
level interpretation. MATS-submittable as a Stage 2 deliverable;
lacks the per-feature-baseline confirmation pass. 4090 busy ~40
hours; Colab H100 ~17–21 hours. Solo pool fits.

**Target (~3–4 weeks, recommended, 2-account pool minimum).** Adds
Width-16K SAE, skip-transcoder features, and Phase E confirmation
pass (3–5 features × 5 α × 200 examples with per-feature baselines,
57k generations). Confirmation-level claims — "feature X shows
Δ-flip-rate Y over all four baseline families at 200 examples per
sweep point with 10 per-feature baselines, 95% CI Z" — are
scientifically sufficient for the phantom-vs-real question. 4090
busy ~60 hours; Colab H100 ~32–46 hours (confirmation dominates).

**Extended (~4–5 weeks, 3-account pool).** Adds the Phase E final
pass (1–2 features × single peak α × 2000 examples with per-feature
baselines, 88k generations). Gets you the large-sample confidence
intervals on the most promising feature(s). 4090 busy ~60 hours;
Colab H100 ~57–86 hours.

**Generous (~4–6 weeks, 3-account pool + Modal overflow).** Adds
cross-layer steering validation (repeating Phase E at a second
layer), subtype-task steering as a comparison, and a full Ma-et-al.-
style LLM-guided falsification pass (not just the paraphrase-
preserving test) on the top SAE features. Only worth it if this
becomes a paper rather than a Stage 2 deliverable. Cost dominated by
LLM judge API calls at that point.

Budget to Target; cut to Minimum if anything breaks. Specifically, if
the pool is running low, cut in this order:

1. Drop the Extended final pass. Target-tier confirmation-level claims
   are sufficient.
2. Cut confirmation to 3 features instead of 5.
3. Cut confirmation α to {-1, 0, +1} instead of {-2, -1, 0, +1, +2}.
4. Drop skip-transcoder features, keep only the two SAE widths.
5. Drop Width-16K, keep only Width-262K.

Do not cut: per-feature baselines at confirmation (the whole point of
that stage), the paraphrase-preserving test in B.3, S2 ontology-
held-out split evaluations, or the B0 baselines.

---

## Phase 0: Invariants, baselines, and splits (do this before Phase A)

A reviewer correctly flagged that my earlier draft treated "probe AUC >
chance" as evidence of reasoning-content probing, when in fact outcome
on InAbHyD is entangled with height, prompt length, ontology size,
output-strategy choice, and confidence. Phase 0 establishes the
baselines and splits that downstream probe AUCs must be compared against,
so the eventual claim is "activation/SAE probes beat a strong metadata
baseline by X on these splits," not "activation probes beat chance."

### 0.1: Pinned artifacts and invariants

Write `docs/stage2_invariants.json` containing:

- **Model identity.** For each of Gemma 3 4B and 27B: HuggingFace model
  revision hash, tokenizer revision hash, SHA-256 of the chat template
  string applied by `src/inference.py::build_messages`.
- **Stage 1 alignment.** SHA-256 of each shipped JSONL under
  `results/full/with_errortype/`. If any JSONL changes during Stage 2,
  probes must be retrained because the probe-behavior correspondence
  has shifted.
- **SAE release IDs.** For each layer we intend to probe at, the
  exact Gemma Scope 2 SAE release ID and checkpoint hash as they
  appear in SAE Lens at layer-selection time (the suffix conventions
  — `-16k`, `-262k`, `-res-all`, etc. — have varied across releases,
  so copy the actual string from `SAE.from_pretrained(release=...)`
  at pinning time rather than hardcoding one here). Pin these at
  layer-selection time (Phase A.2) and do not change after extraction
  starts.
- **GPT judge snapshot.** Paraphrase-preserving test in Phase B.3 and
  any Phase E CoT-classification calls pin to a dated snapshot (e.g.
  `gpt-5.4-2026-03-05`) rather than the moving alias. Store the
  snapshot ID alongside every judge call's output.

### 0.2: Row counts and data inventory

Before extraction, build `docs/stage2_inventory.json` with per-cell
counts from the shipped JSONLs:

- For each `(model, task, height, parse_failed, is_correct_strong)`
  cell: row count.
- For each `(model, task, height)`: total rows, positive count,
  negative count, parse-failed count, rows dropped for any other
  reason.
- Per-cell class balance after `parse_failed=False` filtering, so we
  can flag cells where positive or negative counts fall below the
  100-row threshold for meaningful per-height AUC. Known problem cells
  from Stage 1: `(4B, infer_subtype, h=3)` with 40 positives,
  `(4B, infer_subtype, h=4)` with 45 positives.

### 0.3: Split specifications

Three split families, defined upfront:

- **S1: Height-stratified random row split (70/15/15).** The baseline
  comparability split. Stratified by `height` so each height is
  proportionally represented in train/val/test.
- **S2: Ontology-held-out split.** InAbHyD examples are generated
  from ontology skeletons (tree topologies plus concept/entity name
  pools). Rows sharing an ontology skeleton can leak structural
  artifacts across train/test in S1. To make S2 a true
  structure-held-out split (and not partially a name-held-out split),
  key skeletons by **canonicalized topology**, not by concept-name
  sequence:

  1. Build the parent-child edge set from
     `ontology_fol_structured.inheritance`.
  2. Rename every node to an integer by BFS order from the root, so
     that "root→child_a→child_b" becomes "0→1→2" regardless of what
     the original concept names were.
  3. The skeleton key is the sorted tuple of edges under this
     renaming, plus generator parameters that are stable per skeleton
     (`height`, branching factor if available, task type).

  Two examples that use different concept names but the same topology
  hash to the same skeleton and end up in the same split partition.
  Bucket rows by skeleton, then assign skeletons to train/val/test
  using **stratified group splitting over height and label**
  (scikit-learn's `StratifiedGroupKFold` or equivalent): hold out
  15% of skeletons for test and 15% for val, but within each bucket
  match the train/test/val distribution of `(height,
  is_correct_strong)` cells as closely as possible. Without this,
  S2 could turn out "harder" than S1 partly because the held-out
  skeletons happened to land heavier on h=4 or on the incorrect
  class, making the split a confound of topology-generalization and
  class/height balance. Exact stratification is rarely achievable
  with group constraints, so record the residual imbalance and run
  B0-height and B0-prompt separately on S2 so the baseline comparison
  stays apples-to-apples.

  No ontology skeleton appears in both train and test. This is the
  harder split and the more scientifically interesting result — a
  probe that works only on S1 is likely memorizing skeleton
  structure.
- **S3: Name-scrambled generalization set.** Not a train/test split
  but an out-of-distribution evaluation set, generated fresh in B.3.
  See B.3 for details.

All three splits share a single pinned seed. Record which split an
example belongs to in `results/stage2/splits.jsonl`.

### 0.4: Baseline family B0 — metadata-only probes

Train a family of non-activation baselines that predict `is_correct_strong`
from features computable without looking at the model's internals at all:

**Pre-output baselines (fair comparison for activation probes).**
These features are computable from the prompt alone, before the model
generates any output — the same information available at the pre-CoT
activation position the probe is trained on.

- **B0-height.** Single feature: `height`. Trivial baseline; just
  reports "accuracy drops with depth."
- **B0-prompt.** Features: `height`, `prompt_token_count`,
  `num_theories_axioms`, `num_observations`, `parent_salience`,
  `num_direct_paths`.
- **B0-namefreq.** Adds per-row count of times each concept/entity
  name appears in train-set rows of the same class. Captures memorized
  name-frequency shortcuts.

**Post-output diagnostics (interpretation, not a baseline).** These
features are derived from the generated answer — they're not available
at the pre-CoT activation position. Comparing an activation probe
against these would be an unfair baseline (the baseline sees
information the probe doesn't). Kept as interpretation tools:

- **D0-strategy.** Adds Stage 1's output-strategy label (entity-level
  first vs. concept-level first) to B0-prompt. If a probe's Δ over
  B0-prompt largely disappears when D0-strategy is added, the probe
  was predicting strategy choice — a behavioral mode, not a reasoning
  mechanism. This is diagnostic, not a baseline to beat.
- **D0-parsefail.** Adds Stage 1's `parse_failed` label to B0-prompt.
  Similar role: tells us whether the probe's signal is partly
  commitment/confidence.

Train each B0 and D0 variant on each of S1 and S2, with logistic
regression (same hyperparameter sweep as the main probes). Report
per-height AUC and aggregate AUC. **Every activation/SAE probe AUC is
reported with Δ relative to the strongest pre-output B0 variant on
the matching split**, not absolute AUC alone, and not against the D0
post-output diagnostics.

Expected magnitudes, for calibration: B0-height alone should hit
~0.70 aggregate AUC on 27B property (accuracy drops from 96% at h=1
to 26% at h=4 — that's a very strong signal). B0-prompt may push to
~0.75. If your raw-residual probe hits 0.82, the meaningful number is
+0.07 over baseline, not 0.82 absolute.

### 0.5: Label-shuffle sanity probe

For S1 and S2, train one probe per feature source (raw residual at
layer L2) on shuffled labels. Expected AUC: ~0.50. Any deviation
flags a leak (e.g. stratification accidentally correlating with
something). Do this once as a precondition; no need to sweep.

---

## Phase A: Activation extraction

### A.1: Hidden-states extraction infrastructure

Stage 1's Modal deployment uses vLLM, which doesn't cleanly expose the
residual stream. For Stage 2 we extract activations directly via a
TransformerLens-based pipeline, split across platforms:

- **Gemma 3 4B: local 4090.** Load once in bf16 via the official
  `transformer-lens` package (Gemma 3 support landed in mainline, in
  `transformer_lens.model_bridge.supported_architectures.gemma3`). For
  each of the 44,000 shipped prompts, run a forward pass and capture
  residual stream activations at the target layers and at the last
  pre-CoT token position via a hook on `blocks.{L}.hook_resid_post`
  (or the bridge equivalent — see "TransformerBridge note" below).
- **Gemma 3 27B: Colab Pro H100 (80 GB).** Same TransformerLens setup
  loaded in bf16 (fits in ~54 GB with ~25 GB headroom). Outputs go to
  Google Drive for session-death survival. Modal HF-based deployment is
  the fallback if Colab H100 availability is unreliable — this is NOT
  the existing `gemma3-27b-inference` vLLM app (which doesn't expose
  activations); it's a separate deployment following the same file's
  pattern but with `AutoModelForCausalLM` and a custom JSON endpoint.

**Why TransformerLens rather than raw HuggingFace.** TransformerLens has
cleaner activation-hook APIs (`run_with_cache`, named hook points),
native integration with SAE Lens for Gemma Scope 2 feature extraction in
Phase C, and is the library Ma et al.'s pipeline is built on —
meaning their code (`features/collector.py`, `features/detector.py`,
`steering/steerer.py`) can be adapted with minimal friction. The cost
is that TransformerLens loads the full set of hooks and is somewhat
memory-hungrier than `AutoModelForCausalLM`; not an issue on 4090 for
4B or H100 80GB for 27B.

**TransformerBridge note.** TransformerLens 3.0+ introduced a
`TransformerBridge` system that's the path for newer architectures
including Gemma 3. The expected hook point is still likely
`blocks.{L}.hook_resid_post`, but verify with `model.hook_dict.keys()`
on first load — if the hook naming differs (e.g. bridge uses
`blocks.L.attn.hook_resid_post` or similar variants), pin the actual
string in `docs/stage2_invariants.json` so all extraction code uses
the same identifier.

**Reuse of Ma et al.'s code.** Their `reasoning_features` package was
written against the `huseyincavusbi/TransformerLens` fork that predated
mainline Gemma 3 support. When forking their `features/collector.py`,
`features/detector.py`, and `steering/steerer.py`, replace their
import line (and any fork-specific calls) with imports from the
official `transformer_lens` package. The Cohen's d ranking, SAE-hook
machinery, and steering-intervention logic transfer directly; only
the model-loading and hook-name strings may need adjustment.

**Key design choices:**

- **Layer selection is parameterized.** The full-layer sweep in Phase
  A.2 and the three-layer full extraction in Phase A.3 use the same
  extraction code with different layer lists.
- **Position selection is fixed to last pre-CoT token**, computed by
  applying the Gemma 3 chat template with `add_generation_prompt=True`
  and taking the last index. Reuse `src/inference.py::build_messages`
  which already handles Gemma's missing-system-role concatenation. The
  templated string goes into TransformerLens `model.to_tokens(...)`;
  the last position gets hooked. Following Cox et al.; matches
  `BEHAVIORAL_DATA_PLAN.md` §Interface Contract with Teammate B.
- **Output format: one `.safetensors` file per (model, task, layer)**,
  shape `[N, D]` in bf16, alongside a `.example_ids.jsonl` sidecar
  preserving row-to-example_id alignment. Write to Google Drive (Colab)
  or local disk (4090) in 500-example chunks so partial progress
  survives session death.
- **Batch size: 32 for 4B on 4090, 8 for 27B on H100.** Verify with a
  10-example batch before committing to larger runs — Gemma 3 has a
  4096-token max context and prompts are modest, so these should be
  comfortable, but TransformerLens can surprise you on memory.

**Equivalence check: tiered, not strict byte-identical.** Byte-identical
output from TransformerLens greedy decode versus Stage 1's vLLM greedy
decode is the goal, but small differences in attention kernels,
tokenizer revisions, and tie-breaking can produce divergent byte output
even when the model is effectively the same. Rather than hard-fail on
byte mismatch, run a five-tier check and accept at tier 4:

1. **Tokenizer identity.** Prompt token IDs from TransformerLens'
   tokenizer match Stage 1's tokenization exactly on 50 random prompts.
   Hard fail if any divergence — the model sees different inputs.
2. **Revision hashes.** HuggingFace model revision, tokenizer revision,
   and chat-template checksum match the values pinned in Phase 0.1.
   Hard fail on mismatch.
3. **First-token top-5 logit agreement.** On 50 random prompts at
   `temperature=0`, the top-5 next-token logits from TransformerLens
   and vLLM agree in identity and rank. Small numerical differences in
   logit values are fine; identity and rank are the relevant check.
4. **Greedy-output byte match on 50 rows.** Ideally this passes. If it
   doesn't, escalate to 5.
5. **Label agreement on 200 rows.** Run the Stage 1 `is_correct_strong`
   scoring pipeline over TransformerLens-regenerated outputs; compare
   to the shipped labels. Accept ≥98% agreement. Below that, the
   probe-behavior correspondence is weak enough that Stage 2 results
   aren't trustworthy.

Passing tier 3 with tier 4 failure plus tier 5 ≥98% agreement is a
"proceed with caveat" state — extract activations, but document in the
results that greedy outputs are not byte-identical to Stage 1.
`scripts/validate_activations.py` implements all five tiers and writes
`results/stage2/equivalence_report.json`.

**Determinism check.** Extract the same 20 rows twice in the same
session. Activations must match bit-for-bit at positions we probe.
Catches lurking non-determinism in the extraction path (Flash Attention
non-determinism, etc.).

### A.2: Layer selection pilot

Extract all 62 layers (27B) / 34 layers (4B) at the pre-CoT position on a
500-example stratified subset: 125 per height × 4 heights × `infer_property`
only × both models. This is a one-shot run for layer selection, not a
deliverable.

Per-layer, train a quick `sklearn.linear_model.LogisticRegression` probe on
a 400/100 train/test split, predicting `is_correct_strong`. Rank layers by
held-out AUC.

*Reference priors from related work.* Ma et al. (2026) use Gemma-3-4B at
layers 17, 22, 27 with Gemma Scope 2 Width-16K SAEs, citing token-
concentration-ratio analysis (their Figure 2) that shows middle layers
have lower token concentration and higher entropy than early or late
layers. They treat this as evidence that middle-to-late layers are the
best candidates for reasoning features. This is directly relevant to us
(same model, same SAE family) and suggests our layer sweep will likely
identify winners in a similar range. Cox et al. (2026) find the best
probe layer varies by model and task and report layerwise AUCs across
all layers of their smaller models. Our pilot reproduces the Ma et al.
methodology empirically on our task rather than assuming it transfers.

Pick three layers per model:

- The highest-AUC layer.
- The highest-AUC layer in the first third of the model (captures early
  computation if it exists).
- The highest-AUC layer in the last third (captures late computation).

This spans the model depth-wise while committing to layers that actually
carry predictive signal. Record the three chosen layers and their AUCs in
`docs/layer_selection.json` with a timestamp and the commit SHA of the
pilot code.

Commit only these layers to full extraction. If the highest-AUC layer is
in the last third (likely), the "last third" pick becomes redundant — in
that case substitute the second-highest-AUC layer in the first two thirds
as your third layer.

### A.3: Full activation extraction

For each (model, task) in the full cross product — 2 models × 2 tasks — run
the hidden-states extraction over all 11,000 Stage 1 examples for that cell
at the three chosen layers. Output: one `.safetensors` file per
(model, task, layer) shape `[N, D]` where N ≤ 11,000 (parse-failed rows
optionally included for Phase B's auxiliary 3-way analysis; see B.1.bis)
and D is the hidden dim — 5376 for Gemma 3 27B, 2560 for Gemma 3 4B —
plus a sidecar `.example_ids.jsonl` preserving the row-to-example_id
alignment. Total across the full cross product: 44,000 rows extracted,
distributed across 4 JSONLs.

Storage: 11k × 5376 × 2 bytes × 3 layers × 2 tasks for 27B ≈ 700 MB +
11k × 2560 × 2 bytes × 3 × 2 for 4B ≈ 340 MB ≈ 1 GB total. Google Drive
or a Modal volume handles this comfortably.

Wall-clock: at batch 8 on H100 for 27B, expect ~3–5 forward passes/sec
(no generation, just one forward per row for the last-position activation).
11k examples × 2 tasks / 4 passes/sec ≈ 1.5 hours per task ≈ 3 hours on
27B. 4B on 4090 is much faster, ~1 hour total. Budget half a day with
margin for TransformerLens setup, SAE pinning, and the verification
sweep below.

### A.4: Verification

Before Phase B begins:

- [ ] Three layer files per (model, task) exist with row counts matching
  the Stage 1 JSONLs exactly.
- [ ] `.example_ids.jsonl` order matches Stage 1 JSONL row order (diff
  them explicitly; don't trust implicit ordering).
- [ ] `results/stage2/equivalence_report.json` exists and passes hard
  tiers 1–3 (tokenizer identity, revision hashes, first-token top-5
  logit agreement). Either tier 4 (greedy-output byte match) passes
  or tier 5 (200-row label agreement) is ≥98%.
- [ ] Random spot-check: load one activation, feed it into the model's LM
  head, confirm logits at the first-generated-token position match a
  fresh generation. This catches off-by-one position bugs which would
  corrupt all downstream probe work.

### A.5: Out of scope for Phase A

- Activations at positions other than last pre-CoT.
- Attention patterns, MLP intermediates, anything other than residual
  stream.
- Storing full-context hidden states (only the pre-CoT position).
- Running the extraction on `infer_membership_relation` or on the pilot
  dataset (only `data/full` is extracted).

---

## Phase B: Probe training on raw activations

### B.1: Probe specification

Two probe types per (model, task, layer, train-set):

- **Difference-of-means** (Cox et al. §2.3). Compute
  `μ_correct - μ_incorrect` over training activations; score a test
  activation by cosine similarity to this direction (Cox et al.
  normalize; this affects AUC ordering slightly vs. raw dot product but
  is how their published results are computed). Predict `correct` if
  score > threshold, tuned on val set to maximize balanced accuracy.
  No hyperparameters beyond threshold.
- **Linear (logistic regression)** with L2 regularization. Single
  hyperparameter: regularization strength C, picked from
  `{0.01, 0.1, 1.0, 10.0}` on a val set. sklearn implementation;
  `liblinear` solver.

Both types trained per (model, task, layer) on the split specifications
pinned in Phase 0.3 — **both S1 (height-stratified random) and S2
(ontology-held-out)**. S1 gives comparability with prior work; S2 is
the harder and more scientifically interesting result. Report AUC on
both. Seed pinned.

**Parse-failed rows are excluded from main binary probe training.**
Stage 1's shipped dataset has `parse_failed=True` on roughly 9–10% of
27B `infer_property` rows at h≥2 — these are Gemma 3 hedging
disjunctions ("X is Y or X is not Y") where the model explicitly
declined to commit to a hypothesis. They are genuinely unscorable as
correct-vs-incorrect; training a binary probe on them would inject
label noise. Filter `parse_failed=False` before the train/val/test
split. Document per-cell drop counts in the probe results.

**Evaluation metric: held-out AUC, reported as Δ over the strongest B0
metadata baseline from Phase 0.4, not absolute AUC.** This is the
single most important reporting discipline for this phase: an
activation probe hitting 0.82 AUC when B0-prompt hits 0.75 is a much
weaker result than the same 0.82 when B0-prompt hits 0.55, and
absolute-AUC language obscures the difference. Report both split
families (S1 and S2); report per-height breakdowns because
ceiling/floor effects make aggregate AUC misleading (e.g. h=1 at 96%
accuracy has too few negatives to be meaningfully probed); report
bootstrap 95% CIs (1000 resamples) on all AUCs.

Also report balanced accuracy at the tuned threshold for the
difference-of-means probe, for comparability with Cox et al.'s published
numbers.

### B.1.bis: Auxiliary 3-way analysis (correct / incorrect / parse-failed)

The binary probe above answers "can we predict whether the model got
this one right." A separate, smaller auxiliary analysis answers "can
we predict whether the model will commit to a hypothesis at all." This
matters because a probe that predicts `is_correct_strong` could plausibly
be a "model is about to hedge" detector with the hedged outputs already
filtered out, making the reported signal less reasoning-specific than
it looks.

For 27B `infer_property` only (the cell with the most hedged rows),
train a 3-way softmax classifier on the same S1 split predicting
`{correct, incorrect, parse_failed}` from the raw residual at the
selected layer L2. Report:

- 3-way accuracy vs. a class-majority baseline.
- Binary collapse: project to `correct vs. not-correct` and compare
  AUC against the main binary probe. If the 3-way probe's binary
  collapse matches the main probe's AUC, the main probe isn't losing
  information from the filtering. If it differs substantially, the
  parse-failed rows carry information the main probe can't see.
- Confusion matrix: specifically, is the probe confusing "incorrect"
  and "parse_failed"? If so, the signal is partly commitment/confidence,
  not purely correctness.

This is one additional probe per (27B, infer_property, L2) — trivial
compute cost and valuable interpretation.

### B.2: Transfer and cross-model comparison

**Cross-task transfer (primary transfer diagnostic).** Probe trained
on `(M, infer_property, L)` evaluated on `(M, infer_subtype, L)` test
split, within the same model. Same-model activations means same
`d_model`, so the probe direction is directly applicable. Two
directions × two models × three layers = 12 transfer evaluations.
This is the clean transfer diagnostic the plan relies on for the
phantom-vs-real surface-feature test: a probe that transfers cleanly
across tasks with different surface form cannot be task-specific
lexical-cue detecting.

**Cross-model comparison, not raw transfer.** My earlier draft
specified raw cross-model probe transfer — train a probe on 4B,
evaluate on 27B. A reviewer correctly caught that this is not
well-defined: Gemma 3 4B has `d_model = 2560` and Gemma 3 27B has
`d_model = 5376`, so the 4B probe direction (a vector in ℝ²⁵⁶⁰)
cannot be applied to 27B activations (vectors in ℝ⁵³⁷⁶). Learned
alignment (Procrustes, ridge regression on paired prompts) would
work but is a serious methodological addition, out of scope for a
Stage 2 deliverable.

Replace raw cross-model transfer with **cross-model comparison of
probe properties** — a qualitative side-by-side of what independently-
trained 4B and 27B probes look like:

- At what depth-fraction does the highest-AUC layer land in each
  model? (If both pick a "middle-late" layer at ~0.7 depth fraction,
  that's a point of structural convergence.)
- What's the Δ over the strongest pre-output B0 on S1 and S2 for
  each model?
- What's the per-height AUC profile? Do both models show the same
  shape (e.g., signal emerges at h=2, saturates at h=3)?
- What's the stability profile of the SAE top-features set (Phase
  D.3) for each model?

These are qualitative comparisons, not transfer evaluations; they
don't produce an AUC number, they produce a comparison table.

**Cross-model caveat from Stage 1 (still relevant for interpretation).**
The 4B model falls back to entity-level enumeration ~50% of the time
on `infer_subtype` at h≥3 (Stage 1 draft, "Output strategy
divergence"), while 27B stays concept-level >90% of the time. A 4B
probe is therefore partially distinguishing "concept-level response
vs. entity-level enumeration" at depth. This is another reason raw
cross-model transfer wouldn't be interpretable even if dimensions
matched — the probes may be detecting structurally different things.
For the qualitative comparison above, report strategy-frac alongside
the comparison so this confound is visible.

### B.3: Reasoning-vs-surface-feature decomposition

Three diagnostics per trained probe. The first two directly address the
surface-feature variant of the phantom concern; the third is the direct
analog of Ma et al.'s false-negative construction and lets us test
whether our probes respond to surface form or to reasoning content.

**Prompt-length regression.** Report two complementary tests, not just
partial correlation:

1. *Partial correlation.* Regress probe logit on `prompt_token_count`
   alone, then jointly on `prompt_token_count` and `height`. Report
   the partial correlation of probe logit with `is_correct_strong`
   after controlling for length. If controlling for length kills the
   signal, the probe was a length detector.
2. *Within-height AUC.* Retrain the probe per-height (h=2, h=3, h=4
   separately — h=1 has too few negatives) and report within-height
   AUC. This removes height as a confound directly rather than
   residualizing post-hoc. h=1 may be unstable but a within-height
   probe that still hits meaningful AUC at h=2/3/4 is stronger
   evidence of reasoning-content detection than an all-heights probe.

**Name-scramble ablation, two conditions.** Random strings like
`zqxblat` create rare-token artifacts that confound the test. Run two
scramble conditions instead of one:

1. *Pronounceable nonce words.* Generate 500 eval examples at each
   height with concept/entity names swapped for pronounceable nonce
   words that match the token-count distribution of the originals
   (check via the Gemma tokenizer: target same number of tokens
   per name). Pools of nonce words can be generated once and reused
   across heights; record them in `docs/stage2/nonce_pool.json`.
2. *Natural name substitution.* Same-size eval set, but concept names
   come from a held-out natural-name pool — names in the benchmark's
   morphology list that weren't used in any Stage 1 example. This
   tests generalization without introducing rare-token artifacts at
   all.

A probe that holds up under (2) but not (1) is telling you the nonce
condition was driving an artifact. A probe that holds up under both
is substantially more trustworthy than either alone.

**Paraphrase-preserving test (Ma et al.–style false-negative probe).**
Select 200 correctly-classified examples from the S1 test split (100
where `is_correct_strong=True`, 100 where `=False`). For each, use
**GPT-5.4 via OpenAI** (pinned snapshot, see Phase 0.1) to generate a
meaning-preserving paraphrase that keeps the same theories,
observations, and ground truth but varies sentence order, phrasing
("Every X is Y" vs. "All X are Y"), and entity-introduction order.
Reuse `src/error_classification.py`'s infrastructure:
`get_openai_gpt_credentials()` for auth, `_is_gpt5_plus()` for the
reasoning-model kwargs quirk (`max_completion_tokens` instead of
`max_tokens`, no `temperature=0`). Ma et al. used Gemini 3 Pro via
OpenRouter; GPT-5.4 is the same rigor tier on the other vendor, and
matches Stage 1's judge-model stack.

**Paraphrase validation — do not trust the paraphrasing model alone.**
A paraphrase that silently drops a theory axiom or flips a negation
invalidates the test. Two-stage validation per paraphrase:

1. *Parser round-trip.* Feed the paraphrase through a lightweight
   structural parser that extracts the theories, observations, and
   query as sets of logical atoms (reuse Stage 1's `structured_fol`
   construction pattern from `src/export.py`). Compare against the
   original's `ontology_fol_structured`. Reject paraphrases where any
   axiom, observation, or the query target changes.
2. *Second-judge semantic check.* On paraphrases that pass (1), ask a
   second call to GPT-5.4 (with a different prompt role): "Does the
   paraphrase preserve every fact and the same final question?
   Yes/no." Reject the no's. This catches paraphrases that pass the
   structural parser because the parser is forgiving but change
   meaning in ways a reader would flag.

Budget: expect 20–30% paraphrase rejection rate. Generate 300 to
retain 200 valid pairs. Report the rejection rate and a sample of
rejected paraphrases in the results for transparency.

Run the target model on each validated paraphrased prompt (Phase A
extraction), project onto the probe direction, and measure: (a)
whether the paraphrase flips the model's `is_correct_strong`
(behavioral stability under paraphrase), (b) whether the probe's
prediction flips (probe stability under paraphrase). Mismatches
between (a) and (b) are informative: probe predictions that change
when model behavior doesn't were tracking surface form rather than
reasoning content — Ma et al.'s false-negative failure mode.
Mismatches in the other direction (model behavior changes but probe
doesn't) are also informative and suggest the probe generalizes
beyond surface form.

All three diagnostics are required for the Target tier and above; Minimum
tier may skip the paraphrase-preserving test if time is tight, but the
phantom-vs-real claim is materially weaker without at least one of name-
scramble or paraphrase-preserving.

### B.4: Deliverables

- `results/stage2/probes/{model}_{task}_{layer}.{probe_type}.pkl` — one
  per trained probe, containing the probe parameters, the
  train/val/test indices, the tuned threshold, and metadata (model
  slug, task, layer index, commit SHA).
- `results/stage2/probe_auc.json` — nested dict of
  `{model: {task: {layer: {probe_type: {split: auc}}}}}` plus per-height
  breakdown.
- `results/stage2/probe_transfer.json` — cross-task transfer results
  only, shape `{train_cell: {eval_cell: auc}}`.
- `results/stage2/probe_comparison.json` — cross-model comparison
  table: per-model highest-AUC-layer depth fraction, Δ over B0 on S1
  and S2, per-height AUC profile. See B.2.
- `results/stage2/probe_diagnostics.json` — length-regression partial
  correlations, within-height AUC, both name-scramble conditions,
  and paraphrase-preserving agreement rates per probe.
- One figure: probe AUC vs height, one subplot per (model, task), lines
  for each layer. This is the direct analog of Stage 1's
  `accuracy_vs_depth.png`.
- One figure: cross-task transfer scatter — probe AUC on source cell on
  x-axis, probe AUC on transfer target on y-axis. Points on the diagonal
  = perfect transfer; points below = surface-bound probe.

### B.5: Pre-handoff sanity checks

- [ ] Every (model, task, layer, probe_type) combination has a trained
  probe and a recorded AUC.
- [ ] **Inspect** cells with suspicious train–test AUC gaps (train
  much higher than test → overfit; test much higher than train →
  sampling artifact) and cells where validation AUC clearly exceeds
  both. Don't require strict `train > val > test` ordering; that
  ordering is common but not guaranteed, particularly in small
  per-height cells where validation can beat train by sampling luck.
- [ ] Test AUC at h=1 is interpretable or explicitly marked N/A — at 96%
  accuracy there may be <40 negatives in the test split, giving
  uninformative AUC.
- [ ] Cross-task transfer evaluations use the same probe parameters
  (same layer, same probe_type) trained on the source cell.
- [ ] Length-regression partial correlation computed for every probe at
  Target tier and above.

---

## Phase C: SAE and skip-transcoder feature extraction (skeletal)

Apply Gemma Scope 2 SAEs at Width-16K and Width-262K (and skip-transcoders
at matching widths) to the cached activations from Phase A, at the three
chosen layers.

**Direct reuse target.** Ma et al.'s repo (`GeorgeMLP/reasoning-probing`)
has working code for this exact operation in `reasoning_features/features/collector.py`
(SAE activation collection) and `reasoning_features/features/detector.py`
(Cohen's d / ROC-AUC feature ranking). Fork and adapt rather than writing
from scratch. Their contrastive selection is reasoning-text vs. non-
reasoning-text; ours is correct vs. incorrect, so the grouping logic
needs replacement but the Cohen's d computation and SAE-hooking machinery
transfers directly.

Per activation, store the top-k feature indices and values (k=128 by
default; revisit if probe AUC saturates quickly with k). Output format:
`.safetensors` with sparse (indices, values) pairs, per (model, task,
layer, sae_width).

**SAE availability.** Google's Gemma Scope 2 release provides residual
SAEs and transcoders for every layer of Gemma 3 pretrained and
instruction-tuned models from 270M through 27B. SAE Lens lists
`blocks.*.hook_resid_post` residual SAEs at both 16K and 262K widths
for Gemma-3-27B-IT. So any layer our Phase A.2 pilot selects will
have Gemma Scope 2 SAE coverage — no need to fall back to a nearby
layer for lack of SAE. The remaining concern is pinning the exact
SAE release IDs (captured in Phase 0.1's `docs/stage2_invariants.json`)
so the analysis is reproducible if Google ships an updated release
mid-project.

Key operational details for execution:

- Storage sizing. 44k × 128 × 8 bytes × 3 layers × 2 widths × 2 models
  ≈ 540 MB, manageable.
- Whether to compute feature activations lazily (on probe-training
  demand) or eagerly (one-shot). Eager is simpler; lazy saves disk if
  we decide to skip a width.

Phase D probe training will consume these features directly.

---

## Phase D: Probe training on SAE and skip-transcoder features (skeletal)

Same probe specification as Phase B, but features are now SAE activations
(top-k sparse) instead of raw residual stream. Use the same train/val/test
splits as Phase B (both S1 and S2) for direct comparability.

**Phase D separates three claims that should not be collapsed.** A
reviewer pointed out these are often conflated in the SAE
interpretability literature, to the literature's cost:

1. *SAE representation preserves signal.* The SAE-feature probe AUC
   approaches the raw-residual probe AUC. If raw hits 0.85 and
   Width-262K SAE hits 0.84, the SAE representation isn't throwing
   away the signal. This is a claim about the SAE decomposition's
   completeness.
2. *Individual SAE features localize signal.* A small top-k feature
   set (k ≤ 20) recovers most of the full-feature probe's AUC. Sparse
   regularization on the SAE-feature probe (L1 or elastic-net
   logistic regression, sparsity parameter swept) gives a more
   principled way to check this than "take the top-k of a dense
   probe's direction." If a 20-feature probe hits 0.82 against a
   full-feature probe's 0.84, signal is localized; if the 20-feature
   probe drops to 0.70, it's distributed.
3. *Individual SAE features are causal.* Phase E. Separate phase,
   separate claim; don't let probe AUC leak into causal language.

### D.1: Full SAE-feature probe

Same probe types (diff-of-means + logistic) on full-width SAE features
(16K or 262K). Report AUC on S1 and S2 splits, with Δ over the
strongest B0 baseline just as in Phase B. Answers claim (1).

### D.2: Sparse SAE-feature probe and top-k localization

Fit L1 logistic regression on SAE features, sweeping the regularization
parameter to produce feature-count-vs-AUC curves. Report the AUC at
k = 5, 20, 100, 500 nonzero coefficients. Answers claim (2).

### D.3: SAE-feature stability across seeds and widths

Top-20 SAE features selected by the Phase D.1 probe direction may not
be stable across bootstrap seeds, splits, or SAE widths. If the top-20
list changes substantially across resamples, steering "the top feature"
in Phase E is steering noise. Stability checks:

- *Bootstrap stability.* Train 10 probes on 10 bootstrap resamples of
  the S1 train set; compute the Jaccard overlap of the top-20 feature
  sets across seed pairs. Report the median and range. A stable
  probe has median overlap ≥ 0.5; Jaccard ≤ 0.2 means the top-20
  is largely seed-dependent.
- *Cross-split stability.* Top-20 on S1 vs. top-20 on S2. If these
  are disjoint, the probe is capturing split-specific artifacts, not
  a general signal.
- *Cross-width stability.* Top-k Width-16K features vs. top-k
  Width-262K features. With 16× more features at 262K, an "important"
  reasoning feature should plausibly have a direct counterpart at
  16K (or split into a few). Disjoint top-k across widths is a
  warning sign that feature selection is unstable at one or both
  widths.

**Phase E stability gate — tiered, not exact top-20.** With 262k
features, exact top-20 overlap across seeds and splits may be too
strict — a real distributed signal could spread across correlated
features that each individually cross the top-20 threshold on some
resamples but not others. Relax to a tiered stability gate:

- **Tier S-strong (highest confidence).** Feature appears in top-20
  on both S1 and S2, AND survives ≥50% of bootstrap resamples
  (Jaccard ≥ 0.5 with the full-data top-20 across seeds).
- **Tier S-medium.** Feature appears in top-100 on both S1 and S2,
  has the same sign of probe-direction coefficient across splits,
  and its Phase C activation distribution is similar across splits
  (KL divergence below a threshold to be pinned empirically).
- **Tier S-loose (correlated-cluster membership).** Feature belongs
  to a correlated-feature cluster (Pearson correlation ≥ 0.7 with
  another top-20 feature on the same split) where at least one
  member of the cluster hits S-strong on both S1 and S2.

Phase E should steer features in order of tier: S-strong first, fall
back to S-medium if S-strong is empty, then S-loose. If no features
reach any tier, Phase E is not run and the Stage 2 result is
"representation preserves signal (D.1) but individual features don't
localize stably (D.3), so causal SAE-feature steering wasn't
attempted." That is a legitimate result that reproduces Ma et al.'s
null in a different way; don't force Phase E on unstable features
just to have a Phase E section.

Record the stable-feature set per tier in
`docs/stage2/stable_features.json` before Phase E begins.

### D.4: Deliverables

- `results/stage2/sae_probes/{model}_{task}_{layer}_{width}.{probe_type}.pkl`
  — one per trained SAE probe.
- `results/stage2/sae_probe_auc.json` — AUC on S1 and S2 for each
  full-width and sparse probe, with bootstrap CIs.
- `results/stage2/stable_features.json` — the stable-feature set per
  (model, task, layer, width) that Phase E is allowed to steer.

---

## Phase E: Causal steering validation (skeletal)

**Direct reuse target.** Ma et al.'s `reasoning_features/steering/steerer.py`
implements the exact steering intervention we need (residual-stream
addition at a target layer, with a coefficient γ applied to the SAE
decoder direction). Their `steering/evaluator.py` handles benchmark
scoring. Fork and adapt to score against our InAbHyD parser rather than
AIME/GPQA. Critical: scoring of steered outputs uses
`src/gemma3_parse.py::parse_hypotheses` (the Stage 1 parser) — any
other parser introduces a confound between steering effect and scoring
regime.

### E.1: Four intervention families

Running one orthogonal baseline per feature (the draft this replaces)
is too weak. Every candidate SAE feature is compared against four
intervention families at matched α:

1. **Raw probe-direction steering (Cox-style reference point).**
   Steer along the difference-of-means probe direction from Phase B,
   same α, same layer. This serves as a sanity check on the steering
   setup and a reference against Cox et al.'s published flip rates
   (>50% on their binary tasks). *Failure is informative but not
   dispositive.* InAbHyD differs from Cox et al.'s setup in answer
   format, generation dynamics, parser sensitivity, and α scaling; if
   raw-direction steering doesn't flip behavior, interpret SAE-feature
   steering results cautiously and investigate whether the α range is
   wrong, the target layer is suboptimal, or the intervention mode
   (pre-CoT vs. all-tokens) needs adjusting — but don't conclude
   causal steering is impossible on InAbHyD from raw-direction
   failure alone.
2. **SAE top-feature steering (main experiment).** Steer along the
   SAE decoder direction for each stable feature from Phase D.3.
3. **Random orthogonal residual directions (generic perturbation
   control).** Sample 5–10 random directions orthogonal to both the
   probe direction and the target SAE feature decoder direction;
   norm-match to the SAE feature's decoder norm. Run the same
   intervention. Any flip rate here represents generic-perturbation
   effects that can't be attributed to reasoning content.
4. **Random same-SAE decoder directions (SAE-basis perturbation
   control).** Sample 5–10 random SAE feature decoder directions
   excluding the top-stable-features set, norm-matched. Tests whether
   the flip rate is specific to the selected features or whether any
   SAE direction at the same layer and width produces similar
   behavior change.

Additionally, a **no-op hook baseline** runs the generation with a
hook attached but adding zero to the residual, to catch any
generation-code regressions introduced by the hooking machinery itself.

### E.2: α parameterization in residual-std units

Raw coefficient α values aren't comparable across features or layers
because the SAE decoder direction norms differ. Parameterize
intervention magnitude in **residual-stream standard deviations**:
compute σ_L = std of the raw residual activations at layer L across
the full Stage 1 training set, and express steering as `α · σ_L ·
direction / ||direction||`. Sweep α ∈ {-3, -1, 0, +1, +3} at screening,
{-2, -1, 0, +1, +2} at confirmation if needed. Report Cox et al.'s α
range translated into σ units in the write-up for comparability.

### E.3: Staged budget with explicit generation-count formulas

Phase E dominates Stage 2 compute. The flat "top-10 × 5 α × 5000
examples × 4 baseline families" would be ~250k–1M steered generations
on 27B, infeasible. Staged instead. Baseline-direction sampling
policy matters enormously for the arithmetic — a reviewer pointed out
that "per-feature vs. shared baselines" changes total generation
count by 5–20×. The policy here is **shared at screening, per-feature
at confirmation and final**:

- At screening, random orthogonal and random same-SAE baselines
  establish what generic perturbation at this layer/width looks like.
  That's a layer-level property, not a per-feature property — so
  screening uses one set of shared baselines across all screening
  features. Trades some statistical power against baseline
  specificity, but the screening pass only has to rank features
  coarsely.
- At confirmation and final, we want the stronger per-feature
  guarantee that "this feature's effect is specific, not shared with
  neighbor directions." Baselines are re-sampled per feature.

General generation-count formula for a pass:

    N_gen = examples × alpha_values × (
        candidate_features               # feature steering itself
        + raw_probe_reference            # one direction, all α
        + no_op_controls                 # one "no-op" baseline
        + (per_feature ?
              candidate_features × (n_orth + n_same_sae)
              : n_orth + n_same_sae)
    )

Applying this to each pass:

**E.3.a Screening pass — shared baselines.**
- 20 features, α ∈ {-3, 0, +3} (3 α), 200 examples per point
- 5 shared orthogonal + 5 shared same-SAE + 1 raw-probe + 1 no-op = 12 baseline directions total, not per-feature
- N_gen = 200 × 3 × (20 features + 12 baselines) = 200 × 3 × 32
  = **19,200 generations**

**E.3.b Confirmation pass — per-feature baselines.**
- 3–5 features that survived screening (use 5 as the upper-bound
  estimate), α ∈ {-2, -1, 0, +1, +2} (5 α), 1000 examples per point
- Per-feature: 10 orthogonal + 10 same-SAE = 20 baselines per feature
- Plus 1 raw-probe reference, 1 no-op (shared, computed once at each α)
- N_gen = 1000 × 5 × (5 features + 5 × 20 per-feature baselines + 2 shared)
       = 1000 × 5 × (5 + 100 + 2)
       = 1000 × 5 × 107
       = **535,000 generations** — too expensive

  That's a real problem. To get confirmation to a feasible ~50k,
  either (a) cut per-feature baselines to 5 orth + 5 same-SAE, or
  (b) cut examples per point to 200 at confirmation and save the 1000-
  example count for the final pass, or (c) cut α range to {-2, 0, +2}.
  Realistic confirmation: **1000 × 5 × (5 + 5 × 10 + 2) = 285,000**
  with 10 per-feature baselines, or **200 × 5 × (5 + 5 × 10 + 2) =
  57,000** cutting examples. Go with the latter: 200 examples is
  still enough to detect meaningful flip rate over 10 per-feature
  baselines at this stage. **Confirmation generations: ~57,000.**

**E.3.c Final pass — per-feature, large sample.**
- 1–2 features (use 2), single α of peak margin, 5000 examples
- Per-feature: 10 orthogonal + 10 same-SAE = 20 baselines
- Plus 1 raw-probe, 1 no-op (shared)
- N_gen = 5000 × 1 α × (2 + 2 × 20 + 2) = 5000 × 44 = **220,000** —
  also too expensive. Cut to 2000 examples: 5000 × 44 → 2000 × 44
  = **88,000 generations**. Or keep 5000 examples but reduce
  per-feature baselines to 5 + 5 = 10: 5000 × (2 + 2 × 10 + 2) =
  120,000. Go with **2000 examples + 20 per-feature baselines =
  ~88,000 generations** — gives robust statistical power on both
  the feature and baseline side.

**Revised totals: 19k (screening) + 57k (confirmation) + 88k (final)
≈ 165k generations on 27B.** At H100 throughput of ~0.5–1 gen/sec
for 27B with hooks and TransformerLens overhead, that's 45–90 hours.
Tighter than my earlier estimate implied — this probably exceeds
Colab Pro+'s monthly compute budget and spills to Modal (at
$2–4/hr × 80 hours = $160–320) or a second month of Pro+.

**If budget is tight, cut in this order, not the reverse:**
1. Drop the final pass and report the confirmation pass as the
   headline — "confirmed at 200 examples per sweep point with 10
   per-feature baselines" is scientifically sufficient.
2. Cut confirmation to 3 features.
3. Cut to α ∈ {-1, 0, +1} at confirmation.

Do not cut per-feature baselines at confirmation. The entire point of
confirmation is the per-feature baseline guarantee; without it,
confirmation is just a larger screening pass.

### E.4: Pre-CoT-only vs. all-tokens intervention

My earlier draft defaulted to "steer at every position during
generation." That's a more invasive intervention with a weaker
causal claim: you're perturbing all of generation dynamics, not just
the precomputed reasoning state at the probe position. A reviewer
was right to flag this. Run both modes on the final-pass features:

1. *Pre-CoT-only steering.* Hook fires only at the last-pre-CoT
   position (the position the probe was trained on), adding the
   steering vector once. This is the intervention most directly
   tied to the Phase B causal claim.
2. *All-generation-token steering.* Hook fires at every generation
   position, as in my original draft. Stronger behavior change but
   weaker causal specificity.

If pre-CoT-only steering produces meaningful flip rate above
baselines, the causal story is tight — the reasoning outcome is
encoded at the probe position, and the SAE feature direction there
controls it. If only all-token steering works, the claim softens to
"steering this feature direction somewhere in the model's forward
pass changes outcomes," which is weaker but still informative.

### E.5: Measurement — full dependent-variable set

For each intervention × α combination, record:

- correct → incorrect flip rate (primary)
- incorrect → correct flip rate (secondary; expect low for most
  features but informative if nonzero)
- parse-failure rate (does steering push the model into hedging?)
- output-format degeneration rate (repetition, nonsense — a flag
  that intervention magnitude is too large)
- mean output length (proxy for format degeneration)
- CoT classification on flipped generations (Cox et al. four-cell
  framework, GPT-5.4 judge — see below)
- effect size with bootstrap 95% CI

Report each dependent variable vs. all four baseline families, not
just against the no-op baseline.

### E.6: Prior expectations given Ma et al. and Cox et al.

Ma et al.'s steering experiment on top-3 contrastively selected SAE
features for Gemma-3-12B at layer 22 reduced AIME accuracy from 26.7%
baseline to 10–26.7% under γ=2 steering, and GPQA from 37.9% to
13.6–33.3%. None of the features improved reasoning. Taken at face
value, this is evidence that SAE features selected by their contrastive
method don't causally drive reasoning — the null result for Stage 2.
Cox et al., by contrast, steered difference-of-means probe directions
(not SAE features) and achieved >50% answer flip rates clearly
exceeding orthogonal baselines.

So the likely outcomes for our Phase E are: (a) our SAE-feature
steering looks like Ma et al.'s (minimal or degrading effect,
indistinguishable from orthogonal), reproducing their null result on
InAbHyD — scientifically valid; (b) our SAE-feature steering looks
like Cox et al.'s (clean flip rate above orthogonal baseline), a novel
positive result given the failure mode Ma et al. document; (c)
somewhere in between, which is the most likely and also the most
informative case. Interpret our results against both anchors
explicitly in the write-up.

A particularly informative sub-outcome: raw probe-direction steering
produces meaningful flip rates (confirming the signal exists in
residual space) but SAE-feature steering does not (confirming the
signal isn't localized to the selected SAE basis). That's a clean
result of the shape "the information is there but SAE features don't
capture it as a causal handle," consistent with Ma et al.'s
sparsity-favors-low-dim-correlates argument.

### E.7: CoT trace classification — Cox et al. four-cell framework

For each steering setting where steering flipped the answer (i.e. made
`is_correct_strong` go from True to False), sample up to 50 flipped
generations and classify the CoT using a GPT-5.4 judge (pinned
snapshot, shared with the paraphrase-preserving test) along Cox et
al.'s two axes:

- *Premise truthfulness:* does the CoT contain any false premises?
- *Logical entailment:* does the final hypothesis follow from the
  stated premises, assuming they are true?

This produces a four-way classification: sound reasoning (shouldn't
occur in flipped samples), non-entailment, confabulation, hallucination.
The headline causal-relevance claim becomes stronger if SAE-feature
steering produces confabulation and non-entailment at rates comparable
to what Cox et al. observed with difference-of-means probe steering
on their binary tasks — because then we'd be recovering their
behavioral signature using SAE features (the feature source Ma et
al. flag as potentially cue-driven), which is the direct test our
project is set up to run.

If instead steering mostly produces hallucination, the SAE features
are noisier than direct probe directions and the causal-relevance
argument weakens (this matches Cox et al.'s observation that
hallucination rates rise with steering strength as reasoning
degenerates).

---

## Phase F: Analysis and write-up

Equivalent to Stage 1 Phase 4. Produces:

- Draft Probe Results section for final report: probe AUC across feature
  sources, cross-task transfer, length-regression decomposition, name-
  scramble robustness.
- Draft Causal Validation section: feature steering response curves,
  orthogonal baseline comparison, per-feature interpretability notes,
  Cox et al. four-cell classification rates under steering (non-
  entailment, confabulation, hallucination) per feature.
- Figures: probe AUC vs depth (per feature source), transfer scatter,
  steering response curves with orthogonal baseline shaded, stacked-bar
  of four-cell classification rates under steering (analogous to Cox
  et al. Figure 4).
- Updated Implications section building on Stage 1 Phase 4's list.

---

## Interpretation decision tree

A reviewer pointed out that pre-specifying interpretation protects
against post-hoc rationalization and makes the final write-up much
easier. Columns are the primary result patterns we might see; rows
are the interpretive conclusions they warrant. Pinned before running
Phase E so the interpretation isn't shaped by the result.

| Observed pattern | Interpretation |
| --- | --- |
| Raw residual probe beats B0 by ≥0.10 on both S1 and S2, cross-task transfer works, SAE probe matches raw, SAE steering > both baselines at pre-CoT-only position | Strongest "real feature" story. SAE features are causal handles on InAbHyD reasoning. This would be a genuinely novel positive result given Ma et al.'s null. |
| Raw residual probe beats B0 on S1 and S2, SAE probe matches, SAE steering ≈ random-orthogonal or same-SAE baselines | Predictive but non-causal — phantom SAE feature. Reproduces Ma et al.'s null result in the new task + contrastive-selection-structure setting. Still a scientifically valuable result; the paper's framing becomes "we extended Ma et al.'s falsification to a different failure mode and the null generalizes." |
| Raw residual probe beats B0 on S1 and S2, SAE probe is meaningfully worse than raw | Reasoning signal exists in residual space but is not captured by the selected SAE basis. Probably the most common outcome per Ma et al.'s theory (sparsity suppresses high-dimensional within-behavior variation). Frame as "SAE decomposition incomplete for this signal." |
| Raw residual probe beats B0 on S1 but not on S2 | Probe is likely exploiting ontology-skeleton artifacts. Not a reasoning feature, a dataset artifact. Report the S1/S2 delta; the headline becomes "InAbHyD probes look strong on random splits but don't generalize to new skeletons." |
| Probe fails name-scramble (pronounceable nonce) and natural-name substitution while model behavior holds | Surface-form probe; detects specific lexical patterns rather than reasoning. |
| Probe passes natural-name substitution but fails pronounceable-nonce scramble | Artifact from rare-token statistics in nonce condition; probe is not necessarily surface-bound. Report both and note. |
| Probe fails paraphrase-preserving test (probe flips when behavior stable) | False-negative failure mode à la Ma et al.; probe tracks surface form not content. |
| Raw probe-direction steering (E.1.1) produces meaningful flip rate, SAE steering doesn't | Clean result: signal is in residual space but not localized to SAE features. This is a different positive result that directly tests Ma et al.'s sparsity-favors-low-dim-correlates prediction. |
| Steering primarily increases parse-failure rate or output format degeneration instead of flipping correctness | Perturbation-amplitude effect, weak causal reasoning claim. Back off α or report as "intervention too aggressive at this magnitude." |
| 3-way auxiliary probe (B.1.bis) confuses "incorrect" with "parse_failed" | Main binary probe's signal is partly confidence/commitment, not purely correctness. Soften claim. |
| Bootstrap stability (D.3) shows top-20 Jaccard overlap < 0.3 across seeds | Top-feature list is unstable. Phase E should not steer features from this set; report D.1/D.2 representation/localization results and note that individual-feature steering wasn't possible because selection was unstable. |

Patterns can combine — e.g., a row "Raw probe works + SAE probe works
+ SAE steering null + name-scramble passes + paraphrase passes" reads
as "predictive, surface-robust, non-causal at SAE resolution," which
is an informative 3-axis result. Don't try to force every experiment
into a single binary phantom-vs-real verdict.

---

## Interface contract with downstream

**For MATS / final report submission:** the Stage 2 deliverable is the
union of the Probe Results and Causal Validation sections. The headline
claim the paper makes stands or falls on the Phase D → Phase E story: if
SAE features predict failure and steering those features causally flips
behavior more than orthogonal controls, we have a "real" feature. If
probes work but steering doesn't, we have a "phantom" — an SAE feature
correlated with the behavior without causing it.

**For Teammate B (or teammate split):** the handoff contract from Stage 2
probes to Stage 2 steering is the `results/stage2/probes/` directory plus
the top-features list from Phase D. Whoever runs steering needs only:
(a) the SAE feature indices to steer, (b) the probe-direction magnitudes
(for interpretation), (c) the held-out test-split example IDs (for the
eval set).

**For future work / Stage 3:** the name-scramble ablation dataset and the
per-example activation cache are the Stage 2 artifacts most likely to
enable follow-up experiments. Keep them around.

---

## Sanity checks before handoff

**Integrity checks:**
- [ ] Every activation file's row count matches the Stage 1 JSONL row
  count for its (model, task) cell.
- [ ] Every probe has a fitted model artifact, a recorded AUC, and a
  record of the train/val/test split indices.
- [ ] Cross-task transfer results reference valid probes (i.e. the
  source-cell probe actually exists and is loadable).
- [ ] Steering experiments reference valid features (indices exist in
  the relevant SAE width).

**Sanity checks:**
- [ ] Probe AUC is above 0.5 on at least h=2 and h=3 for every
  (model, task) in the diagonal (non-transfer) evaluations. Below 0.5
  means the probe is systematically mispredicting.
- [ ] **Inspect (don't fail on)** cells where logistic regression AUC <
  difference-of-means AUC. Logistic is more expressive in-sample, but
  regularization and generalization can make it worse on test. A
  reversed ordering is informative — usually it means the data are
  close to linearly separable along the diff-of-means direction and
  logistic's extra capacity is overfitting — but not a hard failure.
- [ ] Phase E effect sizes on the final-pass features exceed *all
  four* baseline family effect sizes (raw probe reference is a
  separate comparison, not a baseline). If any baseline family
  matches the feature effect, the feature isn't causally specific
  and the write-up reports the confound rather than a positive
  result.

---

## Implementation notes

### Why pre-CoT activations only

Cox et al. train probes on the last pre-CoT token because that's where the
model has "decided" what to do without having externalized it via
generation. Probing later tokens mixes the reasoning signal with
output-planning signal; probing earlier misses the point at which the
model's internal state best predicts the outcome.

### Why three layers, empirically selected

All-layers extraction is ~20× the storage and compute of three-layers.
One-layer commitment risks picking the wrong layer; it also foregoes any
comparison across layers, which is often informative (Cox et al. report
layerwise probe AUCs in their Appendix E and find the best layer varies
by model and task). Cox et al.'s actual workflow is to probe every
layer and then steer on the single best; with a Gemma-3-27B-sized model
that's storage-prohibitive for our 44k dataset. Three layers — the
highest-AUC from a small pilot sweep, plus one earlier layer — is a
pragmatic middle ground that preserves some depth comparison while
keeping Phase A tractable. This framing is our choice, not theirs.

### Where the Phase B probe signal could come from, besides reasoning

Four plausible sources, listed roughly in order of how "phantom-like" they
are:

1. **Prompt length.** Taller trees have longer prompts. Length correlates
   with height correlates with accuracy. Phase B.3 length-regression
   catches this.
2. **Entity-name distributional cues.** If certain concept names tend to
   appear in easier examples, the probe could learn name associations.
   Phase B.3 name-scramble catches this.
3. **Task-specific lexical patterns.** `infer_subtype`'s ground truths
   all match "Every X is Y" where Y is a concept, vs property's Y is a
   property. A probe that fails cross-task transfer has likely learned
   these lexical patterns. Phase B.2 transfer catches this.
4. **Genuine reasoning content.** The null hypothesis for the paper:
   probe detects "model successfully consulted ontology chain," which
   transfers across tasks, is robust to name scrambling, and survives
   length regression.

### Seed and reproducibility

Same story as Stage 1: deterministic per-probe seed, recorded in the
probe artifact. For the name-scramble ablation, record both the
name-permutation seed and the generator seed so regeneration is possible.

---

## Deliverables checklist

At the end of Phase 0:
- [ ] `docs/stage2_invariants.json` with model/tokenizer revision
  hashes, chat-template checksum, Stage 1 JSONL SHA-256s, pinned SAE
  release IDs, and pinned GPT judge snapshot ID.
- [ ] `docs/stage2_inventory.json` with per-cell row counts before
  and after `parse_failed` filtering.
- [ ] `results/stage2/splits.jsonl` assigning every row to S1 and S2
  train/val/test.
- [ ] `results/stage2/baselines/b0_*.json` with pre-output B0 baseline
  AUCs on S1 and S2.
- [ ] Label-shuffle probe sanity check passes (~0.50 AUC).

At the end of Phase A:
- [ ] Activation extraction infrastructure working on 4090 (for 4B)
  and Colab H100 (for 27B), with Modal HF-based fallback documented
  if stood up.
- [ ] `results/stage2/equivalence_report.json` showing all 5 tiers
  pass (or tiers 1–3 + tier 5 ≥98% label agreement if byte-match
  fails).
- [ ] Three-layer selection committed and documented in
  `docs/layer_selection.json`.
- [ ] `results/stage2/activations/{model}_{task}_L{layer}.safetensors`
  files for both models × both tasks × three layers (= 12 files).
- [ ] `.example_ids.jsonl` sidecar matching Stage 1 JSONL row order.

At the end of Phase B:
- [ ] Trained probes for every (model, task, layer, probe_type) = 24
  probes, on both S1 and S2 splits.
- [ ] `probe_auc.json` with AUC, Δ over strongest pre-output B0, and
  bootstrap 95% CIs per cell.
- [ ] `probe_transfer.json` with cross-task transfer results.
- [ ] `probe_comparison.json` with cross-model comparison (see B.2;
  not raw transfer).
- [ ] `probe_diagnostics.json` with length-regression, within-height
  AUC, both name-scramble conditions, and paraphrase-preserving
  agreement rates.
- [ ] B.1.bis 3-way auxiliary probe results for 27B × infer_property.
- [ ] Two figures: AUC-vs-depth, cross-task transfer scatter.
- [ ] Draft probe-results prose (1–2 pages).

At the end of Phase C:
- [ ] SAE feature caches for all (model, task, layer, width) = 24
  caches at Target tier.

At the end of Phase D:
- [ ] Full-width + sparse (L1) SAE probes trained per (model, task,
  layer, width) on both S1 and S2.
- [ ] D.3 stability analysis: bootstrap overlap, cross-split overlap,
  cross-width overlap. `stable_features.json` with the tiered
  stable-feature set.

At the end of Phase E:
- [ ] E.3.a screening: 20 features × 3 α × 200 examples with shared
  orthogonal (5) and shared same-SAE (5) baselines, no-op control,
  raw-probe-direction reference. Flip rates + CIs.
  Total ~19k generations per E.3.
- [ ] E.3.b confirmation (Target and above): 3–5 features × 5 α × 200
  examples with 10 per-feature orthogonal + 10 per-feature same-SAE
  baselines. Total ~57k generations per E.3.
- [ ] E.3.c final pass (Extended and above only): 1–2 features ×
  single peak-margin α × 2000 examples with 10 per-feature orthogonal
  + 10 per-feature same-SAE baselines. Total ~88k generations per
  E.3.
- [ ] Pre-CoT-only vs. all-tokens comparison on final-pass features
  (or confirmation-pass features if final isn't run).
- [ ] Cox et al. four-cell classification on flipped generations per
  steering setting with ≥20 flips.
- [ ] Draft causal-validation prose.

At the end of Phase F:
- [ ] Figures, prose, final report sections ready for integration.
- [ ] Interpretation decision tree filled in with observed result
  patterns per (model, task, layer) cell.
