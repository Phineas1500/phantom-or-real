# Final Report Outline

This outline assumes the 27B result remains the main mechanistic story. Add
Gemma 3 4B as a comparison once teammate results are available.

## Title

Phantom or Real? Sparse Feature Probes for Ontology Reasoning Failures

## 1. Introduction

- Task: predict whether a Gemma model will solve a single-hypothesis InAbHyD
  ontology reasoning prompt before generation.
- Statistical framing: learn a predictor of correctness
  `y = is_correct_strong` from prompt-derived metadata and internal
  pre-generation activations.
- Main question: do sparse features expose a real reasoning-success mechanism,
  or do they only provide correlational artifacts?
- Final claim: raw residuals strongly predict success/failure, but tested
  residual SAE features do not cleanly localize the strongest signal.

## 2. Related Work

- Cox et al.: pre-generation activation probes and causal steering for
  reasoning/answer prediction.
- Ma et al.: apparent SAE reasoning features can be lexical or stylistic
  correlates; feature-level claims need falsification.
- Gemma Scope 2: pretrained sparse dictionaries for Gemma 3 activation sites.

Use our result as a bridge: robust raw activation signal exists, but residual
SAEs can still miss the signal that matters for prediction.

## 3. Dataset

- InAbHyD single-hypothesis prompts for `infer_property` and `infer_subtype`.
- Heights 1-4, with 1k/2k/3k/5k examples per task.
- Stage 1 labels from deterministic Gemma 3 inference and strong/weak
  hypothesis scoring.
- 27B behavioral headline: strong accuracy falls from 0.960 to 0.264 on
  property and from 0.973 to 0.055 on subtype.
- Caveat: shipped rows all have `has_direct_member=True`, so that structural
  shortcut slice is vacuous.

## 4. Methods

- Reconstruct prompts exactly with Gemma chat template and last pre-generation
  token position.
- Extract raw residuals at `blocks.{L}.hook_resid_post`.
- Train split-aware logistic probes with validation-selected regularization,
  train-only standardization, bootstrap CIs, and per-height diagnostics.
- Compare to metadata baselines: height, prompt/token metadata, and name
  frequency features.
- Probe Gemma Scope 2 residual SAE features at L45, widths 16K and 262K.
- Decode SAE reconstructions and probe both reconstruction and
  raw-minus-reconstruction error.
- Steering pilot: raw L45 direction vs orthogonal control on a small balanced
  property subset.

## 5. Results

- Behavioral depth effect: accuracy collapses monotonically with height.
- Raw residual probes beat metadata baselines on S1 and S3.
- Metadata residualization diagnostic: adding the raw L45 probe score to rich
  prompt/name metadata gives `+0.06` to `+0.18` AUC over metadata-only probes
  across S1/S3 and both tasks, supporting a genuine conditional activation
  signal.
- Cross-task transfer is positive but weaker than within-task probing.
- Residual SAE probes beat B0 in most settings but trail raw residuals.
- Top-k truncation is not the issue: top-128 already captures all active
  features in tested L45 residual SAEs.
- Reconstruction/error diagnostic is the pivot result: about 95% energy
  reconstructed, but error probes recover raw-level AUC.
- L45 MLP-output and 16K skip-transcoder pilots used bare normalized hooks; keep
  the old rows as exploratory motivation, not as final negative claims.
- Exact-hook L45 MLP-output SAE rerun: encoding `blocks.45.hook_mlp_out`
  improves the MLP-output SAE from old S1/S3 `0.577/0.674` and `0.550/0.702`
  to S1 `0.811/0.878` and S3 `0.807/0.879` for property/subtype. This is
  meaningful and residual-SAE-like, but still below raw exact `hook_mlp_out`.
- Exact-hook L45 262K transcoder probe: after correcting the input to
  `ln2.hook_normalized * ln2.w` and the target to `hook_mlp_out`, the
  Neuronpedia-visible all-layer transcoder is substantially stronger than the
  old bare-normalized run. It reaches S1 property/subtype AUCs `0.795/0.873`
  and S3 AUCs `0.802/0.885`, roughly residual-SAE-like but still below raw.
- Exact-hook L45 16K skip-transcoder rerun: the fair weighted-input rerun
  improves over the old bare-normalized 16K pilot to S1 `0.787/0.868` and S3
  `0.785/0.880`; exact 262K remains slightly better on most comparisons.
- Exact 262K component diagnostic: full latent+skip output has interpretable
  energy explained around `0.67/0.66` for property/subtype. Dense full/error
  components carry more signal than sparse latents alone but still trail raw
  exact activations, supporting partial rather than complete sparse
  localization.
- Sparse feature-family concat: combining residual SAE 16K, residual SAE 262K,
  and exact 262K transcoder features gives the best sparse-only L45 probe so
  far, S1 `0.822/0.884` and S3 `0.814/0.885`, but still remains below raw exact
  activations. This is positive evidence for complementary sparse views, not a
  full bridge to raw.
- Adding exact-hook MLP-output SAE 16K features to the sparse-family concat
  gives a small property gain; lowering the regularization grid improves the
  same four-block concat further to S1 `0.830/0.888` and S3 `0.828/0.888`.
  Adding exact-16K TC moves L45 only marginally to S1 `0.832/0.883` and S3
  `0.829/0.889`.
- L30 residual SAE/multi-layer sparse concat: L30 standalone features are weak
  for property, but adding L30 residual SAE 16K/262K features to the corrected
  L45 sparse family gives the current best sparse-only result, S1
  `0.839/0.887` and S3 `0.834/0.892`. This narrows but still does not bridge
  the raw gap.
- Crosscoder pilot: raw concat over layers `{16,31,40,53}` nearly matches raw
  L45, but the 65K crosscoder over those same layers trails raw concat on every
  task/split. Treat this as an appendix-style multi-layer check supporting the
  main sparse-dictionary cautionary story.
- L40 residual SAE follow-up: L40 is a stable Neuronpedia-visible raw layer, but
  its sparse residual features are weak standalone. Adding L40 to L30+L45 gives
  a tiny S3 gain, S3 `0.835/0.896`, but hurts S1 subtype, S1 `0.839/0.883`.
- Cross-method comparison: after exact-hook correction and sparse-family
  concat, the best sparse-only result moves up but still does not improve over
  raw activations. Crosscoders remain useful as a bounded multi-layer null:
  they trail the fair raw-concat baseline substantially and do not robustly beat
  metadata baselines on S3.
- Dense active-feature probe check: centered dense probes over train-active
  sparse columns, including corrected exact-hook artifacts and the four-block
  concat, do not close the raw-vs-sparse gap, so sparse CSR scaling is not the
  main explanation.
- bf16-vs-fp32 sparse encoding sanity check: re-encoding sampled rows in fp32
  leaves active sets nearly unchanged, so dtype instability is not the main
  explanation.
- Neuronpedia availability check: current public Gemma 3 27B-IT residual
  dashboards do not include L45, so Neuronpedia cannot directly interpret the
  exact top L45 residual SAE features in the main result. Public all-layer
  `gemmascope-2-transcoder-262k` dashboards do include L45, and the corrected
  exact same-source probe/audit did not reveal clean ontology-reasoning
  features.
- Steering pilot is null/inconclusive, not a causal success claim.

## 6. Discussion

- The raw correctness signal is robust and pre-generation.
- Residual SAE features are not enough for a clean localized mechanism in this
  setting.
- The reconstruction-error result is a cautionary finding for SAE-based
  mechanistic claims: high reconstruction energy does not imply retention of a
  behaviorally relevant direction.
- S3 heldout target symbols reduce but do not eliminate lexical-confound risk.
- Future work: alternative sparse dictionaries, name-scrambled regeneration,
  optional Neuronpedia-facing layer-40/53 residual audit, and a stronger
  steering protocol.

## Core Tables/Figures

- Figure 1: strong accuracy vs height for 27B property/subtype:
  `docs/figures/stage2/stage2_behavior_accuracy_27b.png`.
- Figure 2: B0, residual SAE, best sparse concat, and raw L45 probes on S1/S3:
  `docs/figures/stage2/stage2_probe_overview_auc.png`.
- Figure 3: reconstruction vs error probes:
  `docs/figures/stage2/stage2_reconstruction_error_auc.png`.
- Figure 4: sparse feature-family progression:
  `docs/figures/stage2/stage2_sparse_progression_auc.png`.
- Figure 5: exact-hook MLP/transcoder sparse artifacts vs same-site raw
  activations: `docs/figures/stage2/stage2_site_transcoder_auc.png`.
- Small table: exact MLP-output raw vs exact MLP-output SAE, with old pilot
  shown as superseded.
- Small table: raw exact `mlp_in` vs exact 262K transcoder.
- Appendix table: exact 262K transcoder latent/skip/full/error component probe.
- Small table: sparse feature-family concat.
- Appendix table: raw-concat vs crosscoder pilot.
- Appendix table: metadata residualization / metadata-plus-raw-score
  diagnostic.
- Appendix or compact main table: all feature sources ranked by S1/S3 AUC.
- Appendix table: dense-active sparse-feature scaling sanity check.
- Appendix table: bf16-vs-fp32 sparse encoding stability check.
- Appendix table/note: corrected L45 262K transcoder Neuronpedia feature audit.
- Small appendix table: steering pilot summary and null result.

## One-Sentence Abstract Candidate

We find that Gemma 3 27B pre-generation residuals robustly predict ontology
reasoning success, but Gemma Scope residual SAEs only partially expose this
signal; reconstruction diagnostics show the missing predictive component is
concentrated in the small raw-minus-SAE error subspace.
