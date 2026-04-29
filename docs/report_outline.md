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
- Cross-task transfer is positive but weaker than within-task probing.
- Residual SAE probes beat B0 in most settings but trail raw residuals.
- Top-k truncation is not the issue: top-128 already captures all active
  features in tested L45 residual SAEs.
- Reconstruction/error diagnostic is the pivot result: about 95% energy
  reconstructed, but error probes recover raw-level AUC.
- L45 MLP-output raw activations also carry raw-level signal, but the
  MLP-output width-16K SAE is weak. This supports the interpretation that the
  sparse dictionaries tested here miss the predictive direction rather than the
  signal existing only at one residual site.
- L45 skip-transcoder features are stronger than the MLP-output SAE but still
  trail raw `mlp_in`, so computation-oriented sparse features do not rescue the
  main SAE localization claim.
- Crosscoder pilot: raw concat over layers `{16,31,40,53}` nearly matches raw
  L45, but the 65K crosscoder over those same layers trails raw concat on every
  task/split. Treat this as an appendix-style multi-layer check supporting the
  main sparse-dictionary cautionary story.
- Cross-method comparison: crosscoders are middle-tier. They beat the weak
  MLP-output SAE and usually beat the skip-transcoder, but they do not improve
  on residual SAEs and they trail the fair raw-concat baseline substantially.
  On S3, crosscoders do not robustly beat metadata baselines.
- Dense active-feature probe check: centered dense probes over train-active
  sparse columns do not close the raw-vs-sparse gap, so sparse CSR scaling is
  not the main explanation.
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
  and a stronger steering protocol.

## Core Tables/Figures

- Table/Figure 1: strong accuracy vs height for 27B property/subtype.
- Table 2: B0 vs raw L45 probes on S1/S3.
- Table 3: raw vs residual SAE probes on S1/S3.
- Table 4: reconstruction vs error probes.
- Small table: MLP-output raw vs MLP-output SAE pilot.
- Small table: raw `mlp_in` vs skip-transcoder pilot.
- Appendix table: raw-concat vs crosscoder pilot.
- Appendix or compact main table: all feature sources ranked by S1/S3 AUC.
- Appendix table: dense-active sparse-feature scaling sanity check.
- Small appendix table: steering pilot summary and null result.

## One-Sentence Abstract Candidate

We find that Gemma 3 27B pre-generation residuals robustly predict ontology
reasoning success, but Gemma Scope residual SAEs only partially expose this
signal; reconstruction diagnostics show the missing predictive component is
concentrated in the small raw-minus-SAE error subspace.
