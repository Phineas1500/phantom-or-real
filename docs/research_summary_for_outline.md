# Research Summary For Report Outline

Date: 2026-05-01

Purpose: one drafting-oriented place for the project story, experiments, and
findings. `docs/REPORT_NOTES.md` is the full chronological lab notebook;
`docs/stage2_results_pack.md` is the compact quantitative Stage 2 pack;
`docs/stage2_steering_decision_table.md` is the causal-test tracker. This file
is the synthesis layer to use when drafting the final outline.

## One-Sentence Story

Gemma 3 pre-generation activations robustly predict success and failure on
structured InAbHyD ontology reasoning, but the tested sparse dictionary
features and probe-derived directions do not behave like localized causal
reasoning mechanisms: sparse features expose only part of the signal,
raw-minus-SAE reconstruction error preserves much of the predictive signal, and
steering/patching tests show a predictive-versus-causal gap.

## Main Claims

1. InAbHyD gives a real behavioral success/failure axis: accuracy collapses
   sharply with ontology height, especially for `infer_subtype`.
2. Raw pre-generation residual activations predict correctness well above
   metadata baselines, label-shuffle controls, and heldout-target splits.
3. Gemma Scope sparse dictionaries partially expose this correctness signal,
   and exact hook alignment plus bigger/more active dictionaries help, but raw
   activations still lead.
4. The reconstruction-error diagnostic is the central sparse-dictionary
   caution: residual SAEs reconstruct about 95% of residual energy, but the
   raw-minus-reconstruction error recovers almost the full raw-probe AUC.
5. Feature-level audits did not find clean monosemantic ontology-reasoning
   features. The best candidates look like mixtures of common-superclass,
   direct-generalization, fan-in, height, and template signals.
6. Direction steering failed across raw correctness, reconstruction-error,
   sparse-bundle, single-feature, and answer-property directions, despite
   strong or perfect probe AUC.
7. Forced-choice testing showed that recognition is often intact: on 27B rows
   where free-form output was wrong, the model still chose the gold hypothesis
   over its own emitted wrong foil in 14/16 MCQ cases.
8. Full-state patching produced an asymmetry: h1-correct states did not repair
   h4-incorrect prompts above matched noise, while h4-incorrect states reduced
   h1-correct margins more than matched noise in aggregate. This supports a
   conservative "asymmetric deployment/commitment" interpretation rather than a
   clean missing-state repair mechanism.

## Claim Strength Guardrails

Use this table when drafting the abstract, discussion, and limitations. It
separates measured results from plausible interpretations and from claims the
current evidence does not support.

| Strength | Claim | Evidence | Drafting guidance |
| --- | --- | --- | --- |
| Strong | InAbHyD depth creates a robust behavioral success/failure gradient. | 27B strong accuracy falls from h1 to h4 on both `infer_property` and `infer_subtype`; 4B shows the same broad degradation. | Safe main-text claim. |
| Strong | Raw pre-generation activations predict correctness beyond simple metadata. | 27B L45 raw probes beat B0 on S1/S3; label shuffle near chance; metadata-plus-raw diagnostics add conditional signal. | Safe main-text claim. |
| Strong | Tested sparse dictionaries expose only part of the raw correctness signal. | Residual SAEs, corrected exact-hook transcoders, big-L0 transcoders, sparse concats, and crosscoders generally trail raw activations. | Safe main-text claim; note big-L0 and sparse concat improvements. |
| Strong | Residual SAE reconstruction error preserves much of the predictive signal. | Residual SAEs reconstruct about 95% energy, while raw-minus-reconstruction error probes recover near-raw AUC. | Central report claim. |
| Strong | Probe-derived steering directions did not reliably repair free-form answers. | Raw correctness, reconstruction-error, sparse-bundle, single-feature, 4B answer-property, and 27B answer-property steering all failed to produce controlled beneficial repairs. | Safe causal-null claim if scoped to tested settings. |
| Strong | Forced-choice recognition can be intact when free-form generation is wrong. | In the 27B hard-foil setup, baseline MCQ selected gold in 14/16 rows that were free-form wrong. | Safe main-text claim; avoid claiming all errors are recognition-intact. |
| Strong | Forward h1-to-h4 patching did not reveal a clean transplantable repair state. | Clean h1 patches did not consistently outperform matched noise at late `last_prompt` sites. | Safe claim scoped to tested layers, landmarks, and strict natural pairs. |
| Moderate | Reverse h4-to-h1 patching shows an asymmetric disruption effect. | Aggregate corrupt-state breakage exceeded noise at L35-L50, but the effect is driven by lower/mid-headroom pairs and high-headroom pairs are mixed. | Good result, but present as asymmetry before mechanism. |
| Moderate | The failure mode is more like free-form answer deployment/commitment than missing ontology comprehension. | MCQ recognition is high on free-form-wrong rows; h1 states do not repair h4; h4 states can disrupt h1. | Use as interpretation, not as a proved mechanism. |
| Moderate | Sparse features look confounded by height/template/common-superclass structure rather than clean reasoning concepts. | Neuronpedia-facing audit and local mini-dashboard did not find clean ontology-reasoning features. | Good discussion claim; avoid saying no such features exist. |
| Weak/speculative | The wrong-answer state is localized specifically at late `last_prompt` residual sites. | Patching effects are largest or most interpretable there, but not cleanly localized and noise is a serious control. | Do not state as a main claim. |
| Weak/speculative | Reconstruction-error subspace is causal. | Error probes are predictive, and 4B error steering was null. | Do not claim causal error-subspace mechanism. |
| Unsupported | We discovered a monosemantic sparse reasoning feature. | Feature audits and steering do not support this. | Avoid. |
| Unsupported | SAEs cannot represent reasoning-relevant information. | Our results are scoped to tested dictionaries, features, probes, and tasks. | Avoid; say tested dictionaries did not yield a clean localized mechanism. |
| Unsupported | Cox-style answer steering is contradicted. | Cox studies constrained binary answer tasks; our task is structured free-form generation. | Frame as extension to a different regime, not contradiction. |
| Unsupported | Patching proves the exact causal circuit. | Patching was landmark-level residual replacement on 8 strict natural pairs. | Avoid; call it a causal diagnostic/asymmetry. |

## Existing Docs Audit

No single pre-existing doc fully served the outline-drafting role.

| Doc | What it is good for | Gap |
| --- | --- | --- |
| `docs/REPORT_NOTES.md` | Most complete chronological record; includes final forced-choice and patching notes. | Too long and notebook-like for drafting. |
| `docs/stage2_results_pack.md` | Best compact quantitative pack for probes, sparse dictionaries, reconstruction, and 4B comparison. | Report-pack oriented, not a full project narrative; causal results are summarized but not framed as the final story. |
| `docs/report_outline.md` | Skeleton for the final report. | Some framing predates the final forced-choice and asymmetric patching results. |
| `docs/behavioral_results_draft.md` | Good Stage 1 behavioral narrative. | Does not cover Stage 2 mechanism/causal results. |
| `docs/stage2_steering_decision_table.md` | Most current steering and patching decision tracker. | Focused only on causal experiments. |
| `docs/stage2_clean_to_corrupt_patching_plan.md` | Detailed forward/reverse patching design and results. | Too narrow for whole-project outline drafting. |
| `docs/stage2_27b_margin_forced_choice_results.md` | Detailed forced-choice margin result. | Too narrow for whole-project outline drafting. |

Use this file as the top-level drafting source, then pull exact tables from the
specific docs above.

## Stage 1: Behavioral Setup

Task family:

- InAbHyD single-hypothesis ontology reasoning.
- Two tasks: `infer_property` and `infer_subtype`.
- Heights 1-4 with deeper prompts requiring longer inductive/abductive chains.
- Labels come from deterministic model generations and strong/weak hypothesis
  scoring.

27B behavioral headline:

| Task | h1 strong | h2 strong | h3 strong | h4 strong | h4 weak | h4 parse fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `infer_property` | 0.960 | 0.577 | 0.392 | 0.264 | 0.424 | 0.104 |
| `infer_subtype` | 0.973 | 0.325 | 0.114 | 0.055 | 0.240 | 0.014 |

4B behavioral caveat:

- 4B also degrades with height, but its deep `infer_subtype` positives are very
  sparse, so 4B subtype probe numbers should be treated cautiously.
- 4B often uses different output strategies at depth, making it more of a
  comparison model than the core mechanistic story.

Important dataset caveat:

- Shipped rows all have `has_direct_member=True`, so any structural shortcut
  slice based on that field is vacuous.

## Splits And Controls

Main splits:

- S1: random train/validation/test split.
- S3: heldout target-symbol split, intended to reduce target-name leakage.
- S2 was not treated as a main evaluable split.

Metadata baselines:

| Split | Task | Best B0 feature set | Test AUC |
| --- | --- | --- | ---: |
| S1 random | `infer_property` | `b0_prompt` | 0.743 |
| S1 random | `infer_subtype` | `b0_height` | 0.841 |
| S3 target-symbol heldout | `infer_property` | `b0_namefreq` | 0.711 |
| S3 target-symbol heldout | `infer_subtype` | `b0_prompt` | 0.859 |

Control conclusions:

- Label shuffle stayed near chance for 27B S1 raw L45: property 0.493,
  subtype 0.481.
- Adding the raw L45 probe score to rich prompt/name metadata gives conditional
  gains of roughly 0.06 to 0.18 AUC across tasks/splits.
- These controls defend the raw activation result as more than prompt metadata
  or label imbalance.

## Raw Activation Probes

27B main site: `blocks.45.hook_resid_post` at the final pre-generation token.

| Split | Task | Raw L45 test AUC | Bootstrap CI | Delta vs B0 |
| --- | --- | ---: | --- | ---: |
| S1 random | `infer_property` | 0.897 | [0.881, 0.912] | +0.153 |
| S1 random | `infer_subtype` | 0.914 | [0.896, 0.932] | +0.073 |
| S3 target-symbol heldout | `infer_property` | 0.884 | [0.868, 0.901] | +0.173 |
| S3 target-symbol heldout | `infer_subtype` | 0.917 | [0.898, 0.934] | +0.058 |

Other raw findings:

- Raw L45 stays strong under metadata residualization diagnostics.
- Cross-task transfer is positive but not task invariant:
  property -> subtype is 0.862 on S1 and 0.846 on S3; subtype -> property is
  0.786 on S1 and 0.788 on S3.
- Raw probes at nearby/multiple layers also carry signal. L40 is the best
  Neuronpedia-visible compromise layer, but L45 remains the main raw reference.
- Raw exact MLP-output activations carry essentially the same signal as raw
  residuals when exact hooks are used.

4B raw comparison:

| Method | S1 property | S1 subtype | S3 property | S3 subtype |
| --- | ---: | ---: | ---: | ---: |
| Raw L22 | 0.903 | 0.974 | 0.906 | 0.972 |

Interpretation: raw pre-generation activations contain a robust success/failure
readout. The key question is whether sparse dictionaries expose it in an
interpretable or causal way.

## Sparse Dictionary Results

### 27B Residual SAEs

L45 residual Gemma Scope 2 SAEs with top-k sparse features:

| Split | Task | Width 16K AUC | Width 262K AUC | Raw L45 AUC |
| --- | --- | ---: | ---: | ---: |
| S1 random | `infer_property` | 0.786 | 0.806 | 0.897 |
| S1 random | `infer_subtype` | 0.876 | 0.870 | 0.914 |
| S3 target-symbol heldout | `infer_property` | 0.799 | 0.779 | 0.884 |
| S3 target-symbol heldout | `infer_subtype` | 0.865 | 0.867 | 0.917 |

Conclusion: residual SAE features retain some correctness signal but trail raw
activations, especially on property.

### Exact Hook Fixes

Several early sparse results used bare normalized hooks that were later found
to mismatch Gemma Scope exact hook conventions. Exact reruns matter:

- Exact MLP-output SAE 16K improved from the old weak pilot to S1
  `0.811/0.878` and S3 `0.807/0.879` for property/subtype.
- Exact 16K skip-transcoder improved to S1 `0.787/0.868` and S3
  `0.785/0.880`.
- Exact 262K affine transcoder reached S1 `0.795/0.873` and S3 `0.802/0.885`.

Conclusion: hook/scale alignment materially improves learned-dictionary
results, but corrected dictionaries still do not close the raw gap.

### Big-L0 Transcoder

The L45 262K big-L0 exact affine transcoder was the strongest single learned
dictionary. The direct top-512 sparse-feature probe reached:

| Split | Task | Direct sparse-feature AUC |
| --- | --- | ---: |
| S1 random | `infer_property` | 0.853 |
| S1 random | `infer_subtype` | 0.893 |
| S3 target-symbol heldout | `infer_property` | 0.854 |
| S3 target-symbol heldout | `infer_subtype` | 0.894 |

The corresponding component diagnostic, which split the artifact into sparse
latents, affine skip, full output, and target-minus-full error, showed:

| Split | Task | Big-L0 latent AUC | Big-L0 full AUC |
| --- | --- | ---: | ---: |
| S1 random | `infer_property` | 0.837 | 0.863 |
| S1 random | `infer_subtype` | 0.854 | 0.888 |
| S3 target-symbol heldout | `infer_property` | 0.832 | 0.850 |
| S3 target-symbol heldout | `infer_subtype` | 0.869 | 0.877 |

The artifact is denser than small-L0 dictionaries, with mean active feature
counts around 120/113 for property/subtype, so top-128 was not enough for this
run.

Conclusion: denser dictionaries help and produce the best sparse property
result, but the result is still below raw activations.

### Sparse Feature-Family Concat

Combining sparse families helps:

| Split | Task | L45 five-block | L30+L45 all sparse | L30+L40+L45 all sparse | Raw exact `mlp_out` |
| --- | --- | ---: | ---: | ---: | ---: |
| S1 random | `infer_property` | 0.832 | 0.839 | 0.839 | 0.896 |
| S1 random | `infer_subtype` | 0.883 | 0.887 | 0.883 | 0.916 |
| S3 target-symbol heldout | `infer_property` | 0.829 | 0.834 | 0.835 | 0.892 |
| S3 target-symbol heldout | `infer_subtype` | 0.889 | 0.892 | 0.896 | 0.915 |

Conclusion: sparse views are complementary, especially for property. L30 helps
some; L40 gives a small S3 subtype boost; L53 is weaker. No sparse stack
matches raw activations.

### Crosscoder And Dense-Active Checks

- Raw concat over layers `{16,31,40,53}` nearly matches raw L45, but the 65K
  crosscoder over those same layers trails raw concat on every task/split.
- Dense active-feature probes with centered dense matrices do not close the
  raw-vs-sparse gap.
- bf16 vs fp32 sparse encoding is stable, with active Jaccard around 0.99 or
  better for the main artifacts.

Conclusion: the sparse/raw gap is not explained by sparse matrix scaling,
dtype instability, or lack of multi-layer raw signal.

### 4B Sparse Comparison

| Method | S1 property | S1 subtype | S3 property | S3 subtype |
| --- | ---: | ---: | ---: | ---: |
| Raw L22 | 0.903 | 0.974 | 0.906 | 0.972 |
| Resid SAE L22 16K | 0.808 | 0.956 | 0.820 | 0.965 |
| Resid SAE L22 262K | 0.808 | 0.964 | 0.819 | 0.971 |
| MLP-out SAE L22 16K exact | 0.812 | 0.961 | 0.824 | 0.969 |
| Transcoder L22 262K big-affine top512 exact | 0.855 | 0.969 | 0.874 | 0.977 |
| L20+L22 all-sparse concat | 0.842 | 0.969 | 0.850 | 0.976 |

Conclusion: 4B mirrors the 27B property pattern. Subtype is harder to interpret
because deep positive counts are small.

## Reconstruction-Error Pivot

Residual SAE reconstruction/error diagnostic:

| Split | Task | SAE width | Energy explained | Reconstruction AUC | Error AUC | Raw L45 AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| S1 random | `infer_property` | 16K | 0.948 | 0.786 | 0.894 | 0.897 |
| S1 random | `infer_property` | 262K | 0.955 | 0.806 | 0.897 | 0.897 |
| S1 random | `infer_subtype` | 16K | 0.948 | 0.877 | 0.916 | 0.914 |
| S1 random | `infer_subtype` | 262K | 0.954 | 0.870 | 0.915 | 0.914 |
| S3 target-symbol heldout | `infer_property` | 16K | 0.948 | 0.799 | 0.881 | 0.884 |
| S3 target-symbol heldout | `infer_property` | 262K | 0.955 | 0.788 | 0.886 | 0.884 |
| S3 target-symbol heldout | `infer_subtype` | 16K | 0.948 | 0.865 | 0.916 | 0.917 |
| S3 target-symbol heldout | `infer_subtype` | 262K | 0.954 | 0.867 | 0.914 | 0.917 |

Main interpretation:

- The residual SAEs reconstruct about 95% of residual energy.
- Probes on the reconstruction track sparse SAE-feature performance.
- Probes on the small raw-minus-reconstruction error recover almost the entire
  raw-probe signal.
- Therefore, high reconstruction energy is not enough to preserve the
  behaviorally relevant correctness direction.

This is the project-specific contribution that connects most directly to
Ma et al.: sparse decompositions can miss behaviorally important directions
even when the reconstruction looks strong.

## Feature Interpretation And Falsification

Neuronpedia availability:

- Public Gemma 3 27B residual dashboards do not include L45, so the top L45
  residual SAE features could not be audited directly through Neuronpedia.
- Public all-layer `gemmascope-2-transcoder-262k` dashboards include L45, so
  we audited the corrected exact transcoder there.
- Neuronpedia-facing top features looked generic or lexical: examples included
  "for all", "assuming", "technical contexts", "entity classification", code
  fragments, pronouns/scope, and abstract concepts.

Local big-L0 mini-dashboard:

| Feature | Probe assoc. | Property AUC | Subtype AUC | Short interpretation |
| ---: | --- | ---: | ---: | --- |
| 72374 | correct | 0.641 | 0.759 | Direct/simple universal generalization; height-1 confounded. |
| 35036 | incorrect | 0.400 | 0.343 | Error-associated common-supertype/fan-in and wrong-direction cases. |
| 4892 | correct | 0.524 | 0.488 | Common-superclass candidate, weak alone. |
| 75345 | correct | 0.548 | 0.516 | Moderate-depth common-superclass pattern, mixed correctness evidence. |
| 187589 | correct | 0.574 | 0.754 | Subtype-positive simple universal generalization, low-height confounded. |
| 45599 | incorrect | 0.488 | 0.497 | Sparse fan-in/exhaustive-hypothesis-risk feature, weak alone. |

Conclusion: the audit produced plausible feature labels and a useful shortlist,
but not a clean feature-level mechanism. This supports a falsification-oriented
framing: sparse features are hypotheses, not explanations by default.

## Causal Tests

### Steering Overview

Across tested settings, predictive directions did not become reliable control
knobs.

| Experiment | Model/site | Target | Probe result | Steering result | Interpretation |
| --- | --- | --- | --- | --- | --- |
| Raw correctness | 27B L45 residual | `is_correct_strong` | Test AUC 0.8965 | 0 false-to-true flips on 8 rows | Predictive correctness direction is not a repair knob. |
| Single big-L0 features | 27B L45 big-affine TC | Shortlisted sparse features | Candidate features from sparse probe/dashboard | 0 false-to-true; 2 true-to-false | Individual features not causal repair handles. |
| Sparse-probe bundle | 27B L45 big-affine TC | Sparse correctness probe coefficients | Test AUC 0.853 | 0 false-to-true on 8 rows | Distributed sparse direction did not repair. |
| Raw/error/sparse bundle | 4B L22 | `is_correct_strong` | Raw 0.9036, error 0.9005, sparse 0.8602 | 0 strong flips | Same predictive-versus-causal gap at 4B scale. |
| Raw answer-property | 4B L22 residual | Gold answer polarity | val/test AUC 1.000 | 0 polarity flips, 0 predicate flips toward gold | Concrete answer content still did not steer. |
| Raw answer-property | 27B L45 residual | Gold answer polarity | val/test AUC 1.000 | 0 useful toward-gold flips | 27B matches 4B answer-property null. |

Important distinction from Cox et al.:

- Cox-style work steers binary answer/property axes in constrained tasks.
- Our primary target is free-form structured generation over many possible
  ontology hypotheses.
- Even when we made the target more Cox-like with answer polarity and
  forced-choice margins, the direction did not show reliable causal control.

### Forced-Choice Hard-Foil Result

The hard-foil follow-up selected 16 27B S1 h3/h4 property rows that were
free-form wrong and compared the gold hypothesis against the model's own
emitted wrong hypothesis.

Key results:

- Raw L45 gold-polarity probe: `val_auc=test_auc=1.000`.
- Baseline MCQ choice accuracy: 14/16, even though all selected rows were
  free-form wrong.
- No MCQ choice flips and no false-to-true MCQ flips through 2 projection SD.
- On the two rows with MCQ headroom, `toward_gold` did not beat controls.
- Original-prompt margin shifts were not target-specific; at 2 SD,
  `away_gold` increased the original gold-vs-foil margin more than
  `toward_gold`.

Conclusion: this closes the probe-direction forced-choice branch for the
current setup. The result is also substantively useful: the model often
recognizes the correct answer under binary presentation while failing to deploy
it in free-form generation.

### Full-State Patching

Forward clean-to-corrupt patching:

- 8 strict natural pairs: h1 correct and h4 incorrect, sharing the full gold
  hypothesis.
- Patched semantic landmarks rather than raw token index:
  `last_prompt`, `subject`, `predicate`, and `question_stem`.
- Layers: 30, 35, 40, 45, 50.
- Metric: recovery of `gold - model-emitted-wrong-foil` logprob margin.

Late `last_prompt` summary:

| Layer | h1->h4 clean recovery | h1->h4 noise recovery |
| --- | ---: | ---: |
| L35 | 0.108 | 0.036 |
| L40 | 0.079 | -0.001 |
| L45 | 0.046 | 0.085 |
| L50 | 0.071 | 0.115 |

Forward conclusion: h1-correct residual states do not repair h4-incorrect
prompts above matched noise in a clean, stable way. This argues against a
simple missing-state explanation at the tested residual sites.

Reverse corrupt-to-clean patching:

- Same 8 pairs.
- Patched h4 incorrect `last_prompt` states into h1 correct prompts.
- Layers: 35, 40, 45, 50.
- Metric: breakage of h1 `gold - foil` margin; positive breakage means the
  margin dropped.

Side-by-side asymmetry:

| Layer | h1->h4 clean recovery | h1->h4 noise recovery | h4->h1 corrupt breakage | h4->h1 noise breakage |
| --- | ---: | ---: | ---: | ---: |
| L35 | 0.108 | 0.036 | 0.100 | -0.015 |
| L40 | 0.079 | -0.001 | 0.136 | 0.018 |
| L45 | 0.046 | 0.085 | 0.120 | -0.065 |
| L50 | 0.071 | 0.115 | 0.177 | 0.023 |

Reverse caveat:

- The aggregate reverse effect is driven by lower/mid-headroom pairs.
- High-headroom pairs are mixed even in absolute margin deltas.
- The safest report claim is the asymmetry itself, not a strong localized
  mechanism claim.

Patching interpretation:

- Forward patching: no transplantable repair state at the tested sites.
- Reverse patching: wrong h4 states can disrupt h1 margins more than noise in
  aggregate.
- Together with forced-choice recognition, this is consistent with a picture
  where free-form generation enters a wrong-answer deployment/commitment state,
  but the project should present that as interpretation rather than directly
  proven mechanism.

## Relationship To Cox Et Al. And Ma Et Al.

Cox-style connection:

- Similarity: both projects train pre-generation/pre-CoT probes and test
  whether the probed directions are causal.
- Difference: Cox tests binary answer choices, where a one-dimensional answer
  direction can plausibly shift probability mass between two outputs. InAbHyD
  correctness is a free-form structured-generation property over many possible
  hypotheses.
- Our result extends the picture: high probe AUC does not guarantee causal
  control when the target is an emergent meta-property or a structured
  free-form answer.

Ma-style connection:

- Similarity: sparse features that correlate with reasoning/success can be
  confounded, incomplete, or downstream readouts.
- Difference: Ma focuses on falsifying contrastively selected SAE reasoning
  features through token injection and LLM-guided counterexamples; our central
  falsification is predictive-vs-causal and reconstruction-error based.
- Our result adds a complementary caution: even when sparse reconstruction has
  high energy explained, the behaviorally predictive direction can remain in
  the small reconstruction-error component.

Useful taxonomy for discussion:

- Primitive content/answer directions: often predictive and sometimes causal
  in constrained binary settings.
- Lexical or cue-like sparse features: predictive but confounded.
- Emergent correctness/free-form-generation directions: predictive but not
  necessarily causal steering axes.

## Recommended Report Structure

1. Introduction
   - Present the question: are sparse dictionary features "phantom or real"
     mechanisms for ontology reasoning success?
   - State the answer: raw activations predict strongly, sparse features expose
     partial signal, causal tests do not support a localized steering feature.
2. Dataset and behavioral setup
   - InAbHyD tasks, heights, strong/weak scoring, accuracy collapse.
3. Predictive signal in raw activations
   - Raw L45/L22 probes, metadata baselines, heldout-target split, label
     shuffle, cross-task transfer.
4. Sparse dictionaries expose only part of the signal
   - Residual SAEs, exact hook fixes, big-L0 transcoder, sparse concat,
     crosscoder/dense/dtype controls.
5. Reconstruction error is the pivot
   - Show high energy explained vs raw-level error-probe AUC.
   - Use this as the central mechanistic caution.
6. Feature interpretation
   - Neuronpedia availability limits and local mini-dashboard.
   - No clean monosemantic reasoning feature found.
7. Causal tests
   - Steering nulls in compact table.
   - Forced-choice recognition result.
   - Forward/reverse patching asymmetry.
8. Discussion
   - Predictive readouts are not necessarily causal handles.
   - Recognition vs free-form deployment distinction.
   - Relationship to Cox and Ma.
   - Limitations and future work.

## What To Include In Main Text

High-value main-text items:

- Behavioral accuracy vs height.
- Raw vs metadata baseline table.
- Raw vs residual SAE vs best sparse dictionary/concat table.
- Reconstruction/error table or figure.
- Short feature-audit summary.
- Steering/forced-choice/patching table.
- Conservative final interpretation.

Good appendix items:

- Full cross-method sparse feature comparison.
- Exact-hook audit details.
- Dense-active and dtype sanity checks.
- Neuronpedia/top-feature audit examples.
- Full steering condition tables.
- Patching per-pair/headroom breakdowns.

Probably omit from main text:

- Superseded bare-normalized MLP/transcoder rows except as a short methods
  caution.
- Most Slurm/job logistics.
- Every individual sparse concat ablation.
- More steering sweeps unless they directly support the final causal table.

## Key Artifact Index

Narrative and summary docs:

- `docs/REPORT_NOTES.md`
- `docs/stage2_results_pack.md`
- `docs/report_outline.md`
- `docs/behavioral_results_draft.md`
- `docs/stage2_steering_decision_table.md`

Important causal docs:

- `docs/stage2_4b_steering_results.md`
- `docs/stage2_4b_answer_property_steering_results.md`
- `docs/stage2_27b_answer_property_steering_results.md`
- `docs/stage2_27b_margin_forced_choice_results.md`
- `docs/stage2_clean_to_corrupt_patching_plan.md`

Important result reports:

- `docs/stage2_results_pack.md`
- `docs/sae_reconstruction_probe_27b_l45_s1.json`
- `docs/sae_reconstruction_probe_27b_l45_s3_target_symbol.json`
- `docs/feature_mini_dashboard_27b_l45_262k_big_affine_top512.md`
- `docs/answer_property_margins_27b_l45_polarity_hardfoil.json`
- `docs/clean_to_corrupt_patching_27b_property_margin_pilot.json`
- `docs/corrupt_to_clean_patching_27b_property_margin_pilot.json`

Figures already available:

- `docs/figures/stage2/stage2_behavior_accuracy_27b.png`
- `docs/figures/stage2/stage2_probe_overview_auc.png`
- `docs/figures/stage2/stage2_reconstruction_error_auc.png`
- `docs/figures/stage2/stage2_sparse_progression_auc.png`
- `docs/figures/stage2/stage2_site_transcoder_auc.png`
- `docs/figures/stage2/stage2_cross_model_property_auc.png`
- `docs/figures/stage2/stage2_steering_predictive_vs_causal.png`
- `docs/figures/stage2/stage2_forced_choice_hardfoil.png`
- `docs/figures/stage2/stage2_patching_asymmetry.png`

Figure planning:

- `docs/report_figure_plan.md`

## Final Drafting Caution

Do not overclaim a discovered causal reasoning feature. The strongest
defensible claim is that the project found a robust predictive signal and then
systematically falsified several natural causal interpretations of that signal.
The positive mechanistic structure is the reconstruction-error localization and
the forward/reverse patching asymmetry; the negative result is that neither
single sparse features, sparse bundles, nor raw probe directions behaved like
reliable steering handles for free-form ontology reasoning.
