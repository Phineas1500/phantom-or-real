# Stage 2 Forced-Choice And Margin Steering Plan

Date: 2026-04-30

## Purpose

The previous free-form answer-property steering checks found that answer
content is perfectly decodable from raw activations but does not reliably steer
emitted ontology hypotheses. The next diagnostic should test whether this is an
output-space problem rather than a total absence of causal influence.

Run two measurements under the same raw answer-polarity direction:

1. Original-prompt candidate margin: score the original InAbHyD prompt without
   changing the visible task, comparing full-sequence log probabilities for the
   gold hypothesis versus the opposite-polarity foil.
2. MCQ forced-choice margin: reformat the same gold/foil pair as `(A)` versus
   `(B)`, randomize option order, score the gold option margin, and also decode
   the model's explicit choice.

## Decision Logic

| Outcome | Interpretation | Next step |
| --- | --- | --- |
| Original margin and MCQ margin both move toward gold above orthogonal controls | The raw direction has causal answer sensitivity; free-form greedy decoding was too strict a metric. | Scale to 32 rows and consider sparse/reconstruction-error comparisons. |
| MCQ margin/choice moves, original margin does not | Steering works only in the constrained answer pathway. | Report output-channel dependence; consider forced-choice-only sparse comparisons. |
| Original margin moves, MCQ does not | The direction affects free-form hypothesis likelihood but not explicit A/B commitment. | Inspect prompt formatting and run a small strength sweep. |
| Neither margin moves | Cox-style answer steering does not transfer under this direction/task setup. | Move to activation patching or reconstruction-error steering. |

## First Run

Use `scripts/stage2_steer_answer_property_margins.py` through
`scripts/stage2_steer_answer_property_margins_27b_L45_property_smoke.sbatch`.

Settings:

- Model: `google/gemma-3-27b-it`
- Task: `infer_property`
- Site: `blocks.45.hook_resid_post`
- Direction: raw L45 gold-polarity logistic direction
- Rows: 8 balanced S1 h3/h4 test rows
- Conditions: baseline, `toward_gold`, `away_gold`, orthogonal
- Strengths: `0.5` and `1.0` train projection SD
- Metrics: original full-candidate logprob margin, MCQ option logprob margin,
  MCQ generated choice accuracy

Use full candidate sequence log probability rather than first divergent token
because synthetic concept names tokenize unevenly.

Result: completed as Scholar job `452338`; see
`docs/stage2_27b_margin_forced_choice_results.md`. The MCQ margin showed a
small `toward_gold` shift, but MCQ choices were saturated at 8/8 correct at
baseline and original-prompt margins were noisy. Do not scale the exact
opposite-polarity-foil setup without replacing the foil with a harder
model-emitted wrong hypothesis or matched wrong concept.

## Refined Hard-Foil Run

Use `scripts/stage2_steer_answer_property_margins_27b_L45_property_hardfoil.sbatch`.

This is a closure test for the Cox-style diagnosis, not a broad positive-result
hunt. It restricts to rows where the Stage 1 free-form answer was parsed but
strongly incorrect, then compares the gold hypothesis against the model's own
emitted wrong hypothesis. This removes the easy opposite-polarity foil and gives
the forced-choice prompt real headroom.

Settings:

- Model/site/direction: same as the first run, `google/gemma-3-27b-it` at L45
  with the raw gold-polarity direction.
- Rows: 16 balanced S1 h3/h4 test rows, four per height x polarity label, all
  originally incorrect.
- Foil: first parsed Stage 1 hypothesis that differs from the gold structured
  hypothesis.
- Conditions: baseline, `toward_gold`, `away_gold`, orthogonal.
- Strengths: `0.5`, `1.0`, `1.5`, and `2.0` train projection SD.
- Primary metric: paired MCQ-margin delta versus baseline, especially
  `toward_gold` minus matched orthogonal, plus false-to-true MCQ choice flips.

Decision rule: if this is null, we can write the stronger statement that the
probe-derived answer-polarity direction does not move outputs even under a
binary, adversarially-headroomed choice setup. The next branch should then be
activation patching/interchange, because patching asks whether any compact
state is causal rather than whether this direction is causal.

Result: completed as Scholar job `452362`; see
`docs/stage2_27b_margin_forced_choice_results.md`. The hard-foil run produced
zero MCQ choice changes and zero false-to-true flips through `2.0` projection
SD. Mean MCQ-margin changes were tiny, and original-prompt margin changes were
not directionally interpretable because `away_gold` often moved as much or more
than `toward_gold`. Close this probe-direction steering branch and pivot to
activation patching/interchange.

## Backlog After This Diagnostic

1. Activation patching/interchange scan: patch full activations from matched
   easier/correct prompts into harder/incorrect prompts across layer and prompt
   position. This is causal localization, not steering, but it tests whether a
   repair state exists at all.
2. CAA-style mean-difference steering at the localized layer/position, using
   content contrasts rather than correctness contrasts.
3. Reconstruction-error steering: either train an error-component direction and
   intervene in raw space, or decompose candidate directions into
   SAE-reconstructable and residual-error components and compare them as matched
   interventions.
