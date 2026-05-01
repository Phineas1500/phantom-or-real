# Stage 2 27B Margin And Forced-Choice Steering Results

Date: 2026-04-30

Plan reference: `docs/stage2_forced_choice_margin_plan.md`

## Summary

Scholar job `452338` completed the 27B L45 margin plus forced-choice smoke in
14 minutes 17 seconds. The run reused the raw L45 gold-polarity direction from
the answer-property steering setup and measured two graded margins for each
condition:

1. original-prompt full-sequence logprob margin:
   `log P(gold hypothesis) - log P(opposite-polarity foil)`;
2. MCQ forced-choice margin:
   `log P(gold option) - log P(foil option)`, plus decoded `(A)/(B)`.

The result is not a clean steering success. MCQ margins move slightly more in
the `toward_gold` direction than orthogonal controls, but every baseline MCQ
choice was already correct and the original-prompt margins were noisy. Treat
this as weak constrained-margin sensitivity, not evidence of answer repair.

## Artifacts

- Report: `docs/answer_property_margins_27b_l45_polarity_smoke.json`
- Rows: `results/stage2/steering/answer_property_margins_27b_l45_polarity_smoke.jsonl`
- Direction: `results/stage2/steering/answer_property_margins_27b_l45_polarity_direction.npz`
- Log: `slurm_logs/stage2_margin27b_452338.out`

## Setup

- Model: `google/gemma-3-27b-it`
- Task: `infer_property`
- Site: `blocks.45.hook_resid_post`
- Direction target: gold answer polarity
- Probe: `val_auc=1.000`, `test_auc=1.000`
- Rows: 8 balanced S1 h3/h4 test rows
- Selected answer labels: 3 non-negated, 5 negated
- Conditions: baseline, `toward_gold`, `away_gold`, orthogonal
- Strengths: `0.5` and `1.0` train projection SD

## Aggregate Results

| Condition | Mean original margin | Delta vs baseline | Mean MCQ margin | Delta vs baseline | MCQ choice acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 16.648 | - | 15.703 | - | 1.000 |
| `toward_gold_pos0p5sd` | 16.727 | +0.079 | 15.812 | +0.109 | 1.000 |
| `toward_gold_pos1sd` | 16.659 | +0.011 | 15.906 | +0.203 | 1.000 |
| `away_gold_pos0p5sd` | 16.672 | +0.023 | 15.766 | +0.063 | 1.000 |
| `away_gold_pos1sd` | 16.618 | -0.031 | 15.781 | +0.078 | 1.000 |
| `orthogonal_pos0p5sd` | 16.688 | +0.039 | 15.719 | +0.016 | 1.000 |
| `orthogonal_pos1sd` | 16.846 | +0.197 | 15.687 | -0.016 | 1.000 |

Pairwise counts:

- `toward_gold_pos1sd` improved MCQ margin on 6/8 rows and worsened it on 2/8.
- `toward_gold_pos1sd` improved original-prompt margin on 4/8 rows and worsened
  it on 4/8.
- No condition changed the decoded MCQ answer, because baseline MCQ was already
  8/8 correct.

## Interpretation

This run partially supports the output-space diagnosis but does not give a
reportable steering win:

- MCQ margins show a small directional effect under `toward_gold`, especially at
  `1.0` SD, and orthogonal controls do not show the same MCQ increase.
- The effect is tiny relative to the already-large baseline MCQ margin.
- The opposite-polarity foil is too easy: the model already strongly prefers
  the gold option under the MCQ prompt.
- Original-prompt margins do not show a reliable target-specific effect; the
  `orthogonal_pos1sd` original-margin delta is larger than either
  `toward_gold` delta.

Do not scale this exact opposite-polarity MCQ setup as-is. If we continue with
margin/forced-choice steering, the next version should compare the gold
hypothesis against the model's actual wrong hypothesis on baseline-incorrect
rows, or against a matched wrong concept from the same ontology. That would test
the parent/child or wrong-concept failure mode that matters for InAbHyD, rather
than the easier polarity contrast.

## Next Options

1. Gold-vs-emitted-wrong margin/MCQ: select baseline-incorrect rows, parse the
   model's wrong hypothesis, and score/force-choice gold versus that wrong
   candidate. This is now implemented as
   `scripts/stage2_steer_answer_property_margins_27b_L45_property_hardfoil.sbatch`,
   using 16 balanced h3/h4 rows and strengths through `2.0` projection SD.
2. Activation patching/interchange scan: patch full activations from matched
   easier/correct prompts into harder/incorrect prompts to find whether a causal
   repair state exists.
3. CAA-style content vectors: if patching localizes a site, compute
   mean-difference vectors for concrete content contrasts at that site.
4. Reconstruction-error steering: compare interventions along
   SAE-reconstructable versus raw-minus-reconstruction directions.

## Hard-Foil Follow-Up

Scholar job `452362` completed the hard-foil refinement in 33 minutes. It
selected 16 S1 h3/h4 property rows that were originally parsed but strongly
incorrect, balanced by height and gold polarity. The foil for each row was the
model's own emitted wrong Stage 1 hypothesis.

Artifacts:

- Report: `docs/answer_property_margins_27b_l45_polarity_hardfoil.json`
- Rows: `results/stage2/steering/answer_property_margins_27b_l45_polarity_hardfoil.jsonl`
- Direction: `results/stage2/steering/answer_property_margins_27b_l45_polarity_hardfoil_direction.npz`
- Log: `slurm_logs/stage2_hfoil27b_452362.out`

Key results:

| Condition | Mean original-margin delta | Mean MCQ-margin delta | MCQ choice flips | False-to-true MCQ flips |
| --- | ---: | ---: | ---: | ---: |
| `toward_gold_pos0p5sd` | +0.360 | +0.008 | 0 | 0 |
| `toward_gold_pos1sd` | +0.859 | -0.016 | 0 | 0 |
| `toward_gold_pos1p5sd` | +0.918 | +0.062 | 0 | 0 |
| `toward_gold_pos2sd` | +0.403 | +0.109 | 0 | 0 |
| `orthogonal_pos2sd` | -0.078 | +0.031 | 0 | 0 |
| `away_gold_pos2sd` | +1.667 | +0.094 | 0 | 0 |

Baseline MCQ choice accuracy was already 14/16 even though all selected rows
were originally free-form incorrect. On the two true MCQ-headroom rows,
`toward_gold` produced no flips and did not beat controls: mean MCQ-margin
delta at `2.0` SD was `-0.125` for `toward_gold`, `-0.125` for orthogonal, and
`+1.875` for `away_gold`.

Interpretation: this closes the Cox-style forced-choice probe-direction branch
for the current setup. Even under a binary choice format with harder
model-emitted wrong foils and strengths through `2.0` projection SD, the raw
gold-polarity direction did not change choices. The small MCQ-margin shifts are
too small and non-antisymmetric to support a causal-axis claim, and
original-prompt margin movement is not target-specific.
