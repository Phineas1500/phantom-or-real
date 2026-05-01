# Stage 2 27B Answer-Property Steering Results

Date: 2026-04-30

## Summary

The 27B answer-property steering smoke completed as Scholar job `452301`. It
tested a raw L45 residual direction trained to predict the gold answer polarity
for `infer_property`, then applied decode-time steering on a small balanced S1
h3/h4 test subset.

The offline probe was perfectly predictive on the split used for this run
(`val_auc=1.0000`, `test_auc=1.0000`), but steering did not produce controlled
answer movement. `toward_gold` produced zero polarity flips toward gold, zero
predicate flips toward gold, and zero strong false-to-true repairs. The only
false-to-true repair happened under `away_gold_pos1sd`, not `toward_gold`, so it
is not evidence for target-directed steering.

Conclusion: raw L45 answer-polarity information is linearly available, but this
free-form decode-step intervention did not convert it into reliable emitted
answer control. This closes the free-form answer-property steering branch unless
a later forced-choice protocol first shows clean answer-direction control.

## Artifacts

- Report: `docs/answer_property_steering_27b_l45_polarity_smoke.json`
- Rows: `results/stage2/steering/answer_property_27b_l45_polarity_smoke.jsonl`
- Direction: `results/stage2/steering/answer_property_27b_l45_polarity_direction.npz`
- Log: `slurm_logs/stage2_ansprop27b_452301.out`

## Setup

- Model: `google/gemma-3-27b-it`
- Hook: `blocks.45.hook_resid_post`
- Task: `infer_property`
- Split family: S1
- Steering subset: 8 balanced h3/h4 test rows
- Target: gold answer polarity
- Intervention scope: `last_token_each_forward`
- Strengths: `0.5` and `1.0` train projection SD
- Controls: norm-matched orthogonal directions
- Generation: deterministic, `max_new_tokens=160`

Probe details:

- `best_c=0.01`
- `raw_coef_norm=0.0305`
- `train_projection_std=299.21`
- kept rows: 10,025
- train/val/test counts: 6,988 / 1,525 / 1,512

## Condition Results

| Condition | Strong acc. | Weak acc. | Parse fail | Content changed | False -> true | True -> false | Predicate toward gold | Takeaway |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | 0.375 | 0.625 | 0.000 | 0.000 | - | - | - | 3/8 strong, 5/8 weak. |
| `toward_gold_pos0p5sd` | 0.250 | 0.500 | 0.250 | 0.000 | 0 | 1 | 0 | Degradation, no useful answer movement. |
| `toward_gold_pos1sd` | 0.250 | 0.500 | 0.125 | 0.000 | 0 | 1 | 0 | Degradation, no useful answer movement. |
| `away_gold_pos0p5sd` | 0.375 | 0.625 | 0.125 | 0.000 | 0 | 0 | 0 | Flat relative to baseline. |
| `away_gold_pos1sd` | 0.500 | 0.750 | 0.000 | 0.125 | 1 | 0 | 1 | One repair, but in the wrong steering direction. |
| `orthogonal_pos0p5sd` | 0.250 | 0.500 | 0.250 | 0.000 | 0 | 1 | 0 | Matched-control degradation. |
| `orthogonal_pos1sd` | 0.375 | 0.625 | 0.000 | 0.000 | 0 | 0 | 0 | Flat relative to baseline. |

## Changed Cases

- Source row `4926`: baseline was strongly correct. Both `toward_gold`
  strengths and `orthogonal_pos0p5sd` became parse failures. This is best read
  as perturbation/format fragility, not targeted answer steering.
- Source row `6604`: baseline parsed the predicate as invalid `a`, while
  `away_gold_pos1sd` parsed `salty`, matching the gold predicate with the same
  polarity. Because this happened under `away_gold`, it is not directional
  evidence for the trained answer-polarity vector.

## Interpretation

This run matches the 4B answer-property result: answer content can be perfectly
decoded offline, but the current free-form decode-step steering protocol does
not reliably move emitted predicates or polarities. The next steering branch,
if pursued, should constrain the output channel with Cox-style forced-choice
prompts rather than increasing free-form answer-property strength.
