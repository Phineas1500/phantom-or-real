# Gemma 3 4B Answer-Property Steering Results

Date: 2026-04-30

Plan reference: `docs/stage2_4b_answer_property_steering_plan.md`

## Summary

The answer-property steering follow-up completed for Gemma 3 4B layer 22 on
`infer_property` h3/h4 examples. The pulled runner was used, with one local
addition: `--resume` support keyed by `(source_row_index, condition)` so a
partial JSONL can be reused if a long decode run is interrupted.

The raw L22 gold-polarity probe was perfectly predictive on the split
(`val_auc=1.0000`, `test_auc=1.0000`), but decode-time steering did not produce
reliable answer-content movement. The full sweep produced no polarity flips,
no predicate flips toward gold, and no strong-correctness improvements over
baseline. The few answer-content changes were invalid predicate changes or
also appeared in orthogonal controls.

Conclusion: raw answer-property information is linearly available at L22, but
the tested decode-time intervention does not provide specific causal control
over emitted answer content. Per the plan, sparse answer-property steering was
not run because the raw direction did not move answer content above controls.

## Artifacts

Smoke run:

- Report: `docs/answer_property_steering_4b_l22_polarity_smoke.json`
- Rows: `results/stage2/steering/answer_property_4b_l22_polarity_smoke.jsonl`
- Direction: `results/stage2/steering/answer_property_4b_l22_polarity_direction.npz`

Main run:

- Report: `docs/answer_property_steering_4b_l22_polarity_decode_sweep.json`
- Rows:
  `results/stage2/steering/answer_property_4b_l22_polarity_decode_sweep.jsonl`
- Direction: `results/stage2/steering/answer_property_4b_l22_polarity_direction.npz`

## Runs

Smoke test:

| Metric | Value |
|---|---:|
| Source rows | 8 |
| Output rows | 56 |
| Strengths | `0.5,1` |
| Elapsed seconds | 830.5 |
| Parse failures | 0 |
| Hook mismatches | 0 |
| Answer-content changes vs baseline | 0 |
| Polarity flips vs baseline | 0 |
| Predicate flips vs baseline | 0 |

Main polarity sweep:

| Metric | Value |
|---|---:|
| Source rows | 32 |
| Output rows | 320 |
| Strengths | `0.5,1,2` |
| Elapsed seconds | 4659.6 |
| Parse failures | 2 |
| Hook mismatches | 0 |
| Polarity flips toward gold | 0 |
| Polarity flips away from gold | 0 |
| Predicate flips toward gold | 0 |
| Strong false-to-true flips | 0 |

Main by-condition summary:

| Condition | Strong acc. | Parse fail | Polarity match | Predicate match | Answer change |
|---|---:|---:|---:|---:|---:|
| `baseline` | 0.5312 | 0.0000 | 1.0000 | 0.8125 | 0.0000 |
| `toward_gold_pos0p5sd` | 0.5000 | 0.0000 | 1.0000 | 0.8125 | 0.0000 |
| `toward_gold_pos1sd` | 0.5000 | 0.0312 | 1.0000 | 0.8065 | 0.0323 |
| `toward_gold_pos2sd` | 0.5000 | 0.0312 | 1.0000 | 0.8387 | 0.0000 |
| `away_gold_pos0p5sd` | 0.5312 | 0.0000 | 1.0000 | 0.8125 | 0.0625 |
| `away_gold_pos1sd` | 0.5312 | 0.0000 | 1.0000 | 0.8125 | 0.0625 |
| `away_gold_pos2sd` | 0.5000 | 0.0000 | 1.0000 | 0.8125 | 0.0625 |
| `orthogonal_pos0p5sd` | 0.5000 | 0.0000 | 1.0000 | 0.8125 | 0.0312 |
| `orthogonal_pos1sd` | 0.5000 | 0.0000 | 1.0000 | 0.8125 | 0.0312 |
| `orthogonal_pos2sd` | 0.5000 | 0.0000 | 1.0000 | 0.8125 | 0.0312 |

## Interpretation

The probe result confirms that answer polarity is easily decoded from raw L22
activations. The generation result is different: adding the direction at
`blocks.22.hook_resid_post` on every decode step did not flip emitted polarity
and did not repair wrong predicates.

Observed non-flat cases were not useful steering effects:

- Source row `3644`: `toward_gold` at 1sd and 2sd caused parse failures.
- Source row `4680`: `toward_gold_pos1sd` changed the predicate from `slow` to
  invalid `frompor`, which is a predicate flip away from gold.
- Source row `8325`: `away_gold` and orthogonal controls changed invalid
  predicate `a` to invalid `the`.
- Source row `10598`: `away_gold` changed invalid predicate `a` to invalid
  `daumpin`.
- Source row `8765`: several conditions changed strong scoring from true to
  false while parsed predicate and polarity stayed unchanged, including
  orthogonal controls.

The result supports the same predictive-versus-causal gap seen in the prior 4B
correctness steering experiments, now using concrete answer-content labels.

## Validation

Validation completed locally:

```bash
python -m py_compile scripts/stage2_steer_answer_property_direction.py
python -m json.tool docs/answer_property_steering_4b_l22_polarity_smoke.json
python -m json.tool docs/answer_property_steering_4b_l22_polarity_decode_sweep.json
wc -l results/stage2/steering/answer_property_4b_l22_polarity_smoke.jsonl \
  results/stage2/steering/answer_property_4b_l22_polarity_decode_sweep.jsonl
git diff --check
```

`jq` is not installed in this local WSL environment, so Python JSON validation
was used instead.
