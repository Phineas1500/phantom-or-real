# Item WW′ (+ rider WW′-r) — pick-then-steer: the model's own pick among its retrieval heads' candidates as the aim (41st and 42nd NOT CONFIRMED; the table chain closes) (2026-09-06)

Registered: docs/causal_handle_directions.md item WW′ (before any data);
rider WW′-r registered on telemetry after the numbered pick prompt
failed to parse (24 / 90). Design: the retrieval heads' top-3 (and
top-5) rows are quoted as labelled candidates before the standard
prompt, the model names the row holding the answer (greedy), and PASTA
steering (α 0.01, the 14 heads, reading queries only) aims at the
named row. Same rows, image and seeds as WW, so WW's baseline, quote,
steer-gold and steer-heads arms pair row-by-row. Jobs: WW′ **job-35zpa
/ job-6j3g2** ($6.17, numbered pick), rider **job-weedn / job-ehbuf**
($6.03, letter pick). Reader `scripts/tables_pick_reader.py`; outputs
`docs/wwpr_gates.json` (rider) and `docs/wwp_gates.json` (numbered).

## The pick (rider, letter labels; parsed 100%)
| pool | gold among candidates | model names gold | names the first-listed | heads' own top 1 = gold |
|---|---|---|---|---|
| fixable failures, top 3 (90) | 48 | **15** | 51 | 19 |
| fixable failures, top 5 (90) | 66 | 20 | 44 | 19 |
| blind rows, top 3 (120) | 111 | 89 | 99 | — |

Asked to choose among three rows that contain the right one half the
time, the model names it 15 times and the first-listed row 51 times.
Its explicit judgment carries the same preference as its heads.

## Fixable failures (90; WW baseline 0.029)
| arm | gold rate | dP vs baseline [95% CI] | up / down | vs quote (+0.206) | vs steer heads' top 3 (+0.047) |
|---|---|---|---|---|---|
| pick3_steer (rider) | 0.076 | +0.047 [+0.001, +0.101] | 6 / 4 | −0.158 [−0.244, −0.079] | +0.000 [−0.060, +0.062] |
| pick5_steer (rider) | 0.111 | +0.082 [+0.022, +0.146] | 11 / 4 | −0.124 [−0.211, −0.040] | +0.035 [−0.036, +0.106] |
| pick3_steer (numbered; fallback on 78 rows) | 0.111 | +0.082 [+0.032, +0.138] | 11 / 2 | −0.124 | +0.035 |

On the rows where the pick is the gold row, steering at it repairs
+0.358 (rider) / +0.349 (numbered); elsewhere ≈ 0. **41st NOT
CONFIRMED** (bar +0.20).

## Blind draw (120; WW baseline 0.823)
| arm / rule (rider) | net [95% CI] | up / down | gated | collateral on correct | on failures |
|---|---|---|---|---|---|
| pick3_steer, always on | −0.028 [−0.068, +0.013] | 3 / 10 | 120 | −0.042 | +0.036 |
| disagreement-gated pick3 rule | −0.033 [−0.075, +0.008] | 2 / 6 | 53 | | |
| peak-share-gated pick3 rule | −0.008 [−0.042, +0.017] | 1 / 2 | 5 | | |
| pick5_steer, always on | −0.060 [−0.106, −0.018] | 2 / 12 | 120 | −0.071 | −0.012 |

**42nd NOT CONFIRMED.** The disagreement gate fires on 53 rows — the
model's pick differs from the row its own answer came from more often
than it is right — and steering at those picks loses more than it
gains.

## What this closes
The intervention is strong wherever the aim is right (+0.36 on the
picked-gold rows, +0.625 at the gold row in WW, +0.055 blind with an
oracle aim at zero collateral). Every label-free aim tried on the
tables — the keyword retriever (WM), the retrieval heads' top 3 (WN,
WW), and now the model's explicit choice among those candidates —
inherits the same wrong preference among neighbouring rows. Quoting
the top 3 as text stays the best label-free repair on fixable rows
(+0.206) and nets nothing blind. The table chain closes here.

## Program tally after WW′
42 registered directional predictions: **28 confirmed**, 13 not, 1
intermediate; ≈ $208 across 173 H100 jobs.
