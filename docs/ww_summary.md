# Item WW — attention steering aimed by the model's own retrieval heads (38th CONFIRMED: steering at the gold row repairs +0.625; 39th and 40th NOT CONFIRMED: the label-free aim does not compose) (2026-09-06)

Registered: docs/causal_handle_directions.md item WW (before any data).
Steering: PASTA's α-scaling in the 14 retrieval heads pinned from WN
(layers 35 / 41 / 47 / 53) — for every query after the documents, the
attention logits of every key outside the highlighted rows get log α
added before the softmax, on prefill and every decode step; α 0.01
primary, 0.1 as a dose reading. Nothing is written into the rows.
Shards **job-t88ur / job-dfwrm** (45 fixable rows each, 7 arms) and
**job-2rrs2 / job-i7j7a** (60 fresh blind rows each, 5 arms), $15.70;
four attempts lost to an out-of-memory in the eager softmax and its
pre-emptive cancellations, $3.49 (patched job bit-identical to stock
eager attention). k = 8 at temperature 0.7 per arm. Job
`scripts/tables_steer_job.py`, reader `scripts/tables_steer_reader.py`,
outputs `docs/ww_gates.json`, rows `results/loop_screen/ww_{fa,fb,b1,b2}.jsonl`.

## Delivery
810 of 810 steered row-arms fired on every chunk; the pinned heads'
highlight share at the final prompt token rose on 100% of them: mean
0.333 before → 0.842 (gold, α 0.01), 0.970 (heads' top 3, α 0.01),
0.396 (gold, α 0.1), 0.822 (heads, α 0.1), 0.175 (control rows).

## Fixable failures (90 rows; baseline 0.029; the heads' top 3 holds the gold row on 48, the top 1 on 19)
| arm | gold rate | dP vs baseline [95% CI] | up / down |
|---|---|---|---|
| **steer at the gold row, α 0.01** | **0.654** | **+0.625 [+0.528, +0.718]** | **62 / 2** |
| steer at the gold row, α 0.1 | 0.397 | +0.368 [+0.274, +0.464] | 37 / 2 |
| steer at the heads' top 3, α 0.01 | 0.076 | +0.047 [+0.008, +0.094] | 7 / 2 |
| steer at the heads' top 3, α 0.1 | 0.067 | +0.037 [+0.004, +0.076] | 7 / 2 |
| steer at 3 rows of another table (control) | 0.035 | +0.006 [−0.022, +0.035] | 4 / 4 |
| quote the heads' top 3 as text | 0.235 | +0.206 [+0.122, +0.290] | 22 / 2 |

**38th CONFIRMED — STEERING-AT-THE-GOLD-ROW-REPAIRS.** The largest
repair in the program, on the regime where the residual write gave
0.000 at the same rows (WM). The control is inert: the effect is the
row, not the perturbation. The dose runs the opposite way from the
write's: the stronger α 0.01 repairs more than α 0.1.

**39th NOT CONFIRMED — THE-HEADS-AIM-IT-BLIND.** +0.047 is below the
+0.10 bar and the contrast with the control (+0.042 [−0.001, +0.092])
straddles zero. Paired, gold-aimed beats heads-aimed by +0.578.
Conditional on the aim: steering at the gold row repairs +0.591 when
the heads' top 3 contains it and +0.664 when it does not (the repair
never depended on the heads); steering at the heads' top 3 repairs
+0.089 on the 48 rows where the gold row is among them, and on 39 of
those 48 the majority answer is the baseline's. α-scaling redistributes
the heads' attention among the highlighted rows in proportion to the
model's prior, and the prior already prefers the wrong neighbour.
Quoting the same three rows as text repairs four times more (+0.206;
aim hit +0.323 / miss +0.071; paired over steering +0.158 [+0.064,
+0.250]) because the model re-reads and reorders them in a short
context.

## Blind draw (120 fresh rows; baseline 0.823; 99 correct-majority, 21 failures; the heads' top 3 holds the gold row on 0.925)
| arm | dP vs baseline [95% CI] | up / down | on correct rows | on failures |
|---|---|---|---|---|
| **steer at the gold row, α 0.01 (oracle aim)** | **+0.055 [+0.015, +0.102]** | 11 / 3 | 0.000 | +0.315 |
| steer at the heads' top 3, α 0.01 | −0.022 [−0.051, +0.007] | 1 / 8 | −0.027 | 0.000 |
| steer at the heads' top 3, α 0.1 | −0.014 [−0.041, +0.014] | 2 / 7 | −0.021 | +0.024 |
| quote the heads' top 3 | −0.009 [−0.061, +0.045] | 8 / 9 | −0.049 | +0.179 |
| two-stage rule, gate 0.1957 (5 rows gated) | −0.017 [−0.042, 0.000] | 0 / 2 | | |
| two-stage rule, wide gate 0.3201 (24 gated) | −0.017 [−0.050, +0.017] | 1 / 3 | | |

**40th NOT CONFIRMED — BLIND-STEERING-NETS-POSITIVE.** The gate fires
on 5 of 120 fresh rows (the fresh draw's peak shares sit far above WN's
failure strata) and the rule loses two of them. With an oracle aim the
same steering lifts the frame +5.5 points at zero collateral — the
ceiling the aim has to reach.

## What this changes
WM's wall had two halves. The intervention half is open: attention
steering repairs the fixable table failures where the residual write
corrupted them, with an inert control and no collateral when aimed
right. The aim half stays where WN left it — the heads find the row
in their top 3 on half of the fixable failures — and the two do not
compose, because the heads' residual error is the ordering inside
their top 3 and steering cannot reorder what it highlights. The text
route (quote the top 3) reorders but nets nothing blind. Possible
next: a two-stage aim — quote the top 3, let the model name the row,
steer at the named row — bounded by the quote arm's own accuracy;
or close the table chain here.

## Program tally after WW
40 registered directional predictions: **28 confirmed**, 11 not, 1
intermediate; ≈ $196 across 169 H100 jobs.
