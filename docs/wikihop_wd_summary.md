# Item WD — blind yield and collateral: BLIND-LOOP-HELPS / COLLATERAL-HARM (16th prediction CONFIRMED; the abstention rule removes the harm) (2026-09-03)

Registered: docs/causal_handle_directions.md item WD (before any data).
Frame: the WO anonymized frame (507 rows). Doc-dependent pool 293 = 47
hint-repairable (WO's branches reused, same selector) + 246 unrepairable
(seeded 100 run: Y1 **job-iqv3d**, Y2 **job-dd5f2**, 50 rows each);
collateral 60 seeded correct-majority rows (C **job-g5naa**). Frozen
direction = donor mean gold hint-delta over the 47 WO rows (disjoint from
every test row); every candidate at 2× with k=4; std baseline k=8;
selector = output-first (answers-fired, real-text L38 gauge tie-break).
Seed 20260845, context ctx-92f605b4, $3.89 for the three jobs. Row-level:
`results/loop_screen/wikihop_wd_{y1,y2,c}.jsonl`; reader
`scripts/wikihop_wd_gates.py` → `docs/wikihop_wd_gates.json`.

Delivery audit: 10,284 fired branches (Y1 4,016 / Y2 4,228 / C 2,040);
zero-prefill 0, zero-positions 0, unhooked gauge forwards 0; no rows
skipped. Mean branches per row 20.1 / 21.1 / 8.5.

## Yield on the doc-dependent pool (no oracle filter)
| stratum | n | weight | baseline | loop | random branch | loop − baseline |
|---|---|---|---|---|---|---|
| hint-repairable (WO branches) | 47 | 0.160 | 0.029 | 0.362 | 0.156 | **+0.332 [+0.197, +0.468]** |
| unrepairable (Y1 + Y2) | 100 | 0.840 | 0.000 | 0.083 | 0.019 | **+0.083 [+0.033, +0.140]** |
| Y1 alone / Y2 alone | 50 / 50 | | 0.000 | 0.120 / 0.045 | | +0.12 [+0.04, +0.22] / +0.045 [0.00, +0.11] |

**BLIND YIELD (stratum-weighted) = +0.123 [+0.076, +0.174] — the 16th
registered prediction CONFIRMED.** The loop also yields on rows a text
hint could not repair: 8.3% of the unrepairable stratum against a 1.9%
random-branch rate, i.e. the write reaches some rows the hint does not.

## Collateral on correct-majority rows
| | baseline | loop | loop − baseline |
|---|---|---|---|
| always answer | 0.983 | 0.850 | **−0.133 [−0.217, −0.050]** |

The lower bound is below the pre-named −0.10 → **COLLATERAL-HARM** for
the always-answer loop. Eight of 60 rows are broken (every one from
1.0 to 0.0); on each of those eight the top answers-fired rate is not
unique-and-≥ 0.5.

## Abstention variant (pre-registered as descriptive; the deployment mode)
Answer with the loop only when the top answers-fired rate is unique and
≥ 0.5, else keep the baseline answer.

| | abstained | loop − baseline |
|---|---|---|
| hint-repairable stratum | 66% | +0.128 [+0.043, +0.234] |
| unrepairable stratum | 40% | +0.023 [0.000, +0.053] |
| **blind yield, weighted** | | **+0.039 [+0.015, +0.070]** |
| **collateral (correct rows)** | 28% | **0.000 [0.000, 0.000]** |

Under the abstention rule the loop never changes a correct row's answer
(all eight harmed rows abstain) and keeps about a third of the yield.
Frame-level net (descriptive, pool shares 293/507 doc-dependent and
181/507 correct-majority): always-answer ≈ +0.023, abstention ≈ +0.023 —
the same net gain, one of them without breaking anything.

## Verdict
**BLIND-LOOP-HELPS / COLLATERAL-HARM.** As a black box with no oracle
filter, the frozen-write + output-first loop raises accuracy on the
document-dependent failures of a frame it has never seen (CI > 0), at a
cost on already-correct rows that exceeds the pre-named bound when it is
forced to answer — and at no cost when it abstains on non-unique or weak
acceptance. The deployment-relevant reading is the abstention mode:
+0.039 [+0.015, +0.070] on the failure pool, 0.000 on correct rows.

## Program tally after WD
16 registered directional predictions: **14 confirmed**, 2 not confirmed
(13th, 14th). WikiHop chain: W → WH → WR → WL → WF → WX → WA (+rider) →
WG → WB → WO → WD; ≈ $35 across 41 H100 jobs. Nothing further is owed on
WikiHop.
