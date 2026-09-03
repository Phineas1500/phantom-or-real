# Item WE — blind yield and collateral on REAL text: BLIND-LOOP-HELPS / COLLATERAL-HARM (17th prediction CONFIRMED; net deployment value on real text ≈ 0) (2026-09-03)

Registered: docs/causal_handle_directions.md item WE (before any data). The
WD design with nothing changed but the frame: the WF fresh real-text frame
(800 rows, closed-book 0.445). Doc-dependent pool 276 = 59 hint-repairable
(WX cross-fit frozen-write branches reused, same selector) + 217
unrepairable (seeded 100 run: Y1 **job-qa7tt**, Y2 **job-dukdd**);
collateral 60 seeded correct-majority rows of 372 (C **job-4u3y2**).
Frozen direction = donor mean gold hint-delta over the 59 WF rows
(disjoint from every test row); every candidate at 2× with k=4; std
baseline k=8; selector = output-first with the real-text L38 gauge
tie-break. Seed 20260852, context ctx-69b2f31a, $4.78. Row-level:
`results/loop_screen/wikihop_we_{y1,y2,c}.jsonl`; reader
`scripts/wikihop_wd_gates.py --pins docs/wikihop_we_pinned.json --tie-key primary_L38`
→ `docs/wikihop_we_gates.json`.

Delivery audit: 14,244 fired branches (Y1 4,804 / Y2 5,420 / C 4,020);
zero-prefill 0, zero-positions 0, unhooked 0; no rows skipped. Mean
branches per row 24.0 / 27.1 / 16.8. Frozen norm target 3,251 (real-text
donors; the anonymized donors gave 5,435).

## Yield on the doc-dependent pool (no oracle filter)
| stratum | n | weight | baseline | loop | random branch | loop − baseline |
|---|---|---|---|---|---|---|
| hint-repairable (WX branches) | 59 | 0.214 | 0.017 | 0.271 | 0.048 | **+0.254 [+0.153, +0.373]** |
| unrepairable (Y1 + Y2) | 100 | 0.786 | 0.000 | 0.013 | 0.006 | **+0.013 [0.000, +0.033]** |
| Y1 alone / Y2 alone | 50 / 50 | | 0.000 | 0.000 / 0.025 | | 0.000 / +0.025 [0, +0.065] |

**BLIND YIELD (stratum-weighted) = +0.064 [+0.040, +0.093] — the 17th
registered prediction CONFIRMED.** On real text the loop barely reaches
the unrepairable stratum (1.3% of rows, one row in Y1+Y2 fully fixed and
a few partially); the yield is the repairable stratum's, diluted by its
21% share.

## Collateral on correct-majority rows
| | baseline | loop | loop − baseline |
|---|---|---|---|
| always answer | 0.998 | 0.900 | **−0.098 [−0.179, −0.033]** |
| abstain unless one branch is uniquely accepted ≥ 0.5 | 0.998 | 0.982 | −0.017 [−0.050, 0.000] |

The always-answer lower bound is below the pre-named −0.10 →
**COLLATERAL-HARM** (6 of 60 rows broken, each 1.0 → 0.0). Unlike WD,
abstention is not collateral-free here: on one row (WH_dev_4853) a
wrong candidate ("station" against the baseline's correct "railway
station") is the unique fully accepted branch.

## Abstention variant (descriptive)
| | abstained | loop − baseline |
|---|---|---|
| hint-repairable stratum | 51% | +0.051 [0.000, +0.119] |
| unrepairable stratum | 27% | +0.005 [0.000, +0.015] |
| **blind yield, weighted** | | **+0.015 [+0.004, +0.030]** |
| **collateral** | 20% | **−0.017 [−0.050, 0.000]** |

## Real text vs anonymized (pre-named descriptive comparison)
| | anonymized frame (WD) | real-text frame (WE) |
|---|---|---|
| repairable stratum, loop − baseline | +0.332 | +0.254 |
| unrepairable stratum, loop − baseline | +0.083 [+0.033, +0.140] | +0.013 [0.000, +0.033] |
| blind yield, weighted | +0.123 [+0.076, +0.174] | +0.064 [+0.040, +0.093] |
| collateral, always answer | −0.133 [−0.217, −0.050] | −0.098 [−0.179, −0.033] |
| abstention: yield / collateral | +0.039 / 0.000 | +0.015 / −0.017 |
| frame net, always / abstain (pool shares) | +0.023 / +0.023 | **−0.023 / −0.003** |
| frame net with an oracle failure detector (two-stage rule) | +0.104 | +0.029 |

Frame shares: WD 293/507 doc-dependent, 181/507 correct; WE 276/800 and
372/800. The two-stage rule with the real-text gauge as failure
detector (flags 33% of correct rows) nets −0.023.

## Verdict
**BLIND-LOOP-HELPS / COLLATERAL-HARM.** The blind loop raises accuracy on
real-text failures (CI > 0) and costs more than the bound on correct
rows when forced to answer. Read together with WD: the loop's blind
deployment value is about +0.02 per frame on renamed entities and about
zero on real text, because real-text failures outside the
hint-repairable quarter are firm (1.3% reached, against 8.3% when memory
is removed) and correct rows are as fragile either way. The loop is a
repair tool for rows known to be failing; where memory can answer, that
knowledge has to come from outside the loop. This is the same statement
as WA's "the write is strongest where memory cannot answer", now on the
deployment side.

## Program tally after WE
17 registered directional predictions: **15 confirmed**, 2 not confirmed
(13th, 14th). WikiHop chain: W → WH → WR → WL → WF → WX → WA (+rider) →
WG → WB → WO → WD → WE; ≈ $40 across 44 H100 jobs.
