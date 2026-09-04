# Item WK — the knowledge-conflict regime (NQ-Swap): can the blind loop help at the frame level? (2026-09-04)

Registered: docs/causal_handle_directions.md item WK (before any data).
Frame: NQ-Swap (Longpre et al. 2021), 800 named-entity conflict rows —
the Wikipedia paragraph's answer entity replaced by a type-matched
entity, the document's answer is the gold, the original answer is what
memory says (`scripts/nqswap_frame.py`, seed 20260904; paragraph
contexts only; the memory answer absent from the substituted text).
Stage 1 **job-it6ue** ($0.24); stage 2 **job-ayty2 / job-qzigj** (A/B,
own-frame donors) and **job-7t69a / job-x2pj7** (XA/XB, WikiHop WX
donors). Readers `scripts/wikihop_wk_pins.py`, `scripts/wikihop_wk_gates.py`.

## Stage 1 — the conflict regime as Gemma sees it (800 rows, k = 8)
| | NQ-Swap | WikiHop real (WF) | HotpotQA | SQuAD |
|---|---|---|---|---|
| accuracy with documents (vs the document's answer) | **0.578** | 0.466 | 0.613 | 0.897 |
| closed-book vs the document's answer | 0.016 (counterfactual) | 0.445 | 0.521 | 0.692 |
| memory rate (closed-book modal = the original fact) | **0.74** | — | — | — |
| 0/8 failures | 323 (40%) | — | — | — |
| conflict failures (memory ∧ 0/8) | **226** | doc-dependent 276 | 219 | 54 |
| hint-repairable share of those | **0.327 [0.270, 0.385]** | 0.214 | 0.242 | 0.259 |

**27th prediction (FIXABLE-MAJORITY): INTERMEDIATE.** A third of the
conflict failures are fixable by attention alone — above WikiHop's
quarter, well short of the sandbox's majority. The model follows the
document on 58% of samples, reverts to memory on 17%, and answers
something else on 25%. Among the 226 conflict failures, 118 answer with
the memory fact and 105 with another listed candidate; the two kinds are
equally repairable (0.33 vs 0.30). Instrument gate (std within
[0.10, 0.90]) PASS.

## A regularity across frames: the hint either works completely or not at all (descriptive)
Hint-first correct out of 8, on document-dependent failures:
| frame | 0/8 | 1–7 | 8/8 |
|---|---|---|---|
| NQ-Swap conflict failures (226) | 65% | 5% | **30%** |
| WikiHop real (276) | 76% | 5% | 18% |
| WikiHop anonymized (288) | 75% | 3% | 22% |
| HotpotQA (219) | 74% | 5% | 22% |
| SQuAD (54) | 72% | 2% | 26% |
| Qwen3.5-27B on WikiHop (140) | 53% | **43%** | 4% |

On Gemma the fixable share is a discrete class, not a tail of a graded
response: a failure is either a firm belief the hint cannot touch or an
attention lapse the hint removes entirely. Qwen's response is graded.
This is the behavioural face of the write results — the same rows flip
or do not, whatever the dose.

## Stage 2 — the blind frame test (120 rows drawn blind; baseline 0.597; 1,397 branches per arm; delivery valid everywhere)
| blind rule over the 120 rows | own-frame donors | WikiHop vector (WX donors) |
|---|---|---|
| **abstention (REGISTERED, 28th)** | **−0.009 [−0.025, 0.000]** · 0 improved / 2 harmed | −0.003 [−0.009, 0.000] · 0 / 1 |
| always answer | −0.078 [−0.138, −0.025] · 2 / 14 | −0.099 [−0.176, −0.023] · 7 / 21 |
| **grounded two-stage (pre-named rider)** | **+0.037 [+0.008, +0.071]** · 5 / 0 · acts on 5% | **+0.062 [+0.021, +0.113]** · 8 / 1 · acts on 11% |
| oracle two-stage (flag = failure) | +0.019 [−0.015, +0.054] · 4 / 2 | +0.077 [+0.025, +0.133] · 11 / 3 |

**28th prediction (BLIND-LOOP-HELPS-AT-FRAME-LEVEL): NOT CONFIRMED.**
The abstention rule changes nothing because the model's committed
answer always wins its own branch: on 45 of 46 failing rows the
baseline answer has a branch (the candidate list in the prompt gives
every candidate an address — a correction to the registration's "no
address" sentence) and it fires at 1.0, so the unique top branch is the
baseline's and the rule reproduces it. The gold branch fires at only
0.29 (own) / 0.50 (WikiHop) on repairable rows, against 0.96 for the
text hint: a memory conflict resists the write far more than a WikiHop
attention lapse does.

Per stratum (grounded rule, dP over baseline):
| stratum | n | baseline | own donors | WikiHop vector | detector flags |
|---|---|---|---|---|---|
| conflict failure, repairable | 12 | 0.083 | +0.167 | **+0.250** | 67% |
| conflict failure, unrepairable | 23 | 0.022 | +0.065 | **+0.152** | 70% |
| other failure | 11 | 0.023 | 0.000 | 0.000 | 9% |
| correct-majority | 71 | 0.970 | +0.014 | 0.000 | **3%** |

Correct rows resist the write here more than on WikiHop: the model's
own answer's branch fires at 0.99 and every other branch at 0.02 (own)
/ 0.06 (WikiHop). The groundedness check, "is the answer a span of the
passage?", is label-free, costs nothing, catches two thirds of the
conflict failures and 3% of correct rows, and is exactly the detector
the WD/WE addendum said the loop was missing. With it and the WikiHop
vector the whole-frame accuracy rises from 0.597 to 0.659.

## The write on the 12 repairable conflict rows in the draw (descriptive)
| frozen write at L30 | gold-address dP | specificity | text hint |
|---|---|---|---|
| own-frame donors, 1× / 2× | +0.396 / +0.208 [+0.021, +0.438] | +0.210 [+0.049, +0.403] | 0.958 |
| WikiHop vector, 2× | **+0.417 [+0.167, +0.667]** | +0.420 [+0.170, +0.676] | |

The cross-task vector beats the frame's own for the third time.

## Verdict
**ABSTENTION-LOOP-DOES-NOT-CLOSE-BLIND / GROUNDED-TWO-STAGE-HELPS
(rider, descriptive).** The knowledge-conflict regime has a larger
fixable share than WikiHop (a third, not a quarter) but not the
sandbox's majority, and the registered blind rule fails for a
structural reason that also explains the selector gap on WikiHop: the
baseline's own branch always fires. What does work blind is a
label-free groundedness detector in front of the loop — +0.037 /
+0.062 at the frame level, zero to one row harmed — the first blind
frame-level gain on natural text in the program. It was pre-named but
not registered as a prediction, so it is a descriptive result until a
registered replication on a fresh draw (and, ideally, a frame where the
memory answer can still appear in the passage, so the detector is not
helped by NQ-Swap's construction).

## Program tally after WK
28 registered directional predictions: **23 confirmed**, 4 not (13th,
14th, 19th, 28th), 1 intermediate (27th). ≈ $129 across 120 H100 jobs.
