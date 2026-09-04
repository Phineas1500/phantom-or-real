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
