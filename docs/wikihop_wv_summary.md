# Item WV — the content control: CONTENT-MATTERS / SIGN-MATTERS (23rd + 24th predictions CONFIRMED) (2026-09-03)

Registered: docs/causal_handle_directions.md item WV (before any data).
Gemma-3-27B, the 59 WX rows (cross-fit donors as in WX), write at L30 ×
2× at the gold candidate's whole-word mentions, k=4, donor mean
per-position |δ| as the norm target in every arm. Arms: (a) the frozen
hint-delta — WX's records; (b) the sign-flipped hint-delta (jobs
**job-bdnba / job-288qk**); (c) a matched-norm random direction, seed
20260867 (**job-cxsgd / job-zs47i**); (d) a second random draw, seed
20260868 (**job-byrpw / job-zr3sp**). Random directions have cosine
−0.026 / −0.024 / +0.003 / +0.004 with the hint-delta. Each arm also
fires at three seeded non-gold addresses. Attention read (WT settings,
30 rows ≤ 1,600 tokens): **job-n7c9t** (flip), **job-5m7en** (random,
seed 20260867). WV cost ≈ $9 over 13 jobs, ≈ $6.5 of it five first-
attempt behaviour jobs that died at model download on full-disk
machines (platform ticket tkt-qzgmz). Reader `scripts/wikihop_wv_gates.py`
→ `docs/wikihop_wv_gates.json`; rows `results/loop_screen/wikihop_wv_*.jsonl`.

## Repair at the gold address (59 rows, baseline 0.017)
| vector at the gold span, L30 × 2× | gold rate | dP vs baseline | non-gold-address gold rate | specificity |
|---|---|---|---|---|
| **hint-delta (frozen)** | **0.377** | **+0.360 [+0.242, +0.483]** | 0.031 | +0.346 [+0.232, +0.470] |
| sign-flipped hint-delta | 0.000 | −0.017 [−0.051, 0.000] | 0.041 | −0.041 [−0.083, −0.010] |
| random, matched norm (two draws pooled) | 0.000 | −0.017 [−0.051, 0.000] | 0.025 | −0.025 [−0.064, −0.001] |

Row-paired: **hint − random = +0.377 [+0.258, +0.508]** (23rd prediction
CONFIRMED — CONTENT-MATTERS); **hint − flip = +0.377 [+0.258, +0.508]**
(24th prediction CONFIRMED — SIGN-MATTERS). The flipped and random
vectors produce zero gold answers in 236 samples each; a large
perturbation of the wrong content at the right address mildly
suppresses the gold answer rather than helping it.

## Attention at the gold span (final token, mean over heads, write − none; 30 rows)
| layer | hint-delta (WT) | sign-flipped | random |
|---|---|---|---|
| L31 | +0.039 [+0.031, +0.047] | −0.008 [−0.010, −0.006] | −0.006 [−0.008, −0.005] |
| L32 | **+0.055 [+0.041, +0.069]** | −0.007 [−0.009, −0.005] | −0.005 [−0.007, −0.004] |
| L33 | +0.039 | −0.010 | −0.004 |
| L38 | +0.0035 [+0.0020, +0.0048] | −0.005 [−0.007, −0.004] | 0.000 [−0.001, +0.001] |
| L42 | +0.021 | −0.007 | −0.0004 |
| L53 | +0.031 | −0.026 | −0.009 |

The pre-named reading: the random vector routes *less* attention than
the hint-delta — it routes none, and both wrong vectors slightly repel
attention from the span. **The direction carries the routing.**

## Verdict
**CONTENT-MATTERS / SIGN-MATTERS.** The frozen hint-delta is not "a
perturbation at the address": a vector of the same size at the same
span, random or sign-reversed, neither repairs nor routes attention.
Combined with WX (address specificity) and WT (routing), the mechanism
sentence is complete: one learned direction, at the candidate's
mentions, in a mid-depth band, makes the answer position attend to
that span and adopt that candidate; nothing else of the same size does.

## Program tally after WV
24 registered directional predictions: **21 confirmed**, 3 not (13th,
14th, 19th). WikiHop chain W → … → WY → WV complete; ≈ $115 across 101
H100 jobs (≈ $25 lost to platform disk failures).
