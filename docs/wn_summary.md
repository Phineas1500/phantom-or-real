# Item WN — the aimer bake-off on a haystack of tables: can a label-free signal find the row the model needs? (37th prediction CONFIRMED at the bar; the retrieval heads are also a failure detector) (2026-09-06)

Registered: docs/causal_handle_directions.md item WN (before any data).
Rows: from the WM stage-1 rows, all 90 fixable failures (0/8 std, hint
≥ 4/8), 90 seeded unfixable failures (0/8 both ways), 90 seeded
correct-majority rows; two shards (**job-c7iqm / job-8tcs9**, $3.04;
a first pair died on `python` vs `python3`, $0.10). Gemma-3-27B with
eager attention. Per row: one forward capturing the final prompt
token's attention (per head on the ten global layers 5, 11, …, 59;
head means on all 62), one 64-token greedy self-selection, two
teacher-forced forwards for contrastive perplexity. Job
`scripts/tables_aimers_job.py`, reader `scripts/tables_aimers_reader.py`,
outputs `docs/wn_gates.json`, `results/loop_screen/wn_ranks.jsonl`.

## Why
WM showed that on 5k-token documents of tables 40% of the failures are
attention lapses the text hint repairs (0.89), but the keyword
retriever finds the gold row for only 23% of them. Every aimer here is
label-free at use and comes from the model itself: its own span
selection (AutoPASTA), its attention from the answer position, raw and
corrected for positional bias (Found-in-the-Middle), the attention of
its retrieval heads (Wu et al.), and question-conditioned contrastive
perplexity (LongLLMLingua). Gemma-3 alternates five sliding-window
layers (1,024 tokens) with one global layer, so only the ten global
layers can attend across the document.

## Gold-row recall, per aimer and stratum (recall@1 / recall@3)
| aimer | fixable failures (90) | unfixable failures (90) | correct rows (90) |
|---|---|---|---|
| BM25 rows within the top table (WM's aimer) | 0.09 / 0.23 | 0.13 / 0.26 | 0.39 / 0.63 |
| self-selection (copy the row; token F1 match) | 0.27 / 0.36 | 0.29 / 0.41 | **0.96 / 0.98** |
| raw attention, global layers, head mean | 0.19 / 0.37 | 0.24 / 0.39 | 0.82 / 0.92 |
| calibrated attention (÷ frame positional profile) | 0.19 / 0.41 | 0.20 / 0.33 | 0.81 / 0.90 |
| attention, all-62-layer mean | 0.00 / 0.04 | 0.00 / 0.04 | 0.01 / 0.03 |
| **retrieval heads (16, cross-fit)** | **0.21 / 0.52 [0.42, 0.62]** | 0.20 / 0.49 | 0.88 / 0.94 |
| contrastive perplexity | 0.01 / 0.02 | 0.00 / 0.00 | 0.07 / 0.09 |
| union of the model aimers @3 | 0.67 | 0.59 | 0.99 |

Retrieval heads on the fixable failures: recall@5 0.73, @10 0.91.

**37th prediction CONFIRMED — A-LABEL-FREE-AIMER-FINDS-THE-ROW**, at
the bar: 0.522 ≥ 0.50 and the interval excludes 0.23, as registered;
the interval straddles 0.50 itself. Aiming is half a wall: on half the
failures the hint repairs, the model's own retrieval circuit already
ranks the needed row in its top three, where the keyword retriever
managed a quarter.

## The retrieval heads
- Stable across shards: 14 of 16 heads coincide, all in global layers
  35–53 (41: 9, 47: 12, 53: 8, 35: 3 by count over both sets). Head
  (47, 0) alone puts 0.75 / 0.62 of its attention on the gold row on
  the correct rows of shards A / B.
- Per-layer head means, recall@3 fixable / correct: L47 0.46 / 0.94,
  L59 0.39 / 0.87, L53 0.38 / 0.90, L35 0.32 / 0.80, L29 0.13 / 0.41,
  L5–L23 ≤ 0.04 / 0.11. Averaging over all 62 layers destroys the
  signal: the 52 sliding-window layers cannot see the row.
- On failures the heads still place 0.14 (fixable) / 0.15 (unfixable)
  of their attention on the gold row, against 0.48 on correct rows.

## Self-selection follows the answer
The model copies a verbatim row (F1 ≥ 0.9 with some row) 86% / 79% /
87% of the time, but the row it names is the gold row 0.27 / 0.28 /
0.96. Asked which row supports the answer, it reports the row behind
the answer it already gave. Among the fixable rows, retrieval heads@3
alone cover 28, self-selection@1 alone 5, both 19, neither 38.

## Detector reading (descriptive, pre-named)
| label-free signal | fixable | unfixable | correct | AUROC failure vs correct | fixable vs correct | fixable vs unfixable |
|---|---|---|---|---|---|---|
| retrieval heads' peak share on one row (low = failure) | 0.28 | 0.34 | 0.49 | **0.76** | **0.80** | 0.62 |
| entropy of answer-position attention over rows | 3.94 | 3.89 | 3.83 | 0.60 | 0.63 | 0.57 |
| attention mass on the documents | 0.389 | 0.390 | 0.373 | 0.62 | 0.62 | 0.51 |

The confidence of the retrieval circuit — how much of its attention
lands on its single favourite row — separates the model's 0/8
failures from its majority-correct rows at 0.76–0.80 with no label,
no probe and no second generation. Caveat: these strata are the easy
extremes of the frame, not its mixed distribution; a frame-level rule
would have to be registered on a blind draw.

## What this changes
WM's null had two candidate causes: the aimer and the write. The
aimer half is now half-open — the model's own retrieval heads aim at
0.52 / 0.73 (top 3 / top 5) on fixable failures. The write half stays
closed as built: WM showed the residual write corrupts a row (−0.42 at
the gold row) and is inert at the answer's tokens. The admissible step
2 is an intervention on a fresh blind draw, aimed by the retrieval
heads, using attention steering (a PASTA-style bias toward the aimed
rows in the global layers) rather than the residual write, against
quoting the aimed rows and the text hint, with the peak-share detector
as the gate of a two-stage rule. Not launched.

## Program tally after WN
37 registered directional predictions: **27 confirmed**, 9 not, 1
intermediate; ≈ $177 across 161 H100 jobs.
