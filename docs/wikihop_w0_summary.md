# Item W0 — WikiHop grading + capture landed; pins for W1 (2026-09-01)

Registered: docs/causal_handle_directions.md item W (2026-08-19). Mission
https://givemeanode.com/missions/wikihop-loop-port (H100 lane, Gemma-3-27B
bf16). Frame: 800 fresh WikiHop dev rows, seed 20260821, docs ≤ 14,000 chars,
screening rows excluded (`results/loop_screen/wikihop_port_input.jsonl.gz`).
Prompts byte-identical to the screening's std/closed constructions
(`scripts/wikihop_common.py`, verified against `scripts/wikihop_w0_jobs.py` on
all 800 rows).

Jobs: W0a grade **job-73kjw** (vLLM, std k=8 + closed k=8, 12,800
generations, seed 20260821, 975 s). W0b capture **job-a2zut** (v5, the pin:
HF forward per row, float32, whole-word mention addressing). Superseded
captures: job-ren8t (v3: float16 store overflowed — one non-finite entry each
in `L53_final` and `cand_L30`, finite maxima at the float16 ceiling) and
job-m5qdr (v4: float32 but substring addressing). The 2026-08-19 launch
(job-uch77 / job-s3afv) failed on the gated-repo 401; fixed via the org
`hf-token` secret in the job env.

## Scoring (pinned)

Strong label = **normalized exact match** to the gold candidate
(`wikihop_common.normalize_answer`: first line, NFKC, lowercase, leading
bullets/quotes and trailing quotes/asterisks/periods stripped, whitespace
collapsed). Disclosure: the screening's headline 0.523 reproduces exactly
only under *containment* (gold ⊂ output); exact match gives 0.520 on the
screening rows. Item W uses exact match as registered; containment numbers
are reported alongside. Std outputs are 94.3% well-formed (== some listed
candidate), closed 99.8%; the ill-formed remainder is mostly years/dates
answered in place of a candidate and scores as incorrect.

## Behavior (final, from job-73kjw)

| quantity | exact | containment |
|---|---|---|
| std accuracy (800 rows × 8) | **0.463** | 0.467 |
| closed-book accuracy | **0.457** | 0.464 |
| rows failing (<0.5) | 53.3% | — |
| any-correct@8 on failing (<0.5) rows | 0.026 | — |
| rows at 0/8 std (FAILING) | **415** | — |
| doc-dependent failing (0/8 std ∧ 0/8 closed) | **299** (≥150 ✓) | — |

Labels are almost deterministic at temperature 0.7: std correct-count
histogram 0/8 → 415 rows, 8/8 → 350, only 35 rows in between (3 ties at
4/8, excluded from the gauge fit). Std-majority × closed-majority: both pass
254, doc-helps (std pass, closed fail) 116, doc-hurts (std fail, closed
pass) 110, both fail 313. Contamination is heavier than the screening
suggested (closed ≈ std): parametric memory answers ~46% of this frame
outright, so every intervention claim stays scoped to the 299-row
doc-dependent pool. Within that pool, 129 rows commit to the same wrong
answer with and without documents (memory-driven misses) and 170 commit to
a different wrong answer with documents (reading-driven misses) —
descriptive, reported for the W2 slice analysis.

## Gauge (final — final-token states are identical across capture versions)

Logistic (centered, liblinear C=1.0) on final-token states vs std-majority
labels, 5-fold stratified CV (seed 20260821), n = 797:

| layer | CV AUC |
|---|---|
| **L38** | **0.776** (primary = argmax) |
| L43 | 0.775 |
| L48 | 0.757 |
| L53 | 0.750 |

**NATURAL GATE: PASS** (0.776 ≥ 0.65). Out-of-fold AUC restricted to
doc-dependent-failing vs correct-majority rows: 0.790. The gauge layer (L38)
is strictly downstream of the write layer (L30), as the L/N lessons require.
The primary layer differs from Gemma-InAbHyD's late-layer optimum (L53): on
this data the separability peaks earlier and declines monotonically.

## Write (v5 pin)

Class-mean of the gold-candidate mention-mean L30 state, correct-majority
minus incorrect-majority donors, donors = graded rows OUTSIDE the
doc-dependent pool, balanced by seeded subsampling (seed 20260821):
127 per class (371 correct-majority available, 127 incorrect-majority
available outside the pool). |class vector| = 558; cosine with the L38 gauge
weight 0.03. Mention = case-insensitive whole-word span match via the
tokenizer's offset mapping (v3/v4 substring matching hit sub-word spans —
'hawaii' in 'hawaiian', 'ship' in 'relationship' — for 104/800 gold
candidates and 22% of all candidates; no gold candidate loses all mentions
under whole-word matching).

## Amplitude base — measured, and the operationalization pinned before W1

The registration defines the base as the mean per-position L30 state norm.
Measured at candidate-mention positions (v5 records per-position norms and
per-row per-dim mean squares) it is **32,630**, but 98.0% of the squared
norm sits in two massive-activation dimensions (104 and 2733; per-dim RMS
≈ 27,000 and 18,000 vs ≤ 600 for every other dimension). The class-mean
write direction lives in the other 5,374 dimensions (its components on
104/2733 are ~20% of its own norm, i.e. 0.4% of those dims' magnitude).
Excluding the two massive dims, the per-position content-scale RMS norm is
**4,737** — within 1.3× of the anchor dose the closed loop used at Gemma
L30 (`fixed_norm_target` 3,708 in F(ii)-c / the L-series), which the pinned
ladder brackets. (The same quantity estimated from the mention-MEAN vectors
is 3,470; position-specific variance cancels in the mean, so the per-
position measurement is the one pinned.) Read literally, the registered
ladder {0.25, 0.5, 1.0} × 32,630 = {8.2k, 16.3k, 32.6k} would start at 2.2×
the proven dose and top out at 8.8×.

**Pinned (pre-data):** base = 4,737 (massive-dim-excluded), ladder
{1,184, 2,368, 4,737}. The literal rungs {8.2k, 16.3k, 32.6k} are ALSO run in
W1 as descriptive extra rungs (multiples 1.72 / 3.44 / 6.89 of the pinned
base), so the registered reading is measured rather than replaced. The
positive-control gate is read on the pinned ladder; if only a literal rung
passes, that is reported as the literal-ladder branch and re-registered
before W2.

## W1 draw (seed 20260822, pinned)

WH_dev_1021, WH_dev_1499, WH_dev_17, WH_dev_1854, WH_dev_1931, WH_dev_194,
WH_dev_2623, WH_dev_2892, WH_dev_3676, WH_dev_4895, WH_dev_583, WH_dev_893
(213 candidates total, mean 1,559 prompt tokens). W2 pool = the remaining
287 doc-dependent rows.

Artifacts: `results/loop_screen/wikihop_w0_grades.jsonl` (row-level
generations), `results/loop_screen/wikihop_w0_rows.jsonl` (scored rows +
pools), `results/loop_screen/wikihop_w0_pinned.npz` (gauge w/b/mean, class
vector, base), `docs/wikihop_w0_pinned.json` (all pins + donor ids),
`results/loop_screen/wikihop_w0_capture.npz` + manifest (v5 capture,
not committed — 423 MB; npz sha256 7661f25c…, manifest bc1a107f…; job
artifact art-dacis).
