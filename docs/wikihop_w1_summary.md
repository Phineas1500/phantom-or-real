# Item W1 — calibration landed: the L30 write does not move WikiHop answers at any dose; the gauge still separates the gold branch (2026-09-01)

Registered: docs/causal_handle_directions.md item W + the W1 pre-launch note.
Job **job-yp5er** (mission wikihop-loop-port, H100, 931 s). 12 seeded
doc-dependent failing rows (seed 20260822), in-job k=8 baselines, gold fires
k=4 at six rungs (pinned {0.25, 0.5, 1.0} × 4,737 and literal {1.72, 3.44,
6.89} × 4,737 = {8.2k, 16.3k, 32.6k}), 3 seeded non-gold fires k=4 at every
non-middle rung, full per-candidate pass k=4 at the middle rung (0.5× =
2,368; 213 candidates). Write at L30 on every whole-word mention position
of the fired candidate; gauge read at L38 under the write. Row-level:
`results/loop_screen/wikihop_w1.jsonl` (1,860 fired records + 96 baseline);
gates: `docs/wikihop_w1_gates.json` (`scripts/wikihop_w1_gates.py`).

## Delivery audit (independent channels)

| channel | reading |
|---|---|
| hook counters inside `generate` | 1,860/1,860 fired records: ≥1 prefill application, positions written = mention positions; gauge forward hooked on every branch |
| gauge shift under the write (L38, final token) | gold branch +0.78 / +1.36 / +1.52 / +0.77 / +1.21 / −0.29 vs baseline across the six rungs; non-gold +0.19 … +1.38 |
| output channel | outputs differ from the baseline modal answer in 6–15% of samples; at the literal rungs the fingerprint lift is −0.111 [−0.194, −0.028] (the fired non-gold candidate is answered *less* than at baseline) |

The write reaches the late layers and perturbs generation; the null below is
not a hook-scope failure.

## Gate (b) — POSITIVE CONTROL: FAIL at every rung

| rung (norm) | gold correct (k=4) | baseline | dP gold [CI] | non-gold answers-fired | fingerprint lift [CI] |
|---|---|---|---|---|---|
| 0.25× (1,184) | 0.021 | 0.000 | +0.021 [0, +0.062] | 0.111 | 0.000 [0, 0] |
| 0.5× (2,368) | 0.000 | 0.000 | 0 | 0.050 | +0.002 [0, +0.005] |
| 1.0× (4,737) | 0.000 | 0.000 | 0 | 0.056 | −0.056 [−0.139, 0] |
| 1.72× (8.2k) literal | 0.000 | 0.000 | 0 | 0.000 | −0.111 [−0.194, −0.028] |
| 3.44× (16.3k) literal | 0.000 | 0.000 | 0 | 0.000 | −0.111 [−0.194, −0.028] |
| 6.89× (32.6k) literal | 0.000 | 0.000 | 0 | 0.000 | −0.111 [−0.194, −0.028] |

One gold hit in 288 gold-fire samples. The non-gold "answers-fired" rate at
the low rungs is entirely the candidates the model already answers at
baseline (lift = 0): firing a candidate never makes the model say it.
Baseline texture: the modal answer carries 95.8% of baseline samples, is a
listed candidate in 9/12 rows, and is never gold; under the write the
output stays that modal answer in 85–94% of samples at every rung, including
32.6k. Gemma-3-27B's WikiHop commitment is not addressable by the
mention-position class-mean write at this site, at doses from 2× the natural
class-mean difference up to the full state norm.

## Gate (c) — SELECTION SIGNAL: PASS

Gold-branch minus mean non-gold gauge score at the middle rung: **+0.87
[+0.19, +1.60]** (n = 12); argmax-gold 3/12 (candidate lists average 18);
gold gauge rank ≤ 2 in 6/12 rows. The gauge reads the addressed candidate's
correctness from the steered state even though the sampled answer does not
change — the readable half transfers, the steerable half does not.
Descriptive gauge-select dP = 0 (no branch of any row is ever correct).

## Registered consequence

No rung passes (b) → the pre-named fallback applies: one mini layer sweep
{20, 25, 35, 40} at the middle rung, re-registered before W2 (registry note
of 2026-09-01, late). If that also fails, **W-WRITE-DOES-NOT-TRANSFER** is
the landed verdict for the 10th prediction's lever-site bet, with the MDE
stated. The selector is NOT demoted (c passed), which matters only if a
write site is found.
