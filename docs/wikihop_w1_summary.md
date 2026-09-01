# Item W1 — WikiHop calibration: the write does not steer, the gauge still reads (2026-09-01)

Registered: docs/causal_handle_directions.md item W + the W0/W1 pre-launch
note. Job **job-yp5er** (mission wikihop-loop-port, H100, 931 s). 12 seeded
doc-dependent failing rows (seed 20260822), Gemma-3-27B bf16 HF pathway,
write = gold-mention L30 class-mean direction (|v| 558) added at every
whole-word mention position of the fired candidate, gauge = W0's L38
logistic read at the final prompt token under the write. Row-level:
`results/loop_screen/wikihop_w1.jsonl` (1,956 records); reader
`scripts/wikihop_w1_gates.py` → `docs/wikihop_w1_gates.json`.

## Delivery audit (three independent channels)
1. **Hook counters inside `generate`:** all 1,860 fired records show ≥1
   prefill application with the expected number of positions written
   (the L-series bug class is excluded by construction — a zero aborts
   the job).
2. **Gauge channel:** the L38 final-token gauge moves under the write —
   gold-branch score minus baseline +0.78 / +1.36 / +1.52 at 0.25 / 0.5 /
   1.0× base; non-gold +0.19 / +0.21 / +0.77.
3. **Output channel:** at the literal top rung (32.6k per position) the
   output distribution shifts away from baseline (fingerprint lift −0.11,
   CI excludes 0) — the write reaches generation.

## Gates
| rung (× base 4,737) | per-position norm | gold correct (k=4×12) | baseline | non-gold answers-fired | fingerprint lift [CI] | positive control |
|---|---|---|---|---|---|---|
| 0.25 | 1,184 | 0.021 | 0.000 | 0.111 | 0.000 [0, 0] | fail |
| 0.5 (per-candidate pass) | 2,368 | 0.000 | 0.000 | 0.050 | +0.002 [0, +0.005] | fail |
| 1.0 | 4,737 | 0.000 | 0.000 | 0.056 | −0.056 [−0.139, 0] | fail |
| 1.72 (literal 0.25×) | 8,158 | 0.000 | 0.000 | 0.000 | −0.111 [−0.194, −0.028] | fail |
| 3.44 (literal 0.5×) | 16,315 | 0.000 | 0.000 | 0.000 | −0.111 [−0.194, −0.028] | fail |
| 6.89 (literal 1.0×) | 32,630 | 0.000 | 0.000 | 0.000 | −0.111 [−0.194, −0.028] | fail |

**(b) POSITIVE CONTROL: FAIL at every rung.** One gold sample in 288 is
correct. The non-gold "answers-fired" hits are all candidates the model
already answers at baseline (lift 0); at the literal doses even those
disappear. Outputs remain the baseline modal answer 85–94% of the time at
every dose — no degeneration, no steering.

**(c) SELECTION SIGNAL: PASS.** At 0.5×, gold-branch gauge minus mean
non-gold gauge = **+0.87 [+0.19, +1.60]** (row bootstrap, n = 12);
argmax-gold 3/12 (chance ≈ 1/18); gold gauge rank ≤ 2 in 6/12 rows.
Descriptive gauge-select dP = 0 (nothing to select among: no branch
answers differently).

## Reading
The InAbHyD lever site is **readable but not steerable** on WikiHop: the
class-mean write at candidate mentions changes what the L38 gauge sees
(gold more than non-gold — the read half of the loop transfers), but the
sampled answer is pinned to the model's baseline commitment across a
28× range of doses (2× the natural class-mean difference up to 7× the
full state norm). The baseline is extremely committed (95.8% of baseline
samples equal the modal answer; the modal answer is never gold on these
rows), which is the regime the screening selected for and the regime where
the InAbHyD write worked — so the null is not a power artifact of the
baseline (MDE for gold repair ≈ 0.10 at n = 12 × k = 4; observed 0.021).

## Consequence (pre-named)
No rung can be pinned for W2. The registered fallback runs next: a mini
layer sweep at write layers {20, 25, 35, 40} at the middle rung (0.5× that
layer's massive-dim-excluded per-position norm), same rows, same donor
recipe per layer, L30 re-run as a within-job anchor. If no layer passes the
positive control, the landed verdict is **W-WRITE-DOES-NOT-TRANSFER** and
W2 does not launch. Registry: the W1 gate-outcome note.
