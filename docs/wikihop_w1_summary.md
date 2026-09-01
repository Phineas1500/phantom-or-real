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

## Fallback layer sweep (job-erwpa) — FAIL at every layer; verdict lands

Pre-named in the W registration, re-registered 2026-09-01 (late) before any
sweep data. Same 12 rows, write at the middle rung 0.5 × base_L of each
layer's massive-dim-excluded per-position norm, class-mean from the same
127/127 donors, gold k=4 + the same 3 seeded non-gold candidates k=4 (paired
with W1), fresh k=8 baselines per pass, gauge at L43 (downstream of every
write). Row-level: `results/loop_screen/wikihop_w1_sweep.jsonl`; gates:
`docs/wikihop_w1_sweep_gates.json`.

| write layer | dose (norm) | gold correct (k=4×12) | baseline | non-gold answers-fired | fingerprint lift [CI] | gauge Δ gold / non-gold | output = baseline modal (gold / non-gold fires) |
|---|---|---|---|---|---|---|---|
| L20 | 711 | 0/48 | 0/96 | 0.090 | −0.021 [−0.062, 0] | +0.39 / +0.23 | 0.94 / 0.92 |
| L25 | 1,247 | 0/48 | 0/96 | 0.111 | 0.000 [0, 0] | +0.56 / +0.27 | 0.94 / 0.94 |
| L30 (anchor) | 2,368 | 0/48 | 0/96 | 0.111 | 0.000 [0, 0] | +1.69 / +0.77 | 0.94 / 0.91 |
| L35 | 3,483 | 0/48 | 0/96 | 0.083 | −0.028 [−0.083, 0] | +1.20 / +0.49 | 0.94 / 0.91 |
| L40 | 5,502 | 0/48 | 0/96 | 0.069 | −0.042 [−0.111, 0] | +0.24 / +0.10 | 0.94 / 0.90 |

Delivery audit valid at every layer (240/240 fired branches hooked inside
`generate`; gauge moves under every write). Determinism check: the five
fresh k=8 baselines are identical across passes and identical to W1's
(12/12 rows) — the seeded HF sampler reproduces exactly, so every steered
branch is a paired comparison against the same baseline draw.

## Landed verdict — W-WRITE-DOES-NOT-TRANSFER

Across 5 write layers (L20–L40) and, at L30, six doses from 2× the natural
class-mean difference to the full residual norm, the mention-position
class-mean write repaired **1 of 528 gold-fire samples** (W1 288 + sweep
240; the one hit at L30 0.25×) on doc-dependent WikiHop failures, and never
lifted a fired non-gold candidate's answer rate above baseline. **MDE:** a
per-sample repair rate ≥ 0.0125 (rule-of-three bound on 0/240 in the sweep;
≥ 0.061 per layer on 0/48) and a non-gold fingerprint lift ≥ 0.06 (the
widest CI half-width) would have been detected; the InAbHyD lever's oracle
effects (+0.24 to +0.39) and fingerprint (+0.07) lie far above both bounds.
The 10th registered prediction (the L30 lever site transfers to natural
data) does not survive its positive control; **W2 is not run** (its oracle
gate presupposes a passing write site). Scope sentence for the paper: the
closed loop's write half is sandbox-specific at the tested sites and doses;
its gauge/selection half transfers (natural gate 0.776, selection signal
+0.87 CI > 0). The registration's amendment policy was not invoked: no
result was voided; the capture corrections (float16 → float32, substring →
whole-word addressing) and the amplitude-base operationalization were made
before the corresponding data, on independent channels, and are recorded.

## Independent replicate of the sweep (job-32zfc + job-zjr7q, capture job-pepj4)

A second session ran the same pre-named sweep from an independent capture
(pins identical to rounding) with W1's seeded non-gold candidates and the
L38 gauge (L48 for the L40 write). Same reading at every layer:

| write layer | amplitude (0.5× base) | gold correct (k=4×12) | non-gold fingerprint lift [CI] | outputs == baseline modal | gauge shift gold / non-gold |
|---|---|---|---|---|---|
| L20 | 711 | 0/48 | −0.021 [−0.062, 0] | 0.922 | +0.29 / +0.12 |
| L25 | 1,247 | 0/48 | 0.000 [0, 0] | 0.938 | +0.26 / +0.15 |
| L30 | 2,368 | 0/48 | 0.000 [0, 0] | 0.917 | +1.36 / +0.40 |
| L35 | 3,483 | 0/48 | −0.028 [−0.083, 0] | 0.917 | +1.46 / +0.50 |
| L40 | 5,502 | 0/48 | −0.042 [−0.111, 0] | 0.911 | +0.11 / +0.30 (L48 gauge) |

Delivery audit valid on all 240 branches. The verdict above stands
unchanged; this is a within-day replication of the null with independent
seeds and a different gauge read (`docs/wikihop_sweep_gates.json`).
