# Item WA — anonymized-entity WikiHop: HINT-DELTA-TRANSFERS-WITHOUT-MEMORY + LOOP-CLOSES-WITHOUT-MEMORY (2026-09-02)

Registered: docs/causal_handle_directions.md item WA (12th directional
prediction). Frame: 800 new real rows (seed 20260832) → 536 anonymized
rows (seed 20260831; scripts/wikihop_anon_frame.py). Stage 1 **job-agh5n**;
stage 2 **job-b5etu** + **job-zcis4** (60 rows, two shards, seed 20260835,
~$4 total). Every pin frozen from W0 (real-text gauge, L30 site,
whole-word addressing). Row-level: `results/loop_screen/wikihop_wa_shard{0,1}.jsonl`
(5,844 records); reader `scripts/wikihop_wl_gates.py` → `docs/wikihop_wa_gates.json`.
Delivery audit valid on all 4,884 fired records; no candidate skipped.

## Stage 1 — memory removed
| quantity | anonymized (WA) | real frames (W0 / WF) |
|---|---|---|
| closed-book accuracy (CONTAMINATION CHECK, ceiling 0.15) | **0.127** (chance ≈ 0.064) | 0.457 / 0.445 |
| std accuracy | 0.378 | 0.463 / 0.466 |
| hint-first accuracy | 0.534 | — / 0.621 |
| doc-dependent failing | 288 of 536 | 299 / 276 of 800 |
| hint-repairable ∧ doc-dependent | 68 (23.6%) | 27.5% / 21.4% |
| memory-driven share among pinned rows | 17% | 46% / 34% |

## Stage 2 gates (n = 60)
**(a) TEXT CEILING — PASS:** 0.921 vs baseline 0.052; dP +0.869 [+0.794, +0.935].

**(b) ORACLE — PASS at both rungs (the 12th prediction CONFIRMED):**

| rung | gold-write correct (k=4×60) | dP [CI] | gold under NON-gold writes | specificity [CI] | rows repaired (any / 4-of-4) |
|---|---|---|---|---|---|
| 1.0 | 0.417 | +0.365 [+0.256, +0.477] | 0.024 | +0.393 [+0.279, +0.512] | 29 / 20 |
| **2.0 (pinned)** | **0.554** | **+0.502 [+0.377, +0.625]** | 0.036 | **+0.518 [+0.391, +0.641]** | 37 / 32 |

Real text (WL / WF): +0.26 to +0.35. Without memory the write is
STRONGER, and the dose matters more (1× → 2×: +0.14). Gold |δ| per
position 4,768 (real ≈ 3,500–3,700): the hint about a novel name moves
the state more, and the write follows the row's own delta.

**(c) LOOP at 2× — PASS on all three readings, with a weaker selector:**

| reading | WA (anonymized) | WF (real) |
|---|---|---|
| gauge-select over all candidates (16.5 / 25.4 per row) | 0.237 | 0.229 |
| baseline / random branch / SC@8 | 0.052 / 0.109 / 0.067 | 0.017 / 0.044 / 0.017 |
| gauge-select − baseline | +0.185 [+0.094, +0.287] | +0.212 |
| gauge-select − random branch | +0.128 [+0.041, +0.226] | +0.185 |
| oracle → fraction recovered | **0.554 → 43%** | 0.364 → 63% |
| gold branch is argmax (chance) | 0.233 (0.119) | 0.237 (0.070) |
| selection signal | +1.12 [−0.14, +2.44] | +0.86 [+0.09, +1.64] |
| gauge picks gold on rows where the gold branch repairs | 11 of 32 | 10 of 22 |

## Verdict
**HINT-DELTA-TRANSFERS-WITHOUT-MEMORY (pinned 2×) + LOOP-CLOSES-WITHOUT-MEMORY.**
With parametric memory removed by construction, the lever is at its
strongest on natural text (+0.50, the same size as the InAbHyD oracle
repairs, +0.24 to +0.39, or larger) — the sandbox's novel-name regime is
the write's home, and familiarity with the entities was never what it
needed. The selector is the weak half again: the L38 gauge was fit on
real-entity text and separates repaired branches less well on pseudonym
text (43% of the oracle; selection signal CI straddles 0), which points
at refitting the gauge on anonymized rows as the obvious next step for
the loop, not the write. The frozen-direction rider (registered,
conditional on (b)) launches now, cross-fit within the 60 WA rows.

## Where the chain stands (all on Gemma-3-27B, L30 write, W0's L38 gauge)
| item | rows | gold write dP (best rung) | loop vs baseline | verdict |
|---|---|---|---|---|
| WL (real) | 53 | +0.297 | +0.245 | loop closes (11th prediction) |
| WF (real, fresh frame) | 59 | +0.347 | +0.212 | replicated |
| WX (real, frozen direction) | 59 | +0.360 | +0.157 | frozen write transfers |
| WA (anonymized) | 60 | **+0.502** | +0.185 | transfers without memory (12th prediction) |

## Rider (registered, conditional on (b)): the frozen direction within anonymized rows — FROZEN-WRITE-TRANSFERS + FROZEN-LOOP-CLOSES
Jobs **job-qmt3y** (A: test WA shard 1, donors shard 0) and **job-y2rmz** (B:
reverse), ~$2.2, cross-fit on the 60 WA rows, seed 20260836. Frozen
direction = donor mean gold hint-delta at L30, norm target = donor mean
per-position |δ| (4,035 / 4,649); |mean δ| retains 46% / 59% of the
per-position norm. Delivery audit valid on all 4,884 fired records.
Row-level: `results/loop_screen/wikihop_wa_frozen_{a,b}.jsonl`; reader
`--frozen` → `docs/wikihop_wa_frozen_gates.json`.

| reading | frozen (rider) | per-candidate (WA, same rows) |
|---|---|---|
| text ceiling | 0.915 | 0.921 |
| gold-ADDRESS write, 1× / 2× | 0.417 / **0.700** | 0.417 / 0.554 |
| dP, 1× / 2× | +0.367 [+0.263, +0.481] / **+0.650 [+0.535, +0.756]** | +0.365 / +0.502 |
| gold under non-gold addresses, 1× / 2× | 0.042 / 0.048 | 0.024 / 0.036 |
| specificity, 1× / 2× | +0.375 / **+0.652 [+0.537, +0.759]** | +0.393 / +0.518 |
| cosine(frozen direction, row's own gold δ) | **0.88 (median 0.92)** | — (real text WX: 0.68 / 0.76) |
| loop: gauge-select at 2× vs baseline / random | 0.254 vs 0.050 / 0.117 (+0.204 [+0.102, +0.317] / +0.137 [+0.034, +0.245]) | 0.237 vs 0.052 / 0.109 |
| oracle → fraction recovered | 0.700 → 36% | 0.554 → 43% |
| gold branch is argmax (chance 0.119) | 0.233 | 0.233 |

Reading: without memory the hint-delta is almost one universal direction
(cosine 0.88), and that direction, written at the gold address at 2× the
donors' per-position norm, repairs **70% of samples** on rows the model
got wrong 8/8 times — the largest repair in the whole program, above the
sandbox's oracle (+0.24 to +0.39). The same vector at a non-gold address
repairs 4.8%: the identity is the address. The selector remains the weak
half (36% of the oracle; the real-text gauge on pseudonym text).
