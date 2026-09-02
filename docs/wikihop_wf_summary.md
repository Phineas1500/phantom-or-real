# Item WF — fresh-frame replication: LOOP-CLOSES-ON-NATURAL-DATA, REPLICATED (2026-09-02)

Registered: docs/causal_handle_directions.md item WF (the obligation opened
by WL). Fresh 800-row WikiHop dev frame, seed 20260827, disjoint from W0's
frame and the screening (builder `scripts/wikihop_frame.py`, verified to
reproduce W0's frame byte-for-byte). Stage 1 **job-954p4** (std + closed +
hint-first, k=8, seed 20260828); stage 2 **job-87kjh** + **job-jgeug** (59
rows, two shards, seed 20260829, ~$3.5). Every pin FROZEN from W0: the L38
gauge, the L30 write site, whole-word addressing — nothing refit.
Row-level: `results/loop_screen/wikihop_wf_shard{0,1}.jsonl` (7,876
records); reader `scripts/wikihop_wl_gates.py` → `docs/wikihop_wf_gates.json`.
Delivery audit valid on all 6,932 fired records; no candidate skipped.

## Stage 1 (the fresh frame behaves like W0's)
| quantity | WF (fresh) | W0/WR (original) |
|---|---|---|
| std / closed accuracy | 0.466 / 0.445 | 0.463 / 0.457 |
| rows at 0/8 std → doc-dependent (∧ 0/8 closed) | 409 → 276 | 415 → 299 |
| hint-first on the doc-dependent pool | bimodal: 211 at 0/8, 50 at 8/8 | bimodal: 196 / 75 |
| hint-repairable ∧ doc-dependent | **59 (21.4%)** | 79 (27.5%) |
| memory- / reading-driven among them | 20 / 39 | 37 / 42 |

## Stage 2 gates (n = 59)
**(a) TEXT CEILING — PASS:** 0.939 vs baseline 0.017; dP +0.922 [+0.858, +0.975].

**(b) REPLICATION — PASS at both rungs (the 11th prediction, second disjoint draw, frozen recipe):**

| rung | gold-write correct | dP [CI] | gold under NON-gold writes | specificity [CI] | rows repaired (any / 4-of-4) |
|---|---|---|---|---|---|
| 1.0 | 0.326 | +0.309 [+0.199, +0.424] | 0.018 | +0.308 [+0.198, +0.422] | 23 / 18 |
| 2.0 (pinned here) | 0.364 | +0.347 [+0.220, +0.479] | 0.023 | +0.341 [+0.216, +0.471] | 24 / 19 |

WL: +0.297 / +0.264. Across the two draws the two rungs are
indistinguishable (WL pinned 1×, WF 2×); the write works in the 1–2×
window and collapses at 4× (WR).

**(c) LOOP at 2× — PASS on both readings:**

| reading | WF (fresh, n=59) | WL (n=53) |
|---|---|---|
| gauge-select correct (all candidates; 25.4 vs 27.6 branches/row) | **0.229** | 0.255 |
| baseline / SC@8 | 0.017 / 0.017 | 0.009 / 0.000 |
| gauge-select − baseline | **+0.212 [+0.119, +0.314]** | +0.245 [+0.142, +0.363] |
| gauge-select − SC@8 | **+0.212 [+0.119, +0.314]** | +0.255 [+0.146, +0.377] |
| oracle (gold branch) | 0.364 → 63% recovered | 0.274 → 93% recovered |
| gold branch is argmax | 14/59 = 0.237 (chance 0.070) | 20/53 = 0.377 (chance 0.071) |
| selection signal (gold − mean non-gold gauge) | +0.86 [+0.09, +1.64] | +1.56 [+0.65, +2.56] |
| gauge picks gold on rows where the gold branch repairs | 10 of 22 | 10 of 14 |

## Post-hoc validation (2026-09-02, after landing; descriptive): the meter, not the nudges' base rate
Expected accuracy of a randomly chosen branch (mean over each row's
branches) vs the gauge's choice, on the same records:

| draw | gauge-select | random branch | oracle | gauge − random [CI] |
|---|---|---|---|---|
| WL (53) | 0.255 | 0.045 | 0.274 | +0.210 [+0.112, +0.320] |
| WF (59) | 0.229 | 0.044 | 0.364 | +0.185 [+0.096, +0.281] |

The gauge's edge over random selection does not shrink with the number
of candidates: on the tercile of rows with ~45 candidates, gauge-select
is 0.276 (WL) / 0.298 (WF) vs random 0.012 / 0.016. Registered as a
reading in the next item (WX).

## Verdict
**HINT-DELTA-TRANSFERS (both rungs) + LOOP-CLOSES-ON-NATURAL-DATA, REPLICATED.**
The write half is now confirmed on two disjoint draws (53 + 59 rows) with
a frozen recipe and a pre-named specificity gate; the loop half passes both
registered readings on both draws. The honest texture: the write is the
stable half (+0.26 to +0.35 on every draw and rung), the selector the
variable half — it recovers 93% of the oracle on one draw and 63% on the
other, with the gold branch ranked first in 24–38% of rows against a 7%
chance rate. Scope unchanged: compatible-answer (hint-repairable)
failures, 21–28% of doc-dependent WikiHop failures. The registered
fresh-draw obligation is discharged; no further replication is owed on
WikiHop.

## Item W chain, final
| item | verdict | rows |
|---|---|---|
| W0/W1/sweep | W-WRITE-DOES-NOT-TRANSFER (class-mean direction; gauge transfers) | 12 (+ sweep) |
| WH | NO-CEILING (reading-driven filter ≠ commitment-failure filter) | 12 |
| WR | gold hint-delta +0.25/+0.30 CI>0; conjunction failed on a mis-powered fingerprint | 24 |
| WL | HINT-DELTA-TRANSFERS + LOOP-CLOSES-ON-NATURAL-DATA (11th prediction) | 53 |
| WF | REPLICATED on a fresh frame, frozen recipe | 59 |
Total: ~$15 across 21 H100 jobs.
