# Item WR — hint-delta oracle write on the HINT-REPAIRABLE slice (2026-09-02)

Registered: docs/causal_handle_directions.md item WR. Stage 1 screen
**job-iimvi**; stage 2 **job-q8nzz** (H100, 522 s, ~$0.45). 24 seeded
hint-repairable rows (seed 20260825; ≥4/8 correct under the hint-first
text prompt in stage 1; WH rows excluded), Gemma-3-27B bf16. Row-level:
`results/loop_screen/wikihop_wr.jsonl` (1,792 records); reader
`scripts/wikihop_wh_gates.py` → `docs/wikihop_wr_gates.json`. Delivery
audit valid on all 1,128 fired records; mention pairing succeeded for all
94 fired candidates.

## Stage 1 (screen of the 287-row W2 pool)
Hint-repairable 79/287 = 27.5%, sharply bimodal (196 rows at 0/8, 75 at
8/8, 16 between). 42 of 165 reading-driven rows and 37 of 122 memory-driven
rows — the memory/reading split does not predict text repairability. The
repairable memory-driven rows are mostly granularity cases (model answers
'paris' for '1st arrondissement of paris', 'japan' for 'niigata
prefecture', 'mammal' for 'paenungulata'); the unrepairable reading-driven
rows are competing commitments ('czech republic' vs 'czechoslovakia',
'2003' vs '2000'). The boundary is compatible-vs-competing, not
memory-vs-reading. Row-level: `results/loop_screen/wikihop_hint_grades.jsonl`.

## Stage 2 gates
**(a) TEXT CEILING — PASS.** In-job hint-first gold text rate **0.964** vs
baseline 0.000; dP +0.964 [+0.891, +1.000] (23/24 rows at or near 8/8;
WH_dev_1549 at 1/8). Non-gold text hints move the answer to the named
candidate 18.6% of the time (lift +0.094 [+0.021, +0.177]).

**(b) ORACLE WRITE — conjunction FAILS (fingerprint conjunct); gold conjunct PASSES at 1× and 2×.**

| rung (× raw δ) | gold correct (k=4×24) | dP [CI] | recovered fraction | non-gold answers-fired | fingerprint lift [CI] | gauge shift gold / non-gold |
|---|---|---|---|---|---|---|
| 1.0 | **0.250** | **+0.250 [+0.104, +0.417]** | 0.26 | 0.107 | +0.017 [0.000, +0.049] | +0.85 / +0.08 |
| 2.0 | **0.302** | **+0.302 [+0.156, +0.458]** | 0.31 | 0.154 | +0.056 [−0.028, +0.153] | +0.63 / +0.07 |
| 4.0 | 0.021 | +0.021 [0, +0.062] | 0.02 | 0.057 | −0.045 [−0.139, +0.049] | −0.10 / −0.12 |

Gold-δ writes repair a quarter to a third of samples at 1–2× (8 and 11 of
24 rows show repair; five rows go 4/4 at 2×), and the effect collapses at
4× (per-position norm ≈ 15k: over-dose, gauge shift turns negative). This
is the first CI-separated activation repair on natural data in item W
(W1 + sweep: 1/528; WH: one row). The non-gold fingerprint lift never
clears zero, so gate (b) as registered fails at every rung and the
**registered verdict is HINT-DELTA-DOES-NOT-TRANSFER-WHERE-TEXT-DOES.**

## Why the fingerprint conjunct failed, and what certifies delivery instead
The fingerprint (non-gold writes make the model answer the fired
candidate more) was carried over from W1 as the in-frame delivery check.
Its own text ceiling here is small (+0.094): the hint as text only moves
the answer to a named non-gold candidate ~19% of the time, so at the
write's ~30% recovery the expected lift is ~+0.03, below the registered
MDE (~0.05). The conjunct was mis-powered for this slice. The
SPECIFICITY control — the quantity the fingerprint stands in for —
is clean: gold accuracy under NON-gold writes is 0.014 (1×) and 0.029
(2×) vs 0.250 / 0.302 under gold writes (10×); non-gold writes leave the
output at the baseline modal answer 81–88% of the time. The write
carries the candidate's identity; it is not a generic "re-read" push.
Delivery is further audited by hook counters and the gold-specific
gauge shift (+0.85 vs +0.08). Per the amendment policy the registered
verdict stands as written; this section records why it under-reads the
data.

## Descriptive: the loop on natural data
The per-candidate δ_X is answer-free in the loop sense (δ_X comes from a
prompt naming X, for every X). Selecting among the 4 fired branches by
the W0 L38 gauge at 2× gives correct rate **0.167 [+0.052, +0.302]** from a
0 baseline (oracle branch 0.302); the gold branch is argmax in 9/24 rows
(chance 6/24); selection signal +0.59 [−0.35, +1.58] (weak at n=24 with
4 branches). Not registered in WR; reported for the next design.

## Implied next registration (NOT launched)
On the remaining 53 hint-repairable rows (79 − 24 WR − 2 WH): (i)
fresh-draw replication of the gold-δ write at {1, 2}× with the
SPECIFICITY control as the pre-named delivery gate (gold accuracy under
gold writes minus under non-gold writes, CI > 0) replacing the
mis-powered fingerprint; (ii) the answer-free loop over ALL candidates
(δ_X for every X, gauge-select at 2×) with the registered readings
gauge-select dP CI > 0 and paired gauge-minus-self-consistency@8 — the
W2 composition, on the slice where the write demonstrably works. ~$4.
