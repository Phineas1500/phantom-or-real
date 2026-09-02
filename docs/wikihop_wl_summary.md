# Item WL — the closed answer-free loop on natural data: HINT-DELTA-TRANSFERS + LOOP-CLOSES-ON-NATURAL-DATA (2026-09-02)

Registered: docs/causal_handle_directions.md item WL (11th directional
prediction). Jobs **job-g74tz** (shard 0, 27 rows, 2,735 s) and
**job-wq2vj** (shard 1, 26 rows, 2,189 s), ~$4. All 53 hint-repairable rows
of the W2 pool not used by WR or WH (no sampling), Gemma-3-27B bf16, write
at L30, W0's L38 gauge, seed 20260826. Row-level:
`results/loop_screen/wikihop_wl_shard{0,1}.jsonl` (7,536 records);
reader `scripts/wikihop_wl_gates.py` → `docs/wikihop_wl_gates.json`.
Delivery audit valid on all 6,688 fired records (hook counters inside
`generate`, gauge forward hooked); mention pairing succeeded for all
1,672 fired candidates.

## Gates
**(a) TEXT CEILING — PASS.** In-job hint-first gold text rate 0.962 vs
baseline 0.009; dP +0.953 [+0.903, +0.991] (n = 53).

**(b) REPLICATION (the 11th prediction) — PASS at both rungs.**

| rung (× raw δ) | gold-write correct (k=4×53) | dP vs baseline [CI] | gold correct under NON-gold writes | specificity [CI] | rows repaired (any / 4-of-4) |
|---|---|---|---|---|---|
| **1.0 (pinned)** | **0.307** | **+0.297 [+0.189, +0.420]** | 0.030 | **+0.277 [+0.173, +0.388]** | 19 / 14 |
| 2.0 | 0.274 | +0.264 [+0.151, +0.387] | 0.029 | +0.244 [+0.135, +0.361] | 16 / 14 |

WR's +0.25/+0.30 replicates on a disjoint draw with the delivery gate
pre-named this time; 1× vs 2× per row +0.033 [−0.052, +0.127] (no dose
difference; 1× pinned by the larger-dP rule). Non-gold fingerprint
(descriptive): answers-fired 0.095 at 1× (158 fires), 0.044 at 2× (1,408
fires).

**(c) LOOP at 2× — PASS on both readings.** For every candidate X
(mean 27.6 per row), δ_X from a prompt naming X, written at X's
mentions; the W0 gauge read on each branch; generate from the argmax.
No gold label used anywhere in the loop.

| reading | value |
|---|---|
| gauge-select correct rate | **0.255** |
| baseline (k=8) | 0.009 |
| self-consistency@8 (majority vote) | 0.000 |
| gauge-select − baseline | **+0.245 [+0.142, +0.363]** |
| gauge-select − SC@8 | **+0.255 [+0.146, +0.377]** |
| oracle (gold branch) at 2× | 0.274 → loop recovers 93% |
| gold branch is argmax | 20/53 = 0.377 (chance 0.071) |
| gold gauge rank 1 / top-3 | 20 / 27 of 53 (median rank 3 of 27) |
| selection signal (gold − mean non-gold gauge) | +1.56 [+0.65, +2.56] |

Mechanism, visible per row: of the 14 rows where the gold branch repairs
at 2×, the gauge picks gold on 10; on the 37 unrepaired rows it picks
gold on 8 (22%, above chance, irrelevant to accuracy). Loop-correct rows:
15 (12 via the gold branch, 3 via a non-gold branch that answered gold).
The gauge is not identifying the gold candidate; it is identifying the
branch whose state reads as correct, which is the gold branch exactly
when the write worked — the same mechanism as the sandbox loop (L″-r/L‴).

## Verdict
**HINT-DELTA-TRANSFERS (pinned rung 1×) + LOOP-CLOSES-ON-NATURAL-DATA**,
scoped to compatible-answer failures (hint-repairable: the model commits
to a coarser or adjacent true answer and can be talked into the finer
one — 27.5% of the doc-dependent pool; screening evidence in
docs/wikihop_wr_summary.md). What does NOT transfer, established by
W1/sweep: the class-mean direction (no candidate identity) and any
repair of competing commitments or memory misses.

Obligations opened by the PASS (registered separately before launch):
fresh-draw replication on a NEW 800-row WikiHop frame (new seed, new
screen); the answer-free amplitude/basis variant (rank-k from other
rows' deltas, L-series style) is optional — the loop as run is already
answer-free in the selection sense, with per-candidate deltas.

## Item W in one paragraph
W0: gauge reads correctness on WikiHop untuned (AUC 0.78). W1 + sweep:
the class-mean write at L20–40, 2× the natural difference to 7× the
state norm, never moves the answer (0/240 + 1/288) though the gauge
shifts — readable, not steerable, for that direction. WH: the
reading-driven filter is not the commitment-failure filter; the hint as
text fails on 10/12 such rows. WR: hint-repairable rows are 27.5% of
the pool, cut across memory/reading; the hint-delta write repairs
+0.25/+0.30 there. WL: replicated at +0.30/+0.26 on 53 fresh rows with
specificity, and the gauge-selected loop reaches 0.255 from 0.009 with
no label — 93% of the oracle. Total cost of the port: ~$11 across 18
H100 jobs.
