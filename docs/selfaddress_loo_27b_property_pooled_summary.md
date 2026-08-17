# Item L″-r — THE COMPOSITION PASSES: the answer-free repair loop closes (gauge-select +0.241, == oracle behaviorally), and the C4 rider lands FROZEN-TRANSFERS (+0.498) — the frozen artifact generalizes to fresh rows after all

Corrected rerun, jobs 461886–461889 (C0–C3) + 461890 (C4 gold-only
rider), Aug 14–16 (queue-stretched), all COMPLETED under 3:00:00
walls. Pooled by `scripts/stage2_selfaddress_loo_pool.py`
(`docs/loo_r_20260816/` holds the archived row-level record; the
first, execution-invalid run is `docs/invalid_20260814/` + the
registry CORRECTION entry — its history is summarized at the end).

## Gates, in registered order — all PASS

| gate | value | status |
|---|---|---|
| baseline verbatim vs L1 | 456/456 token-identical | PASS |
| delivery (NEW hard gate): gold-branch targets-fired lift over baseline, row-paired | +0.180 [+0.088, +0.276]; rate 0.575 vs 0.395 natural | PASS |
| oracle gate: gold-branch dP | **+0.263 [+0.169, +0.362]** (0.088 → 0.351) | PASS |
| parse | percand 7.3% / baseline 8.3% | flag (same regime as L′; orderings unaffected) |

## The registered PRIMARY — PASS on both conditions

| policy (57 rows) | P(strong) | dP vs baseline [CI95] |
|---|---:|---|
| **gauge_select** | 0.329 | **+0.241 [+0.147, +0.342]** |
| oracle (gold branch) | 0.351 | +0.263 [+0.169, +0.362] |
| random_select (20 draws) | 0.106 | +0.019 [−0.025, +0.061] |
| self_ratify | 0.202 | +0.114 [+0.037, +0.204] |
| bestofN majority (L1-recorded, matched compute) | 0.070 | −0.018 [−0.042, +0.007] |
| bestofN any-correct (oracle sampling ceiling) | 0.263 | +0.175 [+0.088, +0.270] |

Paired contrasts: gauge − bestofN-majority **+0.259 [+0.167, +0.364]**
(the primary's second condition); gauge − random +0.223 CI>0; gauge −
self-ratify +0.127 CI>0; gauge − oracle **−0.022 [−0.066, +0.018]**
(statistically indistinguishable from oracle selection); gauge −
bestofN-any-correct +0.066 straddle (matches the any-correct sampling
ceiling). Selector texture: argmax picks gold on 43/57 rows (0.754).
Fire texture: non-gold branches sit at baseline (0.076) while gold
hits 0.351 — addressing specificity inside one run.

**The loop is closed on this row set**: candidates enumerated from the
prompt (answer-free), donor-free LOO write at each candidate's
positions (answer-free content), gauge scores over the steered states
select the branch (answer-free selection) → +0.241 repair with no
answer key anywhere in the pipeline, beating matched-compute majority
sampling by +0.259.

## C4 rider — FROZEN-TRANSFERS (the question item L never actually tested)

Gold-address fires of the FROZEN donor-free vector at pinned norm,
k=8, same 57 rows: **+0.498 [+0.386, +0.607]** (0.088 → 0.586);
delivery lift +0.259 [+0.147, +0.373] (rate 0.654). The frozen
artifact not only transfers to fresh rows — it exceeds L′'s in-job
refit (+0.279) and matches the +0.447 selected-row anchor. The
"in-job fitting is part of the recipe" reading (drawn from the
invalid L1) is dead: both protocols work; the frozen one is stronger
at the gold address. (Composition-vs-rider gap: the composition's
gold branch used the LOO write at k=4; the rider used the frozen
write at k=8 — protocol difference, flagged descriptively, not a
contradiction.)

## Obligation discharged next

Per the L″ registration, the PASS obligates L‴ — a protocol-identical
fresh-draw replication (seed 20260817, the 57 L-series data rows
excluded via committed list; R0–R3 + RG rider, jobs 462037–462041,
launched Aug 17 after the L‴ registration was committed). On PASS the
closed-loop claim upgrades to replicated-on-two-disjoint-draws.

## Provenance note — the first run and the bug

The first L″ execution (461663–461666) and item L's L1 write-side
were EXECUTION-INVALID: branch generations ran without the steering
hooks (registry CORRECTION entry, commit 03b0eae; data archived under
`invalid_20260814/`). The delivery gate that now leads this table was
registered in that correction and is mandatory for all future
intervention lanes. All L″-r numbers above come from the corrected
code with delivery verified.
