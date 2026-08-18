# Item L‴ — THE REPLICATION LANDS: the closed answer-free loop replicates on a second disjoint draw, larger (+0.390), and so does the frozen transfer (+0.513)

Jobs 462042–462045 (R0–R3) + 462046 (RG rider), Aug 17–18, all under
3:00:00 walls. Fresh 58-row frame (seed 20260817; all 57 L-series data
rows excluded via the committed list). Pooled by
`scripts/stage2_selfaddress_l3_pool.py`. Registration: the L‴ entry
(commit 4bf4460).

## Gates
- DELIVERY (hard gate): gold-branch targets-fired lift +0.179
  [+0.073, +0.287] — PASS.
- No verbatim gate exists on fresh rows (registered in advance).
- Parse: 6.7% / 4.7% — flag-free to flag boundary; orderings robust.
- Comparator adaptation, disclosed: the registered "L1-recorded
  bestofN" cannot exist on disjoint fresh rows (registration gap);
  the comparator used is majority/any over the 8 in-job baseline
  samples (self-consistency@8), the strictest available equivalent.

## The composition, second draw (baseline 0.093)
| policy | dP [CI95] |
|---|---|
| **gauge_select** | **+0.390 [+0.265, +0.515]** |
| oracle | +0.446 [+0.328, +0.565] |
| random_select | +0.016 (null) |
| self_ratify | +0.080 [+0.004, +0.157] |
| self-consistency@8 (majority) | −0.041 [−0.075, −0.011] |
| best-of-8 any-correct ceiling | +0.114 [+0.052, +0.185] |

PRIMARY: PASS (both conditions; paired gauge−majority +0.431
[+0.297, +0.560]). Paired gauge−anycorrect +0.276 CI>0 — on this draw
the closed loop beats even the oracle sampling ceiling. Gauge−oracle
−0.056 [−0.121, −0.004]: the gauge captures 87% of oracle selection's
effect. Picks-gold 49/58 (0.845).

## RG rider — frozen transfer, second draw
+0.513 [+0.399, +0.627] (0.093 → 0.606), delivery rate 0.711.

## Standing claims after two draws
- Closed answer-free loop: +0.241 and +0.390 (disjoint draws, both
  primaries PASS, delivery-gated). Registered obligation discharged;
  no further replication registered.
- Frozen-artifact transfer at gold: +0.498 and +0.513.
- Selector: argmax-gold 43/57 and 49/58 under the LOO write; 54/57
  under the frozen write; selection-signal gates green throughout.
