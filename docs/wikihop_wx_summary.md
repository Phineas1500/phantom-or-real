# Item WX — the frozen, row-independent write: FROZEN-WRITE-TRANSFERS + FROZEN-LOOP-CLOSES (2026-09-02)

Registered: docs/causal_handle_directions.md item WX. Jobs **job-tj7za**
(A: test WF shard 1, 29 rows, donors = shard 0) and **job-7swth** (B: the
reverse, 30 rows), ~$2.2. Cross-fit on the 59 WF rows (fresh frame),
Gemma-3-27B bf16, L30 write, W0's L38 gauge, every other pin frozen.
Row-level: `results/loop_screen/wikihop_wx_{a,b}.jsonl` (7,876 records);
reader `scripts/wikihop_wl_gates.py --frozen` → `docs/wikihop_wx_gates.json`.
Delivery audit valid on all 6,932 fired records; no candidate skipped.

## The frozen write
Fit in-job from the DONOR shard only: the mean over all donor gold-mention
positions of (hint-first state − std state) at L30; unit direction u;
norm target N = donor mean per-position |δ| (3,288 / 3,213 for A / B). The
write at candidate X's whole-word mentions is u × rung × N — the SAME
vector for every candidate; only the address changes. |mean δ| retains
47–48% of the per-position norm (1,557 / 1,529), and u has cosine
**0.68 (median 0.76, range 0.06–0.88)** with each test row's own gold
delta: the hint-delta is largely one shared direction, not a per-row
message.

## Gates (n = 59)
**(a) TEXT CEILING — PASS:** 0.932 vs baseline 0.017; dP +0.915 [+0.850, +0.968].

**(b) FROZEN ORACLE — PASS at both rungs:**

| rung (× N) | gold-ADDRESS correct (k=4×59) | dP [CI] | gold correct under NON-gold addresses | specificity [CI] |
|---|---|---|---|---|
| 1.0 | 0.343 | **+0.326 [+0.216, +0.445]** | 0.017 | **+0.326 [+0.216, +0.445]** |
| 2.0 (pinned) | 0.377 | **+0.360 [+0.242, +0.483]** | 0.031 | **+0.346 [+0.232, +0.470]** |

Per-candidate deltas on the same rows (WF): +0.309 / +0.347. The frozen
direction is as good as the row's own hint-delta. The candidate identity
is carried entirely by WHERE the vector is written.

**(c) FROZEN LOOP at 2× — PASS on both registered readings:**

| reading | frozen (WX) | per-candidate (WF, same rows) |
|---|---|---|
| gauge-select over all candidates (25.4/row) | **0.174** | 0.229 |
| baseline / random branch | 0.017 / 0.048 | 0.017 / 0.044 |
| gauge-select − baseline | **+0.157 [+0.072, +0.254]** | +0.212 [+0.119, +0.314] |
| gauge-select − random branch | **+0.126 [+0.040, +0.221]** | +0.185 [+0.096, +0.281] |
| oracle → fraction recovered | 0.377 → 46% | 0.364 → 63% |
| gold branch is argmax (chance 0.070) | 0.288 | 0.237 |

## Verdict
**FROZEN-WRITE-TRANSFERS (pinned 2×) + FROZEN-LOOP-CLOSES.** The
expectation stated at registration (that per-candidate content would be
essential, by analogy with W1's class-mean null) was wrong: the
hint-DELTA direction, frozen across rows, carries the repair, while the
class-mean direction never did — the two directions differ (W1's
class-mean was correct-minus-incorrect gold-mention states; WX's is the
hinted-minus-unhinted difference), and only the latter is the lever. This
reproduces the sandbox's frozen-transfer result (C4 rider, +0.498) on
natural data, locates the effect in the representation rather than in the
test-time hint, and makes the loop ~25× cheaper (no per-candidate hint
forwards: one write vector, one forward per candidate for the gauge).
The selector is again the weaker half (46% of the oracle recovered vs 63%
with per-candidate deltas — the per-candidate branches are easier to
tell apart). No directional prediction was registered; the result is
exploratory and would need its own fresh-draw replication to be claimed
at the level of WL/WF.
