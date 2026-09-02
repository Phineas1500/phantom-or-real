# Item WG — refitting the gauge on anonymized text: SELECTOR-LIMIT-IS-NOT-DISTRIBUTION (2026-09-02)

Registered: docs/causal_handle_directions.md item WG (13th directional
prediction). Stage 1 capture **job-xjpm3** (536 anonymized rows); gauge fit
on 474 rows with the 60 WA test rows held out: CV AUC L38 0.821 / L43 0.820
/ **L48 0.827 (primary)** / L53 0.814 — natural gate PASS (real-text W0
gauge: 0.776). Stage 2 **job-n3i7u** + **job-yv7ym**: the WA rider re-run
with identical seeds (frozen direction, cross-fit, rungs 1×/2×, loop at
2×), every branch scored in one forward by the anonymized-fit gauges at
L38/43/48/53 and the real-text L38 gauge. ~$3.5. Row-level:
`results/loop_screen/wikihop_wg_{a,b}.jsonl`; readers
`scripts/wikihop_wg_compare.py` → `docs/wikihop_wg_compare.json`,
`scripts/wikihop_wl_gates.py --frozen` → `docs/wikihop_wg_gates.json`.

## Consistency checks
- Generation identity vs the rider: **5,834 of 5,844 branch outputs
  byte-identical** (99.8%); the frozen fits reproduce the rider's norms
  exactly; the gold-address write reproduces **+0.650 [+0.535, +0.756]**.
- The real-text gauge reproduces the rider's loop exactly (0.254).
- Delivery audit valid on all 4,884 fired records.

## The 13th prediction — NOT confirmed
Row-paired (anonymized-fit L48 loop − real-text L38 loop) = **+0.054
[−0.075, +0.192]**. The CI includes 0; pre-named verdict
**SELECTOR-LIMIT-IS-NOT-DISTRIBUTION**.

| selector (same 60 rows, same branches) | gauge-select | oracle recovered | gold branch is argmax | vs baseline 0.050 | vs random 0.117 |
|---|---|---|---|---|---|
| real-text W0 gauge, L38 (the rider's) | 0.254 | 36% | 0.233 | +0.204 [+0.102, +0.317] | +0.137 [+0.034, +0.245] |
| anonymized-fit, L38 | 0.233 | 33% | 0.233 | [+0.079, +0.300] | [+0.026, +0.219] |
| anonymized-fit, L43 | 0.221 | 32% | 0.217 | [+0.062, +0.294] | [+0.011, +0.208] |
| **anonymized-fit, L48 (primary)** | **0.308** | **44%** | **0.350** | [+0.142, +0.388] | [+0.087, +0.307] |
| anonymized-fit, L53 | 0.317 | 45% | 0.367 | [+0.150, +0.398] | [+0.094, +0.317] |

## Reading
Refitting on the right distribution buys a modest, not-significant gain
(+5 points, gold-argmax 0.23 → 0.35) and the later layers (L48/L53) do
better than L38, but the loop still recovers under half of a 0.700
oracle. The gauge separates *baseline* correct from incorrect final
states well (AUC 0.83), yet ranking *steered* branches is a different
task: writing the "attend here" vector at a wrong candidate also
produces a confident-looking final state, so non-gold branches often
outscore the repaired gold branch (the selector-write interference
branch W2 named in advance). The limit is in what the unsteered-fit
probe reads from steered states, not in entity familiarity.

Implied next registration (not launched): a **branch gauge** — a probe
fit on steered branches from donor rows (label = the branch's own k=4
correctness), read on test-row branches; the loop's readings as
registered. This is the first selector design that targets the actual
selection task. Every write-side result stands: +0.650 at the gold
address with one frozen vector.
