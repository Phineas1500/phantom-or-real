# Item WB — the branch gauge: SELECTOR-CEILING (14th prediction NOT confirmed) (2026-09-02)

Registered: docs/causal_handle_directions.md item WB. Capture-only jobs
**job-rm433** (A: WA shard 1 under shard 0's donor vector, 555 branches)
and **job-cydwg** (B: reverse, 434 branches), ~$0.8. Branch outputs and
correctness from the WG records (identical branches, seed 20260836);
branch states (final-token, L38/43/48/53, float32) from the captures.
Reader `scripts/wikihop_wb_fit.py` → `docs/wikihop_wb_gates.json`. State
files (not committed, 50 + 40 MB): `results/loop_screen/wikihop_wb_{a,b}_states.npz`
(sha256 90495ad32920… / 292b23638c6b…); manifests committed.

## Consistency (hard gate) — PASS, exact
The real-text L38 gauge score recomputed from every captured state
matches WG's recorded score with max |diff| = 0.0 on all 989 branches; no
branch unmatched. The captured states are the states the earlier
selectors scored.

## Fit — BRANCH NATURAL GATE PASS
| test shard | donor branches (positive) | donor 5-fold CV AUC L38 / L43 / L48 / L53 | chosen layer |
|---|---|---|---|
| A (shard 1) | 434 (39) | 0.804 / 0.805 / 0.832 / 0.808 | L48 |
| B (shard 0) | 555 (27) | 0.707 / 0.813 / 0.824 / 0.836 | L53 |

A probe fit on steered branches separates repairing from non-repairing
branches out-of-fold (AUC 0.82–0.84).

## Readings (60 rows, identical branches, oracle 0.700)
| selector | gauge-select | oracle recovered | gold branch is argmax |
|---|---|---|---|
| **branch gauge (WB)** | **0.183** | **26%** | 0.233 |
| real-text unsteered gauge, L38 (rider) | 0.254 | 36% | 0.233 |
| anonymized-fit unsteered gauge, L48 (WG) | 0.308 | 44% | 0.350 |

Branch gauge vs baseline 0.050: +0.133 [+0.027, +0.248]; vs random branch
0.117: [−0.017, +0.160]; vs SC@8: [0, +0.242]. **Paired branch − real-text:
−0.071 [−0.200, +0.054]**; branch − anonymized-fit: −0.125 [−0.271, +0.017].
The 14th prediction is NOT confirmed; pre-named verdict **SELECTOR-CEILING**.

## Reading
Classifying branches and ranking them within a row are different tasks.
With only 27–39 repairing branches to learn from, the branch gauge
learns row-level cues (which rows are repairable) that do not
discriminate between candidates within a row, and it ranks worse than
the unsteered probes. Three linear final-token probes — real-text
unsteered, anonymized unsteered, steered-branch — now sit at 0.18–0.31
against a 0.70 oracle on the same 60 rows. On this evidence the
selector's ceiling for this probe family is representational: the
final-token state after a wrong-address write is not linearly
distinguishable from the state after the right-address write by any of
these fits, while the write itself is at +0.65. Selection would need a
different signal (e.g. the answer's own log-probability under each
write, or a probe on the mention positions rather than the final token)
— a new design, not a refit.

## Program tally after WB
14 registered directional predictions: 12 confirmed, 2 not confirmed
(13th, 14th — both about the selector on anonymized text). The write
side of the natural-data loop is closed: one frozen direction, identity
from the address, +0.35 on real text and +0.65 without memory,
replicated, delivery-audited throughout. WikiHop chain cost ≈ $27
across 35 H100 jobs.

## Post-hoc, descriptive (2026-09-02, after WB landed; NOT registered): an output-level selector
Select the branch whose k=4 outputs most often equal its own fired
candidate (ties broken by the gauge) — no probe, no extra forwards, the
model's own acceptance of the write as the signal. On the existing
branches:

| setting | gauge-select | answers-fired select | oracle | af − gauge (paired) |
|---|---|---|---|---|
| WL real, per-candidate (53) | 0.255 | 0.264 | 0.274 | +0.009 [−0.052, +0.075] |
| WF real, per-candidate (59) | 0.229 | 0.271 | 0.364 | +0.042 [−0.047, +0.136] |
| WX real, frozen (59) | 0.174 | 0.271 | 0.377 | **+0.097 [+0.025, +0.182]** |
| WA anonymized, per-candidate (60) | 0.237 | 0.304 | 0.554 | +0.067 [−0.025, +0.167] |
| WA rider anonymized, frozen (60) | 0.254 | **0.350** | 0.700 | **+0.096 [+0.013, +0.183]** |

Never worse than the gauge; significantly better on both frozen-write
settings; 50% of the 0.700 oracle on the rider's branches. This is the
selector signal the registry's next item should test on a fresh draw
(registration before launch; the numbers above are post-hoc).
