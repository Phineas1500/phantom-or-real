# Item WO — the output-level selector: OUTPUT-SELECTOR-BEATS-GAUGE (15th prediction CONFIRMED) (2026-09-03)

Registered: docs/causal_handle_directions.md item WO. Frame: a third
800-row real draw (seed 20260838; disjoint from every prior frame)
anonymized to **507 rows** (seed 20260839). Stage 1 **job-dp3e2**
(std/closed/hint-first k=8, seed 20260840); stage 2 **job-k76n4** (A: test
shard 1, 23 rows, donors shard 0) + **job-6fgby** (B: reverse, 24 rows),
seed 20260842, ~$4 total. The WA-rider design unchanged: cross-fit frozen
hint-delta direction, gold at 1×/2×, 3 seeded non-gold at 1×, every
candidate at 2× with k=4; every branch scored by the anonymized-fit
gauges (L38/43/48/53, primary L48) and the real-text L38 gauge. Row-level:
`results/loop_screen/wikihop_wo_{a,b}.jsonl`; readers
`scripts/wikihop_wo_gates.py --tie-key second_L38` → `docs/wikihop_wo_gates.json`,
`scripts/wikihop_wl_gates.py --frozen` → `docs/wikihop_wo_write_gates.json`.

## Stage 1 (the third frame behaves like the others)
Contamination check PASS: closed-book **0.113** (WA 0.127; real 0.45).
std 0.361, hint-first 0.478; doc-dependent 293 of 507; hint-repairable ∧
doc-dependent **47 (16.0%)**, 43 reading-driven + 4 memory-driven; all 47
used (no cap; ≥ 20).

## Stage 2 — write consistency (gate (a)) — PASS
Delivery audit valid on all 729 fired branches; text ceiling 0.891. Gold-
address frozen write: dP **+0.386 [+0.255, +0.527] at 1×, +0.657 [+0.524,
+0.790] at 2×**; specificity +0.367 / +0.614. The third frame reproduces
the WA rider (+0.367 / +0.650): the write is now at +0.65 on two disjoint
anonymized draws with the same frozen recipe.

## Selection on identical branches (47 rows, 11.6 branches/row, oracle 0.686)
| selector | gauge-select | oracle recovered | gold branch is argmax |
|---|---|---|---|
| **output-first (answers-fired, real-text gauge tie-break) — REGISTERED** | **0.362** | **53%** | 0.383 |
| anonymized-fit unsteered gauge, L48 (WG) | 0.351 | 51% | 0.468 |
| real-text unsteered gauge, L38 (the chain's selector) | 0.218 | 32% | 0.234 |
| random branch / baseline / SC@8 | 0.156 / 0.029 / 0.021 | | |

Output-first vs baseline +0.33 [+0.197, +0.468]; vs random branch [+0.080,
+0.338]; vs SC@8 [+0.213, +0.489].

**15th prediction — CONFIRMED:** paired (output-first − real-text gauge)
= **+0.144 [+0.032, +0.271]**. Secondary: output-first − anonymized-fit
gauge = +0.011 [−0.138, +0.160] (a tie: the WG gauge, fit on the WA
frame, transfers to this frame at 0.351 — its own strongest showing).

## Verdict
**OUTPUT-SELECTOR-BEATS-GAUGE.** The loop on natural text now closes
with ONE frozen vector and NO probe as the primary selector: the branch
the model itself accepts (answers with the fired candidate most often)
is the right one, at 53% of the oracle. A probe fit on the right
distribution matches it; the chain's original real-text probe does not.
Selector standing across the anonymized settings: output-first 0.350
(rider, post-hoc) / 0.362 (WO, registered); anonymized-fit gauge 0.308 /
0.351; real-text gauge 0.254 / 0.218; branch gauge 0.183 (WB).

## Program tally after WO
15 registered directional predictions: **13 confirmed**, 2 not confirmed
(13th, 14th). WikiHop chain: W → WH → WR → WL → WF → WX → WA (+rider) →
WG → WB → WO; ≈ $31 across 38 H100 jobs. Nothing further is owed on
WikiHop; the write is closed, and the selector has a registered,
probe-free form.
