# Item WQ — the WikiHop loop on a second model (Qwen3.5-27B): QWEN-WRITE-TRANSFERS (18th prediction CONFIRMED) / QWEN-SELECTOR-FAILS (19th NOT confirmed) (2026-09-03)

Registered: docs/causal_handle_directions.md item WQ (before any data).
Model Qwen/Qwen3.5-27B (64 layers, hidden 5120, linear-attention
hybrid, thinking off). Frame: the WF fresh real-text frame — the same
800 rows Gemma saw in WF/WX/WE. Stage 1: grade_hint **job-4akjt**
($1.84; two failed attempts $1.04: the shared HF cache is too small for
the 54 GB checkpoint, then vLLM's default max_num_seqs exceeded the
hybrid's Mamba cache blocks) + capture **job-rhh7m** ($0.86). Stage 2:
cross-fit frozen hint-delta write, L43 primary **job-i33vz** (A, 13 test
rows) + **job-7pxb9** (B, 14), L31 descriptive **job-dzy76** + **job-mx57i**;
$5.89. Seeds 20260854–20260857. Row-level
`results/loop_screen/wikihop_wq_{l43,l31}_{a,b}.jsonl`; readers
`scripts/wikihop_wl_gates.py --frozen` → `docs/wikihop_wq_write_gates.json`
(+ `_l31_`), `scripts/wikihop_wo_gates.py --tie-key primary_L48` →
`docs/wikihop_wq_gates.json` (+ `_l31_`).

## Stage 1 — the frame through Qwen's eyes (Gemma in parentheses)
| | Qwen3.5-27B | Gemma-3-27B |
|---|---|---|
| std / closed-book / hint-first accuracy | 0.534 / 0.451 / 0.691 | 0.466 / 0.445 / 0.621 |
| failing 0/8 · doc-dependent (0/8 std ∧ 0/8 closed) | 278 · 137 | 409 · 276 |
| hint-repairable ∧ doc-dependent | **27 (19.7%)**, 15 reading-driven | 59 (21.4%) |
| natural gauge, 5-fold CV AUC by layer | L38 0.836 · L43 0.831 · **L48 0.841** · L53 0.822 · L58 0.808 | L38 0.776 |

Gauge gate PASS (best layer after the write layer = L48, ≥ 0.70); the
L48 gauge is the pinned tie-break. The close-cousin quarter is a WikiHop
fact, not a Gemma fact. Capture finite everywhere; Qwen has no
Gemma-style massive dimensions (mention-mean norm 67 vs 4,737).

## Stage 2 at L43 (registered) — 27 rows, delivery valid on 2,144 fired branches
Text ceiling (gold hint-first, k=8): 0.690, +0.639 [+0.537, +0.731].
| gold-address frozen write | gold rate | dP vs baseline 0.051 | specificity (vs non-gold address) |
|---|---|---|---|
| 1× | 0.120 | +0.069 [−0.009, +0.139] | +0.102 [+0.034, +0.179] |
| **2× (pinned)** | **0.213** | **+0.162 [+0.088, +0.250]** | **+0.189 [+0.105, +0.280]** |

**18th prediction CONFIRMED — QWEN-WRITE-TRANSFERS.** One frozen
vector (donor mean hint-delta, 13–14 donors, per-position norm target
30 against Gemma's 3,251) written at each candidate's whole-word
mentions repairs Qwen at 2× with specificity. Effect about half of
Gemma's on the same rows (Gemma WX 2×: +0.360, gold rate 0.377).

| selector on identical 2× branches (15.9 per row) | select | gold branch is argmax | vs baseline 0.051 | vs random branch 0.042 |
|---|---|---|---|---|
| output-first (answers-fired, L48 tie-break) — REGISTERED | 0.046 | 1/27 | [−0.051, +0.042] | [−0.024, +0.040] |
| L48 gauge-select | 0.046 | 6/27 | [−0.051, +0.037] | [−0.029, +0.043] |
| oracle (always the gold branch) | 0.213 | | | |

**19th prediction NOT confirmed — QWEN-SELECTOR-FAILS.** Neither
selector beats the baseline or a random branch. Decomposition: gold
branch never fires on 14/27 rows, is out-fired by a wrong branch on
11/27, tied-and-lost 1, selected 1. The gold branch is never fully
accepted (answers-fired 4/4 on 0 rows) while some wrong branch is fully
accepted on 18/27 rows; mean acceptance gold 0.21 vs non-gold 0.08 but
the maximum over ~15 wrong branches wins. This is a different failure
from Gemma's (ties at full acceptance with the model's own answer):
under a weaker write, Qwen accepts nudges toward wrong candidates more
readily than toward gold.

## Stage 2 at L31 (pre-named descriptive; relative depth 0.48 = Gemma's L30)
| | gold rate | dP | specificity |
|---|---|---|---|
| 1× | 0.139 | +0.088 [0.000, +0.171] | +0.114 [+0.037, +0.207] |
| **2×** | **0.306** | **+0.255 [+0.134, +0.389]** | **+0.288 [+0.161, +0.429]** |

Loop on the L31 branches: **L48 gauge-select 0.157** (51% of the 0.306
oracle; baseline 0.051, random 0.047; vs baseline [+0.019, +0.222]),
output-first 0.083 ([−0.032, +0.111] vs baseline; gold argmax 3/27).
Decomposition: never fired 13, beaten 9, tied-lost 2, selected 3; gold
fully accepted on 3 rows, some wrong branch on 21.

The write is stronger at the Gemma-analog depth than at Qwen's
class-mean carrier (L43), approaching Gemma's real-text effect; and at
L31 the gauge selector works where the output-first selector does not —
the reverse of Gemma's WO ordering. Both statements are descriptive:
the registered layer was L43 and stays the verdict layer.

## Cross-model table on the shared real-text rows (frozen write, 2×)
| | Gemma L30 (WX, 59 rows) | Qwen L43 (registered, 27) | Qwen L31 (descriptive, 27) |
|---|---|---|---|
| gold-address dP · specificity | +0.360 · +0.346 | +0.162 · +0.189 | +0.255 · +0.288 |
| gold rate (oracle) | 0.377 | 0.213 | 0.306 |
| gauge-select loop | 0.174 | 0.046 | 0.157 |
| output-first loop | 0.271 | 0.046 | 0.083 |
| baseline · random branch | 0.017 · 0.048 | 0.051 · 0.042 | 0.051 · 0.047 |

## Verdict
**QWEN-WRITE-TRANSFERS at L43 (2×) / QWEN-SELECTOR-FAILS.** The
write half of the loop is cross-model: one frozen direction at the
candidate's address repairs a second model with a different
architecture, at a registered layer, with specificity. The selector
half is model-specific: on Gemma the model's own acceptance beats every
probe, on Qwen it is at chance and a probe does better (at L31). This is
item N's sandbox contrast (oracle transfer passes, selector weak),
reproduced on natural text.

## Program tally after WQ
19 registered directional predictions: **16 confirmed**, 3 not
confirmed (13th, 14th, 19th). WikiHop chain: W → … → WE → WQ; ≈ $50
across 52 H100 jobs (WQ $9.63 over 8 jobs, 2 failed at setup).
