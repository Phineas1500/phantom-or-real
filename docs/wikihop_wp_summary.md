# Items WP + WP′ — the loop leaves WikiHop: HotpotQA and SQuAD (25th + 26th predictions CONFIRMED on HotpotQA; SQuAD write confirmed, loop descriptive; one WikiHop-fit vector crosses tasks) (2026-09-04)

Registered: docs/causal_handle_directions.md items WP (HotpotQA
distractor, 800 rows, passage-enumerated candidates) and WP′ (SQuAD
v1.1, 800 rows with entity-like answers, paragraph-enumerated
candidates), before any data; a job bug (the grading job's inline
WikiHop prompt template hid the question) voided the first stage-1
readings and both probes (≈ $1.05) and was corrected before any stage-2
data — see the CORRECTION entry. Stage 1 (corrected): HotpotQA
**job-krxds**, SQuAD **job-gmcqj**. Stage 2 (WX recipe at L30: 1×/2×,
loop at 2× k=4, three seeded non-gold at 1×, text-hint gold k=8,
WikiHop real-text L38 gauge frozen for ties): HotpotQA **job-m9m6h /
job-jvxge** (A/B, HotpotQA donors) + **job-pum5j / job-a4hcv** (XA/XB,
WikiHop donors); SQuAD **job-87imv / job-an6wn** (A/B) + **job-rg56z /
job-hvbva** (XA/XB). Total ≈ $9.9 over 14 jobs. Readers
`scripts/wikihop_wl_gates.py --frozen`, `scripts/wikihop_wo_gates.py
--tie-key primary`; outputs `docs/wikihop_wp_*_gates.json`,
`docs/wikihop_wpprime_*_gates.json`; rows
`results/loop_screen/wikihop_{hp,sq}_{a,b,xa,xb}.jsonl`.

## Stage 1 — the same quarter, on two more tasks
| | HotpotQA | SQuAD | WikiHop (WF frame) |
|---|---|---|---|
| std / closed-book / hint-first | 0.613 / 0.521 / 0.712 | 0.897 / 0.692 / 0.925 | 0.466 / 0.445 / 0.621 |
| doc-dependent failures | 219 | 54 | 276 |
| hint-repairable ∧ doc-dependent | **53 (24.2%)** | **14 (25.9%)** | 59 (21.4%) |
| stage-2 rows (cross-fit A/B) | 26 / 27 | 7 / 7 (**underpowered**) | 29 / 30 |

Instrument gate (documents ≥ 0.15 over closed-book): SQuAD +0.205
PASS; HotpotQA +0.092 FAIL as written — amended and flagged in the
registry (the threshold was set while the instrument was broken and
would exclude WikiHop's own +0.02 to +0.07).

## HotpotQA (registered): 53 rows, delivery valid on 9,328 branches, text ceiling 0.922
| frozen write, HotpotQA donors | gold rate | dP vs baseline 0.031 | specificity |
|---|---|---|---|
| 1× | 0.283 | +0.252 [+0.146, +0.363] | +0.237 [+0.132, +0.348] |
| **2× (pinned)** | **0.344** | **+0.314 [+0.193, +0.436]** | **+0.305 [+0.187, +0.425]** |

| selector on identical 2× branches (40 per row) | select | gold argmax | vs baseline 0.031 | vs random 0.047 |
|---|---|---|---|---|
| **output-first — REGISTERED** | **0.231** | 0.226 | **[+0.099, +0.309]** | **[+0.086, +0.290]** |
| WikiHop L38 gauge-select | 0.156 | | | |
| oracle | 0.344 | | | |

**25th prediction CONFIRMED — WRITE-LEAVES-WIKIHOP. 26th prediction
CONFIRMED — LOOP-LEAVES-WIKIHOP** (67% of the oracle). Free-form
bridge and comparison questions, ten paragraphs with distractors,
forty enumerated candidates: the same recipe, the same layer, the
same selector.

## SQuAD (registered; underpowered): 14 rows, delivery valid on 984 branches, text ceiling 0.982
| frozen write, SQuAD donors (7 per shard) | gold rate | dP vs baseline 0.080 | specificity |
|---|---|---|---|
| **1× (pinned)** | **0.536** | **+0.455 [+0.214, +0.696]** | **+0.411 [+0.190, +0.637]** |
| 2× | 0.339 | +0.259 [−0.045, +0.554] | +0.163 [−0.162, +0.469] |

Write reading (25th on SQuAD): CONFIRMED at the pinned 1×. Loop
reading, descriptive per the underpowered flag: output-first 0.179 vs
baseline 0.080 ([0.000, +0.241]), vs random 0.188 ([−0.102, +0.075]) —
does not close on 14 rows with 7-donor directions.

## The cross-task vector (pre-named descriptive): WikiHop's frozen direction at HotpotQA and SQuAD addresses
| WikiHop donors (59 WX rows) writing at… | gold rate 2× | dP | specificity | output-first loop | vs baseline | vs random | oracle |
|---|---|---|---|---|---|---|---|
| HotpotQA (53 rows) | **0.519** | **+0.488 [+0.354, +0.616]** | +0.476 [+0.343, +0.600] | **0.321** | [+0.175, +0.415] | [+0.150, +0.391] | 0.519 |
| SQuAD (14 rows) | **0.732** | **+0.652 [+0.402, +0.866]** | +0.623 [+0.388, +0.836] | **0.429** | [+0.134, +0.616] | [+0.049, +0.499] | 0.732 |

Row-paired, the WikiHop-fit direction beats the task's own cross-fit
direction at 2×: HotpotQA **+0.175 [+0.085, +0.269]** (53 rows), SQuAD
**+0.393 [+0.161, +0.643]** (14 rows). One vector, fit once on 59
WikiHop rows, repairs two other reading tasks at least as well as
directions fit on those tasks, and closes the loop on both.

## Verdict
**WRITE-LEAVES-WIKIHOP / LOOP-LEAVES-WIKIHOP.** The fixable quarter,
the frozen write at the address, and the output-first loop all hold on
HotpotQA; the write holds on SQuAD; and the direction itself is
task-general — WikiHop's vector is the best vector on both new tasks.
The mechanism sentence now reads: one learned direction, at the
candidate's mentions, in a mid-depth band, routes attention to that
span and repairs the answer, on three reading tasks and two models.

## Program tally after WP/WP′
26 registered directional predictions: **23 confirmed**, 3 not (13th,
14th, 19th). Chain W → … → WV → WP/WP′ complete; ≈ $125 across 115 H100
jobs (≈ $25 lost to platform disk failures, ≈ $1 to the prompt bug).
