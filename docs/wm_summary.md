# Item WM — the retriever-addressed write on a haystack of tables: is a substantial blind gain available where failures are attention lapses? (2026-09-05)

Registered: docs/causal_handle_directions.md item WM (before any data).
Frame: 800 WikiTableQuestions examples whose answer is a unique table
cell, each target table buried among seeded distractor tables (≈ 8.5
tables, ≈ 126 rows, ≈ 11,000 characters, prompts ≈ 5k tokens), the
question prefixed with the target table's column names; label-free
BM25 retrievers over tables and over rows within the top table
(`scripts/tables_haystack_frame.py`). Stage 1 **job-wsfz3**; stage 2
**job-5ztc8 / job-anjwg**. Readers `scripts/tables_haystack_stage1.py`,
`scripts/tables_haystack_gates.py`.

## Stage 1 — the regime (800 rows, k = 8)
| | table haystack | NQ-Swap | WikiHop real |
|---|---|---|---|
| accuracy with documents | **0.684** | 0.578 | 0.466 |
| closed-book | **0.031** | 0.016 (counterfactual) | 0.445 |
| hint (attention ceiling) | 0.783 | 0.716 | 0.621 |
| 0/8 failures | 227 (28%) | 323 (40%) | 276 doc-dependent |
| fixable share of failures | **0.396 [0.330, 0.463]** | 0.327 | 0.214 |
| hint response | 131 at 0/8, 81 at 8/8 | bimodal | bimodal |

No memory to hold (closed-book at chance), the highest fixable share of
any frame, and still a majority of failures the hint cannot move: on
WikiTableQuestions' comparative questions ("the only nation with three
medals", "which came first") pointing at the answer cell is not enough,
because the failure is the comparison, not the attention.

| retriever recall | all rows | correct rows | failures | fixable failures |
|---|---|---|---|---|
| table top-1 (BM25 on header + cells) | 0.971 | | | |
| gold row within top-3 of the top table | 0.444 | 0.533 | 0.242 | **0.233** |

The label-free retriever finds the right row on the questions the model
already answers and misses on the ones it fails, which bounds the blind
arm before stage 2: fixable × hit × repair ≈ 0.40 × 0.23 × 0.6 ≈ 5% of
failures ≈ 1.5 points of the frame. The oracle-row arm gives the
ceiling a perfect retriever would reach.

**What the failures are (descriptive).** Among the 90 fixable failures
the model's wrong answer is a cell of the *right* table in 77 cases: it
found the table and picked the wrong row or column ("which peak has the
most isolation" → a distance instead of the peak). Among the 131
unfixable failures the wrong answer is also a right-table cell in 111
cases, but the questions are ordinal or comparative ("the previous
player signed before X", "the chip with the highest frequency", "the
lowest position") — the model reads the table and computes the wrong
comparison, and an attention nudge at the answer cell does not fix a
comparison.

## Stage 2 — the registered arms (120 blind rows, baseline 0.731)
| arm (k = 8) | accuracy | frame net vs baseline [CI] | up / down |
|---|---|---|---|
| **write @ retrieved rows (REGISTERED, 35th)** | 0.537 | **−0.194 [−0.273, −0.122]** | 4 / 29 |
| write @ retrieved table | 0.419 | −0.312 [−0.402, −0.222] | 5 / 45 |
| write @ gold row (the intended ceiling) | 0.311 | **−0.420 [−0.506, −0.331]** | 0 / 53 |
| control: random rows of another table | 0.741 | +0.009 [+0.001, +0.021] | 5 / 0 |
| text hint naming the answer | 0.817 | +0.085 [+0.028, +0.147] | 15 / 3 |
| quoting the retrieved rows (no write) | 0.657 | −0.074 [−0.157, +0.006] | 12 / 20 |

**35th NOT CONFIRMED. 36th NOT CONFIRMED** (write vs quoting, paired
−0.120 [−0.223, −0.015]). The harm is specific: when the retriever hits
the gold row the write costs −0.434, when it misses −0.004; the control
on an irrelevant table costs nothing. Under the gold-row write the
model's answers come from other rows and tables in every broken sample
— the direction written across a whole row corrupts the row rather
than spotlighting it. Every earlier frame wrote at the answer's own
mention tokens; the rider below runs that operation here.
