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
