# Loop-port screening — WikiHop PASSES the behavioral gates (baseline 0.523, failing slice 47.7%, any-correct@8 on failures 0.035); RACE too easy; contamination disclosed

Registration-free reconnaissance (mission
https://givemeanode.com/missions/loop-port-screening, job job-6t87z,
H100 vLLM, 19,200 generations in a 78-second batch, ≈$2). Gemma-3-27B
on 300 seeded rows each (seed 20260819, length-filtered) of WikiHop
(QAngaroo validation — explicit candidate lists, mentions in the
documents: structurally isomorphic to the InAbHyD loop) and RACE-high.
Four arms x k=8 at temp 0.7: std / free (no candidates) / hint (gold
named) / closed (no documents). Row-level:
`results/loop_screen/loop_screen.jsonl`; client
`scripts/loop_screen_job.py`.

## Gates
| gate | WikiHop | RACE-high |
|---|---|---|
| std accuracy (band 10–60%) | **0.523** ✓; failing (<0.5) 47.7% | 0.858 ✗ (too easy); failing 14.3% |
| error correlation on failures | mean 0.008; **any-correct@8 0.035** ✓ | 0.015; any-correct 0.047 ✓ |
| hint gap on failing rows | **+0.158** ✓ (modest) | +0.276 ✓ |
| closed-book (chance) | 0.483 (0.10) ⚠ | 0.609 (0.25) ⚠ |
| recognition gap (std − free) | +0.193 | +0.782 (free-scoring artifact) |

## Reading
1. **WikiHop is the port target.** In-band baseline, a huge failing
   pool (~2.4k rows extrapolated to full dev), and errors MORE
   committed than InAbHyD's (any-correct@8 0.035 vs ~0.19) — the
   regime where sampling is useless and the loop's selection stage
   has value.
2. **Expectation-setting:** the hint gap (+0.158) is far below
   InAbHyD's (+0.5–0.9) — the commitment-failure fraction is smaller,
   so the write's ceiling on this data is lower; power the port at
   n≥150 failing rows and register MDE accordingly.
3. **Contamination handling:** closed-book 0.483 means parametric
   knowledge answers half of WikiHop. The intervention frame filters
   to DOC-DEPENDENT failures (std-fail ∧ closed-fail) and the
   screening discloses both numbers; no "reading comprehension"
   claims without that filter.
4. RACE: viable only with failure-enrichment from the 3.5k pool;
   held as secondary.

## Port plan implied (next build)
Gemma HF-pathway loop (adapt stage2_qwen_loop_hf.py): stage-1-style
WikiHop labels from the std arm; gauge probe fit on final-token
captures of graded rows; class-mean from correct-vs-incorrect at
candidate-mention positions; candidates + mention addresses come free
from the dataset. All runnable on the H100 lane — no Scholar
dependency.
