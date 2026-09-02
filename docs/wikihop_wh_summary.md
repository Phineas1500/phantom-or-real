# Item WH — hint-delta oracle write on the reading-driven slice: NO-CEILING (2026-09-02)

Registered: docs/causal_handle_directions.md item WH. Job **job-az23x**
(mission wikihop-loop-port, H100, 655 s, ~$0.55). 12 seeded reading-driven
rows (seed 20260823; doc-dependent failing ∧ std modal ≠ closed modal),
Gemma-3-27B bf16. Row-level: `results/loop_screen/wikihop_wh.jsonl` (912
records); reader `scripts/wikihop_wh_gates.py` → `docs/wikihop_wh_gates.json`.
Delivery audit valid on all 576 fired records (hook counters inside
`generate`, gauge forward hooked); mention pairing succeeded for all 48
fired candidates (no skips).

## Gate (a) TEXT CEILING — FAIL (pre-named branch: NO-CEILING)
Hint-first text prompt ("Hint: pay close attention to {gold}." before the
documents), k=8: gold correct rate **0.167** vs in-job baseline 0.000; dP
+0.167, row-bootstrap CI **[0, +0.417]** (n=12; the CI includes 0). Only 2
rows are repaired by the hint as text (WH_dev_933 'civic hospital' and
WH_dev_3277 'playstation 3', both 8/8); the other 10 keep their wrong
answer even when told which candidate to attend to. The rate matches the
screening's hint gap on failing rows (+0.158), now with the hint placed
first. Per the registration the write test is uninformative and no
further arms are run.

The non-gold text hint DOES move answers: answers-fired rate 0.250, lift
over that candidate's baseline **+0.188 [+0.056, +0.347]**. So the text
hint has traction on these rows in general; it is the *gold* candidate
that the model will not commit to on 10 of 12 rows — which says the
"reading-driven" filter (different wrong answer with vs without
documents) does not isolate commitment failures. Several of the 10 look
like granularity or label issues (gold 'work' vs answer 'comic book';
'stadium' vs 'venue'; '1964' vs the hinted-text answer '1965';
'christian reformed church in north america' vs 'christian').

## Gate (b) ORACLE WRITE — not read as a gate (descriptive only)
| rung (× raw δ) | gold correct (k=4×12) | dP [CI] | non-gold answers-fired | fingerprint lift [CI] | recovered fraction | gauge shift gold / non-gold |
|---|---|---|---|---|---|---|
| 0.5 | 0.000 | 0 [0, 0] | 0.111 | +0.049 [0, +0.118] | 0 | −0.26 / +0.14 |
| 1.0 | 0.000 | 0 [0, 0] | 0.111 | +0.049 [0, +0.118] | 0 | −0.21 / +0.09 |
| 2.0 | **0.083** | +0.083 [0, +0.250] | 0.139 | +0.076 [0, +0.194] | **0.50** | −0.05 / +0.06 |

At 2× the write fully repairs one of the two hint-repairable rows
(WH_dev_3277: 4/4 correct, baseline 0/8; its gold δ has per-position
norm 33,499, 7× the others' 2,400–6,400, so the 2× write there is
~67k per position and still does not degenerate the output) and not the
other (WH_dev_933, δ 2,886). This is the first whole-row repair by an
activation write anywhere in item W (W1 + sweep: 1 correct sample in
528). Non-gold fingerprint lift grows with dose (+0.049 → +0.076) with
CIs touching zero. Mean gold |δ| per position 6,269 (1.3× the W0 base).

## Reading
1. On WikiHop, "reading-driven" is not "hint-repairable": even the
   explicit gold hint as text fails on 10/12 rows. The sandbox analog of a
   commitment failure is the *hint-repairable* row (~16% of failing rows
   by both the screening and this job), and neither W1 nor WH sampled that
   slice on purpose.
2. Where the text hint works, the oracle δ direction carries repair
   (1 of 2 rows at 2×), and the fingerprint is dose-dependent — traction
   the class-mean direction never showed. n is far too small to claim it.
3. Implied next registration (NOT launched): a hint-first text arm over
   the whole 287-row W2 pool (vLLM, ~$1) to find the hint-repairable rows
   (expected ~45), then the WH design on 12–24 of them with rungs {1, 2,
   4}× so gate (a) is satisfied by construction and gate (b) is powered.
