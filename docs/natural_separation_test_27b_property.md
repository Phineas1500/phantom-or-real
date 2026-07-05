# Item F(i) Verdict: the Rank-8 Lever Subspace Is NOT Where Natural Outcomes Are Written

Job 458416 (capture, 7m44s, 96/96 rows) + `scripts/stage2_natural_separation_test.py`
(pre-registered in `docs/causal_handle_directions.md` F(i) before capture).
96 balanced fresh rows (24 per height × natural-correct/incorrect cell),
unhinted L30 gold-concept-position states, all 45 basis-provenance rows
excluded. Machine-readable: `docs/natural_separation_test_27b_property.json`.

| readout | CV AUC (natural correct vs incorrect) |
| --- | ---: |
| full 5,376-dim state (ceiling) | **0.807** |
| random rank-8 subspaces (200 draws) | median 0.623 · p95 0.721 |
| dev/composite rank-8 basis | 0.701 — inside the null |
| guard-v2 rank-8 basis | 0.718 — inside the null |

**Pre-registered outcome: NULL.** Neither frozen lever basis exceeds its
random-subspace null's 95th percentile. The rank-8 subspace carries no
privileged information about natural success at these positions — it
separates outcomes no better than chance-level 8-dim slices of a highly
redundant state.

## Reading

1. **The lever is OUR steering wheel, not the model's.** The rank-8 core
   moves behavior when written into (items A/C/E, all causally solid),
   but natural correct and incorrect runs do not differ along it more
   than along random directions. Combined with claim 9 (the core is
   nearly orthogonal to the readable correctness subspace), the coherent
   picture: the core is an exogenous CONTROL CHANNEL that the hint
   recruits — not the internal variable whose natural setting decides
   outcomes.
2. **This vindicates the reviews' central caution.** Both adversarial
   reviews argued "commitment variable" was an inferred label on an
   exogenous mediation result. Part (i) tested it and they were right at
   this layer/position/rank. Paper wording stays at exogenous mediation;
   the discussion gains a sharp, honest sentence: we tested endogeneity
   and it failed — identifying the model's own decision variable remains
   open.
3. **The gauge shows up even here**: 0.807 full-dim decodability of
   natural outcomes at L30 concept positions (balanced, fresh rows) — the
   information is present and redundant (random 8-dim slices reach 0.62),
   consistent with claims 1/3.
4. **Scope limits**: L30, concept positions, rank 8, linear readout,
   mean-pooled positions. A natural decision variable could live at other
   layers/positions or nonlinearly; this null does not close the question
   — it closes the cheapest, most likely version of it.

## Consequence for item F(ii)

The class-mean arm's prior weakens but the experiment sharpens: the
natural correct-minus-incorrect class-mean delta (computable from this
capture) can now be tested RAW and rank-8-projected — if the raw natural
delta repairs but its rank-8 projection does not, the repair channel and
the natural-outcome axis are confirmed distinct causally, not just
correlationally. Pre-registration for F(ii) follows in
`docs/causal_handle_directions.md`.
