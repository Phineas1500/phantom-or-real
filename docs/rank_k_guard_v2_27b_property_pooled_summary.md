# Rank-k Guard v2 (fresh rows) — Pooled Verdict, Jobs 458374 + 458375

Shard outputs: `docs/rank_k_guard_v2_27b_property_shard{0,1}of2.json`;
row-level generations in
`results/stage2/erasure/rank_k_guard_v2_27b_property_shard{0,1}of2.jsonl`.
26 rows pooled (13 per shard; 3 per shard skipped for missing concept
positions out of the 16 selected). Fresh-row selection: parse-ok,
strong-incorrect h3/h4 rows, seeded, excluding the 13 composite-manifest
rows the compact core was built from. Row keys are (shard, row_index);
paired row-cluster bootstrap (10k draws) vs the in-job unhinted baseline.

## Pooled causal arms (26 rows, k=8 samples each)

| arm | P(strong) | dP vs unhinted (CI95) | % of concept-replace |
| --- | ---: | ---: | ---: |
| unhinted_baseline | 0.120 | — | — |
| hinted_baseline | 0.870 | +0.750 [+0.601, +0.885] | — |
| L30_concept_replace | 0.375 | +0.255 [+0.111, +0.413] | 100% |
| L30_random_replace | 0.159 | +0.038 [−0.010, +0.101] | 15% (null) |
| rank4_loo_add_L30 | 0.264 | +0.144 [+0.038, +0.264] | 57% |
| rank8_loo_add_L30 | 0.351 | +0.231 [+0.111, +0.365] | **91%** |

## Pre-registered decision rule → outcome

Rule (recorded in `docs/causal_handle_directions.md` before unblinding):
claim 8 survives if pooled `rank4_loo` **or** `rank8_loo` CI excludes zero
AND reaches ≥70% of the pooled in-job `L30_concept_replace` effect.

**Outcome: survives.** `rank8_loo` excludes zero and reaches 91% of the
concept-replace effect. `rank4_loo` also excludes zero but reaches only
57% — the rank-4 basis under-transfers to fresh rows; rank-8 is the
fresh-row-portable core. The concept-replace arm itself repaired
(+0.255, CI excluding zero), so the scoping branch (concept-replace null
on fresh rows) did not fire — the compact-core claim generalizes beyond
recognition-gap-style rows, not just within them.

## Hint-validated secondary slice (hinted P(strong) ≥ 0.5; 23/26 rows)

| arm | dP vs unhinted (CI95) | % of concept-replace |
| --- | ---: | ---: |
| L30_concept_replace | +0.288 [+0.125, +0.462] | 100% |
| rank4_loo_add_L30 | +0.163 [+0.043, +0.299] | 57% |
| rank8_loo_add_L30 | +0.255 [+0.120, +0.402] | 89% |

Same picture on the validated slice — the result is not carried by
hint-refractory rows.

## Wording for the paper (claim 8 / section 5.4)

A rank-8 leave-one-row-out PCA basis of hinted-minus-unhinted
concept-position deltas, added at L30, recovers 91% of the concept-replacement
effect on 26 fresh strong-incorrect rows disjoint from the rows the basis
family was developed on (rank-4: 57%; random-replace control null). State
rank-8 as the fresh-row-portable core and rank-4 as sufficient only
in-distribution (original held-out guard: rank-4 77%).
