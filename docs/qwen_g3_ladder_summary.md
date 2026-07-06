# Item G3 — Qwen Specificity Ladder at L43: NO PASS (under-powered directional miss)

Job 458465 (Scholar, 2× A40, HF pathway, bf16), completed 2026-07-06
17:41, 49 min. `scripts/stage2_qwen_g3_hf.py`; 15 prepared rows (seed
20260706, G0 rows excluded), 8 samples/arm, L43 (rel depth 0.67, the G2
winner), k=8. Raw rows: `results/stage2/erasure/qwen_g3_ladder.jsonl`;
run metadata: `docs/qwen_g3_ladder.json`. Verdict by the rule registered
in `causal_handle_directions.md` §Item G before unblinding.

## Pooled results (row-cluster bootstrap, 10k, seed 20260704)

| arm | P(strong) | dP vs unhinted | 95% CI |
|---|---:|---:|---|
| unhinted_baseline | 0.192 | — | — |
| hinted_baseline | 0.817 | +0.625 | [+0.458, +0.775] |
| **rank8_loo_add_L43** | 0.275 | **+0.083** | **[−0.025, +0.217]** |
| mean_only_add_L43 | 0.167 | −0.025 | [−0.108, +0.067] |
| rand_subspace_add_L43 (2 draws) | 0.208 / 0.208 | +0.017 / +0.017 | both straddle 0 |
| rand_norm_add_L43 (2 draws) | 0.175 / 0.217 | −0.017 / +0.025 | both straddle 0 |

Paired contrasts (protected comparisons):

| contrast | dP | 95% CI | registered? |
|---|---:|---|---|
| rank8 − rand_norm family | +0.079 | [−0.025, +0.200] | YES (PASS criterion) |
| rank8 − rand_subspace family | +0.067 | [−0.021, +0.171] | descriptive |
| rank8 − mean_only | +0.108 | [+0.017, +0.208] | descriptive |

Baselines reproduce G2 exactly (0.192 / 0.817 — same rows, same seeds,
determinism holds). Parse-fail ≤ 1.7% on every arm.

## Verdict

**PASS criterion not met.** Registered rule: rank8_loo CI excludes zero
AND paired (rank8 − rand_norm family) CI excludes zero. Both CIs include
zero. rank8 lands at **48% of G2's concept-replace anchor** (+0.083 vs
+0.175).

This is the under-powered-honest-miss branch, not a demonstrated null:

- Every control arm is flat (−0.017..+0.025); rank8 is the only arm with
  a positive point estimate of any size. Per-row: 5 up / 8 flat / 2 down.
- The design's MDE at n=15 rows is ≈ 0.12; the observed +0.083 is below
  what this design can resolve. The rank8 − mean_only descriptive
  contrast does exclude zero (+0.108 [+0.017, +0.208]).
- The mid-run "mean-domination" flag resolved AGAINST mean domination:
  mean_only finished at −0.025 (below baseline). Whatever the rank-8
  reconstruction carries in Qwen, it is not its mean component — same
  qualitative structure as Gemma (item C: mean_only 35% of rank8), with
  the Qwen mean share ≈ 0.

## Wording consequences (registered both directions)

- The full-state carrier claim from G2 STANDS: hinted-state
  concept-position replacement at L43 repairs (+0.175 [+0.100, +0.258],
  paired vs random +0.158) — the focus-state carrier is reachable in
  Qwen at a model-specific depth (0.67 vs Gemma's 0.48).
- The COMPACT (rank-8) core is **not established in Qwen**: directionally
  consistent at 48% of the carrier effect, but not separable from its
  noise family at this n. The cross-model sentence for the paper:
  "the carrier replicates across models at model-specific depth; the
  rank-8 compression is Gemma-established, Qwen-directional-only
  (+0.083, MDE 0.12)." The §1 LEVER claims remain Gemma-3-27B scoped.
- Registered gates downstream: **G5 (Qwen3.6) is OFF** (required G3
  PASS). **G4 (answer-free class-mean rider) may run as descriptive
  only** — informative for the paper's cross-model paragraph but cannot
  move a claim.
- NOT licensed: "the lever does not transfer" (miss ≠ null at this MDE);
  "rank-8 transfers" (CI includes zero). Licensed: the two-tier wording
  above.

## Reading

Gemma's ladder separated cleanly at 16 rows because its rank8 effect
(+0.245) was 2× this one. Qwen's carrier is as strong as Gemma's, but
the energy is less concentrated: compressing the L43 delta to rank 8
keeps ~48% of the effect in Qwen vs ~78% in Gemma (0.245/0.316 patch).
A finer G3′ (rank sweep k ∈ {8, 16, 32} or more rows) would resolve
whether Qwen's channel is simply higher-rank — a follow-up registration,
not a rerun.
