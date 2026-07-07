# Item G3′ — Qwen Rank-and-Scale Ladder at L43: RANK BRANCH PASSES, k\* = 16

Job 458468 (Scholar, 2× A40, HF pathway, bf16), completed 2026-07-06
20:19, 64 min. `scripts/stage2_qwen_g3prime_hf.py`; same 15 rows, seeds,
and machinery as G3 (job 458465). Raw rows:
`results/stage2/erasure/qwen_g3prime_ladder.jsonl`. Verdict by the rules
registered in `causal_handle_directions.md` §G3′ before submission.

## Integrity gates (registered)

Both replication arms reproduce G3 **verbatim**: 120/120 generations
token-identical per arm; unhinted 0.192, rank8 0.275 exactly. Max
parse-fail across all 10 arms: 0.8%.

## Pooled results (row-cluster bootstrap, 10k, seed 20260704)

| arm | dP vs baseline (0.192) | 95% CI | paired vs noise family | 95% CI |
|---|---:|---|---:|---|
| rank8_loo | +0.083 | [−0.025, +0.217] | +0.121 | [+0.008, +0.246] |
| **rank16_loo** | **+0.117** | **[+0.017, +0.217]** | **+0.154** | **[+0.062, +0.242]** |
| rank32_loo | +0.125 | [+0.025, +0.217] | +0.163 | [+0.058, +0.271] |
| rank64_loo | +0.167 | [+0.058, +0.267] | +0.204 | [+0.117, +0.287] |
| rank8_fixednorm | +0.125 | [−0.017, +0.275] | +0.142 | [+0.017, +0.275] |
| rand_subspace64 (2 draws) | −0.025 / −0.050 | both straddle 0 | — | — |
| rand_normfull (2 draws) | +0.000 / −0.033 | both straddle 0 | — | — |

Noise families for the paired contrasts: rank arms vs pooled
rand_subspace64; fixednorm vs pooled rand_normfull. As % of the G2
full-replace carrier anchor (+0.175): rank 8/16/32/64 recover
48% / 67% / 71% / 95%.

## Verdict

**RANK BRANCH: k\* = 16** — the smallest k whose dP CI excludes zero
(+0.117 [+0.017, +0.217]) AND whose paired contrast against the
random-64 family excludes zero (+0.154 [+0.062, +0.242]). Registered
wording applies: **"Qwen's channel rank is ~16 — the low-rank motif is
cross-model, its dimensionality model-specific."** Rank 16 recovers 67%
of the carrier; the curve keeps climbing to 95% at rank 64 while random
64-dim subspaces at matched norm sit at or below baseline (−0.025/
−0.050) — the recovery is carried by the specific PCA directions, not
by energy.

SCALE BRANCH: moot per registration (fires only if no k qualifies).
Descriptively: rescaling rank-8 to full-delta norm lifts +0.083 →
+0.125 and beats its matched noise family (+0.142 [+0.017, +0.275]),
while full-norm random directions are inert — amplitude starvation was
real but secondary; direction identity is doing the work. This is an
independent in-model data point for the projection-headroom account
(`projection_headroom_hypothesis.md`).

## Cross-model picture after G2 + G3 + G3′

| coordinate | Gemma 3 27B | Qwen3.5 27B |
|---|---|---|
| carrier layer (rel depth) | L30/62 ≈ 0.48 | L43/64 ≈ 0.67 |
| channel rank (this design) | 8 (78% of carrier) | ~16 (67%; 95% at 64) |
| mean component | 35% of rank-8 | ≈ 0 |
| random-subspace controls | null/destructive | null |

The MOTIF transfers — a small causally-privileged subspace at
concept-mention positions, mid-to-late depth, unreachable by random
subspaces at matched norm. The COORDINATES are model-specific: depth,
rank, mean share. G3's miss is now explained: rank 8 was the wrong
dimensionality for Qwen, not a missing mechanism.

Wording consequences: ledger §3 cross-model bullet upgrades to
"low-rank channel motif cross-model (2/2), dimensionality
model-specific (8 vs ~16)". §1 LEVER claims remain Gemma-scoped per
registration (no promotion licensed — this is one task family, and the
Qwen ladder lacks the full battery: no erasure, no label controls).
G4 (answer-free class-mean rider at L43) remains descriptive-only under
the G-series registration, since that gate named G3, not G3′; if we
want G4 to be claim-bearing, it needs a fresh registration note before
launch.
