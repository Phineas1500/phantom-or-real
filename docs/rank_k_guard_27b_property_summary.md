# Rank-k Guard — Verdict (Job 457012)

Post-finale guard for the L30 compact-core claim, run from
`scripts/stage2_rank_k_guard.py` on the same property rowset and saved
concept-position deltas. The critical control is leave-one-row-out PCA: each
receiver row is reconstructed from a basis fit without that row. Run completed
in 3h14 (1,352 generations: 13 rows x 13 arms x k=8).

## Arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP (CI95) | note |
| --- | --- | --- | --- |
| unhinted_baseline | 0.192 | -- | in-job reference |
| rank1_loo_L30 | 0.288 | +0.096 [+0.019, +0.183] | small but real held-out repair |
| rank2_loo_L30 | 0.260 | +0.067 [-0.029, +0.154] | null-ish |
| rank3_loo_L30 | 0.337 | +0.144 [-0.058, +0.365] | noisy positive |
| **rank4_loo_L30** | **0.385** | **+0.192 [+0.096, +0.288]** | **held-out guard clears: 77% of subset effect** |
| rank6_loo_L30 | 0.327 | +0.135 [-0.010, +0.288] | noisy positive |
| **rank8_loo_L30** | **0.423** | **+0.231 [+0.087, +0.404]** | strongest held-out rung; 92% of subset effect |
| rank4_in_sample_L30 | 0.356 | +0.163 [+0.010, +0.337] | in-sample does not explain the effect away |
| rank8_in_sample_L30 | 0.413 | +0.221 [+0.087, +0.385] | matches held-out trend |

## Verdict

The free-parameter guard passes. The L30 rank-4 effect survives when the PCA
basis is fit without the receiver row: rank4_loo repairs +0.192, clearing the
pre-registered >=70% sufficiency rule relative to the +0.250 subset-replacement
effect. This rules out the main in-sample low-rank-fit artifact.

The dimensionality curve is not a clean rank-4 cliff. Rank-1 now shows a small
held-out repair, rank-4 is the first rung that cleanly clears the sufficiency
bar, and rank-8 is stronger. The precise wording should therefore be:
**a compact low-dimensional causal core is present by rank 4, with useful
additional structure through rank 8; do not claim exact intrinsic dimension =
4.** This still preserves the key positive claim: the found variable is compact,
held-out, localized at L30, and not merely the readable correctness subspace.

## Geometry rider landed

`docs/rank_core_geometry_27b_property_summary.md` projects the held-out-surviving
L30 components onto INLP and Gemma Scope decoders. Verdict: gauge-orthogonal,
dictionary-visible, but not sparse-small.

## Next

- Subtype discriminator: capture ladder plus targeted off-trio/rank-k patch.
- Qwen erasure replication for cross-model epiphenomenality.
