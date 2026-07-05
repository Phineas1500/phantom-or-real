# Statistical Hardening Results (review-response plan W2, 2026-07-04)

`scripts/stage2_bootstrap_hardening.py` over the row-level JSONLs; row
clusters = source_row_index pooled across shards/seeds; 10k draws, seed
20260705; machine-readable output in `docs/stat_hardening_2026-07.json`.
LOO range = min/max of the point estimate under single-row deletion (a
sensitivity band, not a CI). Equivalence margin declared at 5pp (90% CI
within ±0.05).

| target | n rows | dP | percentile CI95 | BCa CI95 | LOO point range | MDE |
| --- | ---: | ---: | --- | --- | --- | ---: |
| claim 12 L35_concept_replace | 16 | +0.117 | [+0.008,+0.254] | [+0.023,+0.305] | [+0.067,+0.125] | 0.123 |
| claim 12 L35_rank4_loo_add | 16 | +0.047 | [+0.000,+0.109] | [+0.008,+0.137] | [+0.021,+0.050] | 0.055 |
| claim 8 guard-v2 rank8 | 26 | +0.231 | [+0.111,+0.370] | [+0.115,+0.375] | [+0.200,+0.245] | 0.130 |
| claim 8 guard-v2 rank4 | 26 | +0.144 | [+0.038,+0.265] | [+0.048,+0.279] | [+0.115,+0.165] | 0.113 |
| claim 8 specificity rank8 | 26 | +0.245 | [+0.115,+0.394] | [+0.120,+0.404] | [+0.215,+0.260] | 0.139 |
| claim 2 erase_raw (ctrl-match) | 16 | +0.031 | [−0.094,+0.172] | [−0.094,+0.172] | [−0.017,+0.067] | 0.133 |
| claim 7 hint-span masking | 13 | +0.000 | degenerate (0 flips) | degenerate | [0,0] | — |
| claim 11 Qwen erase_raw | 16 | +0.070 | [−0.031,+0.180] | [−0.031,+0.188] | [+0.033,+0.092] | 0.105 |

## Readings

- **Claim 8 is bootstrap-robust.** BCa ≈ percentile on all three rank
  estimates; LOO ranges tight and far from zero (worst case +0.115 for
  rank4). The reviewers' undercoverage concern does not bite here.
- **Claim 12: BCa strengthens the bound, LOO confirms row-sparsity.**
  BCa CI95 [+0.023,+0.305] sits farther from zero than percentile — the
  bias correction moves AWAY from the null, so the undercoverage attack
  fails in the direction the reviews assumed. But the LOO band confirms
  their substantive point: dropping the single strongest row nearly halves
  the point estimate (+0.125 → +0.067). Demotion to "suggestive,
  row-sparse" stands (plan W1.3); quote BOTH the BCa bound and the LOO
  band in the caveat.
- **Equivalence claims cannot be rescued at this n.** Claim 2 raw erasure:
  MDE ≈ 0.13 — the design cannot distinguish "harmless" from "helps or
  hurts by up to 13pp"; wording must be "no cost detectable (MDE 0.13)".
  Claim 11 Qwen: MDE ≈ 0.11, same treatment; note every LOO point estimate
  is positive (+0.033..+0.092), consistent with the weak-lever sign
  pattern flagged in the triage — item D's original-correctness split is
  the discriminator.
- **Claim 7's TOST "pass" is degenerate, but a stronger statement exists.**
  Zero strong-flips in 104 paired samples at ceiling → rule-of-three 95%
  upper bound ≈ 3/104 ≈ **2.9pp** on the masking cost. Use "masking costs
  < ~3pp (95%, rule of three)" — tighter than the reviewers' suggested
  "< ~5pp".
