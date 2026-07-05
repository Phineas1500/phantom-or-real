# Readable-Stack Erasure (item D) — Pooled Verdict, Jobs 458403 + 458412 + 458413 + 458414

Stitched from four jobs after the 458403 wall-timeout (rows 4926/5564/
3315/4068 from the partial shard; shard 1/4 + 3/4 quarter-shards; the
`--row-indices` remainder for 6604/8934/6705/7466). Stitching validity
re-verified in-data: row 6604's overlapping baseline and erase_raw cells
are 8/8 sample-identical across jobs. 16 balanced rows (8 naturally
correct / 8 incorrect), 6 conditions × k=8, row-cluster bootstrap (10k,
seed 20260704), paired vs in-job baseline. Pre-registered rules:
`docs/causal_handle_directions.md` item D.

## Pooled conditions (16 rows, baseline P(strong) = 0.391)

| condition | dP vs baseline (CI95) | parse | P(strong\|parsed) |
| --- | ---: | ---: | ---: |
| erase_raw (continuity gate) | −0.016 [−0.094,+0.070] | 0.938 | 0.406 |
| **erase_readable_stack** | **+0.047 [−0.070,+0.203]** | 0.992 | 0.444 |
| erase_random_stack d1/d2/d3 | −0.383 / −0.375 / −0.383 (all CIs < 0) | 0.66–0.98 | 0.008–0.017 |
| random family (pooled) | −0.380 [−0.586,−0.182] | | |

Paired (readable − random family): **+0.427 [+0.214,+0.643]**.

## Pre-registered rules → outcome

- **Continuity gate PASSES**: erase_raw reproduces the historical null
  (−0.016, CI straddles zero).
- **Branch E fires**: erase_readable_stack CI includes zero — deleting
  the ENTIRE rank-9 INLP-reachable correctness subspace at all five
  readable layers, every position, prompt and decode, costs nothing
  (point estimate positive). Matched-rank random stacks with the
  identical estimator are catastrophic (−0.38; P(strong|parsed) collapses
  to ~0.01 — content demolition, not format). The registered 2026-07-04
  prediction (Branch E, variance-asymmetry mechanism) is confirmed.
- **Wording locked per the pre-registration**: this is NON-NECESSITY of
  the readable stack — not special inertness — because the telemetry
  shows the readable stack carries 10–1300× less within-run projection
  variance than the random stacks (the clamp removes almost nothing
  where the readout lives). Claim 3's future-work caveat becomes a
  result; the destruction-on-removal form of the entanglement account
  (arXiv 2605.05715) is excluded in our setting; the weak
  (redundant-carrier) form survives, as the round-1 reviews correctly
  insisted — retrained probes still decode after the scrub.

## Weak-lever probe (review-driven, descriptive — W4.2)

Original-correctness split: erase_raw on naturally-correct rows −0.062,
on incorrect rows +0.031; erase_readable_stack: 0.000 / +0.094. The
weak-lever account (clamping pulls failing rows toward the mean, so it
should visibly hurt correct rows) finds only sub-noise movement in the
predicted directions — no support for a behaviorally meaningful lever
hiding in the readout at this n. The uniform non-negative raw-erasure
sign pattern flagged by review 1 does not survive the correct-row split
as a lever signature.

## Claims consequences (W1 sweep)

- **Claim 2** upgrades: readout-axis non-necessity → readable-STACK
  non-necessity (rank 9 × 5 layers), with the same variance-mechanism
  scoping.
- **Claim 3**: "full-subspace erasure listed as future work" caveat is
  replaced by this result.
- §4 related work: 2605.05715's entanglement account is now directly
  discriminated — their LEACE erasure hurt; our full-stack erasure is
  harmless at 16-row/CI resolution with destructive matched-rank
  controls.
