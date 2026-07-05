# Shuffled-Label Projected Control (item F(ii)-b) — Pooled Verdict, Jobs 458424 + 458425

Shard outputs: `docs/classmean_b_controls_27b_property_shard{0,1}of2.json`;
row-level in `results/stage2/erasure/classmean_b_controls_27b_property_shard{0,1}of2.jsonl`.
26 rows pooled (guard-v2 fresh selection); row-cluster bootstrap (10k, seed
20260704), paired vs in-job unhinted baseline. F(ii)'s real-proj arm reused
via verified determinism. Pre-registered in `causal_handle_directions.md`
F(ii)-b after two round-2 adversarial reviews made this their condition for
publication — both predicting the GENERIC outcome.

## Integrity gate

**PASS — 26/26 rows** of the in-job unhinted baseline reproduce F(ii)'s
per-row sample outcomes exactly (end-to-end determinism re-confirmed; the
reuse of F(ii)'s real-proj arm is valid).

## Pooled arms (26 rows, k=8)

| arm | dP vs unhinted (CI95) |
| --- | ---: |
| real class-mean, projected (from F(ii)) | **+0.341 [+0.202,+0.495]** |
| shuffled-label family (d1–d4 pooled) | **−0.043 [−0.103,+0.006]** |
| sign-flipped real vector | **−0.120 [−0.245,−0.024]** |
| fixed-pooled-norm real vector (donor-free) | **+0.399 [+0.245,+0.558]** |

Per-draw descriptive: d1 −0.053, d2 +0.024, d3 −0.029, d4 −0.115 — no
draw repairs; d4's CI is entirely negative.

Paired contrasts: (real − shuffled family) **+0.385 [+0.245,+0.534]**;
(real − signflip) +0.462 [+0.298,+0.625]; (fixednorm − shuffled family)
+0.442 [+0.293,+0.600].

## Pre-registered rules → outcome

- **LABEL-SPECIFIC**: paired (real − shuffled family) CI excludes zero ✓
  AND the family sits below 50% of the real effect ✓ (it is ~0, slightly
  negative). **The branch fires at full strength.**
- **Sign-flip rider**: actively destructive (−0.120, CI excludes zero) —
  the direction's SIGN carries the effect, exactly the label-content
  prediction and inconsistent with any pure-geometry account.
- **Donor-free rider**: fixednorm **+0.399 [+0.245,+0.558]** — the
  first fully donor-free repair in the program: direction from natural
  correct/incorrect behavior (no hints anywhere), amplitude from OTHER
  rows' pooled scale. Numerically the strongest arm; vs the real
  (donor-calibrated) arm the difference is +0.058 [−0.005,+0.128] —
  statistically indistinguishable, i.e., the donor calibration was never
  the active ingredient.

## What this settles

1. **+0.341 is label-specific.** Identical geometry, subspace, positions,
   and norms with scrambled correct/incorrect labels produce nothing
   (−0.043); flipping the true direction hurts (−0.120). The information
   about what makes answers correct carries the repair.
2. **The round-2 reviews' strongest objection is answered by their own
   demanded control.** The "top-variance cone" account (any
   difference-shaped in-subspace write at this scale repairs) predicted
   the shuffled family would land within +0.341's CI; observed: −0.043.
   Both reviewers' point predictions are falsified in our favor.
3. **F(i)+F(ii)+F(ii)-b coherent picture**: the natural class-mean
   direction is causally potent through its lever-subspace component
   (label-specific, sign-sensitive), while per-row outcomes remain
   non-decodable from that subspace (F(i)). Causal alignment without
   per-row decodable alignment — earned, not asserted.
4. **Claim 8 wording consequence (W1 sweep)**: "the specific 8 PCA
   directions carry the repair" stands, now guarded by an on-distribution
   control; the hint-free thread's next-paper opener is the fixednorm
   result.

Caveats: 26 rows, one model, one task, one layer; the shuffled family's
between-draw spread (d2 +0.024 vs d4 −0.115) is visible at 4 draws;
fixednorm's superiority over real-proj is a point estimate, not a claim.
