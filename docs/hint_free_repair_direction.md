# Hint-Free Repair: From Instrument to Method (research direction, 2026-07-04)

Status: ideation, not pre-registered. Nothing here is a claim of the current
paper; this documents the deployment question the rank-8 result makes askable,
the evidence for and against it being answerable, and the experiment ladder we
would run. Framing rule for the paper: one discussion paragraph, worded as
"the question this paper makes askable," never as a capability we have.

## The question

The rank-8 recipe repairs failing rows it was never tuned on (+0.231 guard v2,
with the same-row/new-seed resample at +0.245 in the specificity run — an internal-consistency check, not a replication). If failures on other hard tasks
have the same shape, does this generalize into a way to raise model accuracy
on difficult tasks — steering without knowing the answer?

## The catch in the current recipe (read this first)

What is shared across rows is only the **basis**. The **coefficients** are
row-specific and come from the row's own donor pass:
`stage2_rank_k_guard_v2.py` computes
`recon_pca = rank_k_reconstruction(prep["concept_delta"], basis)` where
`concept_delta` is *that row's* hinted-minus-unhinted state difference. So
repairing a fresh row still requires running it once **with the hint** — the
recipe consumes the answer key. The hint alone is worth +0.750; the patch
+0.245. As it stands this is a measurement instrument for how commitment is
represented, not an accuracy method. The missing piece is a **policy that
predicts the 8 coefficients from the unhinted state alone**.

## Why it worked here — prerequisites any target task must meet

1. **Capability present, decision wrong.** Recognition 14/14, candidates
   0.839–1.000: the model has everything except the proposal choice. Patching
   unsticks a decision; it cannot add missing knowledge or reasoning.
2. **Homogeneous bottleneck.** Every failure is the same failure (wrong
   target-concept commitment), which is why PCA across rows finds a compact
   shared code. Heterogeneous failure modes would not compress.
3. **A cheap donor condition exists** (the one-line concept hint) to generate
   the deltas at all.
4. **Localization is known** (concept-mention tokens, L30) — and was
   expensive to find. It did not transfer even to our sibling task: subtype
   needed L35, at a quarter of the effect size, carried by 3/13 rows
   (claim 12). Cross-task portability is the weakest link, consistent with
   arXiv 2604.03867 (optimal steering layers vary per input) and 2605.03907
   (token-position-dependent steering).

## Evidence the hint-free version might be buildable

- **The mean component is already hint-free.** `mean_only_add_L30` is one
  fixed vector (no donor pass, no answer knowledge) and yields +0.087
  [+0.019, +0.173] on rows with baseline 0.120 (specificity pooled verdict).
  A constant task-level nudge recovers ~1/3 of the row-specific effect.
- **Only 8 numbers to predict.** The policy's output space is tiny; the
  question is whether the unhinted activations contain the information.
- **The gauge shows failure is predictable** (test AUC 0.902 at L53 from the
  same pre-generation states), so "when to intervene" is available. Whether
  "which commitment" is also linearly available unhinted is open — note the
  rank core is nearly orthogonal to the readable correctness subspace
  (claim 9), so the gauge itself will not hand us the coefficients.
- **Misdirection asymmetry** (claim 8: foreign deltas never misdirect): the
  subspace appears biased toward restoring correct commitments rather than
  arbitrary overrides — a desirable property for a deployed intervention,
  and some protection against a badly-calibrated policy doing harm.

## Evidence for humility

- Effect ceiling: the best patch is a third of the hint's behavioral effect.
- Task transfer already degraded within our own dataset (subtype).
- Row-sparsity on subtype suggests the "one shared variable" prerequisite
  can fail even on near-neighbor tasks.
- White-box access, per-task localization R&D, and a donor condition are all
  required inputs; none is free on a new task.

## Experiment ladder (cheapest first, each gates the next)

1. **Coefficient predictability probe (CPU, offline).** On existing guard/
   specificity rows we already have unhinted states and the true per-row
   rank-8 coefficients. Fit ridge regression from unhinted concept-token
   states (L30) to the 8 coefficients, LOO across rows. Gate: predicted-vs-
   true cosine and R^2 meaningfully above a shuffled-rows null.
2. **Closed-loop repair with predicted coefficients (1 GPU job).** Same arms
   harness as the specificity run: unhinted baseline, mean_only (floor),
   rank8 with true coefficients (ceiling), rank8 with LOO-predicted
   coefficients, rank8 with shuffled-row coefficients (control). Success =
   predicted arm sits materially above mean_only and the shuffle control.
3. **When-to-fire gating.** Combine with the correctness probe: intervene
   only when the gauge says failure. Measures collateral damage on rows that
   were going to succeed (the deployment-relevant metric).
4. **Second task port** (subtype at L35, then Qwen): does the *policy
   recipe* — not the directions — transfer?

Step 1 costs an afternoon and requires no new generation; steps 2–3 are one
sbatch each on the existing guard-v2 harness. If step 1 shows the
coefficients are not decodable from unhinted states, the direction dies
cheaply and the paper paragraph stays as "open question."

## Step 1 result (2026-07-04): coefficients ARE decodable — gate passes

Run on `focus_state_composite_27b_property_states.npz` (13 dev rows, 120
concept positions, L30). Outer LOO by row; per-fold LOO rank-8 basis via
`fit_pca_basis` (mirroring the guard); ridge from unhinted concept-token
states to the 8 coefficients, alpha picked by inner LOO over
{1e2..1e6} (chosen: 1e2/1e3); 50-permutation shuffled-pairing null.

- cosine(predicted, true) = **+0.631** over 120 held-out positions.
- Shuffled null mean +0.014, max over 50 permutations +0.180 → observed
  beats **0/50**.
- Per-row R^2 vs constant-coefficient baseline **+0.408**, positive on
  12/13 rows. (The constant baseline itself is ~0 cosine by construction —
  centered-PCA coefficients average to ~0 across rows — so the null, not
  the baseline, is the meaningful bar.)

Reading: ~40% of the row-specific commitment coordinates are linearly
present in the failing run's own states at the same positions. The donor
pass amplifies information the unhinted model already carries.

Caveats before anyone gets excited: 13 development rows only (the rows the
rank-8 story was built on — the only ones with saved unhinted states);
screening-grade n; cosine ≠ behavioral repair (coefficient error may or
may not matter behaviorally — that is exactly what step 2 measures).
Analysis script: `scripts/stage2_coeff_predictability.py`; deterministic
(seed 20260704).

**Step 2 ran 2026-07-04 (jobs 458409/458410, item E) and landed on the
FAIL branch** (`docs/rank8_predcoeff_27b_property_pooled_summary.md`):
the predictor transferred (out-of-sample cos ≈ +0.59 vs ≈ 0 shuffled) but
prediction quality did not convert to behavior — (pred − shufpred) +0.014
[−0.034,+0.072]. Cause: the dev basis under-transfers (ceiling +0.135 vs
+0.245 under a fresh-row LOO basis on the same rows), collapsing the
dynamic range. The informative surprise: EVERY arm writing into the dev
rank-8 subspace repairs (+0.091..+0.154, all CIs excluding zero, shuffled
coefficients included) while random subspaces (item C) do not —
**subspace ≫ coefficients**. Step 3 does not unlock; the thread folds
into item F (endogeneity test), whose class-mean arm uses a fresh-row
basis and whose deployment question becomes "right subspace + right
scale," not per-row coefficient prediction.

## Relation to the current paper

Goes in §6 Discussion as one paragraph: some failures that look like
capability gaps are single, low-dimensional, causally accessible decision
errors; this paper locates, compresses, and flips them when the target is
known; whether the target can be inferred without the hint is the natural
next question. Do not promise the ladder in the paper; cite mean_only as the
existence proof that a nonzero hint-free effect is real.
