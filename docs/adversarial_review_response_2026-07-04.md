# Adversarial Review Triage (2026-07-04, late night)

Two frontier-model reviews of `docs/review_packet_2026-07-04.md` (fresh
contexts, adversarial prompt). Verdicts: Major Revision; Reject. Both
independently praised the same things (specificity architecture, KV
telemetry discipline, pre-registration hygiene, claim 1, the geometry
computation) and independently attacked the same things — the convergent
attacks are the ones that matter.

## A. Convergent objections we ACCEPT — wording/status debt for the draft

1. **The "epiphenomenal / genuinely inert / destructive matched controls"
   framing must go, everywhere.** Both reviews correctly note the
   control-matching verdict (harmlessness tracks low within-run projection
   variance, not semantic identity; the near-zero-variance height direction
   erases as harmlessly as raw) retracts the strong framing — and that our
   own wording rule concedes this while the litreview positioning line
   ("genuine epiphenomenality ... destructive matched controls") and the
   abstract skeleton ("destructive matched noise") still use it. Fix at
   drafting: the landed claim is **non-necessity of the readout axis (and,
   if item D Branch E lands, of the INLP-reachable rank-9 stack)**. The
   variance confound is structural: no destructive low-variance erasure
   exists in the packet, so "harmless" carries limited bits about identity.
   Title "Gauge, Not Lever" survives only with the §5.2 text scoped to
   non-necessity + the variance account stated as the mechanism.
2. **Headline number swap.** Lead §5.3 with the fresh-row concept_replace
   (+0.255 [+0.111,+0.413]) not the dev-row +0.491; the dev-row number is
   reported as the selected-row upper anchor.
3. **Strike "replicated" for the +0.231 → +0.245 pair** (same rows, new
   seeds — same-row resampling, not replication). Applies to
   `hint_free_repair_direction.md` and any draft text.
4. **Claim 12 demoted from "landed."** Report pooled-new only (+0.117
   [+0.008,+0.254]); DROP the meta-pool with 457170 (hypothesis-generating
   job — winner's-curse contamination; both reviews caught it). Add
   leave-one-row-out sensitivity (3/13 rows carry the effect; dropping row
   3137 roughly halves it). New status: "suggestive, row-sparse,
   layer-localized; not a matched replication" — which is nearly what the
   caveat column already said; the status cell must match it.
5. **Equivalence discipline.** Claims 2/7/11 are equivalence claims
   defended by null CIs with no margin. Either run TOST with a declared
   margin (suggest 5pp) or reword: claim 7's "costs nothing" → "costs less
   than ~5pp detectable at this design"; claim 2/11 "no behavioral cost" →
   "no cost detectable at ±0.09–0.13 CI half-widths."
6. **Bootstrap coverage.** For every claim whose CI bound sits within
   ~0.02 of zero (claim 12; L35 rank4), add BCa or wild-cluster bootstrap
   and LOO-row sensitivity before quoting in the draft. Percentile CIs at
   13–26 clusters undercover.
7. **Disjunctive guard rule.** Acknowledge in methods that guard v2's
   "rank4 OR rank8" pass rule doubles null pass probability; the claim's
   content is "rank-8 is the fresh-row-portable core" and the rank-4
   failure at the 70% bar is reported alongside, not hidden.

## B. Objections ANSWERED by existing artifacts (checks run tonight)

1. **Parse-rate mediation of the repair arms** (both reviews). Computed
   from the row-level JSONLs of 458374/5 + 458401/2 (208 gens/arm): parse
   rates are flat across arms (0.88–0.93 vs baseline 0.904; rank8 add
   0.880–0.899 — slightly LOWER), and P(strong|parsed) preserves the full
   ordering: baseline 0.133; mean_only 0.224; rank8_loo 0.390/0.415;
   concept_replace 0.429; rand_norm 0.016–0.057. P(weak) moves in lockstep
   (0.438 → 0.630 under rank8). The repair is content, not format. Add
   this table to the draft's appendix.
2. **Qwen manipulation check** (review 1: "a silently no-op'd hook
   produces exactly this table"). The Qwen erasure rows log per-row
   `hook_summary` with calls, positions (217/row), and applied-delta
   magnitudes: erase_raw 8.95, erase_orthogonal 7.08, erase_gaussian 11.92
   sd-units (means over 128 rows × 4 layers). The hook fired and moved
   activations comparably across conditions; the null is real. Surface in
   the claim-11 evidence cell.
3. **Post-erasure decodability** (review 1: "if INLP round-10+ still
   decodes after the clamp, nothing was removed and the null is vacuous").
   We agree, and the artifact exists: probe_on_erased shows retrained AUC
   ~0.87–0.89 after 8 INLP rounds at every layer. This is exactly why
   claim 3 is worded as informational redundancy and why item D's Branch E
   wording rule pre-commits to non-necessity of the *erased stack*, not of
   "everything readable." The draft must say this out loud rather than
   leaving the reviewer to discover it.

## C. Convergent demand we ADOPT — the endogeneity test (item F candidate)

Both reviews independently identify the same missing experiment: nothing
in the packet shows the rank-8 subspace is an *endogenous* variable of
natural computation (all causal arms are hint-derived; misdirection nulls
mean the variable was never set to a different value; "commitment" is an
inferred label on an exogenous mediation result). Review 1's demanded
experiment, which we adopt as the item F candidate after items D/E land:

- Capture pre-generation L30 concept-position states for a fresh,
  BALANCED set of naturally-correct vs naturally-incorrect unhinted rows
  (no hints anywhere).
- (i) Test whether the rank-8 subspace linearly separates natural success
  from failure, against matched random-subspace nulls (CPU after capture).
- (ii) Add the rank-8 reconstruction of the natural correct-minus-incorrect
  CLASS-MEAN delta — donor-free and hint-free by construction — to failing
  rows, vs the mean_only floor and rand_norm control.
- Plus review-1's priming discriminator as a third arm when budget allows:
  a content-empty donor frame that mentions the gold concept without
  asserting it ("the word X appears below") — separates
  "recently-seen-token" copy/induction accounts from commitment.
- Plus a collateral-damage slice: the winning intervention applied to
  originally-correct rows (repair jobs currently have none; item D's
  balanced rowset covers erasure only).

Note the synergy with item E's shard-0 surprise (shufpred repairing at
+0.135 while true-coefficient dev recon sits at +0.067): generic
coefficients inside the true subspace repair, content in random subspaces
does not (item C). The class-mean arm in (ii) is the clean version of
exactly that observation. If (i) or (ii) lands, the focus state upgrades
from compressed hint-mediation to an identified endogenous variable — the
strongest possible answer to both reviews' central positive-side attack.

## D. Observations logged for the item D read

- **Sign pattern** (review 1): every raw-erasure dP point estimate in the
  packet is non-negative (+0.031, +0.070, +0.031/+0.063 preview) — the
  weak-lever account predicts this (clamping failing rows' projections
  toward the mean helps). Item D's balanced rowset gives the test: under
  weak-lever, erase_raw/erase_readable_stack should HURT the 8
  originally-correct rows. Add an original-correctness split to the item D
  pooled analysis (descriptive; not a pre-registered rule).
- Both steelmen concede the weak (redundant-carrier) entanglement account
  survives item D regardless of branch; the draft's related-work paragraph
  should claim only what Branch E buys: exclusion of the
  destruction-on-removal form in our setting.

## E. Where the reviews overreach (for the response letter, if ever)

- "No manipulation check in Qwen" — existed in the artifacts (B.2), was
  omitted from the packet summary. Packet error, not experiment error.
- "Parse mediation uncontrolled in repair jobs" — data always existed
  row-level; now computed (B.1).
- "Regression to the mean unhandled" — both reviews eventually credit the
  in-job regenerated-baseline design; review 2 restates it as open after
  crediting it.
- Review 2's "Reject" treats the wording debt (A) as if it were the
  evidence; with A executed, its own §6 concedes the two core results
  (raw-axis non-necessity; direction-specific rank-8 sufficiency on the
  stated slice).

## Execution order

1. Tonight: this triage (done); items D/E verdicts as jobs land.
2. Tomorrow with the verdicts: one wording sweep over
   litreview/skeleton/claims-table per §A (single commit), including the
   claim-12 demotion and meta-pool removal.
3. Pre-register item F (endogeneity test) using the §C design; queue after
   the erasure remainder.
4. B.1/B.2 tables go into the draft appendix when §5 is written.
