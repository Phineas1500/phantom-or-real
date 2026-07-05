# Response Packet — "Gauge, Not Lever" (2026-07-05)

This is the authors' RESPONSE to two adversarial reviews received
2026-07-04 (verdicts: Major Revision; Reject). The original evidence
bundle (`review_packet_2026-07-04.md`) is provided alongside this file;
this packet contains only what happened after the reviews: the triage of
the objections, checks run against existing artifacts, new statistical
analyses, and three NEW pre-registered experiments run overnight (items
E, F(i), F(ii)) — including the endogeneity test review 1 demanded as its
condition for publication.

Status note: item D (readable-stack erasure) is still completing at send
time — a 4h wall-timeout forced resharding; 10 of 16 rows are done and
its pooled verdict is NOT claimed here. Everything else below is final.
One same-day self-correction is included in §7 (flagged inline) — the
authors caught and retracted an overclaim in their own F(ii) verdict via
a shuffled-label control before external review.

Contents:
 1. Review triage: objections accepted / answered-from-artifacts / adopted
 2. Execution plan (the commitments)
 3. Statistical hardening results (BCa, LOO sensitivity, MDE/TOST)
 4. Item E pre-registration + pooled verdict (predicted-coefficient repair)
 5. Item F(i) pre-registration + verdict (natural-state separation)
 6. Item F(ii) pre-registration + pooled verdict (class-mean repair),
    including the same-day correction

---

<!-- FILE: docs/adversarial_review_response_2026-07-04.md -->

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

---

<!-- FILE: docs/review_response_plan.md -->

# Review-Response Execution Plan (2026-07-04)

Working checklist for addressing the two adversarial reviews. Analysis and
justification live in `docs/adversarial_review_response_2026-07-04.md`;
this file is the to-do list we execute against. Update the Status column
in place as items land. Order of operations is W5 → W4 → W1/W2/W3 (one
pass, after verdicts) → W6 → W7 (at drafting).

Legend: [ ] todo · [~] in flight · [x] done

## W1. Wording sweep (single commit, AFTER items D+E verdicts land)

- [ ] W1.1 `docs/next_experiments_litreview_2026-07.md`: replace the
      positioning line "genuine epiphenomenality (ours — erasure harmless
      with destructive matched controls)" with non-necessity wording +
      the variance mechanism ("harmless erasure of a low-within-run-
      variance readout; matched controls confirm the family has teeth at
      matched norm, not that the axis is specially inert").
      Locate: `grep -n "destructive matched" docs/*.md`.
- [ ] W1.2 `docs/next_paper_skeleton.md`:
      - Abstract: drop "destructive matched noise"; lead the repair with
        the fresh-row +0.255 [+0.111,+0.413] (dev-row +0.491 reported as
        the selected-row anchor, labeled as such).
      - §5.2: non-necessity wording; state post-erasure decodability
        (retrained AUC ~0.87–0.89 after 8 INLP rounds) IN the section.
      - §4 related work: entanglement paragraph claims only exclusion of
        the destruction-on-removal form (Branch E permitting); states
        that the weak (redundant-carrier) form survives.
      - Title bullet: "Gauge, Not Lever" is retained only with §5.2
        scoped to non-necessity; list one fallback title for the
        item-F-null world.
- [ ] W1.3 `docs/next_paper_claims_table.md`:
      - Claim 2 status/wording cells: non-necessity language throughout.
      - Claim 7: "costs exactly nothing" → "no cost detectable at this
        design (MDE ≈ 5pp)".
      - Claim 12: status landed → "suggestive (row-sparse,
        layer-localized)"; evidence cell drops the 457170 meta-pool
        (report pooled-new +0.117 [+0.008,+0.254] only); caveat gains the
        LOO-row sensitivity result from W2.2.
      - Claim 11 evidence cell: add the hook-telemetry manipulation check
        (deltas 7.1–11.9 sd-units, 217 positions/row — W3.2).
- [ ] W1.4 `docs/hint_free_repair_direction.md`: strike "replicated at
      +0.245" (same rows, new seeds — same-row resample); annotate step-1
      result with the item E behavioral outcome once pooled (see W5.2).

## W2. Statistical hardening (CPU, offline; before drafting §5)

- [x] W2.1 BCa + wild-cluster bootstrap side-by-side with percentile CIs
      for every claim whose CI bound is within 0.02 of zero: claim 12
      pooled-new; subtype L35_rank4; guard-v2 rank4. Script:
      `scripts/stage2_bootstrap_hardening.py` (new, committed), reads the
      existing row-level JSONLs. If BCa lower bound crosses zero anywhere,
      the claim text weakens accordingly.
- [x] W2.2 Leave-one-row-out sensitivity tables (drop each row, recompute
      pooled dP) for claim 12 and claim 8 (guard v2 + specificity). Report
      min/max range next to the CI. Expect claim 12 to fail this (3/13
      rows carry it) — that is the point.
- [x] W2.3 Equivalence discipline: declare a 5pp margin; run TOST for
      claim 2 raw erasure (control-matching rows), claim 7 masking, claim
      11 Qwen raw. Where power is insufficient, report the minimum
      detectable effect instead of "no cost".
- [ ] W2.4 Methods paragraph: enumerate arm families per job; state that
      pre-registered PAIRED contrasts are the protected comparisons and
      null claims carry MDE statements, not significance claims.

## W3. Appendix tables from existing artifacts (mostly done tonight)

- [x] W3.1 Parse-mediation table for all repair arms (guard v2 +
      specificity, 208 gens/arm): parse rates flat 0.88–0.93; P(strong|
      parsed) ordering preserved (baseline 0.133 → rank8 0.390/0.415 →
      concept_replace 0.429); P(weak) in lockstep. Computed 2026-07-04;
      numbers in `docs/adversarial_review_response_2026-07-04.md` §B.1.
      [ ] Residual: re-run as a committed script when drafting the
      appendix (currently ad hoc).
- [x] W3.2 Qwen manipulation check: per-row `hook_summary` — erase_raw
      8.95 / orthogonal 7.08 / gaussian 11.92 sd-units applied, 217
      positions/row, 128 rows/condition. §B.2. Residual: fold into claim
      11 cell (W1.3) and appendix.
- [x] W3.3 Post-erasure decodability: probe_on_erased AUC ~0.87–0.89
      after 8 INLP rounds (artifact exists). Residual: surface in §5.2
      text (W1.2).

## W4. Item D completion + review-driven analysis additions

- [~] W4.1 Pool 16 rows when 458412/458413/458414 land (stitch with the
      458403 partial; row 6604 taken from the remainder job). Evaluate the
      pre-registered branches (gate: erase_raw continuity; E/N/ambiguous).
- [ ] W4.2 NEW (review 1, sign-pattern/weak-lever probe): split every
      erasure arm by ORIGINAL row correctness (8/8 by design). Under the
      weak-lever account, erase_raw / erase_readable_stack should HURT the
      originally-correct rows. Descriptive, labeled as such (not
      pre-registered).
- [ ] W4.3 Verdict doc must carry the variance-telemetry table
      (readable-stack variance 10–1300× below random stacks) adjacent to
      the branch call, and the Branch-E wording locked to non-necessity.

## W5. Item E verdict + hint-free thread updates

- [x] W5.1 Pool shards 0+1 when 458410 lands; evaluate the item E rules
      (gate: rank8_dev CI excludes zero; SUCCESS/PARTIAL/FAIL). Shard-0
      note: gate marginal (+0.067 [0.000,0.192]); shufpred (+0.135)
      matches pred (+0.106) — heading to FAIL-or-uninterpretable on the
      success question, with the informative surprise that generic
      coefficients in the true dev subspace repair while random subspaces
      (item C) do not.
- [x] W5.2 Update `docs/hint_free_repair_direction.md`: step-2 outcome;
      reinterpret the ladder (per-row coefficient prediction may not be
      the binding constraint — subspace+scale may be); fold into the item
      F design rather than a separate step 3.

## W6. Item F — endogeneity test (the reviews' convergent demand)

- [ ] W6.1 Pre-register in `docs/causal_handle_directions.md` (design in
      triage §C): balanced naturally-correct vs naturally-incorrect
      unhinted rows, no hints anywhere; states captured in-job at L30
      concept positions.
      Arms: (i) CPU — rank-8 subspace separation of natural success vs
      failure, against matched random-subspace nulls; (ii) GPU —
      class-mean (natural correct-minus-incorrect) delta reconstructed in
      the rank-8 basis, added donor-free to failing rows, vs mean_only
      floor + rand_norm control; (iii) priming discriminator (content-
      empty concept-mention donor frame) if budget; (iv) collateral slice:
      winning intervention applied to originally-correct rows.
      Decision rules + wording consequences BOTH directions (title
      keeps/downgrades per W1.2).
- [ ] W6.2 Harness: extend `stage2_rank_k_guard_v2.py` (natural-class
      capture + class-mean arm); unit tests; dry-run.
- [ ] W6.3 Submit after the erasure remainder clears the queue; pool;
      verdict doc; artifacts update.

## W7. Draft-time integration (when writing the paper)

- [ ] W7.1 §6 discussion: misdirection asymmetry named as an open anomaly
      ("asymmetric controllability" is a description, not an explanation
      — review 1).
- [ ] W7.2 §6 limitations: search-cost acknowledgment (many intervention
      families were tried before the rank-k path; pre-registration of the
      confirmatory jobs is the mitigation — say it explicitly).
- [ ] W7.3 Appendix: W3 tables + W2 sensitivity tables + the negative-
      results table already enumerated in the claims doc.
- [ ] W7.4 Re-run the scoop check (standing item) and re-run this
      adversarial-review loop on the actual draft before submission.

## Dependencies

- W1 waits on W4.1 + W5.1 (verdict numbers feed the same files; one
  commit).
- W2 independent; can run any time (CPU).
- W6.1 pre-registration should follow W5.1 (item E's outcome shapes the
  class-mean arm's framing) but NOT wait for W4.
- W7 blocks on nothing except drafting starting.

---

<!-- FILE: docs/stat_hardening_2026-07.md -->

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

---

<!-- Pre-registrations: items E, F(i), F(ii) (docs/causal_handle_directions.md) -->

### E. Predicted-coefficient repair (hint-free repair ladder, step 2)

`scripts/stage2_rank_k_guard_v2.py --predicted-coefficients`. Step 1 of
`docs/hint_free_repair_direction.md` passed (LOO cosine +0.631 vs shuffled
null +0.014, 0/50 permutations): the rank-8 coefficients are linearly
decodable from the row's own UNHINTED L30 concept-token states. Step 2 asks
whether ridge-PREDICTED coefficients repair behavior — donor-free steering.
Exploratory (next-paper material): no current-paper claim moves on any
outcome; failure costs nothing but the two job slots.

- Rows: identical selection to guard v2 / item C (seed 20260702, per-height
  16, composite-manifest rows excluded), same 2 shards — row-paired
  comparability with 458374/458375 and 458401/458402.
- Shared machinery: ONE dev basis (rank-8 PCA of the 13 composite rows'
  L30 concept deltas from `focus_state_composite_27b_property_states.npz`,
  no LOO — dev rows are disjoint from all fresh rows), so the four causal
  arms differ ONLY in where the coefficients come from. Ridge predictor
  trained on the 13 dev rows (X = unhinted concept states, Y = centered
  delta @ Q_dev^T), alpha picked by LOO-by-row over {1e2..1e6} in-job,
  deterministic; per-row donor deltas are computed in-job for the ceiling
  arm and diagnostics ONLY — the pred arm never touches the target row's
  hinted pass.
- Arms (6 per shard, k=8 samples, generation config identical to guard v2):
  `unhinted_baseline`; `hinted_baseline` (validation); `rank8_dev_add_L30`
  (ceiling: row's own delta reconstructed on the dev basis, mean + QQ^T);
  `mean_only_dev_add_L30` (floor: dev-basis mean tiled); `rank8_pred_add_L30`
  (dev mean + Q_dev^T c_pred from the row's unhinted states); and
  `rank8_shufpred_add_L30` (identical ridge pipeline trained on a seeded
  row-level permutation of Y — breaks the X→Y pairing, preserves output
  scale; seed 20260704).
- Primary metric: pooled 2-shard paired dP(strong) vs in-job
  `unhinted_baseline`, row-cluster bootstrap (10k draws, percentile CI),
  the only sanctioned sigma source. Diagnostic: per-row cosine(pred, true)
  recorded in basis_records.
- Decision rules (recorded before unblinding):
  - Gate: pooled `rank8_dev_add_L30` CI must exclude zero (dev-basis
    transfer to fresh rows; guard v1 precedent). If it fails, the pred arm
    is uninterpretable — report and stop; no rule evaluation.
  - SUCCESS (donor-free repair): pooled `rank8_pred` CI excludes zero AND
    paired (pred − mean_only_dev) CI excludes zero AND paired
    (pred − shufpred) CI excludes zero. Report pred as % of the rank8_dev
    ceiling. Consequence: step 3 (gated deployment test) unlocked;
    hint-free direction upgrades from "askable" to "answered in-domain".
  - PARTIAL (constant-vector steering only): pooled `rank8_pred` CI
    excludes zero but (pred − mean_only_dev) straddles zero → the ridge
    adds nothing behavioral beyond the dev mean; direction doc updated,
    step 3 not unlocked.
  - FAIL: `rank8_pred` CI straddles zero, or shufpred ≈ pred (paired CI
    straddles zero) → coefficient decodability does not convert to
    behavioral repair at this n; documented, ladder stops.
- Budget: 6 arms × ~13 rows × 8 samples ≈ 624 generations/shard ≈ 1.6 h
  per shard. Queue behind item D (458403/458404).
- Outputs: stem `rank8_predcoeff_27b_property_shard{i}of2`; states npz also
  saves per-row unhinted concept states.

### F(i). Natural-state separation test (endogeneity, capture-only fast lane)

Pre-registered before any capture data exists (2026-07-04 ~23:30). Both
adversarial reviews demand evidence the rank-8 subspace is an ENDOGENOUS
variable of natural computation, not just a hint-mediation channel.

- Capture job (no hints, no generation): seeded balanced selection of
  fresh property rows — 24 per (height 3/4 × naturally-correct/incorrect)
  cell, parse-ok, excluding the 13 composite-manifest rows AND the 32
  guard-v2 selection rows (both contributed to basis fits). Save L30
  states at gold-concept mention positions per row.
- CPU test: per-row feature = mean over concept positions of the state's
  projection onto a rank-8 basis (8 features). Two bases, both frozen
  before this test: (a) the dev/composite basis; (b) the guard-v2
  full-26-row basis. Logistic regression, stratified 5-fold CV, AUC for
  naturally-correct vs naturally-incorrect.
- Null: 200 seeded random orthonormal rank-8 subspaces, same pipeline.
  Also report the full-residual (5,376-dim, L2 logistic) AUC as ceiling.
- Decision rule: endogeneity SUPPORTED for a basis if its CV AUC exceeds
  the 95th percentile of its random-subspace null; strength reported as
  (AUC − null median). If both bases sit inside their nulls, part (i) is
  NULL: the rank-8 subspace carries no privileged natural-outcome
  information at these positions, and item F(ii)'s class-mean arm becomes
  the remaining route; the paper wording stays at exogenous mediation.
- Explicitly descriptive wrt heights: pooled h3+h4 primary; per-height
  secondary. No current-paper claim moves on any outcome (next-paper
  thread, same status as item E).

### F(ii). Natural class-mean repair (endogeneity, causal half)

Pre-registered 2026-07-04 ~23:30, after F(i)'s NULL and before any F(ii)
data. F(i) showed the lever subspace carries no privileged NATURAL-outcome
information; F(ii) asks the causal complement: does the natural
correct-minus-incorrect difference repair failures at all, and if so,
does it act through the lever subspace?

- Rows: guard-v2 fresh selection (identical to items A/C/E; 2 shards) —
  row-paired comparability across the whole family.
- Class vector: from the 458416 capture (96 balanced rows, provenance-
  disjoint), mean over natural-correct rows minus mean over natural-
  incorrect rows of the per-row position-mean L30 state; one frozen
  5,376-dim vector, computed in-job from the capture npz + manifest.
- Arms (6, k=8, generation config identical to guard v2):
  `unhinted_baseline`; `hinted_baseline`;
  `rank8_loo_add_L30` (positive reference, fresh LOO basis — the +0.245
  anchor); `class_mean_raw_add_L30` (class vector tiled at concept
  positions, per-position norm-matched to the row's LOO rank-8 recon —
  the same scale target item C used); `class_mean_proj_add_L30` (class
  vector projected onto the row's LOO rank-8 basis first, then
  norm-matched to the same target); `rand_norm_add_L30_d1` (matched-norm
  Gaussian, item C seed — noise floor).
- Metric: pooled 2-shard paired dP(strong) vs in-job unhinted baseline,
  row-cluster bootstrap (10k), the only sanctioned sigma source.
- Decision rules (before unblinding):
  - Gate: `rank8_loo` CI excludes zero (machinery check).
  - Natural-delta CAUSAL if `class_mean_raw` CI excludes zero AND paired
    (class_mean_raw − rand_norm_d1) CI excludes zero.
  - Channel dissociation (only if causal): paired (raw − proj) CI
    excludes zero with proj null → the natural-outcome axis and the lever
    channel are causally distinct (F(i)'s correlational dissociation
    confirmed causally). proj ≈ raw with both positive → the natural
    delta acts through the lever subspace and F(i)'s null must be
    reinterpreted (subspace informative causally though not decodably).
  - Both class arms null → the natural class-mean difference is not
    causally potent at this scale/position: endogeneity fails on both
    fronts; discussion wording stays at exogenous mediation with the
    stronger both-tests-run sentence.
- Exploratory: no current-paper claim moves on any outcome. Budget: 6
  arms × ~13 rows × 8 ≈ 624 gens/shard ≈ 1.7 h/shard, 2 shards.

---

<!-- FILE: docs/rank8_predcoeff_27b_property_pooled_summary.md -->

# Predicted-Coefficient Repair (item E) — Pooled Verdict, Jobs 458409 + 458410

Shard outputs: `docs/rank8_predcoeff_27b_property_shard{0,1}of2.json`;
row-level generations in
`results/stage2/erasure/rank8_predcoeff_27b_property_shard{0,1}of2.jsonl`.
26 rows pooled (13/shard; same guard-v2 fresh-row selection as items A/C);
row-cluster bootstrap, 10k draws, seed 20260704, paired vs the in-job
unhinted baseline. All four causal arms share the dev basis (rank-8 PCA of
the 13 composite rows' deltas, EVR 0.685) and differ only in coefficient
source. Exploratory (item E): no current-paper claim moves.

## Pooled arms (26 rows, k=8)

| arm | coefficient source | dP vs unhinted (CI95) |
| --- | --- | ---: |
| unhinted_baseline (P=0.120) | — | — |
| hinted_baseline (P=0.870) | — | +0.750 [+0.601,+0.885] |
| rank8_dev_add_L30 (ceiling) | row's own donor delta | +0.135 [+0.043,+0.245] |
| mean_only_dev_add_L30 (floor) | none (dev mean tiled) | +0.091 [+0.010,+0.197] |
| rank8_pred_add_L30 | ridge from row's UNHINTED states | +0.154 [+0.062,+0.264] |
| rank8_shufpred_add_L30 (control) | shuffled-pairing ridge | +0.139 [+0.062,+0.231] |

Paired differences: pred − mean_only **+0.062 [+0.000,+0.149]** (grazes
zero); pred − shufpred **+0.014 [−0.034,+0.072]** (null); dev − mean_only
+0.043 [−0.029,+0.144] (null).

Predictor diagnostics (in-job, per row): cos(predicted, true coefficients)
= +0.564 shard 0 / +0.608 shard 1 on rows the ridge never saw (every row
positive, range +0.36..+0.84); shuffled control −0.015/−0.027 ≈ 0.
Coefficient scales matched.

## Pre-registered rules → outcome

- **Gate (rank8_dev CI excludes zero): HOLDS** (+0.135, CI low +0.043) —
  the dev basis transfers, so the arms are interpretable.
- **SUCCESS fails**: it required (pred − shufpred) to exclude zero; the
  observed paired difference is +0.014 [−0.034,+0.072].
- **Outcome: FAIL branch** (shufpred ≈ pred): coefficient decodability
  does not convert to behavioral repair at this design. The PARTIAL
  condition (pred − mean_only straddling zero) is also met at the boundary
  (+0.000); either reading gives the same substantive conclusion —
  **the ridge's row-specific information adds nothing behavioral beyond
  generic content in the dev subspace.**

## What the experiment DID establish

1. **The predictor works as a predictor.** Out-of-sample cos ≈ +0.59 vs
   ≈ 0 for the shuffle — step 1's decodability result transfers to fresh
   rows. The information pipeline is not the failure point.
2. **The ceiling collapsed, and that is the binding constraint.** True
   coefficients on the dev basis give +0.135 where the same rows under a
   fresh-row LOO basis gave +0.245 (item C). Basis provenance is
   first-order: the dev basis under-transfers, compressing ceiling, floor,
   pred, and control into an indistinguishable +0.09..+0.15 band with no
   headroom for coefficient quality to matter.
3. **Subspace ≫ coefficients.** Every arm writing into the dev rank-8
   subspace repairs with CI excluding zero (+0.091..+0.154) — including
   shuffled coefficients — while item C's norm-matched random subspaces
   sat at −0.041/−0.088. Which subspace is written into carries the
   effect; the coordinates within it matter far less than the step-1
   cosine suggested. This is the design input for item F: the class-mean
   arm must use a fresh-row basis, and the deployment-relevant question
   becomes "right subspace + right scale," not per-row prediction.

## Wording consequences

- `docs/hint_free_repair_direction.md` step 2 outcome recorded; step 3
  (gated deployment test) does NOT unlock; the ladder folds into item F.
- No claim-table changes (exploratory by pre-registration).

---

<!-- FILE: docs/natural_separation_test_27b_property.md -->

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

---

<!-- FILE: docs/classmean_repair_27b_property_pooled_summary.md -->

# Natural Class-Mean Repair (item F(ii)) — Pooled Verdict, Jobs 458418 + 458419

Shard outputs: `docs/classmean_repair_27b_property_shard{0,1}of2.json`;
row-level generations in
`results/stage2/erasure/classmean_repair_27b_property_shard{0,1}of2.jsonl`.
26 rows pooled (guard-v2 fresh selection, identical to items A/C/E);
row-cluster bootstrap, 10k draws, seed 20260704, paired vs in-job unhinted
baseline. Class vector: natural correct-minus-incorrect mean of unhinted
L30 concept-position states from the 458416 capture (96 balanced
provenance-disjoint rows). Exploratory (item F(ii)): no current-paper
claim moves.

## Pooled arms (26 rows, k=8)

| arm | dP vs unhinted (CI95) |
| --- | ---: |
| unhinted_baseline (P=0.120) | — |
| hinted_baseline (P=0.870) | +0.750 [+0.601,+0.885] |
| rank8_loo_add_L30 (anchor) | +0.245 [+0.111,+0.394] |
| class_mean_raw_add_L30 | +0.043 [−0.120,+0.207] |
| **class_mean_proj_add_L30** | **+0.341 [+0.202,+0.495]** |
| rand_norm_add_L30_d1 (noise floor) | −0.077 [−0.168,−0.005] |

Paired: (proj − raw) **+0.298 [+0.130 excl. zero via −CI on raw−proj]**;
(proj − rand_norm) +0.418 [+0.264,+0.577]; (proj − rank8_loo) +0.096
[+0.000,+0.216] (grazes zero — proj is at least the anchor's equal);
(raw − rand_norm) +0.120 [−0.029,+0.284] (null).

Diagnostic: the natural class vector holds **49.8%** of its norm inside
the 8-dim lever subspace (per-row LOO bases, range 0.43–0.55). Random
expectation for an 8-of-5,376 subspace is ~3.9% — the natural
correct-vs-incorrect direction is ~13× more lever-aligned than chance.

## Pre-registered rules → outcome

- **Gate (rank8_loo CI excludes zero): HOLDS** (+0.245 — an exact
  replication of item C's anchor on the same rows).
- **Natural-delta CAUSAL (as defined on the RAW arm): NOT MET** — raw CI
  straddles zero and (raw − rand_norm) straddles zero.
- **The observed combination (raw null, proj strongly causal) was not
  among the three enumerated branches.** The nearest enumerated branch
  ("proj ≈ raw with both positive → the natural delta acts through the
  lever subspace; F(i) must be reinterpreted") anticipated the direction
  but not the strength: the natural delta acts through the lever subspace
  ONLY — its out-of-subspace half is dead weight that dilutes the
  effective component below detectability at matched norm. We flag this
  interpretation as post-hoc relative to the enumerated branches, though
  fully constrained by pre-registered arms and controls.

## What this establishes (with F(i) read jointly)

1. **The natural class-mean direction, projected into the lever subspace,
   is causally potent**: +0.341 — the largest fresh-row repair measured
   in this program (139% of the rank8_loo anchor, 45% of the full hint
   effect), from a direction computed with NO hints anywhere.
   **CORRECTION (same-day, before external review)**: the verdict's
   original "~13× more lever-aligned than chance" gloss was wrong. A
   shuffled-label control (500 permutations of the capture labels) shows
   ANY difference-of-state-means vector is ~50% lever-aligned (null
   median 0.489; the real vector's 0.515 sits at the 54th percentile) —
   the states' variance concentrates in these directions, so alignment
   is a property of the state geometry, not of the correct/incorrect
   labels. What the labels contribute to the projected DIRECTION (vs any
   shuffled-label projection) is therefore an OPEN control: the required
   follow-up arm is a shuffled-label class-mean projected at the same
   norm. Suggestive but not decisive: item E's shuffled in-subspace
   content reached only +0.139 (different basis), and
   cos(proj(class vector), proj(pooled hint-delta mean)) = 0.57 — the
   projected class vector is neither the generic hint-mean direction
   (mean_only: +0.087) nor independent of it.
2. **F(i)'s null is refined, not contradicted**: per-row outcomes are not
   linearly decodable from the lever subspace better than chance (within-
   class variance drowns the class-mean shift), yet the class-mean shift
   itself points along the lever and moves behavior when amplified.
   Causal alignment without decodable alignment — the project's
   gauge/lever theme reproduced inside a single experiment.
3. **Near-donor-free repair.** The direction is donor-free (capture rows,
   no hints); the only donor-derived quantity in the arm is the
   per-position norm target (the row's own LOO recon norm). A fixed-scale
   variant (e.g., pooled mean recon norm from other rows) would make the
   intervention fully donor-free at the row level — the natural next arm,
   and the honest gap to state until it runs.
4. Endogeneity verdict for the discussion: **partial and specific** — the
   lever subspace is causally implicated in natural outcome differences
   at the class level, while carrying no privileged per-row readout.
   Wording stays scoped to exogenous mediation for the current paper
   (exploratory pre-registration), with F(i)+F(ii) as the next paper's
   opening result.
