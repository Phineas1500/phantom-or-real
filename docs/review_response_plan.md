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
