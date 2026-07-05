# Adversarial Review Round 2 — Triage (2026-07-05)

Two frontier-model meta-reviews of `response_packet_2026-07-05.md` +
`review_packet_2026-07-04.md`. Verdicts: **Major Revision / Major
Revision** (upgraded from Major/REJECT). Both independently: credited the
statistical debt as paid against self-interest, credited the F(ii)
self-correction, graded most §A/§B objections ADDRESSED or PARTIALLY
(blocked only on the unexecuted W1 sweep), and converged on the SAME
condition for publication: the shuffled-label projected control.

## Catches accepted and fixed same-day (all committed)

1. **The F(ii) anchor was a deterministic replay, not a replication**
   (review 1). VERIFIED: 208/208 sample-level outcomes bit-identical to
   item C's rank8_loo (same rows, seeds, arm index). The gate could not
   have failed; "exact replication" wording retracted in the verdict doc.
   Side value: end-to-end determinism is now empirically confirmed, which
   (a) validates stitching partial jobs and (b) lets F(ii)-b reuse F(ii)'s
   real-proj arm with an in-job integrity check instead of regenerating.
2. **Rule-of-three independence slip** (review 1): 104 samples are 13
   row-clusters; honest bound is between 2.9pp (independence) and ~23pp
   (cluster-level). Claim 7's "tighter than 5pp" boast retracted in
   `stat_hardening_2026-07.md`; W1.3 wording updated accordingly.
3. **Hardening not applied to the new boundary contrasts** (both).
   Run (`stat_hardening_newresults_2026-07-05.json`):
   - F(ii) proj vs baseline: +0.341 BCa[+0.202,+0.495], LOO
     [+0.315,+0.355], per-height +0.362/+0.328 — robust, not row-sparse.
   - proj vs anchor: pct[−0.005,+0.216] BCa[+0.005,+0.236] — boundary;
     "139% / largest" superlatives retracted.
   - raw arm MDE 0.159 — "dead weight" unlicensed; raw is undetermined.
   - item E pred − mean_only: BCa[+0.010,+0.178] — excludes zero;
     "prediction adds nothing" softened to "adds nothing beyond shuffled
     content; a small increment over the bare mean is boundary-supported".
   - (proj − raw) reported garbled in the verdict → recomputed directly:
     +0.298 [+0.130,+0.471].
4. **F(ii) parse split was missing where the biggest number lives**
   (both). Run: proj parse rate 0.817 vs baseline 0.904 (mild format
   degradation at amplitude), P(strong|parsed) 0.565 vs 0.133 — content,
   not format. Added to the verdict doc.
5. **F(i) "no better than chance" overstated a ~94th-percentile result**
   (review 1): guard basis 0.718 vs p95 0.721 from 200 draws. Prose
   softened; the pre-registered call stands.
6. **49.8% vs 0.515 convention mismatch** — reconciled in the verdict doc
   (per-row LOO bases vs pooled full basis).

## The convergent demand — adopted as item F(ii)-b (jobs 458424/458425)

Both reviews (and review 2's §3 "top-variance cone" objection — the
strongest remaining attack on the whole program) require the
label-specificity control. Pre-registered before any data
(`causal_handle_directions.md` F(ii)-b) with the reviews' exact riders:

- 4 shuffled-label projected class-mean draws (family-pooled),
- the sign-flipped real vector (label-content predicts null-or-harm;
  geometry predicts repair),
- the fixed pooled-norm variant (the honest "donor-free" claim: direction
  AND scale hint-free),
- in-job unhinted baseline as determinism integrity gate.

Outcomes pre-committed: LABEL-SPECIFIC → F(ii)'s §4 wording earned;
GENERIC (the reviewers' predicted outcome) → +0.341 reads as
high-variance in-subspace amplification, F(ii) reverts fully to exogenous
mediation, and claim 8's W1 wording swaps "the specific 8 PCA directions"
for "an identified high-variance subspace". Queued behind item D's
remainder; results expected this evening.

## Review claims NOT fully conceded

- Review 1's §3 asserts item C's controls are "structurally incapable of
  containing high-variance content" — partially true (isotropic
  rand_norm; uniform-random rand_subspace), and F(ii)-b is the answer;
  but note item C's rand_subspace carried the row's own centered delta
  content (not pure noise) and still nulled — the cone alternative must
  explain that null too. Logged for the F(ii)-b verdict discussion.
- Review 1's "near-donor-free is not honest": accepted for the label
  ("hint-free direction, donor-calibrated amplitude" adopted), with the
  fixednorm rider as the test rather than a concession that the fraction
  on faith is large.

## Standing items

- W1 wording sweep remains gated on item D's pooled verdict (today) —
  now additionally incorporates: claim 7 cluster-aware bound, claim 8's
  conditional wording swap pending F(ii)-b, anchor-wording fixes.
- Item D: 458413 finishing, 458414 next; pooled verdict this afternoon,
  then W4.2's original-correctness split (the weak-lever discriminator).
- Both reviews' trajectory assessments, quoted for the record: "a
  research program whose numbers will hold and whose story will be
  rewritten — likely by the authors themselves"; "the trajectory is
  unusually good... I expect the narrow empirical effect to hold up."
