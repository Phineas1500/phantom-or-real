# Contribution Ledger — What We Claim, What We Cite, What We Share

Drafting north star, synthesized 2026-07-05 from: the July-4 literature
sweep (`next_experiments_litreview_2026-07.md`), the July-5 novelty check
(`novelty_check_newresults_2026-07-05.md`), two adversarial review rounds
(`adversarial_review_response_2026-07-04.md`,
`adversarial_review_round2_2026-07-05.md`), the artifact-checklist triage
(`steering_artifact_triage_2026-07-05.md`), and the landed verdicts C/D/E/
F(i)/F(ii)/F(ii)-b. Every §1 entry has a pre-registered experiment behind
it. Wording rules in the claims table override phrasing here if they ever
conflict.

## 1. UNIQUELY OURS (the paper's claims — no anticipation found under
adversarial search)

1. **The necessity dissociation (raw-null / subspace-potent).** The same
   natural class-mean vector at the same total norm is behaviorally null
   full-dimensional (+0.043 [−0.120,+0.207]) and strongly reparative when
   its energy is concentrated into the 8-dim causally-identified subspace
   (+0.341 [+0.202,+0.495]). In every verified prior work the raw vector
   already works; no one has shown the contrast, hence no one has shown
   where the causal content lives. Evidence: F(ii); anchored by item C
   (random subspaces null, noise destructive).
2. **The label-specificity battery.** Shuffled-label class-means (identical
   geometry/norm/positions) are inert (−0.043 [−0.103,+0.006], 4 draws);
   the sign-flipped vector is actively harmful (−0.120, CI < 0); paired
   real-vs-shuffled +0.385 [+0.245,+0.534]. No steering paper in the
   verified corpus runs ANY of these controls; both round-2 reviewers
   predicted the shuffled arm would match — falsified. Evidence: F(ii)-b.
3. **Causal alignment without decodable alignment.** The subspace that
   exclusively carries the repair holds no privileged linearly-decodable
   information about natural outcomes (per-row AUC 0.70–0.72 inside a
   random-subspace null with p95 0.721; full-dim ceiling 0.807). The
   inverse of the field's decodable-but-not-causal refrain, demonstrated
   within one experiment pair. Evidence: F(i) + F(ii), read jointly.
   (Medium-confidence novelty: the LEACE/amnesic-probing theory lineage
   returned no confirmed claims either way — keep the wording empirical.)
4. **The gauge/lever anatomy (the thesis).** In one model: the readable
   correctness signal is deletable in full — the entire rank-9
   INLP-reachable stack, five layers, every token — at zero behavioral
   cost (+0.047 [−0.070,+0.203]) while matched-rank random stacks are
   catastrophic (−0.38); the causal repair channel is geometrically
   disjoint from it (overlap 0.0002 < random); and the two dissociate in
   both directions (readable-and-inert vs potent-and-unreadable). No
   rival holds both halves. Evidence: items D + C + claim-9 geometry.
5. **Fully donor-free repair with a measured deployment profile.**
   Direction from natural outcomes only, amplitude from pooled scale —
   zero per-instance answer information in the vector — repairs at
   +0.399 [+0.245,+0.558], the program's strongest arm; against
   in-distribution baselines nobody else reports: self-consistency
   (maj-of-8) 0.000, best-of-8 ceiling 0.192, per-row deltas 16↑/10=/0↓.
   Evidence: F(ii)-b fixednorm rider + artifact-triage diagnostics.
   (Position-selection and collateral riders = F(ii)-c, in flight;
   claim the profile only as far as it lands.)
6. **Methodological contribution:** pre-registration with branch-complete
   decision rules before unblinding across 10+ jobs, two registered
   predictions confirmed (item D branch + mechanism), determinism-verified
   job stitching, and the artifact-checklist clearance (7/10 classes,
   including the strongest known critiques). Both external review rounds
   certified the practice ("better than most published work in this
   area").

## 2. ESTABLISHED BACKGROUND (cite, never claim)

- **Correctness/error is linearly decodable pre-generation** — ours
  (claim 1) plus 2605.05715, 2604.13068 (7/7 models), 2604.05655,
  2504.05419, 2602.06022. Table stakes.
- **The readable signal resists steering** — 2605.05715 (29 configs
  null), 2604.13068, 2605.23315 (coins "epiphenomenal correctness" —
  cite at first use of the word). Our steering nulls replicate this.
- **Diff-in-means steering can raise reasoning accuracy** — AdaRAS
  (2601.19847: AIME +13, HumanEval +21, Qwen3-1.7B/4B MLP neurons),
  ALS (2509.18116: MATH-500 76→91 raw class-mean), Valentino et al.
  (2505.12189: syllogisms, ±), trajectory steering (2604.05655: hardest
  GSM8K slices +7.6). Prompt-contrast cousins: PDS (2510.05498),
  Fractional Reasoning (2506.15882). The RECIPE FAMILY exists; we did
  not invent it.
- **Probe-gated ("gauge-gated") steering** — AdaRAS (failure probe AUROC
  0.8347 gates intervention) and 2604.05655 (mid-reasoning predictor,
  12.3% of examples). The closed-loop deployment framing is published.
- **Steering-claim failure modes** — Tan 2407.12404 (aggregate masking),
  Pres 2410.17245 (evaluation), AxBench 2501.17148 (prompting baselines),
  Braun 2505.22637, Ali 2507.11771 (shrinks with scale), Brumley
  2411.07213 (brittleness). We cite these as the bar our diagnostics
  answer.
- **Subspace-illusion precedent** — Makelov/Lange/Nanda 2311.17030 with
  Wu 2401.12631 as counterpoint; our fresh-row/basis-provenance
  discipline is the response.

## 3. SHARED TERRITORY (open questions where we contribute the strongest
evidence, framed as adjudication rather than discovery)

- **Inert vs entangled readouts.** 2605.05715's erasure HURT (−3.6pp) →
  they conclude entanglement; our strictly larger erasure is free with
  destructive matched-rank controls → in our setting the
  destruction-on-removal form is excluded. The weak (redundant-carrier)
  form SURVIVES — retrained probes still decode after the scrub — and the
  paper says so. Wording: non-necessity + the variance mechanism, never
  "genuinely inert."
- **Where steering effects come from.** SAE-RSV (2509.23799) shows raw
  diff-in-means vectors are 93.6% noise and filtering helps; we show a
  causally-identified low-rank filter is (in our setting) NECESSARY, and
  the retained content is label-specific. Same conversation, decisive
  experiment.
- **Effect size on hardest slices.** Largest verified hardest-slice
  repair in the corpus (+0.34–0.40 on a 0.12 baseline vs ~8pts nearest),
  at 27B where CAA-style effects reportedly shrink (Ali) — but one task
  family, one model; scope stated.
- **Cross-model generality.** Qwen raw-axis non-necessity replicates
  (claim 11); the lever is untested elsewhere — item G pre-registered.
  Until it runs, all §1 claims are Gemma-3-27B/property-task scoped.

## 4. NOT CLAIMABLE (rejected wordings)

- "We discovered activation steering can fix reasoning" (lineage exists).
- "We invented probe-gated/closed-loop steering" (published twice).
- "The readout is genuinely inert / epiphenomenal" (only non-necessity
  licensed; term collision).
- "The model KNOWS the answer and we unlock it" (F(i) null kills the
  endogenous-commitment reading at this layer/rank; exogenous-mediation
  wording stands).
- "The 8 directions are THE mechanism" (identifiability caveat; say
  "an identified sufficient low-rank channel").
- "Costs nothing"-style equivalence claims (MDE/cluster-bound wording
  from stat_hardening docs).

## 5. Thesis sentence (approved form)

In a model that visibly predicts its own failures, the readable
correctness signal and the causal repair channel are different objects:
the entire readable subspace can be deleted without behavioral cost,
while repair flows exclusively through a low-rank channel that carries
the label-specific content of natural success yet is itself no more
decodable than chance — gauge and lever, anatomically separated.
