# External adversarial review (2026-08-18) — response ledger

An external LLM review of the blueprint raised 7 major concerns +
secondary items. Disposition (analyses run same night from recorded
data; paper edits in commit history):

1. **Erasure application (the kill-shot concern)** — three-part answer
   in §5.2: identical-estimator destructive controls (shared code
   path), in-run projection-variance telemetry (read off the live
   hook), and a NEW direct application audit (job 462096, registered
   verification addendum): baseline vs post-hook projection deviations
   at all 5 layers. The retrained-probe artifact (0.856→0.841 after
   round 1) is cited as the redundancy check it is, not application
   proof.
2. **A-priori-ness + linear-erasure scope + MDE** — conceded, stated:
   "predictability in retrospect" note added; abstract now "no
   detectable cost (MDE ≈ 0.13)".
3. **"No answer key anywhere"** — rescoped everywhere: "no per-row
   answer information about the test problem at inference time; all
   supervised components fit on disjoint rows".
4. **Frozen-beats-fresh** — quantified: row-paired +0.219
   [+0.127, +0.316]. Read as LOO-refit estimator noise (n≈26 donors)
   + STATIONARY content. Content-vs-command resolved into one story:
   label-specific (shuffles null, flips harm), stationary, expressed
   as commitment at the addressed site.
5. **H / misdirection / wrong-address coherence** — new analysis:
   wrong-address branches shift proposals toward the fired concept
   (+0.056 [+0.026, +0.086] over its baseline rate) while dP(strong)
   stays at baseline. Asymmetry restated as partly estimand artifact;
   superposition interference flagged open.
6. **Rank-8 readability + outlier identity** — overlap analysis:
   components 1–2 near-axis-aligned with outlier dims (participation
   ratios 2, 4); basis energy 95.5% vs top-8 coords 97.8%. Removal
   results energy-bound (statePCA-8 floors; permuted control
   near-free); write-side specificity not outlier-explained
   (shuffled-label vectors share the outlier profile, repair nothing).
7. **Oracle equivalence + off-distribution gauge** — softened to
   could-not-distinguish-at-power with both CIs; 87–92% of oracle
   effect; NEW branch-level gauge-vs-outcome AUC 0.68 / 0.80.

Secondary: existence-proof scoping (enumerable candidates),
clustering-unit note, blueprint housekeeping (9 bets, 4 rounds,
dates, "up to 0.94").
