# Paper Outline (final, 2026-07-05) — supersedes the section plan in next_paper_skeleton.md

Format: ICLR 2026, 4–6 pages + appendix (intro ~1p, dataset ~1/4p,
approach ~1.5p, related work ~1/2p, results ~1.5p, discussion ~1/2p).
Sources of truth: `contribution_ledger_2026-07-05.md` (claims + wording),
`next_paper_claims_table.md` (numbers), pooled verdict docs (evidence).
Wording guardrails: ledger §4 applies everywhere.

## Title

- Primary: **Gauge, Not Lever: Dissociating Readable Correctness from its
  Causal Repair Channel in a 27B Reasoning Model**
- Fallback (if "gauge/lever" reads too cute in review): "The Readable
  Correctness Subspace Is Not Necessary — and the Necessary Channel Is
  Not Readable"

## Abstract (6 sentences)

1. Setup: LLMs visibly encode whether they are about to answer correctly
   (probe AUC ≤ 0.94), and recent work shows this readout resists
   steering — but why, and what does causally control correctness?
2. Erasure result: deleting everything linearly readable about
   correctness — the full rank-9 INLP subspace at all five readable
   layers — costs nothing (+0.05 [−0.07,+0.20]) while matched-rank random
   subspaces are catastrophic (−0.38): the readout is not necessary.
3. Lever result: correctness is instead controlled through a compact
   channel at concept-mention tokens; a hint-derived rank-8 basis repairs
   held-out failures (+0.25), passing random-subspace/noise specificity.
4. Necessity dissociation: a natural correct-minus-incorrect class-mean
   vector is null full-dimensional but repairs strongly (+0.34) when
   concentrated into that subspace; shuffled-label and sign-flip controls
   show the labels carry the effect; a fully answer-free variant reaches
   +0.40 on a 0.12 baseline (self-consistency baseline: 0.000).
5. The channel is causally potent yet not decodably privileged (probe
   inside random-subspace null) — the inverse dissociation of the
   readable-but-inert readout.
6. Scope sentence: one model/task family for the positive results;
   raw-axis non-necessity replicates in a second model; pre-registered
   throughout.

## 1. Introduction (~1 page)

- Hook: the trap — months of steering nulls on a highly readable
  correctness direction; the field hit the same wall (3 preprints, 2026).
- Statistical framing: InAbHyD abduction; predict-then-intervene on
  is_correct_strong; row-cluster bootstrap as the only sigma.
- Thesis sentence (ledger §5 verbatim).
- Contributions (map to ledger §1): (i) readable-STACK non-necessity with
  destructive matched-rank controls + variance mechanism; (ii) localized
  compact causal channel with fresh-row + specificity guards; (iii) the
  raw-null/subspace-potent necessity dissociation; (iv) the
  label-specificity battery (shuffled/sign-flip/noise/random-subspace);
  (v) causal-without-decodable; (vi) answer-free repair with a measured
  deployment profile (collateral beneficial; addressing load-bearing —
  item H outcome slots here); (vii) pre-registration methodology.

## 2. Dataset & Task (~1/4 page)

- InAbHyD generation, heights 1–4, property task primary (subtype +
  Qwen as scope checks); strong/weak scoring; parser; 44k rows/model.
- Row-selection conventions: balanced h3/h4 strong-incorrect fresh draws;
  provenance-disjoint sets (dev/composite 13, guard 26+6, capture 96,
  collateral 16) — one paragraph, one provenance figure in appendix.

## 3. Methods (~1.5 pages)

- 3.1 Probes and the readable subspace (claim 1; INLP stack construction).
- 3.2 Erasure family: mean-projection clamp, matched-rank random-stack
  estimator, projection-variance telemetry, P(strong|parsed) split.
- 3.3 Interchange on the focus state: hint donors, concept-position
  localization, LOO rank-k PCA bases, norm-matching conventions.
- 3.4 The class-mean family: capture protocol (no hints), projection,
  fixed-pooled-norm (donor-free) variant, per-candidate firing (item H).
- 3.5 Control taxonomy TABLE (this is a selling point): random positions,
  random subspaces, matched-norm noise, shuffled labels, sign-flip,
  shuffled-ridge, between-run direction, dose ladders — one row each with
  "what mundane account it kills."
- 3.6 Protocol sidebar: pre-registration with branch-complete rules
  before unblinding; two registered predictions confirmed; determinism-
  verified stitching; stats (cluster bootstrap, BCa checks, MDE for
  nulls).

## 4. Related Work (~1/2 page)

Use skeleton §4 as-is (restructured 2026-07-04/05), four blocks:
- Readable-but-unsteerable trio (2605.05715 entanglement account,
  2604.13068, 2605.23315 term collision) — background we replicate.
- Diff-in-means-for-reasoning lineage (AdaRAS, ALS, Valentino,
  2604.05655; SAE-RSV noise account; PDS/Fractional prompt-contrast) —
  the recipe family exists; none shows necessity, label-specificity, or
  decodability dissociations. Probe-gated steering is theirs.
- Methods bar: 2507.08802 (state assumptions), 2511.04638
  (behavioral-null-space vocabulary), Makelov/Wu (subspace illusions →
  our provenance discipline), 2506.11673 (erasure operation choice).
- Steering-rigor critiques (Tan, Pres, AxBench, Ali, Brumley) — cited as
  the checklist our diagnostics answer (appendix table).

## 5. Results (~1.5 pages, claim → subsection)

- 5.1 The gauge (brief): readable everywhere, scramble-robust, Qwen 0.94.
- 5.2 The gauge is not necessary: raw-axis null (with MDE honesty) →
  full-stack Branch E headline (+0.047 vs −0.38 random; paired +0.427;
  P(strong|parsed) 0.44 vs 0.01); variance telemetry as mechanism;
  entanglement discrimination (their erasure hurt, ours doesn't).
  FIG 1: erasure forest plot + variance inset.
- 5.3 The lever exists and is localized: recognition gap (behavioral),
  concept-position patch repair (fresh-row +0.255 headline; dev +0.491
  labeled anchor), ~6× positional potency, KV/masking route nulls.
  FIG 2: repair arms.
- 5.4 The lever is compact and specific: rank-8 fresh-row guard (91%),
  specificity ladder (mean 35%, random subspaces null, noise destructive,
  paired +0.333). FIG 3: specificity ladder.
- 5.5 The necessity dissociation and what the channel carries: F(i) null
  → F(ii) raw-vs-proj (+0.043 vs +0.341) → F(ii)-b battery (shuffled
  −0.043, sign-flip −0.120, donor-free +0.399) → baselines
  (self-consistency 0.000, best-of-8 0.192) → deployment profile
  (collateral +0.266 beneficial; all-positions null; item H policy
  outcome). FIG 4: the F-series ladder — this is the paper's signature
  figure.
- 5.6 Scope & cross-model: Qwen raw-axis replication (with
  manipulation-check telemetry); the G-series port — recognition gap
  cross-model (G0 hint lift +0.523), carrier at L43 rel depth 0.67
  (G2 +0.175), channel rank k*=16 (G3′; curve 48→95% of carrier at
  ranks 8→64, random-64 controls at/below baseline) — the MOTIF
  transfers, the coordinates (depth, rank, mean share) are
  model-specific; subtype suggestive-only (demoted, LOO-sensitivity
  shown); geometry (core ⊥ readable subspace, 0.0002 < null).
  TABLE: claims × (Gemma/Qwen) × (property/subtype) scope matrix.
  FIG candidate: the two-model rank-recovery curve.

## 6. Discussion & Limitations (~1/2 page)

- What the dissociations mean: information location ≠ causal control, in
  both directions; implications for probe-based monitoring AND for the
  steering lineage (the projection-headroom hypothesis, one paragraph,
  framed as testable prediction).
- Endogeneity honestly: F(i) null at this layer/rank — exogenous
  mediation wording; the label-specific content is ABOUT natural success
  without being the model's own per-row decision variable (as far as
  linear tools see).
- Misdirection asymmetry as open anomaly; identifiability caveat
  ("an identified sufficient channel"); search-cost acknowledgment with
  pre-registration as mitigation.
- Limitations: one task family; n=16–26 rows (MDE statements for all
  nulls); brittleness/entropy-telemetry unrun; addressing gap per item H
  outcome; fp32/hardware scoping for any gorman-lane numbers.
- Future: G4 (Qwen answer-free rider, registered), Qwen3.6 stretch,
  projection-headroom transplant, position policy v2.

## Figures & Tables plan

- Fig 1 erasure forest + variance inset (data: item D pooled + telemetry).
- Fig 2 repair arms fresh-row (guard v2 + KV nulls).
- Fig 3 specificity ladder (item C pooled).
- Fig 4 F-series signature ladder (F(ii)/F(ii)-b/F(ii)-c pooled).
- Table 1 control taxonomy; Table 2 scope matrix; appendix: artifact-
  checklist table, parse/baseline diagnostics, negative-results catalog,
  provenance figure, all pre-registration excerpts.

## Assembly order (drafting plan)

1. §5.2 and §5.5 first (the two headline sections; numbers frozen).
2. §3 methods + control table (mostly transcription from verdict docs).
3. §1 intro + abstract (after results read cleanly).
4. §4 related work (skeleton §4 nearly verbatim).
5. §6 discussion; then figures; then appendix tables.
6. Final pass: ledger §4 wording audit + scoop re-check + one more
   adversarial-review round on the actual draft.
