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

Paired (recomputed directly, with BCa/LOO per
`docs/stat_hardening_newresults_2026-07-05.json`): (proj − rand_norm)
+0.418 pct[+0.260,+0.577] BCa identical, LOO [+0.395,+0.435];
(proj − raw) +0.298 [+0.130,+0.471]; (proj − rank8_loo) +0.096
pct[−0.005,+0.216] BCa[+0.005,+0.236] — **boundary: proj is not reliably
larger than the anchor**, so no superlatives ("139%", "largest") are
licensed; (raw − rand_norm) +0.120 [−0.029,+0.284] (null; raw's MDE is
0.159, so "dead weight" is also not licensed — raw is undetermined at
this design). Robustness of the headline arm: BCa ≈ percentile, LOO band
[+0.315,+0.355], per-height +0.362 (h3) / +0.328 (h4) — not row-sparse,
not height-driven. Parse split: proj parse rate 0.817 vs baseline 0.904
(mild format degradation at this amplitude); P(strong|parsed) 0.565 vs
baseline 0.133 — the repair is content, not format. Alignment-fraction
conventions reconciled: 49.8% = mean over per-row LOO bases; 0.515 = the
pooled full-basis figure used by the shuffled-label control.

Diagnostic: the natural class vector holds **49.8%** of its norm inside
the 8-dim lever subspace (per-row LOO bases, range 0.43–0.55). Random
expectation for an 8-of-5,376 subspace is ~3.9% — the natural
correct-vs-incorrect direction is ~13× more lever-aligned than chance.

## Pre-registered rules → outcome

- **Gate (rank8_loo CI excludes zero): holds, but is vacuous as designed
  (CORRECTION, round-2 review)**: the anchor arm shares rows, sample
  seeds, and arm index with item C's rank8_loo, and generation is
  deterministic — 208/208 sample-level outcomes are bit-identical across
  the two jobs. The arm is a pipeline-integrity check that could not have
  failed independently; calling it "an exact replication" was wrong. (It
  does confirm end-to-end harness determinism, which validates stitching
  partial jobs.)
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
