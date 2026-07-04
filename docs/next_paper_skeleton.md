# Next-Paper Skeleton: Gauge, Not Lever

Drafting skeleton for the causal paper. Sources of truth:
`docs/next_paper_claims_table.md` (claim wording and evidence),
`docs/next_paper_synthesis.md` (bottom line and wording rules),
`docs/next_paper_causal_abstraction_dashboard.md` (per-experiment records),
`docs/causal_status_overview.html` (narrative arc and figures). The Stage-1
probes/SAE report skeleton stays in `docs/report_outline.md`; this document
covers the causal follow-up. Format target: ICLR template per
`docs/REPORT_NOTES.md` guidelines (intro ~1p, dataset ~1/4p, approach 2+p,
related work ~1/2p, results 1+p).

## Title candidates

- Gauge, Not Lever: Correctness Readouts Are Epiphenomenal While a Compact
  Focus State Is Causal in Ontology Reasoning
- The Readable Axis Is Not the Causal One: Necessity Tests and a Compact
  Causal Core in Gemma 3 27B

Wording caution (2026-07-04): arXiv 2605.23315 already publishes the phrase
"epiphenomenal correctness" (cross-model convergence result, no repair). Any
title/abstract use of "epiphenomenal" must cite-and-differentiate at first
use, or the title should pivot to the gauge/lever framing, which is not
collided.

## Abstract skeleton

Readable ≠ load-bearing (claims 1-3) → the causal variable is target-concept
focus, localized and patchable (claims 4-6) → its causal core is compact,
held-out, and outside the readable subspace (claims 8-9) → scope honestly
(claims 7, 11, 12: token-route null, Qwen partial, subtype non-transfer).

Positioning shift (2026-07-04, per docs/next_experiments_litreview_2026-07.md):
three 2026 preprints independently report the readable-but-unsteerable gap
(2605.05715, 2604.13068, 2605.23315), so that gap is now established
background, not a headline. Lead with what none of them has: the positive
identification of the causal carrier (rank-8 core, fresh-row-validated
repair) plus the erasure-based demonstration that the readout is genuinely
inert rather than entangled (2605.05715's rival account).

## 1. Introduction (~1 page)

- Task and statistical framing: InAbHyD single-hypothesis abduction;
  predict-then-intervene on `is_correct_strong`.
- The trap this paper documents: months of steering nulls on a highly
  readable correctness direction, explained by one necessity test.
- Thesis (from synthesis bottom line): correctness is robustly decodable but
  causally epiphenomenal as a readout; the causally potent variable is
  hypothesis selection / target-concept focus, computed during prompt
  processing.
- Contributions list = claims 2 (necessity), 5-6 (focus-state repair +
  localization), 8 (compact core, held-out), 9 (geometry), plus the
  negative-results catalog as a methodological contribution.

## 2. Dataset (~1/4 page)

- InAbHyD generation pipeline, heights 1-4, property/subtype tasks, 44k rows
  per model; strong/weak scoring. Point to Stage-1 report for details.
- Row-selection conventions for causal experiments: balanced h3/h4,
  recognition-gap manifests, k>=8 sampled generations, row-level cluster
  bootstrap as the only sigma source.

## 3. Proposed Approach (2+ pages)

- 3.1 Probes and the readable axis (claim 1; brief — inherited from Stage-1).
- 3.2 Necessity by multi-layer subspace erasure (claim 2): intervention
  family, control set (orthogonal, Gaussian, between-run height direction,
  dose-response), P(strong|parsed) precision-vs-demolition split, and the
  control-matching variance telemetry with its pre-registered decision rule.
- 3.3 Interchange/patching on the focus state (claims 5-6): hint-conditioned
  donors, concept-position localization, matched-noise and shuffled-donor
  controls, misdirection arm.
- 3.4 Compact-core extraction and guards (claim 8): rank-k PCA of
  concept-position deltas, leave-one-row-out bases, sufficiency ratio vs
  subset replacement.
- 3.5 Geometry riders (claims 3, 9): INLP stack overlap, Gemma Scope decoder
  overlap, random-subspace nulls.
- Protocol sidebar: pre-registration discipline (decision rules recorded in
  `docs/causal_handle_directions.md` before unblinding), reused across every
  job.

## 4. Related Work (~1/2 page; restructured 2026-07-04 per docs/next_experiments_litreview_2026-07.md)

- Co-closest prior, differentiate first (cite both — a verifier panel split
  on any "closest" superlative): arXiv 2605.05715 (failure regime decodable
  at 71.6%, 29 fixed-linear-steering configs null; explanation =
  ENTANGLEMENT, LEACE erasure destructive −3.6pp — the opposite causal
  status from our harmless erasure) and arXiv 2604.13068
  (detection-without-correction in 7/7 models incl. Qwen-2.5; self-described
  clean negative). Both stop where our contribution starts: neither
  localizes the causal variable nor demonstrates repair.
- Terminology collision: arXiv 2605.23315 coins "epiphenomenal correctness"
  (cross-model CKA convergence; top-PC ablation flips 1.5–5.5%). Cite at
  first use of "epiphenomenal"; differentiate: within-model dissection with
  a working repair vs cross-model convergence with neither.
- Readable-vs-causal subspace precedent pair: Makelov, Lange & Nanda
  2311.17030 (subspace-patching illusions via dormant pathways) with Wu et
  al. 2401.12631 as the counterpoint (contests the normative framing, not
  the phenomenon) — bounds the novelty claim to our specific form.
- Correlational correctness probing (the "readable" half as background):
  2604.05655 (GSM8K correctness AUC 0.87 mid-reasoning), 2504.05419
  (intermediate-answer correctness, zero intervention vocabulary),
  2602.06022 (CORAL — probe only re-weights outputs post-hoc).
- Low-rank causal carriers precedent (plausibility support, not scoops):
  2509.06608 (trained per-layer steering vectors recover 95.3%/87.8% of RL
  reasoning gains), 2505.15634 (contrastive-delta PCA steering basis),
  2506.18167 (reasoning behaviors readable AND steerable — opposite regime).
- Methods bar our design answers: 2507.08802 (NeurIPS'25 spotlight —
  causal-abstraction vacuity; state encoding assumptions explicitly: fixed
  linear PCA basis, no trained alignment map), 2511.04638 (ICLR'26 oral —
  harmless vs pernicious divergence; adopt the behavioral-null-space
  vocabulary for the probe readout; item-C riders supply the
  on-distribution diagnostics), 2506.11673 (ACL'25 Findings —
  mean-projection/LEACE over INLP; cited for the item-D operation choice).
- Kept from the prior plan: Cox et al. (orthogonal-baseline motivation in
  the control-matching section), ITI / TruthfulQA line, hydra-effect /
  self-repair, SAE/dictionary interpretability (Gemma Scope) for claim 9.
- Guardrails: verify all 2026 citations from primary sources per dashboard
  rule (the litreview's verifier passes cover the eleven above); re-run the
  scoop check over the June–July 2026 arXiv window immediately before
  submission.

## 5. Results (1+ page). Claim -> subsection map

| Subsection | Claims | Key numbers (from claims table) |
| --- | --- | --- |
| 5.1 Readable everywhere | 1, 3 | AUCs across models/scramble; INLP redundancy |
| 5.2 Needed nowhere | 2 | raw/height erasures null vs destructive controls; constant-offset caveat wording per claim 2 |
| 5.3 The causal variable | 4, 5, 6, 10 | recognition gap; +0.491 repair; ~6x positional potency |
| 5.4 Compact core | 7, 8 | token-route null; rank4_loo +0.192 (77%); rank8_loo 92%; guard v2 fresh rows (458374/458375): rank8_loo +0.231 = 91% of concept-replace, rule passed; rank-4 under-transfers (57%) — state rank-8 as the portable core |
| 5.5 Geometry | 9 | INLP overlap below random null; dictionary-visible, not sparse-small |
| 5.6 Scope | 11, 12 | Qwen raw-axis non-necessity only; old-trio null replicated on fresh seeds (458376/458377); L35 replication (458387/458388) fires the positive branch: subtype carrier reachable at L35, +0.117 [+0.008,+0.254], random control null — a modest, row-sparse analogue at a different depth, so 5.6 reads "different layer, weaker handle" rather than "non-transfer" |

- Both slots have pre-registered branches ("Pre-Registered
  Manuscript-Hardening Jobs" in `docs/causal_handle_directions.md`); either
  outcome changes wording, not structure.

## 6. Discussion / Limitations

- Gauge-vs-lever as a cautionary tale for probe-based safety monitoring.
- Free theory framing: the causally-null readout is a concrete instance of
  2511.04638's behavioral null-space; the readable-but-inert stack (if item
  D lands branch E) sharpens it from a direction to a rank-9 subspace.
- Limitations to state plainly: n=13-32 row scales; Gemma-scoped mechanism;
  subtype carrier unresolved; erased object is the probe axis, not all
  decodable information; constant-offset account of control destructiveness.
- Hint-free repair paragraph (one only; wording rules and experiment ladder
  in `docs/hint_free_repair_direction.md`): the rank-8 recipe's coefficients
  still come from the row's own donor pass, so it is an instrument, not an
  accuracy method; mean_only (+0.087, hint-free by construction) is the
  existence proof that a donor-free effect is real; predicting the 8
  coefficients from unhinted states is the question this paper makes
  askable. Frame as open question, never as capability.
- Negative-results appendix: the table enumerated at the bottom of
  `docs/next_paper_claims_table.md` (CAA/raw steering, optimized vectors,
  DAS, AtP, decode gates, recognition-context patch, block-mean/cross-row
  transplants, attractor account), each with its control set.

## Figures plan

- Fig 1: erasure headline (null raw vs destructive controls, with dose-response
  inset from control matching) — data exists in shard JSONs.
- Fig 2: interchange repair arm chart (exists in overview HTML, Part 5).
- Fig 3: rank ladder, finale + held-out guard side by side (+ guard v2 when
  landed).
- Fig 4: geometry — INLP overlap vs random null; Gemma Scope redundancy CDF.
- Fig 5: claims/scope matrix across (Gemma/Qwen) x (property/subtype).
- Reuse Stage-1 figure infrastructure in `scripts/make_plots.py` idiom;
  existing figure plan in `docs/report_figure_plan.md` covers Stage-1 only.

## Assembly order

1. Freeze claims 1-10 text from the claims table wording rules (done rows).
2. Both slots filled: guard v2 passed (458374/458375, rank-8 portable core);
   L35 replication landed the layer-mismatch repair (458387/458388, modest
   and row-sparse). All claims frozen — drafting can start.
3. Draft 5.2 with the softened control wording (claim 2 caveat) from the
   start — do not draft the potent-machinery version and patch it later.
4. Negative-results appendix from the dashboard rows (status=completed nulls).
5. Abstract last.
