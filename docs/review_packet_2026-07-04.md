# Review Packet — "Gauge, Not Lever" (2026-07-04)

Self-contained evidence bundle for external review. Project: mechanistic
analysis of ontology-reasoning correctness in Gemma 3 27B-IT (with Qwen3.5
27B replication arms) on the InAbHyD hypothesis-generation task. All numbers
come from run artifacts in the repo; confidence intervals are row-cluster
bootstrap percentile CIs (10k draws) unless stated otherwise. Two experiments
are still in flight (items D and E below); their pre-registered decision
rules and current partial data are included as such.

Contents:
 1. Claims table (the 12 claims, status, evidence, caveats)
 2. Paper skeleton (framing, related work, discussion rules)
 3. Key experiment summaries (erasure control matching; fresh-row guard;
    rank-8 specificity; KV/hint-span necessity; subtype L35 replication;
    Qwen erasure; rank-core geometry)
 4. Pre-registrations for in-flight items D (readable-stack erasure) and
    E (predicted-coefficient repair), plus item D's partial-data preview
 5. July 2026 literature sweep (novelty/positioning — treat external
    paper claims as unverifiable from your side; focus on internal validity)
 6. Hint-free repair direction memo (exploratory next-paper thread)


---

<!-- FILE: docs/next_paper_claims_table.md -->

# Next-Paper Claims Table (Pre-Registered)

Each claim mapped to landed or queued evidence. Written before job 457002
unblinded, per review: the composite job's outcome fills the last variable
rather than opening a new branch. Companion to the dashboard (evidence rows)
and `docs/causal_handle_directions.md` (designs and pre-registrations).

| # | Claim | Status | Evidence | Caveat / remaining |
| --- | --- | --- | --- | --- |
| 1 | Correctness is linearly readable from pre-generation activations, beyond metadata, across models and name-scrambling. | landed | ICLR-paper probe results; Qwen L53 raw AUC 0.940/0.920; scramble results | none new |
| 2 | The correctness readout direction is causally epiphenomenal: erasable at all readable layers with no behavioral cost. In Gemma, matched control erasures are destructive; in Qwen property, matched controls are also non-destructive. | landed (control-matching caveat) | Gemma jobs 456912/456915-918 (property), 456963-966 (subtype): raw CI straddles zero both tasks; orthogonal CI excludes zero; P(correct\|parsed) separates precision from demolition. Qwen jobs 457191-457194: raw erasure dP=+0.070 CI [-0.031,+0.188] over 512 generations, with orthogonal +0.047 and Gaussian +0.016. Control-matching jobs 457207-457210 (docs/erasure_control_matching_27b_property_k4_summary.md): raw +0.031 CI [-0.094,+0.172]; between-run height control also null (-0.047); orthogonal_s1 -0.344 CI excludes zero with monotone dose-response. | pre-registered variance test fired the constant-offset branch: raw within-run projection variance is far below the controls' (L15 10.65 vs 818/382 sd²) and the near-zero-variance height direction erases as harmlessly as raw. Wording rule: cite control destructiveness as the erasure family having behavioral teeth at matched norm, NOT as the raw axis being specially inert among same-norm directions; the landed claim is non-necessity of the readout axis. Qwen supports raw-axis non-necessity but not Gemma's destructive-control separation |
| 3 | Correctness information is redundant (INLP barely decays) yet no readable axis is load-bearing: directionally epiphenomenal, informationally redundant. | landed | docs/probe_on_erased_activations_27b_property*.json | full-subspace LEACE erasure listed as future work, not claimed |
| 4 | The recognition-generation gap is a proposal failure localized to target-concept focus: recognition 14/14, candidates 0.839, concept-hint 0.955-1.000, self-critique +0.15, gold absent from 16-sample proposals on ~2/3 of rows. | landed | Modal behavioral suite artifacts (candidates, proposal distribution, self-critique, hint gradient v1/v2) | property/Gemma only; subtype behavioral replication queued (task 10) |
| 5 | The focus state is causally accessible: patching hint-conditioned context encodings repairs free-form generation (+0.491 CI [+0.28,+0.71] at k=8) against destructive matched noise. | landed | job 456990; concept-resolved row-paired analysis | misdirection does not transfer through residual patches (asymmetric controllability); property KV finale later shows literal hint-token KV is insufficient |
| 6 | The focus state is positionally localized at concept-mention tokens (~6x per-token potency vs matched random positions); it is not captured by a uniform rank-1 block summary. | landed | job 456999; dilution caveat recorded — "not low-rank" NOT claimed | later rank-k guard upgrades this from rank question open to compact-core positive |
| 7 | Exhaustive necessity over tested routes is null: masking the hint span costs nothing (natural decode attention to hint is 0.5%), the combination arm (masking x reversion) survives at 0.981, and KV transplants are insufficient with attention verifiably flowing. Carrier = unpatched layers; commitment is multiply realized across layers. | landed (survives-both bin) | jobs 457002 + 457009; docs/kv_hint_span_27b_property_summary.md | the token route is ruled out on property; remaining necessity question is layer-exhaustive ablation (future work) |
| 8 | Commitment, not spotlight — and the focus state has a COMPACT CAUSAL CORE: rank-4 PCA add at L30 repairs in both the finale (+0.260 CI [+0.115,+0.423]) and the held-out guard (rank4_loo +0.192 CI [+0.096,+0.288], 77% of subset effect). Rank8_loo is stronger (+0.231 CI [+0.087,+0.404]), so the landed claim is compact low-dimensional core by rank 4 plus useful structure through rank 8, not exact intrinsic dimension = 4. Foreign deltas never misdirect. Fresh-row guard v2 (26 rows disjoint from the core's development set): rank8_loo +0.231 [+0.111,+0.365] = 91% of in-job concept-replace effect, pre-registered rule passes; rank4_loo +0.144 [+0.038,+0.264] = 57%, under the 70% bar — rank-4 under-transfers to fresh rows. | landed (fresh-row guard + specificity controls passed) | jobs 457002 + 457009 + 457012 + 458374/458375 + 458401/458402; docs/kv_hint_span_27b_property_summary.md; docs/rank_k_guard_27b_property_summary.md; docs/rank_k_guard_v2_27b_property_pooled_summary.md; docs/rank8_specificity_27b_property_pooled_summary.md | wording: rank-8 is the fresh-row-portable core; rank-4 is sufficient in-distribution only. Rank-core geometry landed: gauge-orthogonal and Gemma-Scope-visible but redundant, not sparse-small. Specificity (pre-registered item C) PASSED on the subspace-load-bearing branch: rank8_loo +0.245 [+0.111,+0.394] vs rand_norm family −0.088 [−0.180,−0.017] (paired diff +0.333 excludes zero), rand_subspace family null (−0.041), mean_only 35% — the specific 8 PCA directions carry the repair, not their mean/norm/rank/position footprint |
| 9 | Geometry: the causal focus content lies outside the readable correctness subspace, and the held-out rank core is even sharper: rank4/rank8 INLP subspace fractions 0.0001/0.0002, below random null mean 0.0017. Gemma Scope decoders expose the core strongly (top subspace sqrt-energy ~0.98), but redundantly: top20 decoder rows explain only 0.1-1.1% of decoder-overlap mass and 10k-112k rows exceed 0.10 overlap. | landed | docs/focus_state_composite_27b_property_summary.md; docs/rank_core_geometry_27b_property_summary.md | wording: dictionary-visible but not sparse-small; avoid saying Gemma Scope misses it entirely |
| 10 | Unification: each prompt format stores selection state where its candidates live (MCQ at option tokens — context patch null; hints at concept mentions — patch repairs). | landed | jobs 456913 + 456990/456999 read jointly | wording only; no new experiment |
| 11 | Cross-model robustness of the necessity/epiphenomenality result (Qwen). | landed for property raw-axis non-necessity | jobs 457191-457194; docs/qwen35_subspace_erasure_27b_property_sampled_k8_summary.md: baseline P(strong)=0.352, erase_raw=0.422, dP=+0.070 CI [-0.031,+0.188]; L53 test AUC=0.940 | property only; controls are non-destructive, so this is cross-model support that the raw readout axis is not load-bearing, not a replication of Gemma's perturbation-sensitivity profile |
| 12 | Cross-task: the behavioral story replicates (recognition gap 33/48; candidates 1.000; hint-first 0.875) but residual-state patch accessibility does NOT robustly replicate. Old L30/40/45 full patch is null (+0.039 in job 457005; +0.016 in the discriminator replay), and the targeted capture-ladder follow-up selected off-trio L53/L50/L35 but found no decisive repair (best: L35 concept replace +0.102 CI [-0.008,+0.258]; matched concept-minus-random +0.109 CI [+0.000,+0.266]; L53 rank4 LOO add +0.000). Old-trio null independently replicated on two fresh seeds: +0.047 [-0.164,+0.258] and +0.000 [-0.227,+0.227] (jobs 458376/458377, which ran old-trio-only due to an sbatch --export comma bug). L35 targeted replication (jobs 458387/458388) fires the pre-registered positive branch: pooled L35_concept_replace +0.117 CI [+0.008,+0.254] (k=16; meta-pool with 457170 +0.112 [+0.003,+0.260]), matched random control null (+0.023), old-trio null in-job. The subtype carrier IS reachable at L35 — the old-trio nulls were layer mismatch. | landed (layer-mismatch repair bin; modest and row-sparse) | docs/subtype_localization_patch_27b_summary.md; docs/subtype_discriminator_27b_summary.md; docs/subtype_l35_replication_pooled_summary.md; jobs 457005 + 457170 + 458376/458377 + 458387/458388 | wording: weaker, row-sparse analogue of the property repair (3/13 addressable rows carry it; ~quarter of property's effect size); evidence the focus-state variable exists on subtype at a different depth, NOT a matched replication. Claims 5-6's quantitative story stays property-scoped; subtype rank ladder is future work (L35_rank4_loo_add marginal +0.047). |

Negative-results table (for the appendix): CAA/raw steering, optimized
vectors, DAS L45/L50, AtP/exact patching, raw-z decode gates, prompt-margin
gated decode correction, recognition-context patch, block-mean delta add,
cross-row block-delta transplant, attractor-strength account — each with its
control set and the post-hoc explanation the focus-state account provides.

Scope rule: claims 7-9 are filled by landed jobs and saved artifacts;
anything not in this table goes to future work, not into the draft.

---

<!-- FILE: docs/next_paper_skeleton.md -->

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

---

<!-- FILE: docs/erasure_control_matching_27b_property_k4_summary.md -->

# Erasure Control Matching — Verdict (Jobs 457207-457210)

Reviewer-insurance follow-up to the multi-layer correctness-direction erasure
(`docs/causal_handle_directions.md`, "Erasure control matching"). Same erasure
family and layers (`L15/L30/L40/L45/L53`), same 16 balanced S1 h3/h4 property
rows, k=4 per condition per shard (576 generations total). New ingredients:
a deliberately between-run height control direction (`height_ge_4` probe,
cosine vs raw −0.05 to −0.01), dose-response for orthogonal/Gaussian controls
(scales 0.25/0.5/1), and within-forward-pass positional projection-variance
telemetry per direction.

## Arms (row-level cluster bootstrap vs in-job regenerated baseline)

| condition | P(strong) | P(weak) | parse fail | P(strong\|parsed) | dP (CI95) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.359 | 0.766 | 0.062 | 0.391 | — |
| erase_raw_s1 | 0.391 | 0.672 | 0.078 | 0.438 | +0.031 [−0.094, +0.172] |
| erase_height_s1 | 0.312 | 0.625 | 0.062 | 0.318 | −0.047 [−0.188, +0.125] |
| erase_orthogonal_s0p25 | 0.312 | 0.500 | 0.203 | 0.357 | −0.047 [−0.188, +0.094] |
| erase_orthogonal_s0p5 | 0.266 | 0.359 | 0.344 | 0.357 | −0.094 [−0.297, +0.109] |
| erase_orthogonal_s1 | 0.016 | 0.141 | 0.344 | 0.019 | −0.344 [−0.547, −0.156] |
| erase_gaussian_s0p25 | 0.375 | 0.531 | 0.047 | 0.375 | +0.016 [−0.141, +0.188] |
| erase_gaussian_s0p5 | 0.328 | 0.438 | 0.031 | 0.333 | −0.031 [−0.234, +0.172] |
| erase_gaussian_s1 | 0.125 | 0.125 | 0.500 | 0.295 | −0.234 [−0.438, −0.016] |

Strict row flips (row P(strong) crossing 0.5): raw 0/0 (false→true/true→false),
height 1/0, orthogonal_s1 0/4, gaussian_s1 0/3. Dose-response is monotone:
both controls are null-ish at scale 0.5 and destructive only at scale 1.
`P(strong|parsed)` reproduces the earlier precision/demolition split:
orthogonal destroys correctness even among parsed outputs (0.391 → 0.019)
while Gaussian damage is substantially format (parse fail 0.50, parsed
precision 0.295).

## Within-forward-pass positional projection variance (sd², call-weighted)

| layer | raw | height | orthogonal | gaussian |
| --- | ---: | ---: | ---: | ---: |
| L15 | 10.65 | 0.01 | 818.37 | 381.73 |
| L30 | 0.84 | 0.00 | 3.13 | 75.32 |
| L40 | 0.14 | 0.00 | 0.11 | 0.33 |
| L45 | 0.06 | 0.00 | 0.14 | 4.32 |
| L53 | 0.03 | 0.00 | 0.01 | 0.02 |

## Pre-registered decision-rule outcome

The rule (`docs/causal_handle_directions.md` item 1) said: if the raw
direction's within-run positional variance is far below the controls', the
constant-offset account is live and the between-run control governs wording.
That branch fired. At the high-variance layers the raw direction carries far
less within-run structure than the matched controls (L15: 10.65 vs 818/382;
L30: 0.84 vs 3.1/75.3), and the by-construction between-run height direction
(~zero within-run variance) erases as harmlessly as raw (−0.047, CI straddles
zero).

## Verdict

Non-necessity stands and gains a second harmless arm: erasing the correctness
readout axis costs nothing, and neither does erasing a semantically unrelated
between-run (height) axis. But the destructive-control contrast is confounded
by within-run variance: harmlessness tracks low within-forward-pass projection
variance, not the semantic identity of the erased direction, and the
norm-matched controls inject one to two orders of magnitude more positional
perturbation than the raw erasure does. Wording rule for the paper: cite the
Gemma control separation as evidence that the erasure procedure has behavioral
teeth at matched norm, NOT as evidence that the raw axis is specially inert
among same-norm directions. The safe claim remains: the readable correctness
axis is not load-bearing (claim 2's first sentence), with the constant-offset
account explicitly acknowledged.

Artifacts: `docs/erasure_control_matching_27b_property_k4_shard{0..3}of4.json`,
`results/stage2/erasure/erasure_control_matching_27b_property_k4_shard*.jsonl`
and `*_directions.npz`.

---

<!-- FILE: docs/rank_k_guard_v2_27b_property_pooled_summary.md -->

# Rank-k Guard v2 (fresh rows) — Pooled Verdict, Jobs 458374 + 458375

Shard outputs: `docs/rank_k_guard_v2_27b_property_shard{0,1}of2.json`;
row-level generations in
`results/stage2/erasure/rank_k_guard_v2_27b_property_shard{0,1}of2.jsonl`.
26 rows pooled (13 per shard; 3 per shard skipped for missing concept
positions out of the 16 selected). Fresh-row selection: parse-ok,
strong-incorrect h3/h4 rows, seeded, excluding the 13 composite-manifest
rows the compact core was built from. Row keys are (shard, row_index);
paired row-cluster bootstrap (10k draws) vs the in-job unhinted baseline.

## Pooled causal arms (26 rows, k=8 samples each)

| arm | P(strong) | dP vs unhinted (CI95) | % of concept-replace |
| --- | ---: | ---: | ---: |
| unhinted_baseline | 0.120 | — | — |
| hinted_baseline | 0.870 | +0.750 [+0.601, +0.885] | — |
| L30_concept_replace | 0.375 | +0.255 [+0.111, +0.413] | 100% |
| L30_random_replace | 0.159 | +0.038 [−0.010, +0.101] | 15% (null) |
| rank4_loo_add_L30 | 0.264 | +0.144 [+0.038, +0.264] | 57% |
| rank8_loo_add_L30 | 0.351 | +0.231 [+0.111, +0.365] | **91%** |

## Pre-registered decision rule → outcome

Rule (recorded in `docs/causal_handle_directions.md` before unblinding):
claim 8 survives if pooled `rank4_loo` **or** `rank8_loo` CI excludes zero
AND reaches ≥70% of the pooled in-job `L30_concept_replace` effect.

**Outcome: survives.** `rank8_loo` excludes zero and reaches 91% of the
concept-replace effect. `rank4_loo` also excludes zero but reaches only
57% — the rank-4 basis under-transfers to fresh rows; rank-8 is the
fresh-row-portable core. The concept-replace arm itself repaired
(+0.255, CI excluding zero), so the scoping branch (concept-replace null
on fresh rows) did not fire — the compact-core claim generalizes beyond
recognition-gap-style rows, not just within them.

## Hint-validated secondary slice (hinted P(strong) ≥ 0.5; 23/26 rows)

| arm | dP vs unhinted (CI95) | % of concept-replace |
| --- | ---: | ---: |
| L30_concept_replace | +0.288 [+0.125, +0.462] | 100% |
| rank4_loo_add_L30 | +0.163 [+0.043, +0.299] | 57% |
| rank8_loo_add_L30 | +0.255 [+0.120, +0.402] | 89% |

Same picture on the validated slice — the result is not carried by
hint-refractory rows.

## Wording for the paper (claim 8 / section 5.4)

A rank-8 leave-one-row-out PCA basis of hinted-minus-unhinted
concept-position deltas, added at L30, recovers 91% of the concept-replacement
effect on 26 fresh strong-incorrect rows disjoint from the rows the basis
family was developed on (rank-4: 57%; random-replace control null). State
rank-8 as the fresh-row-portable core and rank-4 as sufficient only
in-distribution (original held-out guard: rank-4 77%).

---

<!-- FILE: docs/rank8_specificity_27b_property_pooled_summary.md -->

# Rank-8 Specificity Controls (fresh rows) — Pooled Verdict, Jobs 458401 + 458402

Shard outputs: `docs/rank8_specificity_27b_property_shard{0,1}of2.json`;
row-level generations in
`results/stage2/erasure/rank8_specificity_27b_property_shard{0,1}of2.jsonl`.
26 rows pooled (13 per shard from 16 selected; same guard-v2 fresh-row
selection and skip pattern as jobs 458374/458375). Row keys are
(shard, source_row_index); paired row-cluster bootstrap (10k draws, seed
20260704, percentile [2.5, 97.5]) vs the in-job unhinted baseline. Random
families are pooled across draws and shards (per-row mean over d1–d4);
per-draw numbers are descriptive only, per the pre-registration in
`docs/causal_handle_directions.md` item C.

## Pooled causal arms (26 rows, k=8 samples each)

| arm | dP vs unhinted (CI95) | % of rank8 |
| --- | ---: | ---: |
| unhinted_baseline (P(strong) = 0.120) | — | — |
| hinted_baseline (P(strong) = 0.870) | +0.750 [+0.601, +0.885] | — |
| rank8_loo_add_L30 | **+0.245 [+0.111, +0.394]** | 100% |
| mean_only_add_L30 | +0.087 [+0.019, +0.173] | 35% |
| rand_subspace family (d1–d4 pooled) | −0.041 [−0.154, +0.055] | null |
| rand_norm family (d1–d4 pooled) | −0.088 [−0.180, −0.017] | negative |

Paired differences (per-row, same bootstrap): rank8 − rand_norm
**+0.333 [+0.186, +0.489]**; rank8 − rand_subspace +0.286 [+0.143, +0.441];
rank8 − mean_only +0.159 [+0.048, +0.288]. All exclude zero.

Per-draw descriptive: rand_subspace d1–d4 = −0.043, −0.062, −0.043, −0.014
(all CIs straddle zero); rand_norm d1–d4 = −0.082, −0.067, −0.096, −0.106
(all CIs entirely ≤ −0.005). No draw of either family repairs.

Hint-validated slice (hinted P(strong) ≥ 0.5; 23/26 rows): rank8 +0.272
[+0.125, +0.435]; mean_only +0.098; rand_subspace family −0.046;
rand_norm family −0.099. Same picture.

## Pre-registered decision rule → outcome

- **Gate** (pooled rank8_loo CI excludes zero): **holds** (+0.245, CI low
  +0.111).
- **PASS** required pooled rand_norm CI to include zero AND the paired
  (rank8 − rand_norm) CI to exclude zero. Observed: the paired difference
  excludes zero decisively, and the rand_norm family not only fails to
  repair but is slightly destructive (CI entirely ≤ −0.017 — stricter than
  the "includes zero" wording anticipated). The FAIL branch (rand_norm
  positive CI at ≥50% of rank8) does not fire in any form.
- **Outcome: PASS.** Norm-matched per-row Gaussian noise at the same
  positions and layer does not repair; the rank-8 effect is not a
  perturbation-size artifact.

## Decomposition grid → branch

- mean_only / rank8 = 0.35, below the 0.70 bar → **not** mean-dominated.
- rand_subspace family (LOO mean + row delta projected onto a random
  orthonormal rank-8 basis, per-position norm-matched to the PCA non-mean
  component) is null (−0.041), and (rank8 − rand_subspace) excludes zero
  → **not** mean+magnitude.
- **Branch: the PCA subspace itself is load-bearing.** The specific 8
  directions carry the repair; a random 8-dim subspace dressed to match
  them in mean, norm, position, and rank does nothing, and the mean shift
  alone recovers only about a third of the effect.

## Claim 8 wording consequence

The strongest available upgrade: the fresh-row-portable rank-8 core is
direction-specific, not a norm/mean/rank artifact. The mean component is a
real but minor contributor (+0.087 alone); the non-mean PCA structure
carries the remaining two-thirds and cannot be replaced by random
same-norm structure. Cite jobs 458401/458402 alongside the guard-v2 pass
(458374/458375).

---

<!-- FILE: docs/kv_hint_span_27b_property_summary.md -->

# KV/Hint-Span Job — Verdict (Job 457009)

Pre-registered analysis of `docs/kv_hint_span_27b_property.json` (936
generations, 2h18; 13 rows × 9 arms × k=8; row 5292 self-skipped as
documented). All bins were written before the job ran.

## Arms (row-paired bootstrap vs each arm's reference)

| arm | P(strong) | dP (CI95) | attn-to-span | note |
| --- | --- | --- | --- | --- |
| unhinted_baseline | 0.183 | — | — | |
| hinted_baseline | 1.000 | — | 0.0050 | natural decode attention to hint: 0.5% |
| hint_span_masking | 1.000 | +0.000 [0, 0] | 0.0000 | mask verified; **token route does nothing** |
| masking_x_reversion | 0.981 | -0.019 [-0.058, 0] | 0.0000 | **survives-both bin: carrier = unpatched layers** |
| gold_kv_transplant | 0.192 | +0.010 [-0.058, +0.077] | 0.0179 | attention flows (3.6x natural), no repair: genuine insufficiency |
| wrong_kv_transplant | 0.192 | +0.010 [-0.058, +0.087] | 0.0174 | no misdirection (wrong-tgt 0.221 vs baseline 0.327) |
| perpos_add_own | 0.087 | -0.096 [-0.327, +0.125] | — | rank-full ADD fails where replacement worked |
| restricted_add_x2 | 0.000 | -0.183 [-0.385, -0.029] | — | scale curve x1/x2/x4 = +0.06/-0.18/-0.16: no window |
| **rank_k_L30** | **0.442** | **+0.260 [+0.115, +0.423]** | — | **reading rule fires: compactly structured** |

## Verdicts (pre-registered bins)

1. **The decode-time token route is dead on property.** The hinted run
   naturally puts only 0.5% of decode attention on the hint span; masking it
   costs exactly nothing; transplanting hint KV into unhinted runs repairs
   nothing despite receiving 3.6x the natural attention (telemetry validates
   the machinery — gold-KV null = genuine insufficiency, not splice bug);
   wrong-KV does not misdirect. The pathway-mix account's token route is
   ruled out for property.
2. **Exhaustive-necessity headline: survives-both.** Masking x reversion
   leaves repair at 0.981. With both tested routes removed simultaneously,
   the hint still works: the carrier is the unpatched layers. Necessity
   failure is now exhaustive over tested routes — the redundancy symmetry
   at its third granularity (directions, pathways, layers).
3. **The surprise: the focus state has a compact causal core at L30.**
   The rank-4 PCA reconstruction of the concept-position deltas, ADDED at
   L30 alone, repairs +0.260 — matching the three-layer subset-replacement
   effect (+0.250) within CI (point ratio 104%), where rank-1 was null at
   every scale and the rank-full ADD was null/harmful. Per the pre-registered
   reading rule (k<=4 recovering >=70%): **compactly structured.** The PCA
   gatekeeper pointed at L30 (top-1 = 49% of variance) and the causal test
   confirmed it. Truncation HELPS the additive route: the discarded high-rank
   residual is not just inert but obstructive when added.
4. Scale insurance resolved: x1/x2/x4 is monotone harmful past x1 — the
   rank-1 direction has no working scale; the structure, not the magnitude,
   was missing.

## Rank-k guard (Job 457012)

The post-finale held-out-basis guard passes. Leave-one-row-out PCA at L30
repairs at rank 4: P(strong)=0.385, dP=+0.192 CI [+0.096,+0.288], or 77% of
the +0.250 subset-replacement effect. Rank 8 is stronger (P=0.423, dP=+0.231
CI [+0.087,+0.404], 92% of subset). The result is therefore not an in-sample
low-rank fit, but the ladder is graded rather than a hard rank-4 cliff.

## Rank-core geometry rider

The held-out-surviving L30 core is essentially absent from the INLP readable
subspace: rank4/rank8 subspace fractions are 0.0001/0.0002, below a random
subspace null mean of 0.0017. Gemma Scope gives the complementary result: the
core is dictionary-visible, with top decoder subspace sqrt-energy about 0.98 in
both 16k and 262k residual decoders. But it is not sparse-small: the top 20
decoder rows explain only 1.0% (16k) or 0.1% (262k) of decoder-overlap mass,
with thousands of rows showing nontrivial overlap.

## The program's closing picture (property task)

Concept commitment is written at mention sites redundantly across layers; a
compact low-dimensional causal core is present by rank 4 at L30, with useful
additional structure through rank 8; it is invisible to the readable
correctness subspace but visible in Gemma Scope as a highly redundant decoder
family rather than a compact sparse feature set; and the literal hint tokens,
though present and attended, carry none of it at decode time. The subtype
discriminator has now landed: the capture ladder found large late writes and
selected L53/L50/L35, but targeted off-trio residual-state patches did not
robustly repair (best L35 +0.102 CI [-0.008,+0.258]; L53 rank4 LOO add +0.000).
This scopes the compact-core positive claim to property and leaves subtype in a
residual-route-insufficiency bin rather than a clean layer-mismatch win.

## Next (final experimental items)

- Qwen erasure / cross-model necessity robustness (critical path); erasure
  refinements; optional L35 subtype replication only if we want to chase the
  suggestive but non-landed bump.

---

<!-- FILE: docs/subtype_l35_replication_pooled_summary.md -->

# Subtype L35 Targeted Replication — Pooled Verdict, Jobs 458387 + 458388

Per-seed outputs: `docs/subtype_discriminator_27b_l35rep2_seed{A,B}.json`;
row-level generations in
`results/stage2/erasure/subtype_discriminator_27b_l35rep2_seed{A,B}.jsonl`.
Same 16 recognition-gap manifest rows as job 457170, ladder 30/35/40/45 with
top-offtrio=1 (selects L35 in both seeds, row-mean delta norm 6012.6), fresh
generation/control seeds 20260702 and 20260703, k=8 per seed. First
submission (jobs 458376/458377) lost the ladder to sbatch `--export`
comma-splitting and ran old-trio-only; this rerun set env vars in the
submitting shell.

## Pooled new-seed arms (16 rows, k=16; paired row-cluster bootstrap)

| arm | P(strong) | dP vs baseline (CI95) |
| --- | ---: | ---: |
| baseline | 0.191 | — |
| L35_concept_replace | 0.309 | **+0.117 [+0.008, +0.254]** |
| L35_random_replace | 0.215 | +0.023 [−0.004, +0.062] (null) |
| L35_rank4_loo_add | 0.238 | +0.047 [+0.000, +0.109] (marginal) |
| old_trio_full_replace_L30_40_45 | 0.215 | +0.023 [−0.195, +0.242] (null) |

Meta-pool with 457170 (k=24): L35_concept_replace +0.112 [+0.003, +0.260];
L35_random_replace +0.013 [−0.008, +0.039].

## Pre-registered decision rule → outcome

Rule (recorded before unblinding): pooled-new CI excluding zero lands the
off-trio layer-mismatch repair and upgrades claim 12's wording (subtype
carrier reachable at L35); a null with half-width ≤ ~0.10 closes the hedge
as a bounded null. Either way `L35_random_replace` must stay null.

**Outcome: the positive branch fires.** Pooled-new CI [+0.008, +0.254]
excludes zero; the random control is null; the meta-pool concurs. The
subtype focus state is reachable through residual-state concept replacement
at L35 — the old trio was the wrong set of layers for subtype, and the
repeated old-trio nulls (three independent replicates now: 457005, 457170,
458376/458377, plus in-job here) were a layer-mismatch, not a task-level
absence of the variable.

## Caveats that go into the wording

- **Edge-of-significance magnitude.** +0.117 with a lower CI bound of
  +0.008 is a landed but modest effect — roughly a quarter of property's
  L30 repair (+0.491 interchange; +0.255 concept-replace on fresh rows).
- **Row-sparse.** The repair concentrates in 3 of 13 baseline-wrong rows
  (3080: 0→0.31, 3103: 0→0.62, 3137: 0.06→0.94); the other 10 never move
  at k=16. Three manifest rows are at baseline ceiling and uninformative.
- **No compact core landed on subtype.** `L35_rank4_loo_add` is marginal
  (+0.047, lower bound touching zero) — consistent with the property
  guard-v2 finding that rank-4 under-transfers; a subtype rank ladder was
  not run and stays future work.

## Wording for the paper (claim 12 / section 5.6)

Subtype localization partially replicates once the layer is chosen by the
capture ladder rather than inherited from property: residual-state concept
replacement at L35 repairs +0.117 [+0.008, +0.254] (k=16, matched random
control null), against repeatedly-null old-trio patches. State it as a
weaker, row-sparse analogue of the property repair — evidence that the
focus-state variable exists on subtype and sits at a different depth — not
as a matched replication of property's effect size, and keep claims 5–6's
strong quantitative story property-scoped.

---

<!-- FILE: docs/qwen35_subspace_erasure_27b_property_sampled_k8_summary.md -->

# Qwen 3.5 27B Property Subspace-Erasure Summary

Aggregate of shards `457191`-`457194` (`512` generations; 16 balanced S1 test h3/h4 rows; k=8).

## Arms

| condition | P(strong) | P(weak) | parse fail | dP vs baseline (CI95) |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.352 | 0.906 | 0.008 | - |
| erase_raw | 0.422 | 0.883 | 0.000 | +0.070 [-0.031, +0.188] |
| erase_orthogonal | 0.398 | 0.891 | 0.000 | +0.047 [-0.023, +0.141] |
| erase_gaussian | 0.367 | 0.906 | 0.000 | +0.016 [-0.039, +0.086] |

## Probe Directions

| layer | val AUC | test AUC | train projection std |
| --- | ---: | ---: | ---: |
| L16 | 0.710 | 0.731 | 0.0128 |
| L31 | 0.850 | 0.863 | 0.0381 |
| L40 | 0.896 | 0.907 | 0.0947 |
| L45 | 0.906 | 0.918 | 0.1520 |
| L53 | 0.933 | 0.940 | 0.5555 |

## Verdict

Qwen property replicates the non-necessity side of the Gemma erasure story: ablating the readable raw correctness direction does not degrade behavior (`erase_raw` is +0.070 with CI crossing zero). The matched controls are also non-destructive (`erase_orthogonal` +0.047, `erase_gaussian` +0.016), so this is not the full Gemma control-separation pattern. The safe claim is cross-model support that the raw readout axis is not load-bearing, not that Qwen shows the same perturbation sensitivity profile as Gemma.

Artifacts: `docs/qwen35_subspace_erasure_27b_property_sampled_k8.json` and shard reports `docs/qwen35_subspace_erasure_27b_property_sampled_k8_shard*of4.json`.

---

<!-- FILE: docs/rank_core_geometry_27b_property_summary.md -->

# Rank-Core Geometry Rider

Analysis of L30 rank-core PCA components from `results/stage2/erasure/focus_state_composite_27b_property_states.npz`.

## INLP overlap

| rank | subspace fraction in INLP | null mean | null p95 | component fractions |
| --- | ---: | ---: | ---: | --- |
| 4 | 0.0001 | 0.0017 | 0.0024 | 0.0000, 0.0001, 0.0001, 0.0002 |
| 8 | 0.0002 | 0.0017 | 0.0021 | 0.0000, 0.0001, 0.0001, 0.0002, 0.0002, 0.0002, 0.0008, 0.0003 |

## Gemma Scope decoder overlap

### layer_30_width_16k_l0_small

| rank | top decoder sqrt-energy | top20 fraction of decoder energy | count >=0.10 |
| --- | ---: | ---: | ---: |
| 4 | 0.9786 | 0.010566 | 10460 |
| 8 | 0.9801 | 0.010304 | 11003 |

Top features by subspace sqrt-energy:

- rank 4: 12625:0.9786, 2018:0.9782, 12253:0.9781, 12182:0.9778, 9214:0.9773, 15026:0.9768, 14617:0.9762, 11562:0.9760, 11910:0.9758, 15886:0.9732
- rank 8: 2018:0.9801, 12625:0.9790, 12253:0.9786, 12182:0.9784, 9214:0.9779, 15026:0.9778, 14617:0.9777, 11910:0.9777, 11562:0.9771, 15886:0.9749

### layer_30_width_262k_l0_small

| rank | top decoder sqrt-energy | top20 fraction of decoder energy | count >=0.10 |
| --- | ---: | ---: | ---: |
| 4 | 0.9816 | 0.001132 | 106663 |
| 8 | 0.9823 | 0.001103 | 112506 |

Top features by subspace sqrt-energy:

- rank 4: 1056:0.9816, 108101:0.9808, 14222:0.9804, 14038:0.9802, 1279:0.9795, 159794:0.9787, 7151:0.9778, 57329:0.9769, 3978:0.9764, 85244:0.9763
- rank 8: 1056:0.9823, 108101:0.9817, 14038:0.9813, 14222:0.9809, 1279:0.9798, 159794:0.9790, 3978:0.9783, 7151:0.9782, 57329:0.9781, 20513:0.9781

## Verdict

The held-out-surviving L30 core is essentially outside the INLP readable subspace: rank-4 and rank-8 overlap are far below the random-subspace null. Gemma Scope gives a different answer: decoder rows align strongly with the causal-core subspace and the first few PCA components, so the dictionary exposes the object. But the exposure is highly redundant rather than sparse-small: thousands of decoder rows have nontrivial overlap, and the top decoder rows explain only a tiny fraction of total decoder-overlap mass. The loop therefore closes as gauge-orthogonal but dictionary-visible, not as a compact handful of Gemma Scope features.

---

<!-- Pre-registrations: items D and E (from docs/causal_handle_directions.md) -->

### D. Readable-stack erasure (full-subspace necessity, claims 2–3)

`scripts/stage2_subspace_erasure.py` extended with subspace conditions. The
landed erasure clamps the rank-1 probe axis; claim 3 records that
correctness information is INLP-redundant and lists full-subspace erasure as
future work. arXiv 2605.05715 (the closest rival) found the OPPOSITE causal
status for its decodable direction via LEACE (−3.6pp damage), so this
experiment discriminates entanglement vs epiphenomenality at subspace
granularity. Mean-projection clamp retained as the erasure operation (ACL
2025 Findings 2506.11673: comparable to LEACE, less collateral than INLP).

- CPU pre-step (run before submission, non-blinding — it is probe fitting,
  not behavior): regenerate INLP stacks for all five readable layers
  (15/30/40/45/53, seed 20260472, rounds 8) into
  `results/stage2/erasure/inlp_direction_stacks_27b_property_5layer.npz`
  (new file; the existing 3-layer artifact stays untouched for
  `stage2_rank_core_geometry.py`). Sanity check: recomputed L30 round-0
  direction must match the erasure `_directions.npz` raw unit (|cos| > 0.99).
- Conditions (6, all layers simultaneously, every position, prompt+decode):
  `baseline`; `erase_raw` (rank-1 continuity anchor);
  `erase_readable_stack` (per layer: QR-orthonormalized 9-direction INLP
  stack, each orthonormal component clamped to its train-split projection
  mean); `erase_random_stack_d{1-3}` (matched-rank-9 random orthonormal
  bases per layer, seeded per draw × layer, identical clamp-target
  estimator — train-split per-component projection means — removing the
  estimator mismatch caveat from the control-matching job).
- Rows: same balanced selection as the landed erasure (seed 20260427,
  8 original-correct / 8 original-incorrect, heights 3/4), k=8 sampled at
  temperature 0.7, `--row-shard i/2` → 2 jobs of 8 rows (384 generations
  each).
- Telemetry: per-component within-run positional projection variance on the
  prompt forward, per condition × layer (the constant-offset lens from the
  control-matching verdict).
- Primary metric: pooled row-cluster bootstrap dP(strong) vs in-job
  baseline over the 16 rows; P(strong|parsed) precision-vs-demolition
  split; original-correct slice reported as the necessity-facing secondary.
- Decision rules (recorded before unblinding; no null outcome exists):
  - Continuity gate: `erase_raw` must replicate the landed null (CI
    including zero). If not, investigate before interpretation.
  - Branch E (epiphenomenality upgrade): `erase_readable_stack` pooled dP
    CI includes zero → claim 3's future-work caveat is replaced by a
    result: the entire readable stack at all probed layers is not
    necessary; the entanglement account (2605.05715) is excluded in our
    setting. Wording rule carried over from claim 2: state as
    non-necessity; do NOT claim the stack is specially inert among
    same-rank subspaces unless the variance telemetry shows within-run
    variance comparable to the random stacks'.
  - Branch N (first necessity positive): `erase_readable_stack` dP CI
    excludes zero (negative) AND the paired (readable − random-stack)
    difference CI excludes zero AND P(strong|parsed) degrades (not pure
    format demolition) → claims 2–3 rescope to "axis not necessary, stack
    is"; the paper reframes to reconcile with 2605.05715 at subspace
    granularity.
  - Ambiguous branch: readable stack destructive but not beyond the random
    stacks → perturbation-load account (Cox-style degradation at rank 9);
    necessity not established, inertness not established at rank 9; claim 3
    caveat stays with sharpened wording.
- Outputs: suffix `readable_stack_erasure_27b_property_k8_shard{i}of2`.

## Pre-Registered Exploratory Job (2026-07-04, evening)

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

---

<!-- Item D status note (2026-07-04 late): job 458403 hit its 4h wall with 4 of 8 shard-0 rows complete; remaining rows resubmitted as finer shards (per-sample seeds are shard-layout-independent, so stitching is exact). Sanctioned DESCRIPTIVE preview of the 4 complete rows (all height-3; no rule evaluation): erase_readable_stack P(strong)=0.5625 vs baseline 0.500; erase_raw 0.531; erase_random_stack d1-d3: 11 of 12 row-cells at 0.000. Telemetry: mean per-component prompt projection variance, readable stack vs random stacks = L15 10 vs ~15-20k; L30 1.3k vs 27-90k; L40 3.8k vs 34-96k; L45 3.9k vs 96-177k; L53 8.8k vs 83-108k. A prediction (Branch E most likely; variance asymmetry as mechanism) was registered in the project log before this preview was examined. -->


---

<!-- FILE: docs/next_experiments_litreview_2026-07.md -->

# Literature Review & Next-Experiment Ranking (2026-07-04)

Deep-research sweep of arXiv (early 2025 – mid 2026) across four angles:
probing vs. causal relevance, low-rank activation steering, causal
abstraction / interchange interventions, and interp of reasoning
correctness. 18 primary sources fetched; 25 load-bearing claims verified
against abstracts/full text by 3-vote adversarial check (25 confirmed, 0
refuted). Companion to `docs/next_paper_skeleton.md` §4 (related work) and
`docs/causal_handle_directions.md` (experiment designs).

## Goal 1 — Scoop check: NOT scooped, but the niche is crowding fast

No published work combines both halves of the core dissociation. Three
preprints from March–May 2026 independently report the **negative half** —
correctness/failure signals linearly decodable yet causally null under
probe-direction steering or ablation — but all three stop at the null: none
localizes the causal variable to a low-rank subspace outside the probe
directions, and none demonstrates a positive behavioral repair. The rank-8
carrier (+0.23 fresh-row repair, 91% of concept-replace) remains the
differentiating contribution.

### The three co-closest 2026 preprints

1. **arXiv 2605.05715** — "Decodable but Not Corrected by Fixed
   Residual-Stream Linear Steering" (May 2026, medical Overthinking regime).
   Linearly decodable at 71.6% balanced accuracy; 29 configs across five
   families of fixed linear steering all ≈0; replicated cross-architecture
   (Qwen2.5-7B) and cross-domain. **Critical difference: its explanation is
   representational ENTANGLEMENT** — LEACE-erasing the decodable direction
   *damages* accuracy (−3.6pp, p=0.01) while 10 random-direction erasures
   are null. That is the *opposite* causal status from our harmless raw-axis
   erasure. This paper is simultaneously (a) the first citation to
   differentiate, (b) the direct methodological template for our LEACE
   experiment, and (c) the rival account our erasure results discriminate
   against. Single-author, under review, numbers may move in revision.
2. **arXiv 2604.13068** — hallucination signals detectable by linear probes
   but steering along the probe direction fails to correct in 7/7 models
   (117M–7B; GPT-2, Pythia, **Qwen-2.5**; 0% correction across 42
   model×magnitude configs). Self-describes as "a clean negative result."
   Consequence: the detection-without-correction asymmetry is already
   documented as architecture-general *including Qwen* — a Qwen null
   replication adds little; only porting the positive carrier is novel.
3. **arXiv 2605.23315** — "Convergence Without Understanding" (May 2026).
   **Terminology collision: already publishes the phrase "epiphenomenal
   correctness."** Correctness decodable across 16 models (66% cross-model
   probe transfer) with only 1.5–5.5% prediction-flip under ablation of top
   probe PCs. But it is a cross-model CKA convergence result, not a
   within-model dissection: no causal-variable localization, no repair, no
   Gemma 3, no ontology reasoning. Our title/abstract wording ("Correctness
   Readouts Are Epiphenomenal") must cite and differentiate immediately —
   or adjust phrasing.

### Precedent that bounds the novelty claim

- **Makelov, Lange & Nanda, arXiv 2311.17030**: subspace patching can look
  causal while the effect flows through a dormant parallel pathway —
  readable-vs-causal subspace mismatch articulated in 2023. Wu, Geiger et
  al. (arXiv 2401.12631) dispute the normative "illusion" framing but not
  the phenomenon. Our novelty is the specific form: high-AUC correctness
  readability that is causally null, causal variable in an *identified*
  rank-8 subspace outside the probe directions, working repair, on a
  reasoning task.
- Correctness decodability itself is established, correlational-only prior
  art: arXiv 2604.05655 (GSM8K correctness AUC 0.87 at L29,
  Llama-3.1-8B; steering never uses the probe direction), arXiv 2504.05419
  (intermediate-answer correctness AUC >0.9 on R1-Distill-Qwen-32B; zero
  intervention vocabulary in the full text), arXiv 2602.06022 (CORAL;
  probe used only for post-hoc answer re-weighting). The causal test is the
  gap we fill.
- Low-rank causal carriers of reasoning are precedented (plausibility
  support, not scoops): arXiv 2509.06608 (trained per-layer steering
  vectors recover 95.3%/87.8% of full fine-tuning's reasoning gains),
  arXiv 2505.15634 (PCA-like contrastive-delta steering basis,
  methodologically parallel to our hinted-minus-unhinted PCA), arXiv
  2506.18167 (reasoning behaviors readable AND steerable — the opposite
  regime, process-style behaviors not correctness).

### Positioning consequences for the draft

- The readable-but-not-steerable gap is now **established background**, not
  a headline. Foreground the positive identification of the causal carrier
  + the erasure-based inertness demonstration.
- Frame the erasure result explicitly as discriminating between two
  published accounts of steering nulls: entanglement (2605.05715 — erasure
  destructive) vs. genuine epiphenomenality (ours — erasure harmless with
  destructive matched controls).
- Free theory framing: our causally-null readout maps onto the "behavioral
  null-space" formalism of arXiv 2511.04638 (ICLR 2026 oral); the rank-8
  add must be shown non-"pernicious" in their sense (see experiment 2).
- Three independent groups hit the negative half in a 3-month window — the
  topic is hot and the moat is the positive result. **Re-run the scoop
  check over the June–July 2026 arXiv window immediately before
  submission.**

## Goal 2 — Ranked next experiments (hardening-first)

The two peer-reviewed methodological critiques set the reviewer bar:

- **arXiv 2507.08802** (NeurIPS 2025 spotlight): unrestricted causal
  abstraction is vacuous — expressive nonlinear alignment maps reach 100%
  IIA on randomly initialized models. Cuts in our favor (the rank-8 basis
  is a fixed linear construction) but encoding assumptions must be stated
  explicitly.
- **arXiv 2511.04638** (ICLR 2026 oral): patching/DAS/mean-difference
  interventions frequently push activations off-distribution; formalizes
  harmless (behavioral-null-space) vs. pernicious (dormant-pathway)
  divergences; mean-difference patching specifically "can activate hidden
  pathways."

### 1. Random-basis specificity control + on-distribution diagnostics for the rank-8 core — RUN FIRST

The flagship positive claim (8) currently has no random-BASIS control: the
guard arms include `L30_random_replace` (random positions) but no matched
random rank-8 *add* basis at concept positions. 2507.08802/2511.04638 make
this the first thing a reviewer asks for; if random matched-norm bases also
repair, the specificity claim collapses — existential, so it runs before
anything else.

- Design: guard-v2 harness, one new arm family — 10 random rank-8 bases,
  matched per-component norms to the LOO PCA basis, added at concept
  positions at the same scale; same 26 fresh rows or the 13 manifest rows,
  k=8, row-cluster bootstrap. Offline riders on saved states (no GPU):
  logit-lens/unembedding projection of the 8 components, principal angles
  vs the INLP probe stack per layer (sharpens claim 9's subspace fractions
  into angles), rank sweep 1–16 from the saved concept-position submatrix,
  nearest-neighbor distance of steered L30 states to natural hinted states
  + dormant-unit activation check (2511.04638 diagnostics).
- Cost: one 4h SLURM slot + offline analysis.
- Branches: random bases null → specificity guarded, claim 8 hardened, and
  the on-distribution numbers become a methods sidebar. Random bases repair
  → the repair is generic-perturbation at that norm; claim 8 must be
  re-scoped before drafting (this is exactly why it runs first).

### 2. LEACE full-readable-subspace erasure + layer-exhaustive ablation

Claim 3's listed future work, upgraded to must-do: it is the unique
discriminator against 2605.05715's entanglement account, and it closes
claim 7's remaining necessity question (single-fixed-layer nulls are
unreliable per arXiv 2604.03867: per-input optimal steering layers beat the
best fixed layer by 55–86%, oracle bound). Methodological blueprint: arXiv
2506.11673 (ACL 2025 Findings — mean-projection/LEACE cause far less
collateral damage than INLP in amnesic probing).

- Design: LEACE (or mean-projection) of the full INLP readable stack, all
  readable layers simultaneously, prompt positions; controls per house
  style (orthogonal, Gaussian, height direction, dose-response,
  P(strong|parsed) split) PLUS the 10-random-direction erasure control
  mirroring 2605.05715 so the cross-paper comparison is direct.
- Cost: LEACE fit is CPU on saved activations; one to two 4h slots for the
  behavioral arms (property; subtype optional).
- Branches (no null outcome — same structure as the original erasure
  decision rule): accuracy intact → epiphenomenality upgrades from "the
  readout axis" to "the entire readable subspace," the entanglement rival
  is excluded in our setting, claim 3's caveat is deleted. Accuracy drops →
  the readable subspace is entangled and only the axis is inert; claims
  2–3 re-scope, and the paper aligns with 2605.05715's account at subspace
  granularity — still publishable, framed as reconciliation.

### 3. Subtype carrier: layer × position sweep with token-conditional coefficients

The row-sparseness of the L35 result (3/13 addressable rows) may be
per-row layer mismatch, not a weak carrier: 2604.03867 shows per-input
optimal layers deviate 3.8–6.5 layers from any fixed choice, and arXiv
2605.03907 shows prompt-induced activation shifts are strongly
token-position-dependent (convergent with our concept-mention finding;
motivates token-conditional rather than constant coefficients).

- Design: extend the discriminator ladder to a dense layer grid with
  per-row best-layer readout (report both fixed-layer pooled and
  per-row-oracle numbers, labeled as such), concept-mention-gated
  coefficients; same 16 manifest rows + fresh seeds, k=8.
- Cost: one to two 4h slots.
- Branches: per-row layer selection recovers a dense repair → claim 12
  upgrades toward matched replication (with the oracle/fixed distinction
  stated). Still sparse → current wording stands and the hedge closes
  harder; the property/subtype asymmetry becomes a real task-structure
  finding (open question 4 of the survey).

### 4. Qwen positive-carrier port — DEMOTED to next paper

The null half is already architecture-general including Qwen-2.5
(2604.13068), so a Qwen null replication adds nothing; only porting the
rank-8 carrier is novel, and that is a full pipeline (hint suite, delta
extraction, PCA, guard) — not a hardening item. Keep claim 11's current
scoping sentence. Scope to Qwen2.5-7B/14B (fits 2 GPUs) if reviewers demand
a second architecture; otherwise it is the opening experiment of the next
paper (and the survey's open question 3).

## Writing to-dos (zero compute)

- Related work §4 restructure: co-closest prior = 2605.05715 + 2604.13068
  (cite both; a verifier split 2-1 on any "closest" superlative); the
  terminology-collision paragraph for 2605.23315; Makelov 2311.17030 +
  Wu 2401.12631 as the readable-vs-causal precedent pair; correlational
  correctness-probing block (2604.05655, 2504.05419, 2602.06022); low-rank
  carrier precedent block (2509.06608, 2505.15634, 2506.18167); methods bar
  (2507.08802, 2511.04638, 2506.11673). Existing planned citations (Cox et
  al., ITI, hydra-effect, Gemma Scope) stand.
- State encoding assumptions for the rank-8 construction explicitly
  (fixed linear PCA basis, no trained alignment map) — the 2507.08802
  defense.
- Adopt/engage the behavioral-null-space vocabulary of 2511.04638 for the
  probe-readout result.

## Caveats

- The three closest analogues are non-peer-reviewed preprints from
  March–May 2026; numbers may change in revision. The not-scooped verdict
  is bounded by arXiv coverage; June–July 2026 postings not exhaustively
  swept — re-check at submission.
- 2605.03907 / 2604.03867 study persona/alignment (CAA) steering on
  Llama-2 / Qwen-1.5 / Gemma-2 — their token-position and layer-selection
  lessons are directional analogies for Gemma 3 27B ontology reasoning,
  not demonstrated transfers; the 55%/86% figures are post-hoc oracle
  bounds.
- 2509.06608's vectors are per-layer dense biases (~100–230K params), a
  low-parameter but not literally rank-8 intervention; its gain-recovery
  figures are 95.3%/87.8% (verifier-corrected from the abstract's rounding).

---

<!-- FILE: docs/hint_free_repair_direction.md -->

# Hint-Free Repair: From Instrument to Method (research direction, 2026-07-04)

Status: ideation, not pre-registered. Nothing here is a claim of the current
paper; this documents the deployment question the rank-8 result makes askable,
the evidence for and against it being answerable, and the experiment ladder we
would run. Framing rule for the paper: one discussion paragraph, worded as
"the question this paper makes askable," never as a capability we have.

## The question

The rank-8 recipe repairs failing rows it was never tuned on (+0.231 guard v2,
replicated +0.245 in the specificity run). If failures on other hard tasks
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

**Step 2 is now unlocked**: closed-loop arms on the guard-v2 harness —
unhinted baseline / mean_only (floor) / rank8 true-coefficients (ceiling) /
rank8 ridge-predicted coefficients (LOO) / rank8 shuffled-row coefficients
(control). Pre-register deltas and rules in causal_handle_directions.md
before launch; needs fresh rows' unhinted states captured in-job (the
predictor training set can be the composite rows).

## Relation to the current paper

Goes in §6 Discussion as one paragraph: some failures that look like
capability gaps are single, low-dimensional, causally accessible decision
errors; this paper locates, compresses, and flips them when the target is
known; whether the target can be inferred without the hint is the natural
next question. Do not promise the ladder in the paper; cite mean_only as the
existence proof that a nonzero hint-free effect is real.
