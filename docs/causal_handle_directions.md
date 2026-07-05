# Causal-Handle Directions

Assessment of whether the current intervention nulls justify `causally
inaccessible under tested methods`, and which experiment families could still
produce a positive causal result. Companion to `docs/next_paper_synthesis.md`
and the causal-abstraction dashboard.

## Bottom Line

The current nulls do not yet license `causally inaccessible` as a settled
claim. They license `inaccessible under single-site, single-direction,
additive, low-power interventions targeting the correctness summary`. There
are at least four untested intervention families, two of which reuse
artifacts we already have, and one cheap experiment (multi-layer subspace
erasure) that produces a real causal claim regardless of which way it comes
out.

## The Reframe the Data Already Supports

The strongest single result in the program is the prefix-0 margin finding:
on recognition-gap rows, the wrong/selected hypothesis is already more
prompt-likely than gold before any token is generated (Gemma 13/13, Qwen
14/14), while forced-choice recognition stays correct.

Implication: the causal variable that determines free-form output is
**which hypothesis the prompt computation settled on** (`selected_hypothesis`
in the shared variable set), and it is fully determined during prompt
processing, not during decode.

This reframes every null. The tested interventions — raw L45 direction,
optimized vectors, decode-time gates, +1sd injections during generation —
target a *correctness summary*, mostly at or after the point where the
decision is already made. If the probe reads a downstream evaluation of the
committed answer (a "this is going badly" signal) rather than the commitment
itself, the observed pattern is exactly what that hypothesis predicts:
highly readable, robust across models, and causally inert under steering.
A fuel gauge, not a fuel line.

Working hypothesis for the next paper:

> The causal variable is hypothesis selection, computed during prompt
> processing; the correctness probe reads a downstream evaluation of it.

This hypothesis predicts all existing nulls, predicts the
recognition-vs-generation gap, and makes testable predictions in both
directions: erasing the probe subspace should *not* hurt accuracy, while
patching the selection state *should* change outputs.

## Why the Existing Nulls Are Weaker Than They Look

1. **Power.** `F->T=0` over 14 paired rows with a discrete strong-correctness
   flip metric cannot distinguish a ~15% repair effect from zero. The
   commitment manifest has 57 recognition-gap candidates; the gated decode
   run used 14. Regenerated baseline accuracy is 3/14, so each row is a
   single Bernoulli draw of a stochastic outcome. Every future intervention
   should report **delta P(strong-correct) over k >= 8 samples per row** in
   addition to discrete flips, and should use the full candidate rowset.

2. **Single-site, single-direction, additive.** All tested interventions are
   one layer (L45/L50), mostly one position, rank <= 4, additive +1sd. The
   self-repair / hydra-effect literature shows redundantly encoded variables
   get restored by downstream components when one site is perturbed; a null
   at one site does not aggregate to a null over the computation. Additive
   +1sd is also the weakest variant — **clamping** (project out the current
   value, set the component to the correct-class mean) has not been run.

3. **Only sufficiency was tested seriously.** Repair (`F->T`) requires
   injecting content the residual stream may not carry at that site.
   Necessity — does ablating the correctness subspace degrade behavior? —
   is nearly untested (one weak `T->F` at L50). A reviewer asking "did you
   test necessity or only sufficiency?" currently gets the answer no.

## Candidate Experiments, Ranked

### 1. Multi-layer subspace erasure (necessity test)

LEACE / mean-projection of the correctness probe subspace at **every layer
simultaneously**, prompt positions, with the standard control set
(regenerated baseline, orthogonal subspace, matched Gaussian, positive
control). Cheap, well-powered, and decisive either way:

- Accuracy untouched -> the readout is genuinely epiphenomenal. That is a
  positive, publishable causal claim, not a null.
- Accuracy drops beyond controls -> causal necessity, the first real
  positive causal result in the program.

This is the cheapest path to upgrading `inaccessible under tested methods`
to a real causal statement.

Status (2026-06-16): validated at k=8 for Gemma property (jobs 456912,
456915-456918; docs/subspace_erasure_27b_property_sampled_k8_summary.md), Gemma
subtype (jobs 456963-456966; docs/subspace_erasure_27b_subtype_sampled_k8_summary.md),
and Qwen property (jobs 457191-457194;
docs/qwen35_subspace_erasure_27b_property_sampled_k8_summary.md). Gemma property
pooled over 512 generations: erase_raw delta-P=-0.016 (0.4 sigma from zero)
while erase_orthogonal is -0.367 (3.4 sigma below erase_raw) and erase_gaussian
-0.227 with 48% parse failures. Qwen property: erase_raw dP=+0.070 CI
[-0.031,+0.188], with non-destructive controls. Outcome: the epiphenomenality
branch — the correctness readout direction is not necessary for task behavior.
Remaining tightenings: LEACE-style full concept erasure, wider layer set, and
optional Qwen subtype if cross-task cross-model symmetry becomes necessary.

### 2. Cross-prompt patching: transplant the recognition state into generation

The model demonstrably contains a gold-preferring state — it surfaces under
forced choice on the same rows. So the donor activations for repair exist in
the same model on the same problem. Run the MCQ/recognition prompt, extract
activations over the gold-hypothesis tokens (or the decision position), and
patch them into the free-form run at matched positions.

This sidesteps the fundamental weakness of vector steering: a 1D direction
cannot carry *which* hypothesis is right, but a donor state carries the full
content. A repair here with controls is the positive causal result, and it
directly tests the deployment-gap hypothesis against the
correctness-summary hypothesis.

Status (2026-06-10): completed as a controlled repair null (job 456913,
docs/recognition_state_patch_27b_property_manifest.json). Donors selected
gold on 14/14 rows, but patching the shared ontology-context block
(about 90 tokens, L30/L40/L45) gave F->T=0 with disruption equal to
magnitude-matched noise. The recognition advantage is not carried by the
context-token encodings; the remaining candidates are the hypothesis-option
tokens and the comparison/decision positions, which have no matched
receiver positions — a future variant needs a different transplant design
(e.g., appending scored-hypothesis text or patching the decision position).

### 3. ITI-style multi-head intervention

Inference-Time Intervention (Li et al., TruthfulQA) succeeded exactly where
this setup is failing: probe-readable truthfulness that single-direction
residual steering could not move. Recipe: probe each attention head, select
the top-K heads by probe AUC, shift along each head's own direction
simultaneously during decode. The dashboard currently has **no per-head
probing at all** — that is a gap in its own right; if correctness
concentrates in a few heads, both readout and intervention sharpen. This is
a qualitatively different intervention family per the existing decision
rule, and it is cheap.

### 4. Feature-level clamping with the local dictionaries

The local top-k dictionaries capture the predictive signal in
reconstruction (AUC 0.877-0.919) but have never been used to intervene.
Identify the sparse features carrying the probe signal and clamp them to
correct-class values during prompt processing. Reuses an artifact already
paid for, and tests whether the sparse basis exposes a handle the raw
direction does not.

### 5. Counterfactual minimal pairs for DAS on `selected_hypothesis`

Causal-abstraction interchange interventions want input pairs that differ
minimally in the high-level variable. If clean/corrupt pairs differ in many
surface respects, DAS has a noisy training signal, and rank-4 at one
position is a narrow search. InAbHyD allows constructing minimal pairs where
a **single ontology fact flips which hypothesis is gold**. Re-run DAS over
the `selected_hypothesis` variable (contentful), not binary correctness
(1-bit summary), searched jointly over several layers and the
hypothesis-token positions. The prefix-0 result predicts this version should
work where the correctness-variable version did not.

### 6. Localize the commitment circuit via the prompt margin

`gold_vs_foil_logprob_margin` at prefix 0 is a continuous, behaviorally
validated readout of the causal variable. Run path patching / edge
attribution (AtP*, EAP) with the margin as the metric to find which heads
and MLP paths set it during prompt processing, then intervene on those
components. AtP estimates already track exact patches at L50 (r=0.97), so
the tooling exists — it was pointed at correctness instead of commitment.

### 7. Within-recognition selection-circuit localization (added 2026-06-11)

Successor to experiment 2 after its null: the gold preference demonstrably
exists inside the recognition run, so localize it there, where minimal pairs
are cheap and scoring needs no generation (teacher-forced logprob of `(A)`
vs `(B)`).

Design, on the manifest recognition-gap rows:

- Minimal pair: original MCQ prompt (gold at A) vs option-swapped prompt
  (gold at B). Baselines on both verify the choice follows content, not
  letter position.
- `patch_options_from_swap`: replace the gold-option and foil-option token
  spans in the original prompt with the same-content spans from the swapped
  run (identical text, swapped letter-binding context; small RoPE position
  offsets are a flagged caveat, calibrated by the noise control). If the
  choice flips to follow the donor binding, the per-option comparison
  result is stored in the option spans — a causal handle on
  `selected_hypothesis`.
- `patch_tail_from_swap`: patch the matched post-options instruction block
  and final positions from the swapped run. If this alone flips the choice,
  the decision is already resolved downstream of the options by the patched
  layers.
- Controls: magnitude-matched noise on the same spans, both baselines,
  layer-band flag for later sweeps (start with L30/L40/L45 as in
  experiment 2).
- Metric: per-row choice flips plus gold-vs-foil letter logprob margins;
  k-sample dP not needed since scoring is deterministic.

Payoff: a positive interchange-intervention result on `selected_hypothesis`
inside recognition, which then sharpens the free-form question to "why does
generation not deploy the comparison circuit that these spans/positions
demonstrably carry?"

Behavioral precursor (2026-06-11, Modal endpoint,
docs/candidates_in_context_27b_property_manifest.json): listing candidates in
the free-form prompt flips the recognition-gap rows — baseline
P(strong)=0.036, gold-among-candidates=0.839 (10/14 rows false-to-true),
foils-only=0.232 (upper bound on priming; those successes negate the listed
polarity foil into gold). The deployment gap is candidate availability at
selection time. The interchange experiment above is the activation-level
version of this result.

RE-AIMED at `target_concept` (2026-06-11, after the hint-gradient result in
docs/proposal_hints_27b_property_manifest.json: concept-only hint takes
P(strong) 0.045 -> 0.955, concept+property -> 1.000). The failing variable is
which concept the model hypothesizes about, so the interchange should target
the concept-focus state, not MCQ option spans:

- Minimal pair: baseline free-form prompt (receiver) vs hint-augmented prompt
  (donor) for the same row. Place the hint line BEFORE the ontology context
  in the donor so the shared context-block encodings actually differ under
  causal attention (a hint placed after the context cannot alter
  context-token representations); validate behaviorally that hint-first
  retains the ~0.955 repair before patching. Also keep a hint-after-context
  variant whose donor patch is the post-hint tail positions only.
- `patch_hint_state`: patch the donor's context-block (hint-first variant)
  or post-context tail (hint-after variant) residuals into the receiver at
  matched positions (LCS alignment, reuse
  scripts/stage2_recognition_state_patch.py machinery), then generate.
- Bidirectional control — the strongest causal test: a wrong-concept-hint
  donor. If patching the wrong-concept hint state makes generation
  hypothesize about THAT concept, the patched state causally sets
  `target_concept` in both directions.
- Output metric upgrade: score not only strong correctness but which concept
  the generated hypothesis is about (subject extraction via
  `ontology_fol_structured`-style parsing), so steering claims are
  concept-resolved rather than only accuracy-resolved.
- Controls: regenerated baseline, position-shuffled donor, magnitude-matched
  noise, k>=8 samples per row, layer/position-band sweep flag (start
  L30/L40/L45).
- Decision rule: patched-repair (or patched-misdirection) of at least 3 rows
  beyond both controls establishes `target_concept` as a causal handle; the
  MCQ option-span variant remains a follow-up for the recognition side.

Status (2026-06-12): completed as the first positive activation-level repair
(job 456968, docs/hint_state_interchange_27b_property_manifest.json).
patch_hint_gold lifts P(strong) 0.214 -> 0.714 (+0.500, 8/14 rows
false-to-true) vs matched noise at 0.036 — clears the decision rule for a
repair handle. Qualifiers: the position-shuffled gold donor also repairs
(+0.250), so the effect is partly content-not-position; and patch_hint_wrong
is null (no misdirection, no rise in wrong-concept mentions), so
bidirectional control did not transfer through activations at these sites.
Follow-ups: k>=8 batched rerun for power, position-band sweep, and a
content-vs-position decomposition before claiming a localized
`target_concept` variable.

## Review Response Follow-Ups (2026-06-12)

External review of the status overview accepted the core story and pushed on
five fronts. Actions, in priority order:

1. **Erasure control matching.** Alternative account to rule out: the raw
   probe direction may carry between-run information that is near-constant
   within a run (clamping it shifts every position by roughly one vector,
   which models tolerate), while random/orthogonal directions carry
   within-run-varying structure. Hook telemetry recorded mean |delta| only
   (conflates offset and variance), so this needs a small GPU job: report
   within-run projection variance per direction, add a between-run control
   direction by construction (e.g., a height/difficulty-probe direction),
   dose-response the controls to the scale where parse failures reach
   baseline, and state explicitly that the clamp target is the global scalar
   train-projection mean applied at every position.
   DONE from existing data: P(strong|parsed) separates precision from
   demolition — orthogonal erasure destroys correctness even among parsed
   outputs (property 0.410->0.034, subtype 0.474->0.000), so the controls are
   not mere format demolition; Gaussian damage is substantially format.
   Citable precedent for the writeup: Cox et al. motivate orthogonal
   baselines with exactly this hypothesis — large perturbations along
   semantically irrelevant directions induce general reasoning degradation
   toward random guessing, with larger models more robust to arbitrary
   directions. The control asymmetry here is that phenomenon at higher dose,
   which makes the between-run control the natural decisive addition.
   Pre-registered prediction for the variance test: if the probe direction's
   within-run positional variance is far below the controls', the
   constant-offset account is live and the between-run control is mandatory
   before any writeup; if comparable, the current potent-machinery
   interpretation strengthens. Either outcome is reportable.
   DONE (jobs 457207-457210, docs/erasure_control_matching_27b_property_k4_summary.md):
   the constant-offset branch fired. Raw within-run projection variance is far
   below the controls' at the high-variance layers (L15 10.65 vs 818/382 sd²;
   L30 0.84 vs 3.1/75.3), and the between-run height control — near-zero
   within-run variance by construction — erases as harmlessly as raw (dP
   -0.047 vs raw +0.031, both CIs straddle zero), while orthogonal_s1 remains
   destructive (-0.344, CI excludes zero) with monotone dose-response and the
   precision-vs-demolition split reproduced. Consequence: keep the
   non-necessity claim; do not cite control destructiveness as evidence the
   raw axis is specially inert among same-norm directions.
2. **Sigma accounting.** DONE: row-level cluster bootstrap is now the primary
   test in both erasure summaries; raw CIs straddle zero, all control CIs
   exclude zero; the null rules out effects above ~0.1 at n=16 rows. Naive
   per-generation sigmas demoted to planning numbers. Policy: any future
   sigma figure uses row-level bootstrap or a mixed model.
3. **Interchange power.** DONE: job 456990 (docs/hint_state_interchange_27b_property_k8_summary.md).
   Repair confirmed at +0.491 CI [+0.277, +0.705]; shuffled donor retains
   +0.393 (carrier predominantly content; ordered increment +0.098 with CI
   excluding zero); misdirection null on both accuracy and the
   concept-resolved targets_wrong metric. Framing: asymmetric
   controllability. Consequence for item 5: the add-to-unhinted low-rank
   delta arm is now the priority — content-dominance raises its prior.
4. **Bidirectionality.** Hypothesis: the hint works partly through
   decode-time attention back to the literal hint tokens, which a
   context-only patch cannot reproduce. Tests: transplant the donor KV cache
   at the hint positions, or patch with the hint span structurally retained
   but content-neutral. If misdirection still fails through activations, the
   framing is asymmetric controllability (token-reachable vs
   intervention-reachable states diverge, connecting to the non-surjectivity
   result).
   Attractor-strength reanalysis (2026-06-12, zero compute): NOT supported.
   Per-row wrong-hint concept shift vs baseline modal-answer share gives
   Spearman rho=+0.17 (p=0.56, wrong sign for the account); repair vs
   attractor rho=-0.27 (p=0.35). Row-level misdirection shifts are
   heterogeneous (-0.88 to +0.88, mean ~0) with no attractor pattern, so the
   easy-information/hard-displacement account does not explain the
   asymmetry in this data; the KV transplant remains the discriminating
   test.
5. **Hint-delta program (handle -> variable) — re-scoped after k=8 into one
   batched job.** The k=8 content-dominance result (shuffled donor retains
   +0.393 of +0.491) makes the mean delta the single highest-information
   experiment left, and the shuffle arm already answered the ordering half
   of content-vs-position. One combined batched 4h job on the same 14 rows:
   (a) **delta-add sweep**: mean (hinted - unhinted) over the context
   positions, one vector per layer, added to the unhinted run at a small
   scale sweep — if it reproduces a large fraction of +0.39, the focus
   state is a low-rank causal variable;
   (b) **subset-band patching**: full patch restricted to concept-mention
   token positions vs matched random subsets — which positions' content
   carries it;
   (c) **cross-row delta transplant**: row A's delta into row B should
   misdirect toward A's concept if the delta is concept-specific — a
   cleaner misdirection test than the wrong-hint donor since both deltas
   are gold-validated.
   Save the per-layer delta vectors; three geometry checks are then free:
   cosine vs the correctness probe direction (near-orthogonality would be
   the paper's sharpest contrast: the causal direction is not the readable
   one), overlap with Gemma Scope decoder rows (sparse-dictionary thesis),
   and delta-norm by layer. Either outcome is publishable: low-rank gives a
   clean DAS-style variable; failure gives `causally distributed` its
   sharpest datapoint with a demonstrated repair attached.

   Pre-registered analysis plan for job 456999 (recorded before data):

   - **Conditional control**: if delta_add is positive, run the
     matched-norm random-direction add arms before any writeup
     (`--with-random-add`, implemented; 112 generations per scale). Addition
     and replacement are different operations, so the 456990 noise control
     does not cover generic-add effects. If delta_add repairs AND cross_row
     misdirects toward the donor concept, the pair jointly rules out generic
     perturbation; the random-add arm is mandatory only if cross_row is
     inert.
   - **Geometry analyses on the .npz**: (a) project each delta onto the full
     INLP readable-subspace stack per layer (directions saved by the
     extended probe-on-erased check), not only the canonical probe axis —
     "outside even the redundant readable subspace" is the maximal contrast;
     substantial projection onto INLP rounds 2-8 would be the alternative
     publishable nuance. Report against the chance level: at d~5376 random
     cosines concentrate around 1/sqrt(d) ~ 0.014, so 0.08 is six times
     chance, not "orthogonal". (b) Pairwise cosines among per-row deltas:
     clustering implies a shared focus component; mutual near-randomness
     implies concept-specific deltas — a geometric preview of the cross_row
     behavioral result, and agreement between the two is itself a result.
   - **Anchoring**: no full-patch arm is in-job, so "fraction of the content
     effect reproduced" is computed by row-paired cross-job differencing
     against 456990's shuffled (+0.393) and full (+0.491) arms on the same
     14 rows — not pooled-mean comparison. Subset arms should sit below the
     456990 full patch as a consistency check.
   - **Outcome grid**: delta_add+/subset_concept+ = low-rank pointer written
     at concept mentions (cleanest result). delta_add+/subsets equal =
     low-rank broadcast focus bias, position-free, consistent with the
     shuffle finding. delta_add-/subset_concept+ = positionally localized
     but high-rank (patchable spans, no summarizing vector). Both- =
     strongest `causally distributed with a working handle`. cross_row has
     three readings: misdirects toward donor concept (direction-level
     misdirection works; the wrong-hint-donor failure becomes the anomaly,
     plausibly coherent-vs-conflicted donor states); inert (deltas are
     context-bound); row B improves toward its OWN gold (delta carries
     generic hypothesize-well content, connecting to the parsimony hint's
     0.241).
   - **Null-branch caution**: flat s1/s2 licenses only "rank-1 insufficient
     at tested scales" — a wider scale sweep and a rank-k PCA add (top
     components of per-position deltas) are owed before "not low-rank"
     appears anywhere.
   - **Necessity gap**: all six arms are sufficiency-flavored. The
     erase-delta-from-hinted-runs arm (does the 1.000 collapse?) is what
     completes the variable claim; it pairs with the KV transplant as the
     next cluster job.
   - Calibration note on the attractor reanalysis: at n=14 the CI on rho
     spans roughly +/-0.5, so rho=+0.17 is "unsupported", not "refuted".

   VERDICT (2026-06-12, job 456999;
   docs/hint_delta_27b_property_summary.md): outcome-grid bin
   delta_add(-)/subset_concept(+) — the focus state is positionally
   localized but high-rank. Concept-position patching +0.250 CI
   [+0.107,+0.420], random subsets null, mean-delta null at s1/s2,
   cross-row inert (donor targeting 0.027). Geometry: deltas lie below
   chance projection onto the full INLP readable subspace (0.009-0.021 vs
   0.041) and share a causally inert common component (pairwise cos
   0.23-0.38). Random-add control not triggered (delta_add null);
   subset_random served as the matched control for the positive arm.

   REVIEW CATCH (2026-06-12): the mean-delta null and the geometry numbers
   are diluted - averaged over ~90 tokens when the causal content sits at
   5-15 concept-mention tokens, with the inert shared component dominating
   the position-mean. Supportable phrasing until the restricted tests run:
   "positionally localized; not captured by a uniform rank-1 summary".
   Cross-row inertness is overdetermined by the same dilution.

   NEXT CLUSTER JOB (composite, ~10-11 arms x 14 rows x k=8 ~ 1,250
   generations ~ 3h batched - one 4h slot):
   - baseline;
   - reverse subset patch (necessity, dilution-immune): hinted receiver
     with the UNHINTED run's concept-position states - does the 1.000
     collapse? Exactly symmetric to the positive sufficiency arm;
   - complement patch: everything except concept positions - decomposes
     the missing half of +0.491 (additive split vs superadditive
     surroundings);
   - restricted delta add: concept-position-restricted mean delta at
     concept positions, scales ~1/2/4/8;
   - own-position delta: row B's concept-position delta at B's own
     positions (the sharpened low-rank test);
   - spotlight test: row A's concept-position delta at B's gold-concept
     positions - transfers-and-repairs = movable attention-allocation
     operator ("spotlight"); inert/disruptive = concept commitment written
     in place;
   - KV transplant at hint positions — the WRONG-HINT donor's KV, scored on
     targets-hinted-concept (the open question is why misdirection transfers
     through tokens but not residual patches; a gold-KV arm re-demonstrates
     known repair and is cut if budget binds).

   Pre-registered before job 457002 unblinds (review, 2026-06-12):
   - **Reversion metric on reverse_subset**: score arm 3 not only for
     collapse but for targets-modal-wrong-answer (modal answer taken from
     the in-job baseline arm per row). Collapse-to-attractor would mean the
     intervention toggles the variable between its two natural values — set
     by the hint, reverted to default when unset: bidirectional control
     achieved through the necessity direction after the injection direction
     failed.
   - **Conditional link 1 (necessity null)**: if the hinted repair survives
     the reverse patch, that is evidence the hint works through decode-time
     attention to the literal hint tokens rather than context encodings
     alone — the KV job jumps the queue rather than staying deferred.
   - **Conditional link 2 (rank-1 null at both scales)**: the offline PCA
     spectrum on the concept-position submatrix gatekeeps a future causal
     rank-k arm — concentrated spectrum (2-3 components) earns the rank-k
     add before any rank claim; flat spectrum plus the rank-1 null jointly
     support the claim with no further cluster time.
   - **Scale-sweep insurance**: null-at-1/partial-at-4 earns a x2 follow-up
     before interpretation; a two-point sweep can straddle a window.
   - **Hinted-baseline calibration**: read the in-job hinted baseline FIRST;
     its gap from the Modal 1.000 calibrates the harness tax for every
     behavioral-vs-patched comparison in the program.
   - **KV job redesign (deferral consequence)**: in a dedicated job the
     gold-hint KV arm flips from redundant to essential — it is the positive
     control for the cache-splicing machinery; a null wrong-KV without a
     working gold-KV beside it is uninterpretable. Three arms minimum:
     baseline, gold-KV (machinery check), wrong-KV (the question), scored on
     targets-hinted-concept.

   KV/HINT-SPAN JOB SPEC v2 (post-457002, pre-registered):
   - Ablation spec pinned: **decode-only attention masking** of the
     hint-span key positions (not full KV zeroing) — that is what tests the
     actual sentence "available to decode-time attention"; if repair
     survives decode-only masking, the commitment was already written
     elsewhere during prompt processing.
   - **The combination arm is the discriminating one**: hint-span ablation x
     concept-position reversion, together. Three candidate carriers exist
     in the hinted run (hint-span KV; context encodings at the 3 patched
     layers; context encodings at the ~57 unpatched layers). The 457002
     necessity null excluded only the second. Ablation alone cannot
     separate the survivors. Bins: collapses only under the combination ->
     the two pathways are enumerated and jointly exhaustive (strongest
     claim); survives both -> the carrier is the unpatched layers and the
     layer claims need rework; collapses under ablation alone ->
     decode-time attention is the dominant carrier.
   - Wrong-KV bins crossed with the above: misdirects (token-level
     misdirection localizes to the KV pathway) / does not (asymmetric
     controllability extends to the cache).
   - Arms (kill-safe order; review catch: five arms reference the UNHINTED
     baseline, which must be in-job — baselines have wobbled 0.196 -> 0.161
     -> 0.163 across jobs, hence the house rule): unhinted_baseline,
     hinted_baseline, hint_span_masking, masking_x_reversion (combination),
     gold_KV_transplant (machinery control + attention telemetry),
     wrong_KV_transplant, perpos_add_own, restricted_add_x2, rank_k_L30.
     (perpos_add_own added pre-implementation: the ladder rung "per-position
     additive, drops replacement" was never run as the OWN-row version — the
     composite ran only the foreign per-position spotlight and the rank-1
     mean; the rank-k reading rule needs the rank-full additive reference.)
     KV positional convention — CORRECTED at implementation (source
     verification, TL 3.0): "append-with-donor-phases" is unrealizable —
     TransformerLens caches PRE-rotary keys and re-applies rotary from
     mask-cumsum positions on every forward (abstract_attention.py:280-282),
     so tail-appended donor K necessarily acquires the phases of its splice
     slots. The de facto pinned convention is therefore RE-ROTATE-TO-SLOT:
     the donor hint behaves geometrically as if its text were appended
     after the receiver context — a coherent counterfactual prompt layout.
     Values are position-free and exact; decode positions shift by
     +span_len as pre-registered. Gold-KV control + attention telemetry
     adjudicate machinery validity (CPU integration test on pythia-70m
     passed: splice attention 0.11 mean, masking exactly 0.0, seeded
     determinism verified before any GPU use). Masking and
     combination arms do not touch the splice machinery, so a splice bug
     discovered via telemetry costs only the splice-dependent arms, never
     the exhaustive-necessity headline. ~8 arms ~ 900 generations ~ one
     slot.

   457005 BINS (pre-registered ~2h before landing): positive = row-paired
   CI excludes zero. full_patch repairs AND subset_concept repairs with
   subset_random null -> localization replicates cross-task (claim 12
   lands). full_patch repairs but subset_concept does not -> localization
   is task-dependent; the localization claim scopes to property and claim
   12 records the scoping. Both null -> focus-state causal accessibility
   itself is task-dependent; claims 5-6 scope to property and the
   discussion gains a task-dependence subsection. subset_random non-null ->
   control failure; no interpretation until understood.
   - **Redundancy symmetry (discussion spine)**: the program has produced
     the same lesson twice at different granularities — directional
     necessity failed for the gauge (information redundant across
     directions, the INLP curve) and positional necessity failed for the
     lever (commitment redundant across pathways: token span + in-place
     writes, plausibly across layers). The variable is multiply realized;
     interchange sufficiency is the right probe of a multiply-realized
     variable, and exhaustive necessity requires ablating all realizations
     at once — exactly the combination arm. Connects to the Geiger
     causal-abstraction framing as organizing citation.
   - Geometry quantified (for prose): empirical null band for random
     vectors is 0.040 +/- 0.009; observed restricted-delta means 0.013-0.014
     (z ~ -2.8 per delta); 86% of all 360 per-position deltas individually
     sit below null-2SD. At most one flagged sentence of mechanism
     speculation (readable subspace dominated by global summary components;
     deltas are local content edits).
   - KV spec v2 additions (review, 2026-06-12): (a) **attention-mass
     telemetry to spliced positions during decode** — disambiguates a
     gold-KV null between splice failure (no attention reaches the splice:
     bug, fix first) and genuine insufficiency (attention flows, no repair:
     itself a finding); (b) **positional convention pinned explicitly**
     before implementation — spliced keys carry donor RoPE phases; decide
     donor-absolute vs re-rotated vs appended, since silent misconvention is
     how cache machinery fails without erroring; (c) **gold-KV positive bin
     pre-registered**: KV-alone repair means both pathways (context-encoding
     writes and decode-time span access) are individually sufficient — the
     strongest version of the multiply-realized frame, a bin not a surprise;
     (d) **rank-k reading rule**: k <= 4 recovering >= 70% of the
     subset-patch effect = "compactly structured"; anything else =
     distributed-with-structure. Read by rule, not vibes.
   - Geometry clustering note: the 86%-of-360 figure is descriptive; the
     360 per-position deltas are clustered (positions within rows, layers
     within rows). Any joint p-value clusters at row level (n=14). The
     per-delta z ~ -2.8 is the correct per-unit statement.
   - Necessity-null wording rule: "no detectable effect at ceiling" — 112
     generations at 1.000 cannot see sub-ceiling degradation (a reversion
     costing 10% of the repair margin is invisible); token logprobs were
     not logged, caveat accepted rather than rerun.
   - Qwen scoping: keep Qwen erasure on the critical path OR scope the
     cross-model readability sentence in the paper — readability claimed
     cross-model with Gemma-only epiphenomenality is the first asymmetry a
     reviewer circles.

   Spec refinements (review, 2026-06-12):
   - **Expressivity ladder, explicit**: subset replacement (done, +0.250) ->
     per-position additive delta (drops replacement) -> own restricted
     rank-1 (drops rank) -> spotlight = foreign restricted rank-1 (drops row
     identity). Each rung removes one property; where the effect falls off
     IS the answer.
   - Spotlight mapping rule (pre-registered): use the row-restricted MEAN
     vector, so donor/receiver mention-count mismatch is moot — the
     spotlight is exactly the own-rank-1 object with the row swapped. Also
     run a per-position spotlight with cycled position mapping, which stays
     interpretable even if the rank-1 rungs come back null.
   - **Necessity controls (the erasure lesson)**: an in-job HINTED-baseline
     arm (the 1.000 is a Modal prompt-level number; the hooked pairing
     reference must be measured in-job), and a reverse-patch-at-RANDOM-
     positions control in the hinted run — without it, a collapse is
     vulnerable to "injecting unhinted states anywhere destroys the repair",
     the demolition-vs-precision objection again.
   - **Budget reality**: 456999 realized ~270 generations/hour, so ~1,250 is
     ~4.6h — over the cap before the new controls. Order arms by priority
     (hinted baseline + reverse pair first, then spotlight, KV-wrong, then
     the delta ladder), write results incrementally per arm, and split the
     ladder into a second job if the realized rate after two arms says so.
   Job must SAVE donor/receiver concept-position states per row/layer
   (~10MB): enables offline recomputation of the readable-subspace
   projection on restricted deltas, rank-k PCA on the concept-position
   submatrix, and the Gemma Scope 2 SAE feature-diff (sparsity in the
   learned basis - closes the loop to the sparsely-lossy thesis) with no
   further cluster time.
   Unification note for the writeup: the recognition-patch null and the
   hint-patch success are one account - MCQ selection happens at option
   tokens, hints rewrite the context at concept mentions; each prompt
   format stores selection state where its candidates live.

## Cross-Cutting Protocol Upgrades

Apply to every experiment above:

- Metric: delta P(strong-correct) over k >= 8 samples per row, alongside
  discrete paired flips.
- Rowset: all 57 recognition-gap candidates from
  `docs/commitment_rowset_manifest.json`, not the 14-row subset.
- Strength: sweep magnitudes and include a clamping variant, not only
  additive +1sd.
- Controls unchanged: regenerated baseline, orthogonal, matched Gaussian,
  positive control, exact-patch validation where applicable.

## Decision Rules

- Experiment 1 (erasure): report as a positive claim either way —
  epiphenomenality or necessity. No null outcome exists for this design.
- Experiments 2-5: keep the existing rule — at least 3 paired
  false-to-true examples (or a delta P shift exceeding matched noise by
  2 sigma) with passing controls pivots the next paper toward a positive
  causal story; otherwise the thesis stays `causally inaccessible under
  tested methods`, now with necessity tested.
- If experiment 1 shows epiphenomenality *and* experiment 2 or 5 repairs,
  the paper's thesis becomes the selection-vs-evaluation dissociation:
  the probe reads evaluation, selection is the causal handle.

## Suggested Order

1. Erasure (cheapest, decisive either way).
2. Recognition-state cross-prompt patching (highest upside for a positive
   causal result).
3. ITI-style heads and dictionary clamping (cheap, reuse existing infra).
4. Minimal-pair DAS and margin-circuit localization (heavier; run if 1-3
   leave the question open or a positive result needs mechanism).

## Pre-Registered Manuscript-Hardening Jobs (2026-07-02)

Two targeted runs before the report draft freezes claims 8 and 12. Both are
power/robustness upgrades of already-landed designs; no new intervention
family.

### A. Property rank-k guard v2 (fresh-row expansion of claim 8) — DONE (jobs 458374/458375, 2026-07-03)

**Outcome: claim 8 survives on fresh rows.** Pooled 26 rows (3 per shard
skipped for missing concept positions): `rank8_loo` +0.231 [+0.111, +0.365],
91% of the pooled `L30_concept_replace` effect (+0.255 [+0.111, +0.413]) —
passes both prongs. `rank4_loo` +0.144 [+0.038, +0.264] excludes zero but
reaches only 57% — rank-4 under-transfers; rank-8 is the fresh-row-portable
core. `L30_random_replace` null (+0.038, CI spans zero). Hint-validated
slice (23 rows) concordant: rank-8 at 89%. Full verdict in
`docs/rank_k_guard_v2_27b_property_pooled_summary.md`.

`scripts/stage2_rank_k_guard_v2.py`. The 457012 guard passed on the same 13
rows that defined the compact core. This run re-tests sufficiency on rows
that contributed to neither the PCA bases nor the original row selection.

- Rows: 32 fresh property rows (16 h3 + 16 h4), drawn seeded from
  `results/full/with_errortype/gemma3_27b_infer_property.jsonl` with
  `parse_failed == False`, `is_correct_strong == False`, excluding the 13
  composite-manifest rows. Run as 2 row-shards of 16 (interleaved for height
  balance).
- Arms (k=8, in-job references): `unhinted_baseline`; `hinted_baseline`
  (hint-first prompt — per-row hint validation and ceiling);
  `L30_concept_replace` (hinted states at concept positions — the in-job
  subset-effect denominator); `L30_random_replace` (matched random-position
  control); `rank4_loo_add_L30`; `rank8_loo_add_L30` (LOO PCA bases fit on
  the shard's other 15 rows).
- Primary metric: paired dP(strong) vs `unhinted_baseline`, row-level
  bootstrap, shards pooled (32 rows).
- Decision rule: claim 8 survives if pooled `rank4_loo` or `rank8_loo` CI
  excludes zero AND reaches >=70% of the pooled in-job `L30_concept_replace`
  effect. If `L30_concept_replace` itself fails on fresh rows (CI includes
  zero), the compact-core claim gets scoped to recognition-gap-style rows —
  report the scoping, not a guard failure. Secondary (reported either way):
  the same contrasts restricted to hint-validated rows (per-row
  `hinted_baseline` P(strong) >= 0.5), since sufficiency through hint deltas
  presupposes the hint works on that row.

### B. Subtype L35 targeted replication (resolves the claim-12 hedge) — DONE (jobs 458387/458388, 2026-07-03)

**Outcome: positive branch fires.** Pooled new seeds (16 rows, k=16):
`L35_concept_replace` +0.117 [+0.008, +0.254], CI excludes zero;
`L35_random_replace` null (+0.023 [−0.004, +0.062]); old-trio full replace
null in-job (+0.023). Meta-pool with 457170 (k=24): +0.112 [+0.003, +0.260].
The subtype carrier is reachable at L35 — the old-trio nulls were layer
mismatch. Caveats for wording: modest (~quarter of property's repair),
row-sparse (3/13 addressable rows carry it), `L35_rank4_loo_add` only
marginal (+0.047) so no compact core landed on subtype. Full verdict in
`docs/subtype_l35_replication_pooled_summary.md`.

Delivery note: the first submission (jobs 458376/458377) lost the ladder to
sbatch `--export` comma-splitting and ran old-trio-only; rerun with env vars
set in the submitting shell.

Jobs 458376/458377 never ran the L35 arm: the submit line passed
`SUBTYPE_DISC_LADDER=30,35,40,45` inside `sbatch --export=ALL,...`, and
sbatch splits `--export` on commas, so the ladder collapsed to `[30]` (plus
the auto-appended old trio) and no off-trio layer was selected. The two runs
are accidental old-trio replicates on fresh seeds — old-trio full replace
+0.047 [−0.164, +0.258] (seedA) and +0.000 [−0.227, +0.227] (seedB), both
null, consistent with the original old-trio null. The L35 question is
untouched; design, seeds, and decision rule below are unchanged. Resubmitted
with env vars set in the submitting shell (no `--export` commas), tags
`_l35rep2_seedA` / `_l35rep2_seedB`.

Existing `scripts/stage2_subtype_discriminator.py`, no code change: ladder
`30,35,40,45`, `top-offtrio-layers=1` (selects L35), `rank-k=4`, k=8, same 16
manifest rows, two seed shards (20260702 / 20260703; fresh random-position
controls and generations), tagged outputs so job 457170 artifacts are
untouched.

- Primary metric: `L35_concept_replace` minus baseline, pooled across the two
  new shards (effective k=16), row-level bootstrap; meta-check pooling with
  the 457170 shard (effective k=24).
- Decision rule: pooled-new CI excluding zero lands the off-trio
  layer-mismatch repair and upgrades claim 12's wording (subtype carrier
  reachable at L35). CI including zero with half-width <= ~0.10 closes the
  hedge as a bounded null: residual-state concept replacement is insufficient
  on subtype at all tested layers, and L35 stops being a replication target.
  The matched `L35_random_replace` must stay null for either reading.

## Pre-Registered Submission-Hardening Jobs (2026-07-04)

Two experiments motivated by the literature sweep in
`docs/next_experiments_litreview_2026-07.md`, recorded before any data.
Both reuse landed designs; the first defends the flagship positive claim
against the random-basis critique (arXiv 2507.08802, 2511.04638), the second
discriminates against the entanglement account of steering nulls
(arXiv 2605.05715) and closes claim 3's full-subspace future-work item.

### C. Rank-8 specificity controls (random-basis guard for claim 8)

`scripts/stage2_rank_k_guard_v2.py --specificity-controls`. Claim 8's guard
arms include a random-POSITION control but no random-BASIS control; if
matched-norm random bases repair as well as the LOO PCA basis, the
compact-core specificity claim collapses. Existential and cheap, so it runs
before anything else.

- Rows: identical selection to guard v2 (seed 20260702, per-height 16,
  composite-manifest rows excluded), same 2 shards — enables cross-job
  row-paired comparison with 458374/458375.
- Arms (12 per shard, k=8, generation config identical to guard v2):
  `unhinted_baseline`; `hinted_baseline`; `rank8_loo_add_L30` (positive
  reference, in-job); `mean_only_add_L30` (the LOO pooled mean delta tiled
  at concept positions — decomposition rung); `rand_subspace_add_L30_d{1-4}`
  (LOO mean + the row's centered delta projected onto a random orthonormal
  rank-8 basis, per-position norm-matched to the PCA non-mean component;
  one seeded basis per draw, shared across rows; distinct draws per shard →
  8 bases pooled); `rand_norm_add_L30_d{1-4}` (pure Gaussian per-position
  vectors, norm-matched to the full rank-8 reconstruction per position;
  seeded per row × draw → 8 draws pooled).
- The ladder decomposes the rank-8 add: full structure → mean + matched
  content in a random subspace → mean alone → matched-norm noise. Each rung
  removes one property; where the effect falls off is the answer.
- Primary metric: pooled 2-shard paired dP(strong) vs in-job
  `unhinted_baseline`, row-cluster bootstrap (10k), random families pooled
  across draws and shards; per-draw reported descriptively.
- Decision rules (recorded before unblinding):
  - Gate: pooled `rank8_loo_add_L30` CI must exclude zero (replicating
    +0.231). If it fails, machinery anomaly — no interpretation of the
    controls, investigate before any claim edits.
  - Specificity PASSES: pooled `rand_norm` CI includes zero AND the paired
    (rank8 − rand_norm) difference CI excludes zero. Claim 8 gains a
    specificity-guarded sentence; on-distribution riders move to a methods
    sidebar.
  - Specificity FAILS: pooled `rand_norm` CI excludes zero (positive) at
    >= 50% of the rank8 effect → generic-perturbation confound; claim 8
    rescopes and §5.4 drafting pauses pending redesign.
  - Structure decomposition (wording pre-committed, reported either way):
    `mean_only` >= 70% of rank8 → "the fresh-row-portable core is dominated
    by the shared LOO mean delta; rank-8 structure adds the remainder";
    `rand_subspace` ≈ rank8 (paired difference CI includes zero) while
    `rand_norm` is null → "the content is the mean plus a per-position
    magnitude profile; the specific PCA directions beyond the mean are not
    privileged"; both clearly below rank8 → "the PCA subspace itself is
    load-bearing".
- Outputs: stem `rank8_specificity_27b_property_shard{i}of2` (guard-v2
  artifacts untouched).
- Offline riders on saved artifacts (CPU, after unblinding, non-gating):
  logit-lens of the 8 components; principal angles vs the INLP stack; rank
  sweep 1–16; nearest-neighbor distance of steered L30 states to natural
  hinted states + dormant-unit check (the 2511.04638 pernicious-divergence
  diagnostics), using `focus_state_composite_27b_property_states.npz`
  hinted/unhinted blocks as the on-distribution reference.

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

### F(ii)-b. Shuffled-label projected control (label-specificity of +0.341)

Pre-registered 2026-07-05 morning, after two round-2 adversarial reviews
independently made this arm their condition for publication, and BEFORE
any F(ii)-b data. Adjudicates whether F(ii)'s class_mean_proj (+0.341) is
label-specific or generic to difference-shaped vectors in the lever
subspace (the reviews' "top-variance cone" alternative — the program's
only on-distribution specificity control to date).

- Rows/harness/norms: identical to F(ii) (guard-v2 fresh selection, 2
  shards, k=8, per-position norm target = the row's LOO rank-8 recon).
- The F(ii) real-proj and baseline data are reused via verified
  determinism (208/208 identical anchor outcomes); an in-job
  unhinted_baseline is regenerated as an integrity check and must match
  F(ii)'s row-for-row.
- New arms (7 per shard):
  `unhinted_baseline` (integrity);
  `shuflabel_proj_add_L30_d{1-4}` — class-mean vectors from seeded
  label permutations of the SAME 96-row capture (derangement-free simple
  permutations, seeds 20260705+d), projected per-row onto the identical
  LOO bases, norm-matched identically; pooled as a family across draws
  and shards, per-draw descriptive (item-C convention);
  `signflip_proj_add_L30` — the REAL class vector negated, projected,
  norm-matched (label-content predicts null-or-harm; pure-geometry
  predicts repair);
  `fixednorm_proj_add_L30` — the real projected vector at a FIXED
  per-position norm equal to the pooled mean recon norm of the OTHER
  rows (leave-one-out pooled scale): the fully donor-free variant.
- Decision rules (before unblinding):
  - Integrity gate: in-job unhinted_baseline reproduces F(ii)'s per-row
    outcomes (any mismatch → determinism broken → stop, no reuse).
  - LABEL-SPECIFIC: paired (real_proj − shuflabel family) CI excludes
    zero AND the shuflabel family sits below 50% of real_proj's dP.
    Consequence: F(ii)'s endogeneity-adjacent wording ("causal alignment
    without decodable alignment") is earned; claim-8 wording keeps "the
    specific directions".
  - GENERIC (the reviews' predicted outcome): paired (real_proj −
    shuflabel family) CI includes zero → +0.341 reads as "high-variance
    in-subspace amplification"; F(ii) §4 reverts fully to exogenous
    mediation; claim 8's "specific 8 PCA directions" wording is replaced
    by "an identified high-variance subspace" in the W1 sweep.
  - Sign-flip rider (descriptive unless decisive): signflip repairing at
    ≥50% of real_proj strongly supports GENERIC regardless of the
    primary contrast; signflip null-or-negative with LABEL-SPECIFIC
    primary supports content.
  - Donor-free rider: fixednorm_proj CI excluding zero = the first fully
    donor-free repair (direction and scale both hint-free); reported
    regardless of primary outcome.
- Exploratory: no current-paper claim moves except via the W1 wording
  consequences enumerated above. Budget: 7 arms × ~13 rows × 8 ≈ 728
  gens/shard ≈ 1.8 h/shard, 2 shards, queued behind item D's remainder.

## Pre-Registered Cross-Model Port (2026-07-05)

### G. Qwen3.5-27B positive-carrier port (gorman lane; next-paper opener)

Pre-registered before any Qwen causal-repair data. Claim 11 established the
Qwen gauge (L53 AUC 0.940) and raw-axis non-necessity; item G asks whether
the LEVER — the focus-state repair motif — exists in Qwen: is "commitment
written at concept-mention tokens, low-rank, causally repairable" a
cross-model motif or a Gemma idiosyncrasy? Either answer is a next-paper
result. Primary model: Qwen/Qwen3.5-27B (evidence continuity). Qwen3.6-27B
(same architecture class, released 2026-04) is a stretch replication (G5),
run only if G2–G3 land.

Runs as self-staging fp32 jobs on gorman-gpu
(`scripts/gorman_stage_and_run.sbatch`); Scholar bf16 numbers are NOT
stitchable — all comparisons are within-job or within-lane.

- **G0 — calibration gate (in-job, first)**: unhinted + hinted baselines,
  k=8, on the 16 balanced rows of jobs 457191–457194. Gate: unhinted
  P(strong) within the row-cluster CI of the Scholar-era baseline (0.352);
  hinted lift ≥ +0.30. Hinted-lift failure = no addressable failure mode
  at this prompt format → STOP, port reported as behaviorally blocked
  (itself informative: the recognition-gap motif would be Gemma-scoped).
- **G1 — row selection**: in-job screen (reusing the sampled-baseline
  machinery from the Qwen erasure family) for parse-ok strong-incorrect
  h3/h4 rows, seeded, balanced, 16 selected / expect ~13 prepared.
- **G2 — layer sweep (the localization step Gemma taught us to expect)**:
  concept_replace vs random_replace at relative depths ≈ {0.40, 0.50,
  0.60, 0.67, 0.75} of Qwen's stack, k=8, same rows. Winner = largest
  pooled concept_replace dP whose CI excludes zero AND whose paired
  (concept − random) CI excludes zero; ties → deeper layer. No winner →
  STOP: "carrier not reachable at swept depths" (mirrors the subtype
  old-trio lesson — report as layer-mismatch candidate, sweep finer only
  in a follow-up registration).
- **G3 — compact core + specificity at the winning layer**: the item C
  ladder compressed — rank8_loo_add, mean_only_add, rand_subspace_add
  (2 draws), rand_norm_add (2 draws), k=8. PASS = rank8_loo CI excludes
  zero AND paired (rank8 − rand_norm family) CI excludes zero. Report
  rank8 as % of G2's concept_replace.
- **G4 — F-series rider (budget permitting)**: natural-state capture
  (balanced correct/incorrect, no hints) + class_mean_proj at the winning
  layer with one shuffled-label draw and the sign-flip. Descriptive
  unless G3 passed.
- **G5 — Qwen3.6-27B stretch**: rerun G0–G3 with the model name swapped
  (in-job screening supplies rows; no stage-1 artifacts exist for 3.6).
  Registered now to avoid garden-of-forking-paths later; runs only after
  a G3 PASS on 3.5.
- Stats: row-cluster bootstrap (10k, percentile), paired vs in-job
  unhinted baseline, pooling per phase; random families pooled across
  draws. Wording consequences: G3 PASS → "the focus-state control channel
  is a cross-model motif (2/2 models tested), with model-specific layer
  and basis"; G2/G3 nulls after a passing G0 → "the lever is not
  reachable by the Gemma recipe in Qwen — motif not established beyond
  Gemma," reported with equal prominence.
- Budget: G0–G3 ≈ 2,000–2,500 generations ≈ 12–18 h fp32 on 5× V100 —
  one gorman job with staging overhead; G4/G5 a second job. Exploratory:
  no current-paper claim moves on any outcome.
