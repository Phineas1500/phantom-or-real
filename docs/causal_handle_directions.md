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

### G3′. Qwen rank-and-scale ladder at L43 (Scholar)

Pre-registered 2026-07-06 evening, after G3's verdict (rank8 +0.083
[−0.025,+0.217] — under-powered directional miss at 48% of the G2
carrier anchor) and before any G3′ data. Question: is Qwen's channel
the same mechanism at a different geometry — higher rank, or starved
amplitude — or does the carrier resist low-rank compression entirely?

Design: same 15 prepared rows, seeds, and machinery as G3 (job 458465);
one Scholar bf16 job, L43, `scripts/stage2_qwen_g3prime_hf.py`. Arms
(seed arm-indices fixed so the two replication arms regenerate G3
verbatim):

- `unhinted_baseline` (seed-ai 0) and `rank8_loo_add_L43` (seed-ai 2) —
  determinism gates; must reproduce G3's 0.192 / 0.275 exactly.
- `rank16_loo_add_L43`, `rank32_loo_add_L43`, `rank64_loo_add_L43`
  (seed-ai 10/11/12) — LOO PCA reconstructions, the saturation curve.
  151 pooled positions ⇒ rank 64 is well-posed under LOO; effective
  ranks recorded.
- `rank8_fixednorm_add_L43` (seed-ai 13) — rank-8 reconstruction
  rescaled per position to the FULL delta's norm (the dilution-tax
  test).
- `rand_subspace64_add_L43_d1/d2` (seed-ai 14/15) — random orthonormal
  rank-64 projections, norm-matched to the rank-64 reconstruction; by
  energy monotonicity this dominates the k<64 random families, so it is
  THE noise family for all rank arms.
- `rand_normfull_add_L43_d1/d2` (seed-ai 16/17) — random directions at
  full-delta per-position norm; the matched control for fixednorm.

Decision rules (before unblinding):
- Gate: both replication arms match G3 exactly; mismatch → debug, no
  unblinding of new arms.
- RANK BRANCH: winner = smallest k ∈ {8,16,32,64} whose dP CI (vs
  in-job unhinted, row-cluster bootstrap 10k, seed 20260704) excludes
  zero AND whose paired (rank_k − rand_subspace64 family) CI excludes
  zero. Wording: "Qwen's channel rank is k* — the low-rank motif is
  cross-model, its dimensionality model-specific"; report % of G2's
  +0.175.
- SCALE BRANCH: if no k qualifies but fixednorm-8 beats baseline AND
  paired (fixednorm − rand_normfull family) excludes zero: "rank 8
  suffices at full-delta amplitude — the miss was dilution, not rank."
- Both → report both; rank statement takes precedence, scale informs
  deployment.
- Neither → "the Qwen carrier resists low-rank compression at L43
  (k ≤ 64) and rescaled rank-8 — full-state only in this design"; the
  landed two-tier wording stands unchanged.
- Not licensed on any outcome: promotion of §1 lever claims beyond
  Gemma; only the ledger §3 cross-model bullet moves.
- Budget: 10 arms × 15 rows × 8 samples = 1,200 generations ≈ 65 min
  on 2× A40. Exploratory.

### G4′. Qwen answer-free class-mean at L43/rank-16 (claim-bearing, keyed to G3′)

Pre-registered 2026-07-06 late evening, after G3′'s k\*=16 verdict and
before any G4′ data. Supersedes the original G4 rider (whose
claim-bearing gate named a G3 PASS that did not occur; G3′ passed
instead, so a fresh registration is required for this to move a claim).
Question: does the ANSWER-FREE, label-specific content of natural
success transfer cross-model — the F(ii)/F(ii)-b finale in Qwen, using
Qwen's own coordinates (L43, rank 16)?

Design (one Scholar bf16 job, `scripts/stage2_qwen_g4_hf.py`):

- **Class-mean source rows**: screened pool of 32 naturally-correct +
  32 naturally-incorrect candidates (stage-1 strong labels as the
  screen), heights 3/4 balanced, seed 20260707, excluding G0_ROWS ∪ the
  15 G3-ladder rows. In-job class label = majority of 4 unhinted
  samples at temp 0.7 (ties dropped); target ≥ 20 confirmed per class,
  take the first 20 per class in seeded order. States captured at L43
  concept-mention positions on the unhinted prompt (one forward pass
  per row); per-row position-mean; class-mean vector = mean(correct) −
  mean(incorrect). Zero per-instance answer information reaches any
  test row: direction from OTHER rows' natural outcomes, amplitude
  from pooled scale.
- **Test rows**: the 15 G3-ladder rows (disjoint from source rows by
  construction). Basis: per-test-row LOO rank-16 PCA of the hint
  deltas (identical to G3′'s rank16 arm — fresh-row discipline).
- **Amplitude (donor-free pooled scale)**: the class-mean vector is
  rescaled so its per-position add norm equals the pooled mean
  per-position norm of the G3′ rank-16 reconstructions (one scalar,
  computed over all 15 rows — no row-specific information). The raw
  arm uses the same total norm.
- **Arms** (8, seed formula as G3′ with seed-ai in parentheses):
  `unhinted_baseline` (0) and `rank16_loo_add_L43` (10) — verbatim
  replication gates vs G3′; `class_mean_raw_add_L43` (20) — full-dim,
  matched norm; `class_mean_proj16_add_L43` (21) — projected into the
  LOO rank-16 basis, matched norm; `shuffled_label_proj16_L43_d1/d2`
  (22/23) — class labels shuffled within the 40 confirmed source rows
  (seeded draws), mean recomputed, identical projection + norm;
  `signflip_proj16_add_L43` (24) — negated proj vector;
  `rand_norm16_add_L43_d1` (25) — random direction at the same pooled
  norm.

Decision rules (before unblinding):
- Gates: both replication arms reproduce G3′ verbatim; mismatch →
  debug, no unblinding.
- **PRIMARY — cross-model answer-free repair**: `class_mean_proj16` dP
  CI (vs in-job unhinted, row-cluster bootstrap 10k, seed 20260704)
  excludes zero AND paired (proj16 − shuffled-label family) CI
  excludes zero. PASS wording: "the answer-free, label-specific repair
  transfers cross-model when expressed in each model's own coordinates
  (layer, rank)"; ledger §1 claims 1–2 each gain a one-line Qwen scope
  note (single job, 15 rows, no erasure battery); ledger §3
  cross-model bullet upgrades.
- **Dissociation rider** (reported either way, claim-bearing only if
  PRIMARY passes): raw-vs-proj paired contrast; registered expectation
  raw ≈ null (the F(ii) signature). If raw repairs as well as proj,
  the necessity-dissociation claim stays Gemma-scoped and we say so.
- **Sign-flip**: registered prediction ≤ 0 (null or harmful). A
  positive sign-flip CI would impeach label-specificity — reported
  with equal prominence.
- **Shuffled matches real** (both reviewers' original prediction,
  falsified in Gemma): if paired (real − shuffled) straddles zero
  while proj16 beats baseline, the repair is geometry-generic in Qwen
  — label-specificity does NOT transfer; reported with equal
  prominence, no claim moves.
- Null PRIMARY → MDE statement (expect ≈ 0.12 at n=15); wording:
  "channel established (G3′), content transfer not established at this
  power" — no claim moves.
- Stats: row-cluster bootstrap only; shuffled family pooled across
  draws; parse-integrity gate (parse-fail < 5% per arm) before any
  verdict.
- G5 (Qwen3.6) gate is NOT re-keyed: it remains off. Deliberate —
  drafting outranks a third model.
- Budget: screening ~256 gens + capture ~40 forwards + 8 arms × 15
  rows × 8 samples = 960 gens; ≈ 80–90 min on 2× A40, one job.

### G6. Qwen content-transfer ladder (protocol-matched class-mean; Scholar)

Pre-registered 2026-07-06 ~23:00, after the G4′ null and before any G6
data. Motivation and audit in `qwen_content_transfer_followups.md`:
three signals say G4′'s null is starvation, not absence (sign-flip
CI < 0; proj responders a strict subset of rank-16 responders, per-row
corr +0.73; only positive non-gate point estimate). Every protocol
difference vs Gemma F(ii) gets its own arm.

Design (one Scholar bf16 job, `scripts/stage2_qwen_g6_hf.py`):

- **Class-mean sources (protocol-matched to Gemma)**: 96 rows — 24 per
  (height ∈ {3,4} × class) cell, stage-1 strong labels, NO majority
  screen (Gemma's exact selection), seed 20260708, excluding G0 ∪ all
  test rows. Capture forwards-only at L43 gold-concept positions;
  per-row position-mean; class-mean = mean(correct) − mean(incorrect).
- **Test rows**: the 15 G3-ladder rows (gates verbatim) + ~9 fresh
  failing rows (select_fresh_rows, seed 20260708, excluding G0 ∪
  original 16) → ~24 rows, MDE ≈ 0.09. Per-row generation seeds are
  row-keyed, so gate arms stay verbatim on the shared 15.
- **Basis provenance**: ALL bases fit on the ORIGINAL 15 rows' hint
  deltas only (LOO for those rows; the 9 new rows never contribute to
  any basis — fresh-row discipline preserved, and the rank16 gate stays
  verbatim).
- **Norm convention (protocol-matched)**: per-position, per-row targets
  = that row's rank-k reconstruction norms (Gemma F(ii) convention),
  NOT a pooled scalar.
- **Arms** (9, seed-ai in parens): `unhinted_baseline` (0) and
  `rank16_loo_add_L43` (10) — verbatim gates on the shared 15;
  `protomatched_proj16_add_L43` (30) — 96-row class-mean → LOO rank-16
  projection → per-position norm match [PRIMARY];
  `protomatched_proj64_add_L43` (31) — rank-64 projection, rank-64 norm
  targets [named secondary: channel-coverage];
  `protomatched_proj16_x2_L43` (32) — PRIMARY vector at ×2 [named
  secondary: amplitude/dilution];
  `class_mean_raw96_add_L43` (33) — unprojected, same norm targets
  [dissociation rider];
  `shuffled96_proj16_L43_d1` (34) — 96-label permutation, identical
  projection/norms; `signflip96_proj16_L43` (35) — negated [registered
  prediction: ≤ 0, again]; `rand_norm_perpos_add_L43_d1` (36) — random
  direction at the same per-position norms.
- **F(i)-analog rider (CPU, in-job, descriptive)**: from the 96
  captured row-means — CV AUC of natural correct vs incorrect in the
  rank-16 and rank-64 slices vs 200 random matched-rank slices vs
  full-dim ceiling. Diagnoses whether natural-outcome information is
  decodable at L43 concept positions at all.

Decision rules (before unblinding):
- Gates: unhinted + rank16 verbatim vs G3′ on the shared 15 rows;
  parse-fail < 5% per arm.
- **PRIMARY**: protomatched_proj16 dP CI (24 rows, row-cluster
  bootstrap 10k, seed 20260704) excludes zero AND paired (proj16 −
  shuffled96) CI excludes zero. PASS wording: "answer-free content
  transfers cross-model under protocol-matched conditions"; claim 13
  caveat upgrades; §1 claims still remain Gemma-scoped (no erasure/full
  battery in Qwen).
- **Named secondaries** (reported with CIs either way; localization
  reading): proj64 and proj16_x2 each vs baseline and paired vs the
  shuffled/rand family at ×1 (limitation acknowledged: no ×2 control —
  a ×2-only pass is flagged amplitude-confounded and would need its own
  control before any claim).
- Sign-flip: registered prediction ≤ 0.
- Dissociation rider: raw96 vs proj16 paired (descriptive).
- All-null at MDE ≈ 0.09 → licensed sentence: "content transfer
  genuinely differs across models at matched protocol" — a real
  cross-model contrast, reported as such.
- G4′'s verdict stands regardless; this is next-paper material unless
  PRIMARY passes, in which case the current paper's cross-model section
  MAY cite it as a one-line addendum (decision at drafting).
- Budget: 96 capture forwards + 9 arms × ~24 rows × 8 ≈ 1,730 gens
  ≈ 110 min on 2× A40, one job.

### G6′. Raw-transfer control battery (the G6 nail-down; Scholar)

Pre-registered 2026-07-07 ~01:15, after G6's verdict and before any G6′
data. G6 landed the inversion descriptively (raw96 +0.120 CI>0,
raw−proj16 +0.099 CI>0) but raw96 was a descriptive rider: no
shuffled-RAW control exists, raw−rand missed its CI (+0.089
[−0.010,+0.193], one draw), and proj64's pass carries the
control-unmatched flag. G6′ supplies exactly the missing controls.

- Rows/gates/norms/seeds: identical to G6 (24 test rows, 96-source
  class-mean recomputed identically in-job, per-position rank-16 recon
  norm targets; unhinted + rank16 verbatim gates; class_mean_raw96 arm
  regenerated with G6's seed-ai 33 — must reproduce G6 verbatim as a
  third gate).
- New arms (seed-ai): `shuffled_raw96_d1/d2` (40/41) — label-permuted
  class-means (draws seeded control_seed+10+d), UNPROJECTED, identical
  norm targets [the label control for raw];
  `signflip_raw96` (42) — negated raw [registered prediction: ≤ 0];
  `rand_norm_perpos_d2/d3` (43/44) — two more amplitude-matched random
  draws [pooled with G6's d1 → 3-draw family];
  `shuffled96_proj64_at64norms_d1` (45) — shuffled-label vector
  projected into rank-64 at rank-64 norms [closes proj64's flag].
- Decision rules (before unblinding):
  - Gates: all three replication arms verbatim; parse-fail < 5%.
  - **PRIMARY**: raw96 (from gate arm, pooled 24 rows) CI excludes
    zero AND paired (raw96 − shuffled-raw family) CI excludes zero.
    PASS wording: "answer-free, label-specific content transfer works
    in Qwen through the FULL state — Gemma concentrates, Qwen
    distributes; the recipe is cross-model, the compression is not."
    Claim 13 upgrades; §1 claims remain Gemma-scoped.
  - Secondary: raw96 − rand family (3 draws pooled) CI; signflip-raw
    ≤ 0; proj64 − shuffled-proj64 (both at 64-norms) CI [de-flags or
    kills the proj64 pass].
  - Shuffled-raw matches real → the raw effect is geometry/amplitude-
    generic; reported with equal prominence, inversion wording demoted
    to "raw acts, mechanism unresolved."
  - Budget: 6 new arms + 1 regenerated gate arm... 7 arms × 24 rows
    × 8 + capture ≈ 1,450 gens ≈ 95 min, one job.

### J1. J-aware F(i)-analog on Qwen 3.5 (consumption-readable staging test; Scholar)

Pre-registered 2026-07-07 afternoon, before any J1 data. Design doc:
`qwen_jspace_connection.md`. Question: is Qwen's natural-outcome
information at L43 concept positions staged in CONSUMPTION-readable
form — visible through the model's own downstream transformation —
even though it is not linearly probe-readable there (G6 F(i)-analog:
full-dim AUC 0.504)? Motivated by the workspace paper's Tuned-Lens ≪
J-Lens gap and Nanda's Qwen-3.6 replication of the J-machinery.

Method (no generation; fully deterministic; one short Scholar job,
`scripts/stage2_qwen_j1_hf.py`):
- Rows: the same 96 balanced sources as G6 (seed 20260708, identical
  selection code path; 48/48 by stage-1 strong labels).
- Base forward per row captures (fp32): h_L43 at concept positions
  (row-mean — the G6 feature), h_L43 at the final prompt token,
  h_L53 at the final prompt token (positive control: stage-1 probes
  read ~0.94 in this region), h_pen (layer index 62 of 64) at the
  final prompt token.
- **Directional-response features (finite-ε J-analog)**: perturbed
  forwards injecting ε·T at L43 concept positions (existing add-hook
  machinery), ε = 0.1 relative to each position's state norm;
  y = (h_pert − h_base)/ε read at the penultimate layer, final token
  (primary) and at L53 final token (secondary). Tangents: T_real =
  the row's own concept-position states ("10% more of what is already
  staged"); T_null = 10 seeded random-Gaussian draws, norm-matched per
  position (seeds 20260709+d). 12 forwards/row ≈ 1,150 total.
- CPU battery (in-job, same 5-fold CV-AUC as the F(i)-analog):
  raw_concept (gate), raw_L43_final, raw_L53_final (control),
  response_real@pen, response_real@L53, response_null family (10),
  plus 200 label-shuffle nulls for response_real.

Decision rules (before unblinding):
- Gate: raw_concept reproduces G6's 0.504 within ±0.02 (identical
  states, deterministic battery).
- Positive control: raw_L53_final ≥ 0.75 (expect ~0.9). Failure →
  pipeline suspect, STOP, no unblinding of response features.
- **PRIMARY**: response_real@pen AUC ≥ 0.70 AND above the
  random-tangent family p95 AND above the label-shuffle p95 →
  **CONSUMPTION-READABLE CONFIRMED**: the address was never empty;
  wording "Qwen stages outcome information in a form its own
  computation reads but a linear probe does not; the raw-write repair
  (G6′) is in-language." Ledger §3 and the explainer's mystery
  paragraph update; no §1 claim moves.
- Partial branch: above both nulls but < 0.70 → "consumption-readable
  signal present but weak" — descriptive only.
- Null branch: at/below the random-tangent p95 → hypothesis rejected
  AT THIS ADDRESS/READ; the mystery stands; next step is the Q1
  layer × position map with a J-projected column (separate
  registration). Random-tangent family ABOVE raw 0.504 by itself
  (any-tangent separation) → "the transformation, not the content,
  differs by class" — reported as its own finding, wording reserved.
- Next-paper scope throughout; current-paper §6 may cite outcome in
  one hypothesis-flagged sentence only. 3.6 bridge (Nanda's exact
  checkpoint) registered as conditional follow-up on PRIMARY confirm.
- Budget: ~1,150 forwards, no sampling ≈ 30–45 min on 2× A40.

### J2. Position-separated staging: fresh-row confirmation + staging-site steering (Scholar)

Pre-registered 2026-07-07 evening, after J1's verdict and before any J2
data. J1's descriptive cell found raw_L43_final = 0.806 (outcome info
at the final prompt token, same layer as the concept-position write
port). That cell was unblinded once, descriptively; J2 makes it
claim-bearing and runs the first causal test of the staging site.

Design (one Scholar bf16 job, `scripts/stage2_qwen_j2_hf.py`):

**Part A — fresh-row confirmation + position sweep (capture + CPU).**
- 96 FRESH balanced sources (24/cell, stage-1 labels, seed 20260710,
  excluding G0 ∪ all test rows ∪ the J1/G6 96 sources).
- Capture per row (fp32): L43 & L53 at {concept-position mean, final
  prompt token, random-position mean (8 seeded non-concept, non-final
  prompt positions; 2 draws)}.
- Battery: CV AUC per cell + 200 label-shuffles for the headline cell.
- **A-PRIMARY (confirmation)**: fresh raw_L43_final AUC ≥ 0.70 AND
  above its label-shuffle p95 → POSITION-SEPARATED STAGING CONFIRMED
  (info at final token, chance at concept positions, same rows).
  Expected companions (descriptive): concept-mean ≈ 0.5 on fresh rows;
  random-position cells fill the map (is the info final-token-specific
  or late-position-generic?).
- Fail branch: fresh cell < 0.70 → J1's 0.806 was a one-shot
  fluctuation; report both numbers, claim nothing, position sweep
  still fills the Q1 map.

**Part B — staging-site steering (generation; the gauge-not-lever
question asked of Qwen's staging site).**
- Class-mean from Part A's FINAL-TOKEN states (correct − incorrect;
  sources disjoint from test rows, so donor-free by construction).
- Test rows: the 24 G6 rows; `unhinted_baseline` gate (seed-ai 0, must
  reproduce 0.177 verbatim).
- Arms (seed-ai): `finaltok_classmean_050` (50) — vector rescaled to
  0.5 × that row's final-token state norm, added at the final prompt
  token during prefill; `finaltok_classmean_100` (51) — 1.0 × norm;
  `finaltok_shuffled_100` (52) — label-shuffled vector (seed
  20260710+1), 1.0 ×, the matched control. 8 samples/row.
- **B-REGISTERED PREDICTION** (from Gemma's gauge-not-lever thesis):
  writing the outcome class-mean AT THE STAGING SITE does NOT repair —
  both dose arms' dP CIs include zero. Branches: (i) prediction holds
  → "in Qwen too, the readable staging site is a gauge, not a lever —
  the causal write-port is elsewhere (concept positions), completing
  the cross-model symmetry"; (ii) either dose repairs (CI > 0) AND
  beats shuffled paired (CI > 0) → MAJOR DIVERGENCE: Qwen's staging
  site is causally potent — its gauge IS a lever — reported with equal
  prominence, immediate follow-up battery required before any claim;
  (iii) repairs but shuffled matches → geometry/norm artifact at this
  site, no claim. Harmful outcomes reported descriptively.
- Stats: row-cluster bootstrap (10k, seed 20260704) on the 24 rows;
  parse-fail < 5% gate.
- Scope: next-paper item; current paper's §6 line may add "confirmed
  on fresh rows" if A-PRIMARY passes. No §1 movement on any branch.
- Budget: 96 capture forwards + 4 arms × 24 rows × 8 = 768 gens
  ≈ 55–70 min total.

### F(ii)-c. Deployment riders: position leak + collateral slice (Scholar)

Pre-registered 2026-07-05 ~13:30, after F(ii)-b's LABEL-SPECIFIC verdict
and before any F(ii)-c data. Two questions block the deployment reading of
the donor-free result (fixednorm +0.399): (1) position selection currently
uses GOLD-concept mentions — a sliver of answer information at inference
time; (2) collateral cost on rows the model was getting right is unmeasured
(the reviews' collateral-slice demand, item W4.2-adjacent).

- Failing side (guard-v2 selection, 26 rows, single job, shard-count 1):
  `unhinted_baseline`; `fixednorm_proj_add_L30` (LOO basis, gold
  positions — in-family reference; expected ≈ +0.399 via determinism-
  adjacent design but freshly generated here since arm indices differ);
  `fixednorm_allpos_add_L30` (same frozen vector and fixed scale, written
  at ALL taxonomy-concept mention positions in the prompt — no gold
  knowledge at inference).
- Correct side (16 FRESH naturally-correct rows, 8 per height, seeded
  20260706, excluding composite + guard + capture rows):
  `correct_unhinted_baseline`; `correct_fixednorm_add_L30` (frozen vector
  projected onto the FULL failing-rows basis, gold positions, same fixed
  scale) — the false-positive-firing cost.
- Decision rules (before unblinding):
  - POSITION-FREE if fixednorm_allpos CI excludes zero AND reaches ≥50%
    of fixednorm_proj (paired). Then the intervention needs no answer
    information at inference at all. If allpos is null while gold-pos
    repairs, position selection is load-bearing and the deployment
    claim keeps the "which concept" caveat.
  - COLLATERAL-SAFE if correct-side dP point ≥ −0.10 AND CI lower bound
    ≥ −0.20; COLLATERAL-HARMFUL if the CI sits entirely below −0.10;
    in-between = underpowered, reported as such.
- Stats: row-cluster bootstrap (10k, percentile) per side; paired
  contrasts within the failing side. Exploratory: no current-paper claim
  moves. Budget: (3 arms × 26 + 2 arms × 16) × 8 ≈ 880 generations ≈
  2.6 h — one Scholar slot behind 458414.

### H. Position-selection policy (deployment gap closure, Scholar)

Pre-registered 2026-07-05 ~20:15, before any data. F(ii)-c established
that the donor-free vector's ADDRESSING is load-bearing: gold-position
writes repair (+0.447), all-position writes cancel (+0.029 n.s.). Item H
asks whether the address can be chosen without answer information, by
firing the vector at each candidate concept separately.

- Rows: the 26 failing guard rows (single job, shard-count 1). Vector,
  basis, and scale identical to F(ii)-c's fixednorm arm (LOO rank-8
  basis; pooled-others norm; class vector from the 458416 capture).
- Arms:
  `unhinted_baseline` (k=8);
  `fixednorm_gold` (k=8, gold positions — in-family reference);
  `percand_fire` — for EACH taxonomy concept c in the row (~5–10), one
  generation batch (k=4) with the vector written ONLY at c's mention
  positions; rows record `fired_concept`, `fired_is_gold`, and
  `targets_fired_concept` (does the output's hypothesis subject equal the
  fired concept?).
- Pre-named offline selection policies (evaluated from percand outputs,
  no further GPU):
  P1 self-ratification (PRIMARY): choose the candidate whose fires most
  often propose the fired concept itself; tie → most parseable, then
  lowest candidate index. Rationale: if the vector amplifies commitment
  to the marked concept only where commitment is licensed, the right
  address ratifies itself.
  P2 global majority: pool all fires' parsed hypotheses, majority vote.
  P3 oracle (ceiling, not a policy): the gold-candidate fire.
- Decision rules (before unblinding):
  - Mechanism readout (reported regardless): wrong-concept fires' pooled
    dP vs baseline, and targets_fired_concept rate at gold vs non-gold
    candidates. Misdirection-asymmetry prediction: wrong-concept fires
    do NOT pull hypotheses to the fired concept.
  - Gate: gold-candidate fires (k=4 slice of percand) repair with CI
    excluding zero (the k=4 analogue of +0.447; if k=4 is too noisy the
    gate falls back to the k=8 fixednorm_gold arm).
  - POLICY-VIABLE: P1's selected-fire P(strong) beats baseline (paired
    CI excludes zero) AND reaches >=50% of the oracle fire (P3).
  - POLICY-FAILS: P1 and P2 both below 50% of oracle → position
    selection needs a trained policy or gauge-scored reruns; the
    "answer-free in content, not addressing" caveat stands as final for
    this paper.
- Exploratory: no current-paper claim moves; feeds the discussion's
  deployment paragraph and the next paper. Budget: 208 + 208 + ~26×7×4
  ≈ 1,150 generations ≈ 2.9 h — one Scholar slot, queue currently empty.

### G (amendment, 2026-07-06, before any G data): venue moved gorman → Scholar

Recorded before G0 runs. gorman's queue is saturated (5-day neighbors);
Scholar's is empty. Moving to Scholar bf16 REMOVES the original
registration's fp32/no-stitching caveat: G numbers become directly
comparable with the claim-11 Qwen artifacts (same hardware/dtype/harness
family). Operational resharding for the 4h wall (G2 as 2 row-shards) is
venue bookkeeping, not a design change. Phase gating unchanged: G0 must
pass before G2 is submitted; G2's winner selection before G3. Row source:
results/full/with_errortype/qwen35_27b_infer_property.jsonl (same schema
as Gemma stage-1); G0 rows = the 16 balanced rows of jobs 457191-457194.

## Pre-Registered Necessity Battery (2026-08-09)

### K. Necessity ablation on natural successes (Gemma L30; is the lever a channel-in-use?)

Pre-registered 2026-08-09, before any data. The program holds only
SUFFICIENCY evidence for the lever: adding content in the rank-8
subspace repairs failures (C, F-series), but nothing has ever been
REMOVED from it. Item K asks whether the subspace is necessary for
natural success — a channel the model itself uses when it gets rows
right — or a write-only port (sufficient exogenous handle, not the
endogenous carrier). This fills the causal quadrant the paper's framing
implicitly promises (gauge necessity: null, item D; lever sufficiency:
strong, item C; lever necessity: unmeasured) and supplies the missing
"corrupt" cell for the break/repair asymmetry question. Claim scope
throughout: SUBSPACE-necessity of the 8 fitted PCA directions at this
site/protocol — never "the model's own decision variable" (F(i)'s null
stands; endogeneity wording moves only as far as the branch below
licenses).

Design — Scholar bf16, constraint J, gpu:2; FOUR jobs: correct-side
shards 0–2 (16 rows each) + an anchor-only shard 3. Harness: extend
`scripts/stage2_rank_k_guard_v2.py` with a `--necessity` mode —
`build_necessity_arms`, a `make_position_project_out_hook` sibling of
the existing add hook (orthogonal-complement projection of a frozen
basis at manifest positions during prefill), correct-row interleaved
sharding, and an anchor-only shard mode. The sharding and hook are NEW
code; "reused verbatim" applies to row selection, capture loading, add
hooks, scoring, and bootstrap. Correct-row routing keeps the
label-prefix convention (every correct-side arm carries the `correct_`
prefix). Seeding: guard_v2's formula is kept (sample_seed +
source_row_index·10007 + arm_index·101; one seed per (row, arm)
k-batch); `--necessity` ports the Qwen-harness explicit arm-index
tuples so indices are pinned, not positional. The anchor arms are
pinned to F(ii)-c's positional indices 0/1, so the 458431
token-for-token gates are well-defined; correct-side arms take
explicit indices 0 and 60–68 (collision-free vs existing allocations
0, 2, 10–17, 20–25, 30–36, 40–45, 50–52; guard_v2 positional indices
never exceeded 11). Shard 3's row keys for the 26 guard rows reproduce
458431's shard-count-1 formula exactly (verified pre-submission), so a
gate failure is diagnostic of a code change, not keying drift.

- **Correct-side rows**: 48 FRESH naturally-correct rows, 24 per height
  (3/4), stage-1 strong labels, seed 20260809, `select_correct_rows`
  code path; exclusions: composite 13, the guard-v2 selection's 32
  rows (26 prepared + 6 skipped), the 458416 capture 96, and F(ii)-c's
  16 correct rows pinned explicitly (3109, 3134, 3471, 3680, 3685,
  3738, 4270, 5235, 6312, 6367, 6411, 7047, 7812, 8388, 9855, 10524 —
  no existing code path excludes these; passed via `--exclude-rows`).
  Expected in-job baseline ≈ 0.73 (single-sample stage-1 selection is
  ~73% stable under k=8 resampling, F(ii)-c); all deltas are vs the
  in-job `correct_unhinted_baseline`, so selection instability is
  absorbed by design.
- **Failing-side anchor (shard 3 only)**: the 26 guard rows;
  `unhinted_baseline` and `fixednorm_proj_add_L30` regenerated at
  F(ii)-c's arm indices — must reproduce job 458431 token-for-token
  (two gates); the anchor dP (+0.447 by gate) is the in-job repair
  reference for the asymmetry readout.
- **Frozen inputs (nothing refit in-job)**: basis = rank-8 PCA of the
  26 prepared guard rows' L30 concept hint-deltas, loaded frozen from
  job 458431's archived states
  (`results/stage2/erasure/classmean_c_deployment_27b_property_shard0of1_states.npz`,
  26 `L30_row{r}_concept_delta` keys) — numerically identical to
  F(ii)-c's in-job full-pool fit. (NOT
  `hint_delta_27b_property_manifest_deltas.npz`, which holds only the
  composite rows.) Class vector = the 458416 96-row natural capture
  (`natural_state_capture_27b_property_L30.npz` + manifest). Amplitude
  = F(ii)-c's correct-side constant `fixed_norm_target` = 3708.2628
  (job 458431; verified single-valued across all 16 correct rows),
  pinned as a registered constant for ALL shards (correct shards have
  no failing rows to recompute from) — `correct_signflip_fixednorm_100`
  is thereby the exact negation of F(ii)-c's
  `correct_fixednorm_add_L30`. Positions = gold-concept mentions in
  the PROMPT, applied at prefill; modified states persist through the
  KV cache (mechanics identical to the add arms; output-token
  positions out of scope — nulls are scoped "at this site/protocol").
- **Correct-side arms** (k=8 each; explicit arm index in parens):
  `correct_unhinted_baseline` (0);
  `correct_ablate_rank8_gold_L30` (60) — project the L30 residual at
  gold positions onto the orthogonal complement of the rank-8 basis
  [K-PRIMARY];
  `correct_ablate_rand8_gold_L30_d1/d2` (61/62) — matched-rank random
  orthonormal 8-dim subspaces (seeded draws), same positions/layer —
  the control family for generic rank-8 removal (2 draws for wall fit
  vs item C's 4 and G6′'s 3 — flagged as a power limitation; per-draw
  descriptive only);
  `correct_ablate_perm8_gold_L30` (63) — the fitted basis under one
  seeded permutation of its 5376 coordinates applied to all 8 vectors
  (preserves spectrum and per-vector norms; destroys subspace
  identity) — the structure-matched flag-layer control. Registered
  note: a "row-shuffled deltas" basis is vacuous for PCA
  (sample-permutation-invariant), hence coordinate permutation;
  `correct_signflip_fixednorm_100/200` (64/65) — the F(ii)-c
  correct-side vector NEGATED at 1×/2× the pinned constant [the
  corrupt cell]. No 0.5× rung: at the registered MDE it could only
  straddle, and its gens fund the perm8 control;
  `correct_rand_norm_gold_d1/d2` (66/67) — random directions at the
  1× pinned norm — the matched-norm control for signflip at 1×. The
  2× dose has NO matched control: any 2×-only effect is flagged
  control-unmatched/amplitude-confounded (G6 precedent);
  `correct_fixednorm_100` (68) — the POSITIVE F(ii)-c correct-side arm
  regenerated on the fresh 48 [collateral fresh-draw secondary].
- **Telemetry rider (CPU, descriptive, no branch)**: per-row sign and
  magnitude of ⟨class-mean, state⟩ at gold positions pre-intervention
  (sign-stability diagnostic), and per-arm projection-variance
  telemetry as in item D.

Decision rules (before unblinding):

- Gates: (i) both anchor arms verbatim vs 458431 — mismatch → debug,
  no unblinding; (ii) pooled `correct_unhinted_baseline` ≥ 0.55 —
  below → selection suspect, debug, no unblinding of correct-side
  arms; (iii) parse-fail < 5% per arm.
- **Scoring metric, pre-named**: every branch condition is evaluated
  on dP(strong) with unparsed counted as not-strong. An arm breaching
  the 5% parse gate is flagged and P(strong|parsed) is reported
  alongside as the precision-vs-demolition split (erasure-battery
  precedent); an arm above 20% parse-fail has its branches VOID →
  debug. (Parse failure under ablation is itself evidence about
  breaking; the not-strong convention keeps it inside the primary
  metric rather than leaving a post-hoc metric choice.)
- Stats: row-cluster bootstrap (10k, seed 20260704, percentile) per
  side; paired contrasts within the correct side; random families
  pooled across draws. MDE registered honestly: single-arm ≈ 0.11
  floor (0.19/√3 from F(ii)-c's n=16 — an arm pinned near ceiling;
  mid-range break arms plausibly 0.12–0.15 from per-row binomial
  variance at p ≈ 0.5); paired-contrast MDE expected ≈ 0.09–0.13 (G3′
  pairing precedent; correct-row covariance unknown). EVERY branch
  resting on a straddling CI carries a mandatory MDE statement.
- **K-PRIMARY (necessity) — NO registered prediction on direction;
  genuine uncertainty is the point. Branch partition over (ablate
  sign-status × paired (ablate − rand8 family) × rand8-family
  sign-status)**:
  1. CHANNEL-IN-USE: ablate CI entirely < 0 AND paired CI entirely
     < 0. Sub-case (a), rand8 family straddles: wording "removing the
     repair subspace at concept positions breaks natural successes
     that matched-rank random removal leaves intact —
     subspace-necessity at the lever's own site." Sub-case (b), rand8
     family CI < 0: wording "breaks beyond generic rank-8 removal
     damage — necessity supported with a generic-fragility
     component"; §6 movement one notch weaker. Either sub-case: §6
     endogeneity upgrades from "exogenous mediation" toward
     channel-in-use (never identity); drafting MAY add a registered
     addendum to §5.5/§6. FLAG LAYER: if the paired (ablate − perm8)
     CI does not also exclude zero in the breaking direction, the
     verdict carries a structure-generic-removal flag and the wording
     is the hedged sub-case-(b) form regardless of rand8 status (G6
     proj64-flag mechanics).
  2. WRITE-ONLY PORT: ablate CI straddles (anchor positive by gate).
     Wording: "the lever repairs but its removal costs nothing
     detectable at this site/protocol (MDE stated) — an exogenous
     write-port; consistent with multiple realization or redundant
     carriage of the signal elsewhere, not established by this null"
     (interpretive pointer to claim 7, not a finding). §6 keeps
     "exogenous mediation" with direct support.
  3. PROJECTION-DAMAGE CONFOUND: ablate CI < 0 AND rand8 family CI
     < 0 AND paired straddles → generic rank-8 removal damage;
     necessity UNRESOLVED at this design (paired MDE stated); no
     wording movement; a lower-rank/dose follow-up requires its own
     registration.
  4. BREAKS-SPECIFICITY-UNRESOLVED: ablate CI < 0 AND paired
     straddles AND rand8 family straddles → "breakage detected;
     specificity over generic removal unresolved at paired MDE X" —
     no wording movement, equal prominence. (The modal partial
     outcome at this power; named so it cannot force a post-hoc
     call.)
  5. INVERSE-SPECIFICITY: ablate CI straddles AND rand8 family CI < 0
     → "the lever subspace is specifically dispensable while generic
     removal costs" — descriptive, wording reserved. PRECEDENCE: when
     this and WRITE-ONLY PORT both fire, INVERSE-SPECIFICITY subsumes;
     both are reported, and the WRITE-ONLY §6 sentence still applies
     with the generic-fragility context noted.
  6. Catch-all: any remaining cell (ablate CI entirely > 0; paired CI
     entirely > 0, i.e. rank8 removal costs less than generic; rand8
     family CI entirely > 0) → reported descriptively, wording
     reserved, no movement.
- **K-SECONDARY (the corrupt cell) — registered prediction (i), item
  K's ONLY registered prediction** (the program's fourth sign-flip
  prediction, after F(ii)-b, G4′, G6′): `correct_signflip_fixednorm_100`
  dP CI entirely < 0 AND paired (signflip_100 − rand_norm family) CI
  entirely < 0 → label-specific corruption of natural successes;
  prediction CONFIRMED. Direction + specificity only — magnitude
  unregistered (prior sign-flip effects were floor-censored on failing
  rows: −0.120 at an absolute floor near 0; Qwen −0.094 likewise).
  Failure modes, equal prominence:
  - signflip_100 CI < 0, paired straddles, rand_norm family CI < 0 →
    corruption is norm-generic at these positions (contrast with the
    repair side, where labels carry the effect); prediction miss.
  - signflip_100 CI < 0, paired straddles, rand_norm family straddles
    → corruption detected, label-specificity unresolved at paired MDE
    (stated); prediction miss, no corrupt-cell claim.
  - signflip_100 straddles AND signflip_200 straddles → full-ladder
    null → state-dependence ("the vector only acts on failing rows")
    — its own finding, wording reserved.
  - signflip_100 straddles, signflip_200 CI < 0 → dose-threshold
    reading, flagged control-unmatched at 2× (no rand_norm at that
    norm); no label-specificity claim; prediction miss.
  - signflip_100 CI entirely > 0 → impeaches label-specificity of the
    F-series vector — reported with equal prominence, immediate
    follow-up required before any F-series claim is next cited (G4′
    convention).
  - Remaining cells → descriptive catch-all, wording reserved.
- **Asymmetry readout (descriptive; no thresholds, no claims)**:
  headroom fractions with CIs — break fraction B = |signflip_100 dP| /
  baseline_correct (share of available down-room consumed) vs repair
  fraction P = anchor dP / (1 − anchor baseline) (= 0.447/0.880 ≈ 0.51
  by gate). B > P is noted as attractor-compatible asymmetry; B ≤ P
  as symmetric-or-repair-favored. Cross-population caveat explicit
  (different rows, different headrooms); VOID if signflip_100
  straddles or the dose ladder is non-monotone.
- **Collateral secondary (wording-only; NOT a registered
  prediction)**: `correct_fixednorm_100`. REPLICATES = dP CI entirely
  > 0 vs the fresh baseline (direction only). Effect-size consistency
  is assessed descriptively on the headroom fraction dP /
  (1 − baseline) vs F(ii)-c's 0.266/0.273 ≈ 0.97 (ceiling censoring
  makes point-vs-point comparison confounded by where the fresh
  baseline lands). CI straddles or < 0 → F(ii)-c's
  collateral-beneficial wording gains a fresh-row caveat in ledger
  §1.5 (MDE stated); equal prominence.
- Scope: exploratory for the current paper; no §1 claim moves on any
  branch; §6's endogeneity paragraph may cite the outcome in one
  sentence either way; CHANNEL-IN-USE additionally licenses the
  drafting decision named above. Qwen analog (raw-state necessity at
  L43 — no compact basis exists there) deferred to its own
  registration.
- Budget (rates from THIS harness's logged jobs — 458401/458402 ≈ 6.8
  gens/min on 2× A40; 458431 = 880 gens in 2.16 h): correct side 10
  arms × 48 rows × 8 = 3,840 gens; anchor 2 × 26 × 8 = 416. Four
  jobs: shards 0–2 = 16 correct rows each (1,280 gens ≈ 3.1–3.3 h)
  and shard 3 anchor-only (416 gens ≈ 1.1 h) — each inside the 4 h
  constraint-J wall with margin. A 2-shard split would breach the
  wall at this harness's measured rate and is registered out.

### K′. Energy-matched necessity controls (content vs energy adjudication)

Pre-registered 2026-08-09 evening, after item K's verdict and before
any K′ data. Item K's registered telemetry revealed that the fitted
rank-8 delta basis carries 97.7% of the state norm at gold positions
(removal ≈ 30,452 of ‖state‖ ≈ 31,163), while the matched-RANK
controls removed 1.3–3.3% — so K's landed wording was scoped to
state-content necessity and the §6 endogeneity movement was DEFERRED.
K′ supplies the missing adjudication: does the break track the deleted
CONTENT or the deleted ENERGY? No registered prediction — the branches
are the result either way. Claim scope inherits K's (site/protocol;
never channel-identity).

Design — three Scholar bf16 jobs (constraint J, gpu:2), SAME row
machinery as item K: identical 48-row selection (seed 20260809, same
exclusions), identical 3-shard split, `--necessity-prime` mode in
`scripts/stage2_rank_k_guard_v2.py`. No anchor job: the verbatim gates
ride in-shard.

- **Frozen inputs (all from artifacts already on disk; nothing refit)**:
  the K delta basis (458431 states npz, identical load path + pinned
  norm check); the 458416 capture's 96 per-row position-mean unhinted
  states, from which (a) `capture_mean` = the mean state vector and
  (b) `statepca8` = the top-8 RIGHT SINGULAR VECTORS of the UNCENTERED
  96 × d state matrix (the energy-optimal rank-8 subspace of natural
  states — deterministic SVD, frozen).
- **Arms** (k=8; explicit seed indices in parens; both gate arms keep
  their item-K indices so seeds are identical):
  `correct_unhinted_baseline` (0) and `correct_ablate_rank8_gold_L30`
  (60) — VERBATIM GATES: must reproduce jobs 459836/459837/459838
  token-for-token per shard;
  `correct_meanablate_gold_L30` (70) — replace each gold-position
  state with `capture_mean` (mean-ablation: destroys row- and
  position-specific content, preserves a typical state; K′-PRIMARY);
  `correct_statepca8_ablate_gold_L30` (71) — project out the
  `statepca8` subspace (matched-ENERGY generic removal);
  `correct_ablate_rank1_gold_L30` (72), `_rank2_` (73), `_rank4_`
  (74) — project out the top-1/2/4 components of the frozen delta
  basis (energy-vs-dimension ladder; per-rung removed-energy
  telemetry);
  `correct_ablate_dose012_gold_L30` (75) — partial removal h → h −
  0.12·proj_delta8(h), the α that norm-matches the removal to the
  item-K add family (0.12 × 30,452 ≈ 3,650 ≈ the pinned 3,708);
  `correct_keeponly8_gold_L30` (76) — h → proj_delta8(h): DELETE the
  5,368-dim complement (≈ 21% of norm), keep only the 8 delta
  dimensions (sufficiency-at-site rider).
- **Hook**: `make_position_project_out_hook` generalized with
  `alpha` (partial removal) and `keep` (project-onto);
  `correct_meanablate` uses the existing replace hook with the tiled
  frozen mean. Formulas as written above, registered exactly.
- **Telemetry (descriptive, no branch)**: per-arm removed/kept norm
  fractions; per-row ‖h − capture_mean‖/‖h‖ for the mean-ablation
  (the effective perturbation size); principal-angle cosines between
  the delta basis and `statepca8`.

Decision rules (before unblinding):

- Gates: both gate arms verbatim vs the matching item-K shard
  (mismatch → debug, no unblinding); parse per item K (5% flag with
  P(strong|parsed) alongside, > 20% voids the arm; branches scored on
  dP(strong), unparsed = not-strong); pooled baseline ≥ 0.55.
- Stats: identical to item K (row-cluster bootstrap 10k, seed
  20260704; paired within-row contrasts; MDE statement mandatory on
  every branch resting on a straddling CI).
- **Interpretability pre-rules (artifact-derived, computable before
  launch; recorded at registration)**: (a) if the mean principal-angle
  cosine between `statepca8` and the delta basis exceeds 0.9, the
  statepca8 arm is labeled CONFOUNDED-BY-OVERLAP and carries no
  adjudication weight (descriptive only); (b) if the median
  ‖h − capture_mean‖/‖h‖ across rows exceeds 0.5, the mean-ablation
  carries a MEAN-FAR flag — the frozen mean is not a "typical state"
  for these positions and the CONTENT-NECESSITY wording weakens to
  its hedged form (flag reported alongside, branch structure
  unchanged). COMPUTED AT REGISTRATION from the frozen artifacts:
  (a) principal-angle cosines delta8 vs statepca8 = [0.988, 0.846,
  0.758, 0.421, 0.415, 0.235, 0.091, 0.022], mean 0.472 → statepca8
  is INTERPRETABLE (the subspaces share the single giant-norm
  direction, then diverge); its expected removed-norm fraction on the
  archived K rows is 0.993 (≥ the delta basis's 0.977 — a true
  matched-energy control). (b) median ‖h − capture_mean‖/‖h‖ = 0.175
  on the 46 archived K rows → NO MEAN-FAR flag; mean-ablation is a
  ≈ 0.18·‖h‖ perturbation that preserves the typical state.
- **K′-PRIMARY branch partition over (meanablate sign × paired
  (meanablate − ablate_rank8) sign)**:
  1. CONTENT-NECESSITY: meanablate CI entirely < 0 AND paired CI < 0
     or straddling (breaks as hard as zero-ablation) → row-specific
     content at gold positions is load-bearing even at preserved
     typical energy. Item K's DEFERRED §6 movement fires in the
     hedged channel-in-use form ("row-specific content at concept
     positions is necessary; carried within the delta-span state
     content"); wording downgraded per the MEAN-FAR flag if set.
  2. PARTIAL: meanablate CI entirely < 0 AND paired CI entirely > 0
     (breaks, but less than zero-ablation) → both energy and content
     contribute; content share = meanablate dP / ablate_rank8 dP
     reported with CI; §6 movement fires only in the weaker
     "substantial share" form if the share's CI floor ≥ 0.33,
     otherwise stays deferred.
  3. ENERGY-ACCOUNT: meanablate CI straddles zero (MDE mandatory) →
     item K's break is a state-deletion artifact; K's state-level
     wording is FINAL; thin-subspace necessity NOT established at
     this site; the deferred §6 movement is CANCELLED (not deferred
     again).
  4. Catch-all: meanablate CI entirely > 0, or any arm CI > 0 outside
     named branches → descriptive, wording reserved.
- **Riders (descriptive labels, no claims)**: statepca8 vs
  ablate_rank8 paired (subject to pre-rule (a)); keeponly8 labeled
  SUFFICIENT-AT-SITE if its dP CI lower bound ≥ −0.15, else
  INSUFFICIENT-AT-SITE; rank ladder reported as break-vs-rung with
  removed-energy per rung (descriptive turn-on point: first rung with
  CI < 0 and point ≤ 50% of ablate_rank8's); dose012 compared
  descriptively (cross-job) to item K's rand_norm family (−0.351) and
  signflip_100 (−0.696) at the matched ≈ 3,700 perturbation norm.
- Scope: exploratory; no §1 claim moves on any branch; K′ only
  resolves item K's deferred/pending wording as specified above.
  Qwen analog remains deferred.
- Budget: 9 arms × 48 rows × 8 = 3,456 gens as three 16-row shards
  (1,152 gens ≈ 2.8–3.0 h each at the harness's measured 6.8
  gens/min), all inside the 4 h wall; sequential under the 2-GPU QOS
  cap ≈ 9 h wall-clock.

## Pre-Registered Self-Addressing Battery (2026-08-12)

### L. Self-addressing the lever: gauge-selected candidate sweep (closing the addressing gap answer-free)

Pre-registered 2026-08-12, before any data. F(ii)-c/H established that
the donor-free repair is answer-free in CONTENT but not in ADDRESSING:
gold-position writes repair (+0.447), all-position writes cancel
(+0.029 n.s.), and every pre-named answer-free position policy failed
(item H; the vector is a positional commitment command). Item L runs
the composition item H's mechanism readout points at: enumerate the
candidate concepts answer-free from the prompt (the taxonomy is IN the
prompt), fire the lever at each candidate's mention positions in
separate branches, and let the GAUGE — the stage-1 readable
correctness signal, so far established as deletable and unsteerable —
select the branch. If the gauge can rank its own steered futures, the
loop is answer-free end-to-end and the gauge finally has a causal job
(selector, not lever). NO registered prediction on the primary —
genuine uncertainty is the point; branch-complete rules below. Venue:
Scholar bf16, 2× A40, constraint J (lane-locked as always; the
givemeanode H100 lane is explicitly NOT used for registered arms —
different numerics break in-lane comparability).

**Frozen inputs (nothing fit in-job).** Repair vector and protocol
identical to item K's correct-side positive arm, now fired on failing
rows: class-mean from the 458416 96-row capture, projected into the
frozen 26-row delta basis (458431 states npz, pinned-norm check
3708.2628 as in K), per-position norm = the pinned constant, dose 1×.
Gauge direction = row 0 of `L53_inlp_stack` in
`results/stage2/erasure/inlp_direction_stacks_27b_property_5layer.npz`
(the stage-1 L53 probe direction; AUC 0.902); gauge score of a state =
its dot product with that unit direction (ranking only — no bias or
calibration needed; L0 characterizes the scale). Candidate set per row
= `all_concept_names` (taxonomy concepts named in the prompt FOL —
answer-free by construction; 7–10 per row on this dataset). Candidate
positions via the existing `rel_by_concept` machinery. Per-candidate
generation seeds use item H's inner formula (sample_seed +
row·10007 + 555 + cand_index·13) — protocol-matched; no collision
with H is possible (disjoint rows).

**L0 — gauge-on-steered-states calibration (the gate job; runs first).**
The three K-report vetters converged on this prerequisite: the stage-1
probe was trained on natural states and may mis-score steered ones.
One Scholar job, ~1.2 h:
- (a) NATURAL SANITY: one forward per row of the 458416 capture set
  (96 rows); capture L53 final-prompt-token states; CPU battery: gauge
  AUC for natural strong-correct vs incorrect. GATE: AUC ≥ 0.75
  (expect ≈ 0.90; below → artifact or pipeline suspect, STOP, no L1).
- (b) STEERED-SHIFT (descriptive, no gate): on 12 seeded rows from the
  26-row guard set (seed 20260812, first 12 in seeded order),
  per-candidate fires with the frozen vector; capture the steered
  pass's L53 final-token state per branch; report the score
  distribution shift steered-vs-unsteered.
- (c) SELECTION-SIGNAL GATE: on those 12 rows, row-paired contrast
  (gauge score of the GOLD-candidate branch − mean score of non-gold
  branches), row-cluster bootstrap CI. PASS (CI > 0) → L1 PRIMARY as
  registered below. FAIL → L1 still runs, but gauge-select is DEMOTED
  to exploratory and L1's registered outcome becomes the
  oracle-transfer measurement + selector-texture report (stated in
  advance; no re-registration needed).
- (d) k=2 generations per L0 branch (~204 gens) — descriptive
  (score-vs-outcome linkage preview only; no rule keys to them).

**L1 — the sweep.** 64 FRESH failing rows (32 per height 3/4, stage-1
strong labels, seed 20260812, `select_fresh_rows`, excluding: composite
13, guard-v2 32, capture 96, F(ii)-c 16, K/K′ 48). Four 16-row shards.
Arms (k per arm; explicit seed indices in parens):
- `unhinted_baseline` (0), k=8.
- `percand_fire_L30` (80): for EACH candidate concept, k=4 generations
  with the frozen vector written at that candidate's mention positions;
  the steered prefill's L53 final-token state is captured per branch
  (one extra forward per branch). The GOLD-candidate branch doubles as
  the ORACLE arm.
- `matched_bestofN_unsteered` (81): n_cands(row)×4 unhinted samples —
  the compute-matched sampling baseline (same generation count as the
  row's full candidate sweep).
Offline selection policies (pre-named; deterministic functions of the
recorded data, evaluated with row-cluster bootstrap over rows):
- GAUGE-SELECT [PRIMARY]: per row, pick the branch with the highest
  gauge score; the row's outcome = that branch's P(strong) over its
  k=4.
- RANDOM-SELECT: uniform branch choice, pooled over 20 seeded draws
  (seed 20260812+d) — the selector-null.
- SELF-RATIFY: item H's P1 (pick the candidate whose fires most often
  propose the fired concept) — the non-gauge comparison policy.
- ORACLE: the gold branch (ceiling, not a policy).
- BEST-OF-N: majority-vote and any-correct readouts of arm 81
  (the sampling counterfactual at matched compute).

Decision rules (before unblinding):
- Gates: parse-fail per item-K conventions (< 5% pass; 5–20% flag with
  P(strong|parsed) alongside; > 20% voids the arm; all scoring on
  dP(strong), unparsed = not-strong); pinned-norm check in-job as in K.
- **ORACLE GATE (evaluated first):** gold-branch dP vs in-job baseline,
  CI > 0. FAIL → headline = "the frozen-basis donor-free protocol does
  not transfer to fresh failing rows at pinned norm" (equal
  prominence; the K-convention protocol has never been run on failing
  rows — this is a genuine risk, not a formality); selectors reported
  descriptively, no selector claims, and the addressing question
  returns to the registry unresolved.
- **L1-PRIMARY** (requires L0(c) PASS and the oracle gate):
  GAUGE-SELECT dP vs baseline CI > 0 AND paired (gauge-select −
  matched-bestofN) CI > 0. PASS wording: "the gauge selects the
  lever's address with no answer information: the first gauge+lever
  closed loop, answer-free in content, amplitude, and addressing;
  latent repair beats compute-matched sampling." §5.5/§6 gain one
  registered paragraph; no §1 claim moves.
- Named branches (equal prominence):
  1. Gauge-select beats baseline but NOT matched-bestofN (paired
     straddles or < 0) → "answer-free selection works but does not
     beat compute-matched sampling here" — no method claim; MDE
     stated.
  2. Gauge-select null while oracle positive → registered FINDING:
     "the gauge does not rank its own steered futures" — the
     gauge/lever separation extends to selection; report random-select
     to separate selector-failure (random ≈ gauge) from
     branch-degradation (random < gauge).
  3. Random-select ≈ gauge-select, both CI > 0 → selection is
     signal-free: firing at ANY candidate helps on average (consistent
     with item H's ~50% pull) — the composition works but the gauge
     contributes nothing; wording reserved.
  4. Oracle positive at < 50% of the guard-row anchor scale
     (descriptive comparison to +0.447; different protocol, so
     descriptive only) → transfer-attenuation noted alongside any
     other branch.
  5. Catch-all: any arm CI > 0/< 0 pattern not named above →
     descriptive, wording reserved.
- Secondaries (reported with CIs, no claims): gauge-select as a
  fraction of oracle; gauge vs self-ratify paired; gold-vs-non-gold
  fire outcome rates (item-H mechanism texture under the frozen
  protocol); per-branch score-vs-outcome AUC (the L0(d) linkage at
  scale).
- Stats: row-cluster bootstrap (10k, seed 20260704) on all contrasts;
  MDE statement on every straddling CI (expect ≈ 0.11 single-arm at
  n=64).
- Scope: exploratory; no §1 claim moves on any branch; the paper's
  §5.5 addressing caveat is REPLACED by this item's outcome sentence
  (whichever branch fires — that is the point).
- Budget (measured 6.8 gens/min): L0 ≈ 204 gens + ~200 forwards ≈
  1.2 h, one job. L1: per row ≈ 8 + n_cands×4 + n_cands×4 ≈ 64–88
  gens → ≈ 4,900 gens + ~550 capture forwards over 4 shards of 16
  rows ≈ 3.0–3.3 h each, inside the wall; ≈ 13 h sequential under the
  QOS cap. givemeanode credits ($500 confirmed) reserved for
  lane-agnostic future items, not used here.

### L′. Protocol-transfer adjudication (F(ii)-c protocol on the item-L rows)

Pre-registered 2026-08-13 evening, after item L's oracle-gate FAIL and
before any L′ data. Item L found the frozen K-convention protocol
(pooled 26-row basis, pinned norm 3708.2628) produces nothing at gold
positions on 57 fresh failing rows (−0.004 [−0.022, +0.015]), while
the F(ii)-c protocol (per-row LOO bases, in-job pooled-others norm)
repaired the guard rows at +0.447. Two live explanations with very
different consequences: (1) PROTOCOL — in-job LOO fitting/norms are
load-bearing; the paper's §5.5 claims stand with a protocol-scope
sentence; (2) FRESH-DRAW FRAGILITY — the repair itself is
guard-draw-specific; §5.5 must rescope before anything posts. L′
separates them with one arm on the same rows. NO registered
prediction — the adjudication is the point. Venue: Scholar bf16,
2× A40, constraint J.

Design — two shards (0of2/1of2, interleaved) over the identical
64-row item-L selection (seed 20260812, identical exclusions;
row-keyed generation seeds are layout-independent, so the baseline
arm must reproduce item L's baselines token-for-token regardless of
the different shard layout). `--selfaddress-prime` mode in
`scripts/stage2_rank_k_guard_v2.py`; the arm rides the EXISTING
`fixednorm_proj_add` machinery — no new intervention code.

- **Arms** (k=8; explicit seed indices): `unhinted_baseline` (0) —
  VERBATIM GATE vs the item-L1 recorded baselines on all shared rows;
  `fixednorm_proj_add_L30` (90) — the F(ii)-c construction,
  protocol-matched: class vector from the 458416 capture (frozen),
  projected into the per-row LOO rank-8 PCA basis of the shard's
  in-job hint-deltas (LOO within shard, matching the F(ii) in-job
  convention; ~31 contributing rows per basis vs F(ii)'s 12), scaled
  to the in-job pooled-others recon norm, written at gold concept
  positions. In-job norms and basis EVRs recorded per row.
- Gates: (i) baseline verbatim vs L1 (mismatch → debug, no
  unblinding); (ii) parse per item-K conventions (5% flag with
  P(strong|parsed); >20% voids; scoring on dP(strong), unparsed =
  not-strong).
- **L′-PRIMARY** (pooled ~57 prepared rows, row-cluster bootstrap
  10k, seed 20260704), branch-complete:
  1. TRANSFERS: fixednorm_proj dP CI > 0 → "the F(ii)-c protocol
     repairs fresh failing rows — item L's null is attributed to the
     frozen-pooled protocol (basis provenance / norm convention), not
     fresh-draw fragility." Consequences: §5.5 keeps its claims plus
     one protocol-scope sentence (in-job LOO fitting is part of the
     recipe); this arm doubles as the donor-free repair's first
     fresh-draw replication (report % of +0.447 descriptively);
     item L's headline is refined accordingly.
  2. NULL: CI straddles (MDE statement mandatory; expect ≈ 0.09 at
     n=57) → "the repair does not transfer to this fresh draw under
     either protocol — fresh-draw fragility." Consequences: §5.5
     gains a mandatory scoping caveat (guard-draw-specific until
     resolved); the rescue-set map becomes top priority; equal
     prominence.
  3. NEGATIVE: CI < 0 → descriptive, wording reserved.
- **Secondary (within-row protocol contrast, descriptive unless
  PRIMARY fires branch 1)**: paired (L′ fixednorm_proj − item-L1
  recorded gold-branch fire) per shared row, row-cluster bootstrap;
  k asymmetry (8 vs 4) recorded as a limitation. CI > 0 under branch
  1 → the protocol difference is confirmed within-row.
- Scope: exploratory; no §1 claim moves; §5.5 wording moves ONLY per
  the branch consequences named above. Registered-prediction count
  unaffected (none made).
- Budget: 64 rows × 2 arms × 8 = 1,024 gens + ~64 hinted-capture
  forwards per shard → 2 shards ≈ 1.3–1.5 h each at the measured
  rate, inside the wall; ≈ 3 h sequential.

### L″. The composition, with both halves proven (gauge-selected addressing × the transferring write)

Pre-registered 2026-08-14, after L′'s TRANSFERS verdict and before any
L″ data. Item L established the selector (gauge-select == oracle over
steered branches); item L′ established the write (F(ii)-c LOO
protocol: +0.279 on the fresh rows). L″ runs their composition: fire
the LOO-protocol vector at EACH candidate concept's positions in
parallel branches, gauge-select the branch — answer-free in content,
amplitude, and addressing. NO registered prediction (the halves work
separately; composition can still interfere — that is the question).
Venue: Scholar bf16, 2× A40, constraint J.

**Row-reuse statement (deliberate, stated openly).** Same 64-row
item-L selection (seed 20260812). These rows have unblinded twice
(L1, L′); L″'s question — does gauge addressing recover the
demonstrated per-row repair? — is anchored to their recorded arms, and
within-row anchoring is the point. A PASS earns a fresh-draw
confirmation (L‴) before any wording stronger than "closed on the
adjudication rows"; registered now to prevent drift.

Design — four 16-row shards, `--selfaddress-loo` mode: item L's
per-candidate sweep with the write swapped from the frozen vector to
the F(ii)-c construction (class vector from the 458416 capture,
projected into the row's LOO rank-8 basis of in-shard hint-deltas,
scaled to the in-job pooled-others recon norm — identical to item H's
per-candidate write machinery), k=4 per branch, L53 gauge score
captured per steered prefill (item-L machinery).

- **Arms**: `unhinted_baseline` (seed index 0) — VERBATIM GATE vs
  L1/L′ (third regeneration); `percand_loo_fire_L30` (95) — the new
  arm. The compute-matched sampling baseline is NOT re-run: item L1's
  recorded `matched_bestofN_unsteered` (same rows, same n_cands×4
  count, row-keyed seeds, baseline-verbatim-anchored jobs) is the
  registered comparator, reused cross-job with that justification.
  L′'s recorded gold-addressed fire (+0.279) is the oracle anchor;
  the new arm's own gold branch is its k=4 internal replication
  (descriptive consistency check, F(ii)-c-bonus style).
- Gates: baseline verbatim vs L1 (mismatch → debug, no unblinding);
  parse per item-K conventions (dP(strong), unparsed = not-strong;
  5% flag / 20% void).
- **Selector suite** (offline at verdict, as item L): GAUGE-SELECT
  [PRIMARY] (argmax gauge score; row outcome = that branch's k=4
  mean); RANDOM-SELECT (20 seeded draws, 20260812+d); SELF-RATIFY;
  GOLD branch (oracle).
- **L″-PRIMARY**: GAUGE-SELECT dP vs in-job baseline CI > 0 AND
  paired (gauge-select − L1-recorded bestofN MAJORITY-VOTE) CI > 0.
  Majority-vote is the comparator because it is the answer-free
  sampling policy at matched compute; the verifier-assisted
  any-correct ceiling (0.263 on these rows) is reported for
  calibration, not as a pass condition. PASS wording: "the
  answer-free loop closes on the adjudication rows: gauge-selected
  addressing with the transferring write repairs fresh failures and
  beats the compute-matched sampling policy — the first closed
  gauge+lever composition." §5.5/§6 gain one registered paragraph;
  no §1 claim moves; L‴ fresh-draw confirmation required before any
  generalization language.
- Named branches (equal prominence):
  1. Gauge-select CI > 0 but paired vs bestofN-majority straddles →
     repairs but sampling-parity unresolved; MDE stated.
  2. Gauge-select null while the arm's own GOLD branch CI > 0 →
     SELECTOR–WRITE INTERFERENCE: the LOO write disrupts the gauge's
     ranking (compare gauge-vs-random-select to separate
     selector-failure from branch-degradation); its own finding,
     wording reserved.
  3. The arm's own gold branch null (fails to reproduce L′'s +0.279
     at k=4 within arm structure) → protocol instability across arm
     structure; verdict is confined to reporting this; flag for the
     rescue-set analysis.
  4. Wrong-address collateral: non-gold branches pooled CI < 0
     (misaddressed LOO fires actively harm) reported alongside any
     branch — the deployment-relevant texture.
  5. Catch-all: descriptive, wording reserved.
- Stats: row-cluster bootstrap (10k, seed 20260704); MDE on every
  straddling CI (expect ≈ 0.11 single-arm at n=57).
- Scope: exploratory; no §1 claim moves on any branch.
- Budget: per shard ≈ 16 rows × (8 baseline + ~8.7×4 branch gens) ≈
  685 gens + ~140 gauge forwards ≈ 1.9–2.2 h → four shards inside the
  wall, ≈ 8.5 h sequential under the QOS cap.
