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

## Pre-Registered SJ/OC Factorization (2026-08-14)

### M. Self-judgement vs objective correctness: what does the gauge read, and does the lever care?

Pre-registered 2026-08-14, before any capture or intervention data.
Motivation: 2607.16799 (probe-side only) reports that correctness
probes track the model's self-judgement (SJ), not objective correctness
(OC), on conflict rows — and explicitly invites the causal follow-up
that requires a validated lever. Both SJ censuses are in hand
(docs/sj_census_2k_summary.md): under the STRICT question the verbal
channel is ~degenerate (8.2% yes; the pre-named confidence-blindness
case), under the SOFT question ("Was your answer right? Reply with a
single word: yes or no.") it discriminates (50.2% yes; 68/32 split;
conflict cells populated at n=319/303). The registration pins the SOFT
protocol's labels and cites BOTH censuses — the strict census is the
sensitivity bound, not a discarded pilot. Venue: Scholar bf16, 2× A40,
constraint J (hooked lane); SJ labels are recorded properties of the
census rows (black-box, lane-agnostic per the census docs) — no new
elicitation.

**Frozen inputs.** SJ label of a row = its k=4 majority under the soft
question, read from `results/sj_census/sj_census_2k_soft.jsonl`
(ties excluded). OC label = stage-1 `is_correct_strong`. Row frame:
seeded balanced factorial draw, 32 rows per cell from the four
(OC × SJ) cells, seed 20260816, drawn from the census's 2,000-row
frame — 128 rows total; the M2 intervention row sets are the OC=0
cells of this same frame (first 24 per cell in seeded order). Gauge
direction = row 0 of `L53_inlp_stack` (as item L). Repair protocol =
L′'s in-job LOO fit (the lesson of L/L′: in-job fitting is part of the
recipe), pinned norm 3708.2628, dose 1×, generation constants
identical to L′'s prime arms (k, token budget, parse gates, in-job
unhinted baselines).

**M0 — capture + factorial battery (one job, ~2 h).** One forward per
frame row (original stage-1 prompt; no interventions); capture
final-prompt-token states at the stage-1 5-layer stack (L53 primary).
Offline CPU battery, 4-fold CV stratified by cell:
- (a) NATURAL SANITY GATE: gauge AUC for OC on the frame ≥ 0.75
  (expect ≈ 0.90; below → STOP, pipeline suspect).
- (b) Direction estimation: dir_SJ = (mean SJ=yes − mean SJ=no)
  averaged within OC levels (factorially balanced); dir_OC symmetric,
  averaged within SJ levels. Held-out decodability of each (AUC).
- (c) T1, THE IDENTITY TEST (no registered prediction — genuine
  uncertainty): on held-out CONFLICT rows only, sign-agreement of the
  frozen gauge score with SJ vs with OC; row-cluster bootstrap on the
  agreement difference. Branches: GAUGE~SJ iff agreement(SJ) ≥ 0.70
  AND difference CI > 0 (upgrades the paper's deletable-gauge reading
  to 2607.16799's semantic identity, causally grounded); GAUGE~OC iff
  the mirror rule fires (sharpens AGAINST 2607.16799 — the gauge reads
  upcoming correctness even where verbal self-report disagrees);
  GAUGE-MIXED otherwise (report both agreements + MDE; no upgrade).
- (d) Differential rank-1 erasure (descriptive support, not gating):
  projecting out dir_SJ should drop held-out SJ decodability ≥ 0.10
  AUC while moving OC decodability ≤ 0.03 if the directions factor;
  symmetric check for dir_OC. Failure of factorization is reported as
  SJ/OC-ENTANGLED-AT-SITE and demotes T1 branches to descriptive.
- (e) SJ-READABILITY GATE for T1: held-out AUC(dir_SJ) ≥ 0.60; below
  → SJ-NOT-LINEARLY-READABLE-AT-SITE fires (itself a publishable
  contrast with the verbal channel's census fragility) and T1 is
  descriptive only.

**M1 — (reserved; no arm).** The strict-question replication of M0's
battery is explicitly NOT registered — protocol shopping in reverse.
The strict census stays a sensitivity citation.

**M2 — does the lever care about self-judgement? (one to two jobs,
~4 h each).** Rank-8 LOO repair (protocol frozen above) on two 24-row
sets: CONFIDENT-WRONG (OC=0, SJ=yes) and ORDINARY-WRONG (OC=0,
SJ=no). REGISTERED PREDICTION (the program's 8th): the lever ignores
SJ — both sets repair with dP(strong) CI > 0 and the across-set
paired difference CI contains 0 (MDE stated; expect ≈ 0.18 at n=24).
Branches: LEVER-IGNORES-SJ (prediction confirmed — the lever is
OC-side machinery, completing the gauge~SJ / lever~OC factorization
whichever way T1 lands); SJ-GATED-LEVER iff difference CI < 0
(confident-wrong repairs worse — self-judgement modulates
repairability; genuinely new object); SJ-BOOSTED-LEVER iff CI > 0;
REPAIR-FAILS-ON-FRAME iff either set's repair CI includes 0 — then
the L‴-style fresh-fit diagnostics run before any interpretation, and
the verdict is stated against L′'s +0.279 anchor with MDE.
Determinism: M2 baselines are fresh in-job unhinted arms (no verbatim
gate against prior jobs is possible on this frame — stated now, not
discovered later); seeds row-keyed by the house formula with arm base
120.

**Order and timing.** M0 launches when the Scholar queue drains after
the L-series (L″ now; L‴ if obligated). M2 launches only after M0's
gates are read (its row sets do not depend on M0's outcomes — the
dependency is the sanity gate, not the science). Analysis scripts
follow the L-series pool pattern and land in scripts/ before launch.

### CORRECTION (2026-08-14, late) — items L (write-side) and L″: EXECUTION-INVALID; corrected rerun registered as L″-r

**The defect.** In `scripts/stage2_rank_k_guard_v2.py`, the
selfaddress/selfaddress-loo branch loop applied the intervention hooks
to the gauge-scoring forward pass (`gauge_read`) but NOT to the branch
generation call: `generate_sample_batch` at the per-candidate fire was
never wrapped in `with model.hooks(...)` (every other intervention
lane wraps generation; the L′ arms loop does, which is why L′ is
unaffected). Every "steered branch" generation in items L (L1) and L″
was therefore an ordinary unhinted sample. Discovered during the L″
pooled analysis when the oracle branch read −0.009 against L′'s
+0.279 on the same rows with a byte-identical write construction;
confirmed at the data level: branch generations show NO item-H
steering signature (gold-branch targets-fired 0.40--0.41 ≈ the 0.395
natural rate, in both L1 and L″), and the L″ baseline verbatim gate
passed 456/456 (pipeline integrity; the defect is hook scope, not
stitching). Fixed in the same commit as this entry (generation now
wrapped in the branch hooks; baseline and matched-bestofN arms remain
intentionally unsteered).

**What is withdrawn.**
- L1's ORACLE-GATE-FAILS verdict as a test of frozen-protocol
  transfer: the write never touched the generations, so
  frozen-transfer-to-fresh-rows is UNTESTED, not refuted.
- Every behavioral selector equivalence from L1/L″ (gauge-select ==
  oracle == random at the outcome level was trivially guaranteed when
  all branches are baseline draws).
- L′'s "within-row protocol contrast" INTERPRETATION (fresh-fit beats
  frozen): its comparator was the invalid L1 gold branch, so it
  reduces to fresh-fit vs baseline — which is exactly L′'s primary
  (+0.279) and stands.
- L″'s primary outcome (composition FAIL) — void, not a null.

**What stands.**
- L0 gates: natural-state gauge AUC 0.936; selection-signal
  (gold − non-gold steered-state score) +31.8 [+23.8, +40.1].
- L″ per-shard selection-signal replication (e.g. +21.8
  [+15.7, +28.3], shard C0).
- STATE-LEVEL selector evidence: gauge argmax over steered-state
  branch scores picks the gold branch 54/57 rows (0.947) under the
  frozen write (L1) and 43/57 (0.754) under the LOO write (L″) —
  the gauge reads steered states and separates the gold address.
- L′ in full: +0.279 [+0.182, +0.377] fresh-draw repair under the
  in-job LOO protocol (arms-loop execution, hooks verified in code).
- All determinism/verbatim gates (they measure stitching, which was
  never at fault).

**L″-r — the corrected composition (registered now, before any
corrected data exists).** Identical to L″ in every registered
respect — same 64-row frame (selection seed 20260812), same LOO write
construction, same pinned norm, same candidate enumeration, same
row-keyed seeds (baselines therefore verbatim-gate against L1 again),
same shard layout C0..C3, same venue (Scholar bf16 2×A40 constraint
J) — with the single code correction that branch generations now run
under the branch hooks. Decision rules, comparators, and branches are
those of the L″ registration verbatim (PRIMARY: gauge-select dP CI>0
AND paired gauge-vs-bestofN-majority CI>0; selector-write
interference and gold-instability branches as named there; L‴
obligation on PASS unchanged). Additionally registered rider, C4
(one job, ~1.5 h): the ORIGINAL item-L frozen-write oracle-gate test,
executed correctly — gold-address fires only, frozen donor-free
vector at pinned norm, k=8, on the same 64 rows — so the
frozen-transfer question L1 failed to test gets its answer;
branches: FROZEN-TRANSFERS (dP CI>0), FROZEN-DOES-NOT-TRANSFER
(CI<0 or straddle-with-MDE), stated against L′'s +0.279 anchor.

**Process lesson (recorded for the paper's honesty ledger).** The
defect was catchable in advance by a steering-signature positive
control (does the fired arm move targets-fired above baseline?) — a
gate the position-policy lane (item H) effectively had and the
selfaddress lane lacked. L″-r's analysis adds it as a hard gate:
percand arms must show targets-fired lift over baseline (CI>0) for
the write to count as delivered; a rerun failing that gate is
execution-invalid by pre-registered rule, not by post-hoc audit.

### L‴ — fresh-draw replication of the closed loop (registered 2026-08-17, before any data; obligated by L″-r PASS)

L″-r PASSED its registered primary (gauge-select dP +0.241
[+0.147,+0.342]; paired vs bestofN-majority +0.259 [+0.167,+0.364];
oracle gate +0.263; delivery gate +0.180 CI>0; verbatim 456/456), and
the C4 rider landed FROZEN-TRANSFERS (+0.498 [+0.386,+0.607] at gold,
delivery lift +0.259). Per the L″ registration, the PASS obligates a
fresh-draw replication before any claim upgrade. L‴ is that
replication, protocol-identical to L″-r + C4:

- **Fresh frame:** per-height 32, selection seed 20260817, excluding
  the 57 L-series data rows (docs/l3_exclude_rows.txt, committed) on
  top of the standing composite-manifest exclusions. (7 rows of the
  original L selection produced no data — position-skips — and cannot
  contaminate; a fresh draw that re-selects them re-skips them.)
- **Arms:** R0–R3 = the L″-r composition shards (LOO write, candidate
  branches, gauge scores; same seeds formula, same pinned norm); RG =
  the gold-only frozen-write rider (k=8).
- **Gates:** delivery gate (gold-branch targets-fired lift over
  baseline CI>0 — hard, execution-invalid on failure); natural parse
  gates. No verbatim gate exists for the new rows (no prior jobs
  touched them — stated now); the frozen-artifact pinned-norm
  recomputation gate applies as always.
- **Decision rules:** identical to L″ (PRIMARY: gauge-select dP CI>0
  AND paired gauge-vs-bestofN-majority CI>0; oracle gate first;
  selector-write-interference and gold-instability branches as
  registered). Rider branches: FROZEN-TRANSFERS / does-not, against
  the +0.498 and +0.279 anchors.
- **On PASS:** the closed-loop claim upgrades to
  replicated-on-two-disjoint-draws; no further replication obligation
  is registered. On FAIL: the L″-r result stands as
  single-draw-with-gates and the paper says exactly that.
- Venue: Scholar bf16 2×A40 constraint J; 3:00:00 walls (submit-filter
  constraint).

### K″ — site-specificity of the necessity coda (registered 2026-08-18, before any data)

K/K′ established that at the LEVER'S site (gold concept-mention
positions, L30), destroying the state's content at preserved energy
floors natural success (mean-replacement −0.764, zero survivors). The
paper's limitations name the open control: is any position's content
this necessary, or only the lever's site? K″ answers it with the K′
machinery re-aimed at control position sets, on the identical 46-row
correct frame (same correct_seed/per-height/exclusions → same rows;
baselines re-executed and verbatim-gated against item K).

**Arms** (per row, k=8, arm seed indices 130/131; capture-mean state
and all constants identical to K′'s meanablate):
- correct_unhinted_baseline (seed index 0 — the verbatim gate arm);
- meanablate at RANDOM positions: the prep loop's row-keyed
  matched-count random position set (`random_rel`, sample_seed+row);
- meanablate at NON-GOLD CONCEPT positions: the union of other
  taxonomy concepts' mention positions minus gold overlap,
  subsampled to the gold count with seed sample_seed+row+7777
  (pool shortfalls recorded per row).

**Gates.** Baseline verbatim vs item K on shared (row, sample);
parse gates; per-row ablated-position counts recorded (matched-count
telemetry — the removal-side delivery check).

**PRIMARY — the program's 9th registered directional prediction:
necessity is SITE-SPECIFIC.** Row-paired contrast
Δ = P(strong | random-site meanablate) − P(strong | K′-recorded
gold-site meanablate), same rows, cross-job reuse justified by the
verbatim-gated shared baseline (K′ jobs 459847/459849/459850).
Prediction: Δ > 0 (random-site replacement is cheaper). Branches:
SITE-SPECIFIC iff CI > 0 (prediction confirmed); SITE-GENERAL iff CI
contains 0 (MDE stated — the coda's sentence gains "content at any
matched-count position set is comparably necessary"); SITE-INVERTED
iff CI < 0.
**Secondary (no prediction):** the same contrast for the non-gold
concept arm — NONGOLD-CHEAP / NONGOLD-COSTLY-TOO / NONGOLD-WORSE as
the three branches; adjudicates whether concept mentions in general,
rather than the gold concept's site, carry the load.
Absolute dP vs the in-job baseline is reported for every arm.

**Venue/plan.** Scholar bf16 2×A40 constraint J; 3 shards of 16 rows
(matching the K sharding), 3:00:00 walls; pooled analysis offline.

## Pre-Registered Qwen Closed-Loop Port (2026-08-18)

### N. The closed answer-free loop in Qwen3.5-27B: does the recipe survive the position-anatomy inversion?

Registered 2026-08-18, before any data. Gemma's closed loop
(L″-r/L‴: gauge-selected steered branches repair +0.241/+0.390,
== oracle) composes a subspace-separated gauge and lever. Qwen
separates gauge from lever BY POSITION (J1/J2): outcome info is
readable at the FINAL TOKEN (0.66–0.81 across draws; 0.806 at
L43_final on the J1 draw) and absent at concept positions, while the
working write is the RAW class-mean at concept positions, L43
(G6′: +0.120 [+0.016,+0.224]; real-vs-shuffled +0.182; signflip
harmful). The port asks whether the loop closes across that
inversion: candidates enumerated from the prompt, the raw class-mean
fired at each candidate's positions, the final-token gauge scoring
each steered branch. NO registered prediction — the gauge is weaker
(0.81 vs 0.90) and the write smaller (+0.12 vs +0.40); genuine
uncertainty in both directions. Venue: Scholar bf16 2×A40 constraint
J (the Qwen lane); 3:00:00 walls.

**Frozen protocol constants (G6′'s, unchanged):** write = raw 96-row
capture class-mean, per-position norm-matched
(classmean96_raw_perpos_norm), L43, dose 1×; candidate set =
all_concept_names from the prompt FOL; candidate positions via
concept_positions; house scoring; row-keyed seeds
(sample_seed + row·10007 + arm·101 + offsets), branch seed base
95·101 + cand·13 (the L-series convention); k=8 baselines, k=4 per
branch.

**N0 — gauge fit + gates (one job, ~1.5–2 h).** Calibration draw: 64
balanced rows (32 natural-correct / 32 incorrect, heights 3–4, seed
20260819, excluded from all N1 selection). (a) Capture final-token
states at L43 and L53; fit class-balanced logistic gauges; the
PRIMARY gauge site is pinned NOW as L43_final (the carrier layer);
L53_final is recorded descriptively. Freeze the fitted L43_final
direction+bias as the port's gauge artifact (committed before N1
launches). NATURAL GATE: 5-fold CV AUC ≥ 0.65 (expect ≈0.8; below →
STOP). (b) On 12 seeded failing rows (seed 20260819, disjoint):
per-candidate steered fires; SELECTION-SIGNAL GATE: row-paired
gold-branch minus mean non-gold gauge score, CI > 0 → N1 primary as
registered; FAIL → selector demoted to descriptive, N1's registered
outcome becomes the oracle-transfer measurement (stated now).
(c) DELIVERY HARD GATE on the same 12 rows: gold-branch
targets-fired lift over baseline CI > 0, else execution-invalid by
rule (mandatory since the L-series correction).
**N1 — the composition (96 fresh failing rows, seed 20260820, six
16-row shards).** Arms: unhinted_baseline (k=8) + percand raw fire
(k=4/branch). Gates first: delivery (targets-fired lift CI>0, hard);
parse. ORACLE GATE: gold-branch dP CI > 0 (a scaled G6′ replication;
its FAIL branch = WRITE-DOES-NOT-COMPOSE, stated against the +0.120
anchor with MDE, and selector analyses become descriptive). PRIMARY
(if oracle passes): gauge-select dP CI > 0 AND paired
gauge-minus-self-consistency@8 CI > 0 (the comparator is majority
vote over the row's own k=8 in-job baseline samples — registered
directly this time; no cross-job object). Pre-named branches:
SELECTOR-FAILS (oracle passes, gauge-select straddles → the
position-anatomy inversion breaks the loop at the reading step — a
publishable contrast with Gemma); selector-write interference and
gold-instability as in the L-series. On PASS: fresh-draw replication
obligation, CONTINGENT on cluster access ("if Scholar access ends
before the replication can run, the claim is reported as
single-draw-with-gates and marked unreplicated" — stated now, not
after).

**Amendment to item N (2026-08-18, same evening, BEFORE any N data
exists):** the write amplitude. G6′'s classmean96_raw_perpos_norm
takes per-position norm targets from the test row's OWN hint-delta
reconstruction — answer-adjacent scale that would contaminate the
loop's answer-free claim. N instead pins the port's analog of Gemma's
frozen 3708.26 constant: the amplitude is a SCALAR per-position norm
target, frozen in N0 as the mean rank-16 reconstruction norm over the
12 N0 gate rows' hint-deltas (recorded in the N0 artifact; N1
consumes the frozen number — answer-free for every test row and
identical across shards). Consequence disclosed in advance: the
oracle gate is now measured under this pooled-amplitude adaptation,
so its FAIL branch reads "the write does not compose under the
answer-free amplitude" against the +0.120 per-row-norm anchor, not
as a G6′ contradiction.

### Verification addendum (2026-08-18): erasure application audit

Prompted by external review concern 1 (an erasure no-op is behaviorally
indistinguishable from the §5.2 null). Non-claim-bearing audit, 4 item-D
rows: clean forward records projections onto the registered rank-9
stacks at all five layers; erased forward runs the registered clamp
hooks with an audit hook appended at the same hook points. Deliverable:
per-layer baseline deviation-from-target vs post-hook max deviation
(a no-op leaves them identical; correct application drives the second
to bf16 quantization scale). Complements the standing application
evidence: identical-estimator destructive controls (same code path,
different matrix) and the in-run projection-variance telemetry (read
off the live hook).

**Amendment 2 to item N (2026-08-18 evening, from the N0 gate outcomes,
before any composition data).** N0 did its job and caught two design
errors. (1) GAUGE SITE: the registered L43_final read is structurally
blind to the L43 write — a same-layer injection at concept positions
cannot reach the same layer's final-token state; the selection signal
came back identically zero on all 12 rows (a wiring fact, not a
finding). The gauge site moves to L53_final (J1: 0.808, ten layers
downstream of the write); the gauge is refit at L53 and the N0 gates
re-executed. (2) DELIVERY METRIC: the Gemma-style gold-targeting lift
is ceiling-broken in Qwen — failing rows already target the gold
concept ≈0.9 at baseline (Qwen fails on the hypothesis, not the
concept). The hard delivery gate becomes the NONGOLD fingerprint:
non-gold-branch targets-fired lift over each concept's own baseline
rate, CI>0 (on the existing N0 data: +0.069 [+0.028, +0.111], PASS),
with gold-branch behavioral repair reported alongside (N0 gate rows:
0.167 → 0.354). Natural gate stands as passed (CV AUC 0.741 ≥ 0.65,
noted below J1's 0.806). All other rules unchanged.

### Policy note + registered riders (2026-08-18, late — external review round 2)

**Amendment policy (now explicit, retroactively describing every
amendment made):** a registered result may be voided or amended only
on evidence from an INDEPENDENT TELEMETRY CHANNEL (missing delivery
fingerprint; identically-zero degeneracy; deviation audit) — never on
effect size or surprise. Each amendment cites its channel: L/L″ void
(fingerprint absence + code diff), L″ comparator (structural
impossibility), N amendment 2 (identically-zero selection signal +
targeting ceiling).

**Rider F-R1 — rank-6 remainder write (one job, ~2 h, Gemma lane):**
the rank-8 recipe with the two outlier-aligned components
(participation ratios 2 and 4) removed, at matched norm, on the
26-row F-series guard set, k=8, fresh in-job baselines + delivery
telemetry. Branches: REPAIR-SURVIVES (CI>0 at ≥50% of the rank-8
anchor — the channel is not the broadcast dims), REPAIR-COLLAPSES
(CI straddles/negative — the channel is partly the model's
high-bandwidth broadcast dims; label controls still bound the
interpretation), PARTIAL otherwise. No prediction.

**Rider L-R2 — donor-count ladder (one job, ~2.5 h, Gemma lane):**
the L′ in-job LOO refit re-run with donor pools of 5/10/20/40 rows
(seeded nested subsets of the L′ donor set) on 16 seeded L′ rows,
k=8. Adjudicates frozen-beats-fresh: CONVERGES-TO-FROZEN (monotone
approach within CI at 40) → estimator-n explanation; PLATEAUS-BELOW →
stationarity is real and claimable. No prediction.

Both riders queue AFTER the N-series (access-contingent: if Scholar
access ends first, they are reported as registered-but-unrun).

**N1 analysis addition:** a Qwen-native hypothesis-level delivery
diagnostic (does the fired branch's proposed hypothesis change
relative to baseline, scored on parsed hypothesis identity rather
than concept naming) — descriptive, defined before N1 unblinding.

**N0 (corrected, job 462097) gate outcomes — recorded before N1
unblinding:** natural gate PASS (L53_final CV AUC 0.773); delivery
gate PASS (nongold fingerprint +0.069 [+0.029, +0.113]); SELECTION-
SIGNAL GATE FAIL (+0.392 [−0.178, +0.949], n=12; argmax-gold 4/12).
The pre-named FAIL branch applies: gauge-selection is DEMOTED TO
DESCRIPTIVE for N1, whose registered outcome is the oracle-transfer
measurement (gold-branch dP vs the +0.120 anchor) plus selector
texture. Interpretive note, pre-data: Qwen's baseline gold-targeting
is ~0.9, so steered branches differ less from one another than in
Gemma — the selection stage targets a commitment failure Qwen may
not have; this is the SELECTOR-FAILS contrast the registration
pre-named, now partially visible at calibration.

**Erasure application audit — LANDED (job 462099, 2026-08-18):**
suppression 18–57× at every layer (baseline mean |proj−target| 2.8/30.6/60.9/64.7/112.4
at L15/30/40/45/53 vs post-clamp max 0.05/0.91/2.30/3.52/5.94 — bf16
quantization scale); moved mass equals baseline deviation. The clamp
applied. Review concern 1 closed with data.

**Riders launched (2026-08-18, late; pre-data notes):** F-R1 as
registered (guard frame, anchor arm + rank-6 remainder at matched
others-mean norm, seed indices 0/1/140). L-R2 amendment before any
data: TEST rows capped at 12 (of the registered 16) to fit the
cluster's 3-hour submit-filter wall — the donor POOL remains the full
64-row L-frame preparation; donor subsets are seed-20260818 nested
prefixes; seed indices 150–153; MDE correspondingly wider, stated at
verdict.

## Pre-Registered WikiHop Loop Port (2026-08-19)

### W. The closed answer-free loop on natural data (WikiHop): does the recipe leave the synthetic sandbox?

Registered 2026-08-19, before any port data. Screening (job-6t87z,
docs/loop_screen_summary.md) qualified WikiHop: baseline 0.523,
failing 47.7%, any-correct@8 on failures 0.035 (more committed than
InAbHyD), hint gap +0.158 (modest — power accordingly), closed-book
0.483 (contamination — the doc-dependent filter below is mandatory).
Venue: givemeanode H100 lane, Gemma-3-27B bf16 HF pathway — a NEW
lane (no legacy verbatim constraints; seeds row-keyed throughout).
Lessons baked in from the L/N series: gauge strictly downstream of
the write layer; the delivery fingerprint is validated against an
in-frame positive control at W1 BEFORE becoming a hard gate for W2;
amendment policy (independent-channel rule) applies.

**Frame.** 800 fresh WikiHop dev rows, seed 20260821, docs ≤ 14,000
chars, excluding the 300 screening rows. Prompts identical to the
screening's std/closed constructions (scripts/loop_screen_job.py).
Labels: strong = normalized exact match to the gold candidate.
FAILING row: std any-correct@8 = 0. DOC-DEPENDENT failing row:
failing AND closed any-correct@8 = 0 (memory cannot rescue). All
intervention claims are scoped to the doc-dependent frame; both
contamination numbers are always reported.

**W0 — grading + capture (two jobs, ~$3).** (a) vLLM: std k=8 +
closed k=8 on all 800 (seed 20260821). (b) HF capture, one forward
per row on the std prompt: final-token states at layers
{38, 43, 48, 53} plus per-candidate mention-mean states and position
counts at L30. Offline fits, pinned in the W0 artifact before W1:
GAUGE = logistic on final-token states vs std-majority labels;
primary layer = argmax 5-fold CV among {38,43,48,53}; NATURAL GATE
CV ≥ 0.65. WRITE = class-mean of candidate-mention L30 states,
correct-majority minus incorrect-majority donor rows (donors =
graded rows OUTSIDE the doc-dependent failing pool, balanced).
AMPLITUDE base = mean per-position L30 state norm (measured); W1
ladder {0.25, 0.5, 1.0} × base.

**W1 — calibration (one job).** 12 seeded doc-dependent failing rows
(seed 20260822): gold-only fires across the amplitude ladder (k=4)
plus one full per-candidate pass at the middle rung; k=8 baselines.
Gates: (a) natural gauge gate from W0; (b) POSITIVE CONTROL — some
rung shows gold-address repair direction AND the in-frame delivery
fingerprint (answers-fired rate: normalized output == fired
candidate, lift over that candidate's baseline rate, CI > 0 at
nongold); the rung is pinned for W2; (c) selection signal
(gold-branch minus non-gold gauge score, CI > 0) — FAIL demotes the
selector to descriptive for W2 (stated now, as in N). If NO rung
passes (b): pre-named fallback — one mini layer sweep {20, 25, 35,
40} at the middle rung, re-registered before W2; if that also fails,
W-WRITE-DOES-NOT-TRANSFER is the landed verdict (the InAbHyD lever
does not leave the sandbox at these sites/doses; MDE stated).

**W2 — the composition (≥150 doc-dependent failing rows, ~3 jobs).**
Baseline k=8 + per-candidate fires k=4 at the pinned rung/layer.
Delivery hard gate = the W1-validated fingerprint. ORACLE GATE —
**the program's 10th registered directional prediction: the
gold-address write repairs doc-dependent WikiHop failures (dP CI >
0)** — the lever-site-transfers-across-tasks bet. PRIMARY (if oracle
passes and selector not demoted): gauge-select dP CI > 0 AND paired
gauge-minus-self-consistency@8 CI > 0. Branch-complete: WRITE-FAILS
(oracle straddles/negative — 10th prediction refuted, MDE stated,
scope sentence lands); SELECTOR-FAILS/WEAK; selector-write
interference; gold-instability. On PRIMARY PASS: fresh-draw
replication obligation on a disjoint dev draw (access-independent —
this lane survives Scholar).

### W0 landed + W1 pre-launch pins (2026-09-01, before any W1 data)

**W0 jobs:** grade job-73kjw (v3 resubmission; the 2026-08-19 launch
job-uch77/job-s3afv failed on a gated-repo 401, fixed via the org
`hf-token` secret in the job env). Capture: job-a2zut (v5, THE PIN).
Two capture corrections were made before any fit was pinned, each on an
independent execution-quality channel: (1) v3 (job-ren8t) stored float16
and the run log carried an overflow warning — the artifact had non-finite
entries and finite maxima at the float16 ceiling (Gemma's residual
stream has entries to 6×10⁴); v4/v5 store float32. (2) The v3/v4
substring mention regex matched sub-word spans ('hawaii' in 'hawaiian')
for 104/800 gold candidates (manifest position counts); v5 addresses
whole-word matches. No fit result was voided; the pins are v5's.
Full numbers: docs/wikihop_w0_summary.md.

**Behavior (final):** std 0.463 / closed 0.457 (exact match; containment
0.467 / 0.464) — contamination heavier than the screening's 0.483:
memory answers ~46% of the frame. FAILING (0/8) = 415 rows;
DOC-DEPENDENT FAILING = **299 rows (≥150 ✓)**. Labels bimodal (415 at
0/8, 350 at 8/8, 35 between).

**Natural gate (final): PASS** — 5-fold CV AUC L38 0.776 / L43 0.775 /
L48 0.757 / L53 0.750; primary = **L38** (argmax), strictly downstream of
the L30 write.

**Write pin:** gold-candidate mention-mean L30 class-mean, 127 vs 127
balanced donors outside the doc-dependent pool (seed 20260821);
|class vector| 558.

**Amplitude base — operationalization pinned pre-data.** The registered
base ("mean per-position L30 state norm") measures 32,630 at mention
positions, but 98.0% of that norm² is two massive-activation dims
(104, 2733) that the write direction does not use. The massive-dim-
excluded per-position RMS norm is **4,737**, within 1.3× of the closed
loop's Gemma-L30 anchor dose (fixed_norm_target 3,708), which the ladder
brackets. PINNED base = 4,737; registered ladder {0.25, 0.5, 1.0}× =
{1,184, 2,368, 4,737}. The literal reading is NOT discarded: rungs at the
literal {0.25, 0.5, 1.0} × 32,630 (= 1.72 / 3.44 / 6.89 × pinned base)
run in the same job as descriptive extras. Positive-control reading is on the pinned ladder;
a pass only on a literal rung is reported as the literal-ladder branch
and re-registered before W2.

**W1 job spec (launched after this note was committed at 2258f4e: job-yp5er, context ctx-13603f41, 2026-09-01 21:42 UTC):** 12 rows
(seed 20260822: WH_dev_1021, 1499, 17, 1854, 1931, 194, 2623, 2892, 3676,
4895, 583, 893); in-job k=8 baselines; gold fires k=4 at every rung
(3 pinned + 3 literal); full per-candidate pass k=4 at the middle
pinned rung (0.5× = 2,368); extra telemetry beyond the registration:
3 seeded non-gold candidates fired (k=4) at every non-middle rung so the
delivery fingerprint is readable per rung. Write = class-vector
direction × rung norm added at every whole-word mention position of the
fired candidate (all batch rows, prefill only); gauge read at L38 final
token under the write. Delivery is verified per branch by hook counters
inside `generate` (prefill applications and positions written); a
branch with zero applications aborts the job (the L-series bug class,
now a hard execution check). Reader: scripts/wikihop_w1_gates.py.
Gates as registered: (b) positive control per rung = gold repair
direction (gold k=4 correct rate − in-job baseline rate > 0) AND the
in-frame fingerprint at non-gold fires (answers-fired rate minus that
candidate's baseline rate, row-bootstrap CI > 0); (c) selection signal
at the per-candidate rung (gold-branch − mean non-gold gauge score, CI
> 0; FAIL demotes the selector to descriptive for W2). Fallback layer
sweep {20, 25, 35, 40} at the middle rung only if no pinned or literal
rung passes (b) — re-registered before W2 (would need a fresh capture
at those layers).

### W1 gate outcomes + fallback layer sweep re-registered (2026-09-01, late — before any sweep data)

**W1 (job-yp5er) outcomes, recorded before the fallback runs.** Delivery
audit VALID on three channels (hook counters inside generate on all
1,860 fired records; gauge shift under the write +0.8 to +1.5 for gold
branches; output-channel perturbation with a negative fingerprint at
the literal doses). **(b) POSITIVE CONTROL: FAIL at all six rungs** —
gold repair 1/288 samples (dP +0.021 [0, +0.062] at 0.25×, 0 elsewhere);
non-gold fingerprint lift 0.000 / +0.002 / −0.056 at the pinned rungs
and −0.111 [−0.194, −0.028] at every literal rung. Outputs remain the
baseline modal answer in 85–94% of samples at every dose. **(c)
SELECTION SIGNAL: PASS** — +0.87 [+0.19, +1.60], n=12; argmax-gold
3/12. Selector not demoted. Full readings: docs/wikihop_w1_summary.md.

**Fallback (pre-named in the W registration) — layer sweep, now
pinned:** (i) capture v6 (job-jj4fc) on the 800-row frame: candidate mention-mean
states, per-position norms and per-dim mean squares at L20/25/35/40
(same forward, whole-word addressing, float32; no final-token capture);
(ii) per-layer pins by scripts/wikihop_sweep_fit.py — class-mean from
the SAME 127/127 donor sets W0 pinned, amplitude base = massive-dim-
excluded per-position norm at that layer (W0's rule), middle rung =
0.5 × base_L; (iii) one sweep job (job-erwpa, 2026-09-01 22:46 UTC) on the 12 W1 rows: for each write
layer in {20, 25, 35, 40} plus L30 as a within-job anchor, fresh k=8
baselines, gold fires k=4 and the same 3 seeded non-gold candidates
k=4 (row seeds identical to W1, so the non-gold fingerprint is paired
with W1's), no per-candidate pass; gauge read at **L43** (CV 0.775,
downstream of every write layer including L40) — descriptive only.
Gate per layer = positive control as registered (gold repair direction
AND non-gold fingerprint lift CI > 0). A passing layer is pinned and
W2 re-registered at that site (with a per-candidate selection pass at
that site added to W2's first shard, since (c) was measured only at
L30). If no layer passes: **W-WRITE-DOES-NOT-TRANSFER** lands as the
verdict on the 10th prediction's lever-site bet; MDE from the sweep's
CIs (fingerprint: 144 non-gold samples per layer, CI half-width ≈ 0.08
at W1; repair: 48 gold samples per layer) stated at verdict.
Descriptive additions declared now: L30 anchor re-run; gauge scores at
L43 for every branch.

### W1 gate outcomes (job-yp5er, landed 2026-09-01) + fallback layer sweep re-registered before any sweep data

**Delivery audit (independent channels):** all 1,860 fired records
carry ≥1 prefill hook application inside `generate` with the expected
positions written; the L38 gauge moves under the write (gold-branch
gauge − baseline +0.78 / +1.36 / +1.52 at 0.25 / 0.5 / 1.0× base;
non-gold +0.19 / +0.21 / +0.77); at the literal top rung (32.6k) outputs
still change relative to baseline (fingerprint lift −0.11). The write
is delivered. Row-level: results/loop_screen/wikihop_w1.jsonl; reader
output docs/wikihop_w1_gates.json.

**(a) Natural gate:** PASS (W0, L38 0.776).
**(b) POSITIVE CONTROL: FAIL at every rung** — pinned {0.25, 0.5, 1.0}
× 4,737 and literal {1.72, 3.44, 6.89}×. Gold-fire correct rate 0.021 /
0 / 0 / 0 / 0 / 0 (1 of 288 gold samples; in-job baseline 0/96).
Non-gold answers-fired rate 0.111 / 0.050 / 0.056 / 0 / 0 / 0, but the
lift over each candidate's baseline rate is 0.000 [0, 0] / +0.002
[0, +0.005] / −0.056 [−0.139, 0] / −0.111 [−0.194, −0.028] ×3: the
answers that "match the fired candidate" are the ones the model
already gives at baseline. Outputs stay at the baseline modal answer
85–94% of the time at EVERY dose including 32.6k; no degeneration.
**(c) SELECTION SIGNAL: PASS** — gold-branch minus mean non-gold gauge
score at 0.5× = +0.87 [+0.19, +1.60] (row bootstrap, n=12); argmax-gold
3/12 (chance ≈ 1/18); gold gauge rank ≤ 2 in 6/12 rows.
Reading: on WikiHop the InAbHyD lever site is READABLE (the write moves
the gauge, gold more than non-gold) but NOT STEERABLE (the sampled
answer does not move) at doses from 2× the natural class-mean
difference to 7× the state norm. No W2 rung can be pinned.

**Pre-named fallback, now re-registered (before any sweep data):** one
mini layer sweep at write layers **{20, 25, 35, 40}** at the middle
rung (0.5× that layer's massive-dim-excluded per-position norm), same
12 rows, same class-mean recipe per layer (gold-mention mean, the same
127 vs 127 donor rows, whole-word addressing), gold k=4 + the same 3
seeded non-gold candidates k=4 per row per layer, fresh k=8 baselines
per layer; L30 at 0.5× re-run in the same job as a within-job anchor
(descriptive). Requires one capture job (candidate-mention states +
per-dim mean squares at the four layers; W0_WRITE_LAYERS) and one
sweep job. [Launched after the pins committed at ee933f9: capture
job-pepj4 (pins: |v| 212/474/792/1129, bases 1,421/2,494/6,965/11,004
at L20/25/35/40); sweep split into job-32zfc (L20/25/35 + L30 anchor,
L38 gauge) and job-zjr7q (L40, L48 gauge), 2026-09-01 22:49 UTC.] Gauge: L38 for writes at 20/25/35 (downstream); for the L40
write the W0-fitted **L48** gauge (CV 0.757) is read instead —
descriptive only; the sweep's gate is the positive control (repair
direction AND non-gold fingerprint CI > 0), which needs no gauge.
Outcome branches: a layer passes (b) → it is pinned for W2, re-
registered with the selector status from W1 (not demoted); no layer
passes → **W-WRITE-DOES-NOT-TRANSFER** is the landed verdict for item
W (MDE: with 12 rows × k=4 the gold-repair rate detectable at CI > 0
is ≈ 0.10; the non-gold fingerprint MDE ≈ 0.06), W2 does not launch,
and the 10th prediction is recorded as UNTESTED-AT-COMPOSITION /
lever-site-does-not-transfer (the read half transfers: selection
signal PASS).

### W — LANDED: W-WRITE-DOES-NOT-TRANSFER (2026-09-01, late)

Fallback layer sweep (job-erwpa; docs/wikihop_w1_summary.md) FAILS the
positive control at L20/25/30/35/40 at the middle rung: gold repair 0/48
per layer (0/240 pooled; 1/528 pooled with W1's six L30 rungs), non-gold
fingerprint lift 0.000 / 0.000 / 0.000 / −0.028 / −0.042 (no CI above
zero), delivery audit valid on all 240 branches, gauge (L43) shifted
under every write, seeded baselines reproduced exactly across passes
and against W1. Pre-named consequence: **W-WRITE-DOES-NOT-TRANSFER**
is the landed verdict — the InAbHyD lever does not leave the sandbox at
these sites/doses. MDE: repair ≥ 0.0125 per sample (sweep pooled) or
fingerprint lift ≥ 0.06 would have been detected; the InAbHyD oracle
(+0.24 to +0.39) and fingerprint (+0.07) lie above both. The 10th
registered directional prediction (lever-site transfer) is therefore
refuted at its positive control; W2 (oracle/PRIMARY on ≥150 rows) is
NOT RUN, as registered. What transfers: the gauge (natural gate 0.776 at
L38; selection signal +0.87 [+0.19, +1.60]) — the readable half. Cost of
item W: ~$9 across 9 H100 jobs. Any selector-only study on the 287-row
W2 pool would be a new registration.

**Independent replicate of the sweep (2026-09-01, 22:49–23:12 UTC;
recorded after the verdict above, pre-named design, no amendment).** A
second session ran the same pre-named sweep from its own capture
(job-pepj4; per-layer pins identical to job-jj4fc's to rounding: |v|
212/474/792/1129, bases 1,421/2,494/6,965/11,004) as two jobs —
job-32zfc (L20/25/35 + L30 anchor, L38 gauge) and job-zjr7q (L40, L48
gauge) — with W1's three seeded non-gold candidates per row (unpaired
seeds). Same verdict on every layer: gold repair 0/48 per layer (0/240
pooled); non-gold fingerprint lift −0.021 [−0.062, 0] / 0.000 / 0.000 /
−0.028 [−0.083, 0] / −0.042 [−0.111, 0] at L20/25/30/35/40; delivery
audit valid on all 240 branches; fired outputs equal the baseline modal
answer 91–94%; gauge shift under the write gold/non-gold +0.29/+0.12
(L20), +0.26/+0.15 (L25), +1.36/+0.40 (L30), +1.46/+0.50 (L35; L38
gauge), +0.11/+0.30 (L40; L48 gauge). Row-level:
results/loop_screen/wikihop_sweep_{a,b}.jsonl; reader output
docs/wikihop_sweep_gates.json. The duplication (two sessions, ~$1 extra)
is disclosed here; it adds a within-day replication of the null with a
different gauge read and independent seeds, and changes nothing in the
verdict.

## Pre-Registered Item-W Follow-up (2026-09-01, late)

### WH. Hint-delta oracle write on the reading-driven slice: does the ORIGINAL lever direction transfer where document-side writes have traction?

Registered 2026-09-01 (late), before any data. Exploratory follow-up
to the landed W verdict; the 10th prediction stays refuted; NO new
directional prediction is registered here.

**Why.** W1 and the sweep tested a SUBSTITUTE direction (the class-mean)
on a sample where 7 of 12 rows were memory-driven (the std modal answer
equals the closed-book modal answer: the documents were not being
read, so no document-side write can help). The InAbHyD lever was
hint-delta-derived. And the W1 residue points at the slice: under gold
writes at the pinned rungs, outputs left the baseline modal answer on
15/80 samples over the 5 reading-driven rows vs 1/112 over the 7
memory-driven rows — traction only where the documents matter, but
the class-mean direction did not point at the gold.

**Frame.** READING-DRIVEN slice = doc-dependent failing rows (0/8 std ∧
0/8 closed) whose normalized std modal answer ≠ normalized closed modal
answer, within the W2 pool (W1 rows excluded): **165 rows** (122
memory-driven excluded). Calibration draw: 12 rows, seed 20260823,
pinned in docs/wikihop_wh_pinned.json (WH_dev_1480, 2061, 2136, 305,
3277, 330, 3988, 4678, 4766, 793, 802, 933).

**Prompts.** std (unchanged). HINT-FIRST: "Hint: pay close attention
to {X}." placed BEFORE the documents (the screening's hint-after
construction cannot produce a mention-position delta under causal
attention — the mentions precede the hint). The hint-first text arm is
therefore also a fresh behavioral measurement (its gap is not the
screening's +0.158).

**Write (oracle in content).** Per-position hint-delta at L30 for the
fired candidate X: δ_j = h_hint-first(X)[p^h_j] − h_std[p^s_j] at the
j-th whole-word mention of X, mentions paired by order (the hint line's
own mention excluded; mention counts and token ids at paired positions
must match, else the candidate is skipped and logged). Rungs {0.5, 1.0,
2.0} × raw δ (per-position, no norm matching); measured per-position
|δ| reported against the W0 base 4,737. Framed as ORACLE TRANSFER (as
item N's oracle measurement); an answer-free version (rank-k basis from
other rows' deltas, L-series style) is a separate registration only
on a pass.

**Arms per row (one job, ~$1).** std baseline k=8; TEXT hint-first
arms: gold k=8, the 3 seeded non-gold candidates (seed 20260823 + row)
k=4 (the text ceilings); DELTA writes at L30: gold k=4 and the same 3
non-gold k=4 at each rung; W0's L38 gauge read under every write;
delivery counters inside `generate` (a zero aborts the job).

**Gates.** (a) TEXT CEILING: hint-first gold text correct rate − in-job
baseline, row-bootstrap CI > 0. FAIL → NO-CEILING: the hint does not
repair these rows even as text; the write test is uninformative;
reported, no further arms. (b) ORACLE WRITE (the test): some rung with
gold-δ write dP (vs baseline) CI > 0 AND the non-gold fingerprint
(answers-fired rate − that candidate's baseline rate) CI > 0 →
**HINT-DELTA-TRANSFERS** (oracle, reading-driven scope; rung pinned;
next step registered separately). Otherwise **HINT-DELTA-DOES-NOT-
TRANSFER**: the direction that built the InAbHyD loop fails on natural
data at the most favorable slice; item W closes with no escape hatch.
(c) Descriptive: recovered fraction = write dP / text dP per rung;
non-gold text answers-fired lift (fingerprint ceiling); |δ| norms;
gauge shifts; per-row texture (memory- vs reading-driven is fixed by
construction here). MDE: 48 gold samples per rung → repair rate ≥ 0.10
detectable; fingerprint lift ≥ 0.06. Reader: scripts/wikihop_wh_gates.py.
[Launched after registration commit dbb62bf and tooling commit ad03f43:
job-az23x, context ctx-be35c137, 2026-09-02 01:45 UTC.]

### WH — LANDED: NO-CEILING (job-az23x, 2026-09-02)

Gate (a) TEXT CEILING FAILS: hint-first gold text repairs 2/12 rows
(rate 0.167, dP +0.167 [0, +0.417]); 10 rows keep their wrong answer
even when told the gold candidate as text. Pre-named branch: the write
test is uninformative; no further arms. Descriptive: non-gold text hints
do move answers (+0.188 [+0.056, +0.347]); the 2× δ write fully repairs
one of the two hint-repairable rows (4/4; the first whole-row repair by
any activation write in item W) and the non-gold fingerprint rises with
dose (+0.049 → +0.076, CIs touching 0); delivery audit valid on all 576
fired records. Reading: the reading-driven filter does not isolate
commitment failures; the hint-REPAIRABLE slice (~16% of failing rows)
is the sandbox analog and was never sampled on purpose. Implied next
registration (not launched): hint-first text screening of the 287-row
W2 pool to find hint-repairable rows, then the WH design on 12–24 of
them with rungs {1, 2, 4}×. Full numbers: docs/wikihop_wh_summary.md.

### WR. Hint-delta oracle write on the HINT-REPAIRABLE slice (registered 2026-09-02, before any data)

Exploratory follow-up to WH's NO-CEILING. WH showed the "reading-
driven" filter does not isolate commitment failures (the gold hint as
text fails on 10/12 such rows), while on the 2 rows it does repair, the
2× δ write fully repaired one. WR selects the rows where the text hint
works — the sandbox analog of a commitment failure — so that the
ceiling gate holds by construction and the oracle-write gate is
powered. No new directional prediction is registered (the WH lead is
n=2); the expectation stated for the record is that the write repairs
a minority of hint-repairable rows at the higher rungs.

**Stage 1 — text screen (one vLLM job, ~$1).** Hint-first text prompt
(gold named, hint before the documents) k=8, temperature 0.7, seed
20260824, on ALL 287 rows of the W2 pool (docs/wikihop_w0_pinned.json;
the 12 WH rows included for consistency but excluded from the draw).
HINT-REPAIRABLE row := ≥ 4/8 hint-first samples exact-match the gold
(baseline is 0/8 by pool construction). Reported: the hint-repairable
rate of the pool (screening prior ≈ 16% of failing rows), split by
memory-driven vs reading-driven (descriptive).

**Stage 2 — the write test (one H100 job, ~$1.5).** Up to 24 rows drawn
with seed 20260825 from the hint-repairable set minus the WH rows
(pinned in docs/wikihop_wr_pinned.json before launch; if fewer than 12
exist the stage is reported as UNDERPOWERED and not run). The WH
design unchanged except rungs **{1, 2, 4}× raw δ** (WH: nothing at
0.5/1, traction at 2): std baseline k=8; hint-first TEXT arms gold k=8
and 3 seeded non-gold k=4 (fresh seeds — the in-job text rate is the
regression-to-the-mean check on the screen); per-position L30 δ writes
at paired whole-word mentions, gold k=4 and the 3 non-gold k=4 per
rung; W0's L38 gauge under every write; hook counters (a zero aborts).

**Gates.** (a) TEXT CEILING (re-measured in-job): hint-first gold text
rate − baseline, CI > 0 — expected to hold by selection; FAIL →
NO-CEILING (selection artifact; reported). (b) ORACLE WRITE: some rung
with gold-δ write dP CI > 0 AND non-gold fingerprint lift CI > 0 →
**HINT-DELTA-TRANSFERS** (oracle, hint-repairable scope; rung pinned;
the answer-free version and a fresh-draw replication become the next
registrations). Otherwise **HINT-DELTA-DOES-NOT-TRANSFER-WHERE-TEXT-
DOES**: the original lever direction fails even on rows the same hint
repairs as text — item W closes for good on the write side. (c)
Descriptive: recovered fraction per rung; dose-response; |δ| norms;
per-row table; the WH rows' stage-1 re-measurement vs their WH text
arm. MDE: 24 rows × k=4 = 96 gold samples per rung → repair ≥ ~0.08
detectable against a 0 baseline; fingerprint ≥ ~0.05. Reader:
scripts/wikihop_wh_gates.py (unchanged).
[Stage 1 launched after registration commit 18c6339 / tooling cab0b51:
job-iimvi, context ctx-1697ef9c, 2026-09-02 02:16 UTC.]

**Stage 1 landed (job-iimvi, 2,296 generations).** Hint-repairable
(≥4/8): **79/287 = 27.5%** of the W2 pool — sharply bimodal (196 rows at
0/8, 75 at 8/8, 16 between); above the screening prior (16%, hint-
after). Split: 42 of 165 reading-driven (25%) and 37 of 122 memory-
driven (30%) — the memory/reading filter does not predict text-hint
repairability, so hint-repairability is the selector. WH rows re-
measured consistently (3277 and 933 at 8/8, the other ten at 0/8).
Stage-2 draw (seed 20260825, WH rows excluded): 24 rows pinned in
docs/wikihop_wr_pinned.json; not underpowered. Stage 2 launches on
these 24 with rungs {1, 2, 4}× as registered. [Stage 2 launched after
pins commit 3d151aa: job-q8nzz, context ctx-18fff412, 2026-09-02 02:41 UTC.]

### WR — LANDED: registered verdict HINT-DELTA-DOES-NOT-TRANSFER-WHERE-TEXT-DOES (gate (b) conjunction fails on the fingerprint conjunct); gold conjunct PASSES (job-q8nzz, 2026-09-02)

(a) TEXT CEILING PASS: 0.964 vs 0 [+0.891, +1.000]. (b) gold-δ write dP
**+0.250 [+0.104, +0.417] at 1×, +0.302 [+0.156, +0.458] at 2×**, 0.021
at 4× (over-dose collapse); non-gold fingerprint lift +0.017 [0.000,
+0.049] / +0.056 [−0.028, +0.153] / −0.045 — never CI > 0, so the
registered conjunction fails at every rung and the verdict is recorded
as written. Delivery: audit valid on all 1,128 fired records; the
SPECIFICITY control (gold accuracy under non-gold writes 0.014 / 0.029
vs 0.250 / 0.302 under gold writes; non-gold outputs stay at the
baseline modal answer 81–88%) shows the repair is candidate-specific.
The fingerprint conjunct was mis-powered for this slice (its own text
ceiling is +0.094; expected write lift ~+0.03 < MDE 0.05) — recorded
as a design fault of the registration, not amended post hoc.
Descriptive: gauge-select over the 4 fired branches at 2× = 0.167
[+0.052, +0.302] from a 0 baseline (oracle 0.302); gold argmax 9/24.
Implied next registration (not launched): fresh-draw replication on
the remaining 53 hint-repairable rows with the specificity control as
the pre-named delivery gate, plus the answer-free loop over all
candidates (the W2 composition on the slice where the write works).
Full numbers: docs/wikihop_wr_summary.md.

## Pre-Registered WikiHop Closed Loop on the Hint-Repairable Slice (2026-09-02)

### WL. Specificity-gated replication of the hint-delta write + the answer-free loop over all candidates: does the closed loop have a home on natural data?

Registered 2026-09-02, before any data. Obligated by WR: the gold
hint-delta write repaired +0.25/+0.30 (CI > 0) at 1×/2× on 24
hint-repairable rows, but WR's registered conjunction failed on a
mis-powered fingerprint conjunct and the delivery evidence that
mattered (specificity) was not pre-named. WL fixes both and adds the
composition. **Registered directional prediction (the program's 11th):
the gold hint-delta write at L30 repairs hint-repairable WikiHop
failures (dP CI > 0 at 1× or 2×) with candidate specificity.** The
loop reading carries NO prediction (WR's 4-branch gauge-select was
+0.167 [+0.05, +0.30] with a weak selection signal).

**Frame.** ALL 53 hint-repairable rows of the W2 pool not used by WR
stage 2 or WH (79 − 24 − 2; no sampling; pinned in
docs/wikihop_wl_pinned.json, two seeded shards). Baseline 0/8 std and
0/8 closed by pool construction; ≥4/8 under the hint-first text prompt
by the stage-1 screen.

**Arms per row (two shard jobs, ~$4 total).** std baseline k=8;
hint-first TEXT gold k=8 (ceiling re-measure, fresh seeds; the
non-gold text arm is dropped — its ceiling is too low to gate on).
For EVERY candidate X in the list: hint-first forward naming X →
per-position δ_X at X's whole-word mentions (paired, token-id
verified); δ_X written at L30 at **2×** with k=4 (the loop pass; gold
included). Gold additionally at **1×** k=4, and the 3 seeded non-gold
(seed 20260826 + row) additionally at 1× k=4 (specificity at 1×). W0's
L38 gauge read under every write; hook counters inside `generate` (a
zero aborts). 4× dropped (WR: collapse).

**Gates.** (a) TEXT CEILING: in-job hint-first gold rate − baseline CI
> 0 (holds by selection; FAIL → NO-CEILING, reported). (b) REPLICATION
(the 11th prediction): at 1× or 2×, gold-δ write dP CI > 0 AND
SPECIFICITY = per-row (gold-write correct rate − mean non-gold-write
correct rate) CI > 0. Non-gold fingerprint (answers-fired lift) is
DESCRIPTIVE only. PASS → HINT-DELTA-TRANSFERS (replicated, oracle,
hint-repairable scope), rung pinned as the better of the two. FAIL →
WR was a fluctuation; WRITE-SIDE-CLOSED. (c) LOOP (only read if (b)
passes; the composition, answer-free in the loop sense — δ_X comes
from a prompt naming X, for every X, no gold used): gauge-select over
ALL candidate branches at 2×: (i) gauge-select correct rate −
baseline CI > 0 AND (ii) gauge-select − self-consistency@8 (majority
of the 8 baseline samples, correct or not) CI > 0 → **LOOP-CLOSES-ON-
NATURAL-DATA** (scoped: compatible-answer failures; fresh-draw
obligation on a new WikiHop frame follows). (i) passes, (ii) fails →
LOOP-BEATS-BASELINE-NOT-SC. Neither → SELECTOR-FAILS (write transfers,
gauge cannot pick among ~20 branches; item-N pattern). Descriptive:
argmax-gold rate vs 1/n_candidates, selection signal (gold − mean
non-gold gauge), oracle-vs-loop ratio, per-row table, dose (1× vs 2×).
MDE: 53 rows × k=4 = 212 gold samples per rung → repair ≥ ~0.05;
specificity ≥ ~0.05; loop repair ≥ ~0.06 (row bootstrap, n=53).
Reader: scripts/wikihop_wl_gates.py.
[Launched after registration commit 561c60c / tooling 2fb37fe: shard 0
job-g74tz (27 rows), shard 1 job-wq2vj (26 rows), context ctx-4a0a1a50,
seed 20260826, 2026-09-02 12:29 UTC.]

### WL — LANDED: HINT-DELTA-TRANSFERS (pinned 1×) + LOOP-CLOSES-ON-NATURAL-DATA (job-g74tz + job-wq2vj, 2026-09-02)

(a) TEXT CEILING PASS 0.962 [+0.903, +0.991]. (b) REPLICATION PASS at
both rungs — **11th directional prediction CONFIRMED**: gold-δ dP
+0.297 [+0.189, +0.420] at 1× and +0.264 [+0.151, +0.387] at 2×;
specificity +0.277 [+0.173, +0.388] / +0.244 [+0.135, +0.361] (gold
under non-gold writes 0.030 / 0.029); delivery audit valid on all 6,688
fired records; 1× pinned (larger dP; per-row 1×−2× +0.033 [−0.05,
+0.13]). (c) LOOP PASS on both readings: gauge-select over all
candidates (27.6 branches/row) at 2× = **0.255** vs baseline 0.009
(+0.245 [+0.142, +0.363]) and vs SC@8 0.000 (+0.255 [+0.146, +0.377]);
oracle 0.274 (93% recovered); gold argmax 20/53 (chance 0.071);
selection signal +1.56 [+0.65, +2.56]; the gauge picks gold on 10 of the
14 rows where the gold branch repairs. Scope: compatible-answer
(hint-repairable) failures, 27.5% of the doc-dependent pool. Obligation
opened: fresh-draw replication on a new WikiHop frame (to be registered
before launch). Full numbers: docs/wikihop_wl_summary.md.

### WF. Fresh-frame replication of the closed loop on WikiHop (registered 2026-09-02, before any data — the obligation opened by WL's PASS)

**Frame.** 800 NEW WikiHop dev rows, seed **20260827**, docs ≤ 14,000
chars, drawn from the 3,480 rows not in W0's frame or the screening
(builder scripts/wikihop_frame.py — verified to reproduce W0's frame
byte-for-byte on all 800 rows; the fresh frame overlaps neither).
File results/loop_screen/wikihop_fresh_input.jsonl.gz.

**Stage 1 — one vLLM job (~$1.5).** std k=8, closed k=8 and hint-first
k=8 on all 800 rows, seed 20260828. Pools (pre-defined, identical to
W0/WR): DOC-DEPENDENT failing = 0/8 std ∧ 0/8 closed; HINT-REPAIRABLE
= ≥ 4/8 hint-first. WF rows = ALL doc-dependent ∧ hint-repairable rows
(no sampling), pinned in docs/wikihop_wf_pinned.json before stage 2;
if fewer than 20, UNDERPOWERED (reported, stage 2 not run). Reported:
std/closed/hint accuracies, pool sizes, the hint-repairable rate of the
doc-dependent pool (W0/WR: 27.5%), the memory/reading split (descriptive).

**Stage 2 — the WL job, unchanged, on the WF rows (two shards, ~$5).**
Every pin FROZEN from W0: the L38 gauge (w, b, mean), the L30 write
site, whole-word addressing, no refit of anything. Gold hint-delta at
**1× (pinned) and 2×**, k=4; 3 seeded non-gold at 1× (specificity);
every candidate's δ_X at 2× (the loop); std baseline k=8; hint-first
gold text k=8 (ceiling re-measure); hook counters (a zero aborts).
Seed 20260829.

**Gates (identical to WL).** (a) TEXT CEILING CI > 0. (b) REPLICATION:
at 1× or 2×, gold-δ dP CI > 0 AND specificity CI > 0 → the 11th
prediction replicates on a disjoint frame with a frozen recipe. (c)
LOOP at 2×: gauge-select over all candidates − baseline CI > 0 AND −
SC@8 CI > 0 → **LOOP-CLOSES-ON-NATURAL-DATA, REPLICATED**; (i) only →
LOOP-BEATS-BASELINE-NOT-SC; neither → SELECTOR-FAILS-ON-FRESH-FRAME.
(b) fails → WL-DOES-NOT-REPLICATE (the natural-data claim shrinks to
suggestive). No amendments except on independent-telemetry evidence.
MDE at the expected n ≈ 60–80: repair ≥ ~0.05; loop repair ≥ ~0.06.
Readers: scripts/wikihop_wf_pins.py (stage 1), scripts/wikihop_wl_gates.py
(stage 2).
[Stage 1 launched after registration commit 53d19ba: job-954p4, context
ctx-8d9a2c7b, 2026-09-02 13:58 UTC.]

**Stage 1 landed (job-954p4, 19,200 generations).** Fresh frame: std
0.466 / closed 0.445 / hint-first 0.621; 409 rows at 0/8 std;
DOC-DEPENDENT failing **276**; hint-first on the doc-dependent pool
bimodal (211 at 0/8, 50 at 8/8, 15 between); HINT-REPAIRABLE ∧
doc-dependent = **59 rows (21.4% of the pool; W0/WR 27.5%)**, 39
reading-driven + 20 memory-driven. Not underpowered. WF rows = all 59,
pinned in docs/wikihop_wf_pinned.json (two shards, 30 + 29). Stage 2
launches on them with every pin frozen. [Stage 2 launched after pins
commit 545b645: shard 0 job-87kjh (30 rows), shard 1 job-jgeug (29 rows),
context ctx-299ba896 (W0 pinned npz sha256 d3b7c9ab…, frozen), seed
20260829, 2026-09-02 14:25 UTC.]

### WF — LANDED: HINT-DELTA-TRANSFERS (both rungs) + LOOP-CLOSES-ON-NATURAL-DATA, REPLICATED (job-87kjh + job-jgeug, 2026-09-02)

Fresh frame (seed 20260827; disjoint from W0 and the screening), 59
doc-dependent ∧ hint-repairable rows (21.4% of the 276-row doc-dependent
pool), every pin frozen from W0. (a) TEXT CEILING PASS 0.939 [+0.858,
+0.975]. (b) REPLICATION PASS at both rungs: gold-δ dP **+0.309 [+0.199,
+0.424] at 1×, +0.347 [+0.220, +0.479] at 2×**; specificity +0.308 /
+0.341 (gold under non-gold writes 0.018 / 0.023); delivery audit valid
on all 6,932 fired records; pinned rung 2× on this draw (WL: 1× — no
dose difference on either). (c) LOOP PASS on both readings: gauge-select
over all candidates (25.4/row) at 2× = **0.229** vs baseline 0.017 and
SC@8 0.017 (+0.212 [+0.119, +0.314] on both); oracle 0.364 (63%
recovered; WL 93%); gold argmax 14/59 (chance 0.070); selection signal
+0.86 [+0.09, +1.64]; gauge picks gold on 10 of the 22 rows where the
gold branch repairs. The 11th prediction is confirmed on two disjoint
draws with a frozen recipe; the loop closes on both. The fresh-draw
obligation is DISCHARGED — no further WikiHop replication is owed. Scope
stands: compatible-answer failures. Full numbers: docs/wikihop_wf_summary.md.

### WX. Frozen, row-independent write: is the natural-data lever a property of the representation, or of the per-question hint computation? (registered 2026-09-02, before any data)

**Why.** WL/WF's loop computes each candidate's nudge from a hinted
prompt naming that candidate (~25 extra forwards per question). The
sandbox's closed loop used a FROZEN write — one direction fit on donor
rows, identity supplied by the ADDRESS (which candidate's mentions) —
and it transferred (+0.498, the C4 rider). On WikiHop the class-mean
direction at the address failed everywhere (W1/sweep). WX asks whether
a frozen direction derived from HINT-DELTAS (not the class-mean) at
the candidate's address carries the repair. Expectation stated, no
directional prediction registered: W1 suggests the candidate-specific
content is essential and the frozen direction will underperform; a pass
would make the loop ~25× cheaper and locate the effect in the
representation rather than in the test-time hint.

**Frame.** The 59 WF rows (fresh frame), cross-fit: job A tests shard 1
(29 rows) with shard 0 (30 rows) as donors; job B the reverse. Pinned
in docs/wikihop_wx_pinned.json. Every other pin frozen from W0.

**Frozen write.** In-job, from the donor rows only: δ = mean over all
donor gold-mention positions of (hint-first[gold] state − std state)
at L30; u = δ/|δ|; norm target N = donor mean per-position |δ|. The
write at candidate X's whole-word mentions = u × rung × N with rungs
{1, 2} (the same per-position amplitudes as WL/WF's 1×/2×). The same
vector for every candidate — only the address changes.

**Arms per test row.** std baseline k=8; hint-first gold text k=8
(ceiling); frozen write at the GOLD address k=4 at 1× and 2×; 3 seeded
non-gold addresses (seed 20260830 + row) k=4 at 1× and 2×
(specificity); every candidate's address at 2× k=4 (the loop); W0's L38
gauge under every write; hook counters (a zero aborts). Descriptive:
cosine between u and each test row's own gold δ (one hint forward per
test row), per-position norms.

**Gates.** (a) TEXT CEILING CI > 0. (b) FROZEN ORACLE: at 1× or 2×,
gold-address dP CI > 0 AND specificity (gold-address − mean non-gold-
address correct) CI > 0 → **FROZEN-WRITE-TRANSFERS**; else
**FROZEN-WRITE-FAILS** (the per-candidate content is essential; W1's
null generalizes to hint-derived frozen directions). (c) FROZEN LOOP
(read only if (b) passes): gauge-select − baseline CI > 0 AND
gauge-select − RANDOM-BRANCH expectation CI > 0 (the random-branch
comparator is now a registered reading) → FROZEN-LOOP-CLOSES. MDE at
n = 59: repair ≥ ~0.05. Cost ~$3 (two jobs). Readers:
scripts/wikihop_wl_gates.py (extended with the random-branch reading).
[Launched after registration commit 3c792c9 / tooling f8d56cd: job A
job-tj7za (test shard 1, donors shard 0), job B job-7swth (test shard 0,
donors shard 1), context ctx-ae10e32e, seed 20260830, 2026-09-02 17:10 UTC.]

### WX — LANDED: FROZEN-WRITE-TRANSFERS (pinned 2×) + FROZEN-LOOP-CLOSES (job-tj7za + job-7swth, 2026-09-02)

Cross-fit on the 59 WF rows. Frozen direction = donor mean gold
hint-delta at L30 (30/29 donors; |mean δ| 47–48% of the per-position
norm; cosine 0.68 with test rows' own deltas), same vector at every
candidate's address at N = donor mean per-position |δ| (3,288 / 3,213).
(a) TEXT CEILING PASS 0.932. (b) FROZEN ORACLE PASS at both rungs:
gold-address dP **+0.326 [+0.216, +0.445] at 1×, +0.360 [+0.242,
+0.483] at 2×**; specificity +0.326 / +0.346 (gold under non-gold
addresses 0.017 / 0.031); delivery audit valid on all 6,932 fired
records. Equal to the per-candidate deltas on the same rows (WF +0.309 /
+0.347): the identity is carried by the ADDRESS. (c) FROZEN LOOP PASS:
gauge-select 0.174 vs baseline 0.017 (+0.157 [+0.072, +0.254]) and vs
random branch 0.048 (+0.126 [+0.040, +0.221]); oracle 0.377 (46%
recovered; per-candidate 63%); gold argmax 0.288 (chance 0.070). The
stated expectation (frozen fails) was wrong; exploratory, no
prediction registered; a fresh-draw replication would be owed before
claiming it at WL/WF level. Cost ~$2.2. Full numbers:
docs/wikihop_wx_summary.md.

## Pre-Registered Contamination-Free Bridge (2026-09-02)

### WA. Anonymized-entity WikiHop: does the lever work when memory cannot answer — the sandbox's novel-name regime on natural text?

Registered 2026-09-02, before any data. Motivation: on real WikiHop ~45%
of answers come from parametric memory (closed-book ≈ std), and the
lever's scope was defined behaviorally (hint-repairable). The sandbox
used novel names. WA removes memory by construction and asks whether the
write and the loop still work — the straight line from InAbHyD to real
text. **Registered directional prediction (the program's 12th): on
anonymized WikiHop, the hint-delta write at L30 repairs hint-repairable
failures with candidate specificity (gold-δ dP CI > 0 AND specificity
CI > 0 at 1× or 2×).** The loop and the frozen rider carry no prediction.

**Frame.** 800 new real-entity rows (seed 20260832; disjoint from W0,
screening and WF frames; builder scripts/wikihop_frame.py), anonymized
by scripts/wikihop_anon_frame.py (seed 20260831): every proper-noun
candidate and the query subject (numbers and common-noun candidates
untouched) is renamed consistently across documents / candidate list /
answer / subject with a seeded pseudonym; rows kept only when the gold
is a renamed name and still mentioned in the documents → **536 rows**
(mean 11.9 entities renamed, 15.7 candidates; no original name leaks;
98.4% of candidates mentioned in the documents).
File results/loop_screen/wikihop_anon_input.jsonl.gz.

**Stage 1 — one vLLM job (~$1.5).** std / closed / hint-first, k=8, seed
20260834, on all 536 rows. CONTAMINATION CHECK (reported, pre-named
expectation): closed-book accuracy should collapse toward chance (real
frames: 0.445–0.457) — if it stays above 0.15 the anonymization leaked
and the item is reported as INVALID-FRAME. Pools as before: DOC-DEPENDENT
(0/8 std ∧ 0/8 closed), HINT-REPAIRABLE (≥ 4/8 hint-first); WA rows = all
doc-dependent ∧ hint-repairable rows, capped at 60 by a seeded draw (seed
20260833) if more; UNDERPOWERED if < 20. Reported: std accuracy and the
hint-repairable share on anonymized rows vs real (27.5% / 21.4%).

**Stage 2 — the WL job unchanged (two shards, ~$5), every W0 pin frozen**
(the L38 gauge was fit on real-entity text; whether it still reads
correctness on pseudonym text is part of the test). Gold δ at 1×/2× k=4,
3 seeded non-gold at 1×, every candidate's δ at 2× (the loop), std
baseline k=8, hint-first gold text k=8; hook counters. Seed 20260835.

**Gates.** (a) TEXT CEILING CI > 0. (b) ORACLE (the 12th prediction) →
**HINT-DELTA-TRANSFERS-WITHOUT-MEMORY** / else **WRITE-NEEDS-FAMILIAR-
ENTITIES** (the lever depends on parametric familiarity with the
candidates — a real scope limit). (c) LOOP at 2×: gauge-select −
baseline CI > 0 AND − random branch CI > 0 → LOOP-CLOSES-WITHOUT-MEMORY;
else SELECTOR-FAILS-WITHOUT-MEMORY (the real-text gauge does not read
pseudonym text). **Rider (launched only if (b) passes):** the WX frozen
direction, cross-fit within the WA rows — does the shared direction
survive anonymization? (descriptive, ~$2). MDE at n ≈ 40–60: repair ≥
~0.06. Readers: scripts/wikihop_wf_pins.py (stage 1, --label "item WA",
--max-rows 60), scripts/wikihop_wl_gates.py (stage 2).
[Stage 1 launched after registration commit 5c09028: job-agh5n, context
ctx-7e88dcf3, 2026-09-02 17:56 UTC.]

**Stage 1 landed (job-agh5n, 12,864 generations).** CONTAMINATION CHECK
PASS: closed-book accuracy **0.127** (real frames 0.445–0.457; chance
≈ 0.064; below the pre-named 0.15 ceiling) — memory no longer answers.
std 0.378 (real 0.466), hint-first 0.534. 326 rows at 0/8 std;
DOC-DEPENDENT 288; hint-first on the pool bimodal (216 at 0/8, 63 at
8/8, 9 between); HINT-REPAIRABLE ∧ doc-dependent **68 (23.6%; real
frames 27.5% / 21.4%)**, 50 reading-driven + 10 memory-driven among the
60 drawn (the memory-driven share falls from 34–46% to 17%, as
anonymization should make it). WA rows = 60 (seeded cap, seed
20260833), pinned in docs/wikihop_wa_pinned.json (two shards, 30 + 30).
Stage 2 launches on them with every W0 pin frozen. [Stage 2 launched
after pins commit 341030a: shard 0 job-b5etu, shard 1 job-zcis4, context
ctx-4dfc962f, seed 20260835, 2026-09-02 18:21 UTC.]

### WA — LANDED: HINT-DELTA-TRANSFERS-WITHOUT-MEMORY (pinned 2×) + LOOP-CLOSES-WITHOUT-MEMORY (job-b5etu + job-zcis4, 2026-09-02)

Contamination check PASS (closed-book 0.127). (a) TEXT CEILING PASS
0.921 [+0.794, +0.935]. (b) ORACLE PASS at both rungs — **12th
directional prediction CONFIRMED**: gold-δ dP **+0.365 [+0.256, +0.477]
at 1×, +0.502 [+0.377, +0.625] at 2×**; specificity +0.393 / +0.518
(gold under non-gold writes 0.024 / 0.036); 32/60 rows at 4/4 at 2×;
delivery audit valid on all 4,884 fired records. Larger than on real
text (+0.26 to +0.35). (c) LOOP PASS: gauge-select 0.237 vs baseline
0.052 (+0.185 [+0.094, +0.287]), vs random branch 0.109 (+0.128
[+0.041, +0.226]), vs SC@8 0.067; oracle 0.554 (43% recovered — the
real-text gauge is weaker on pseudonym text; selection signal +1.12
[−0.14, +2.44]); gold argmax 0.233 (chance 0.119). Frozen-direction
rider launches (registered conditional on (b)). Full numbers:
docs/wikihop_wa_summary.md. [Rider launched: job A job-qmt3y (test WA
shard 1, donors shard 0), job B job-y2rmz (reverse), context
ctx-73d0e500, seed 20260836, 2026-09-02 18:55 UTC; pins
docs/wikihop_wa_frozen_pinned.json; reader --frozen.]

### WA rider — LANDED: FROZEN-WRITE-TRANSFERS (pinned 2×) + FROZEN-LOOP-CLOSES on anonymized rows (job-qmt3y + job-y2rmz, 2026-09-02)

Cross-fit on the 60 WA rows. Frozen donor-mean hint-delta direction,
cosine **0.88** (median 0.92) with test rows' own deltas (real text:
0.68). Gold-ADDRESS write dP **+0.367 [+0.263, +0.481] at 1×, +0.650
[+0.535, +0.756] at 2×** (gold correct 0.700 from a 0.050 in-job
baseline); specificity +0.375 / +0.652 (gold under non-gold addresses
0.042 / 0.048); delivery audit valid on all 4,884 fired records. Above
the per-candidate deltas on the same rows (+0.365 / +0.502) and above
the sandbox oracle repairs (+0.24 to +0.39): the largest repair in the
program. Frozen loop: gauge-select 0.254 vs baseline 0.050 (+0.204
[+0.102, +0.317]) and vs random branch 0.117 (+0.137 [+0.034, +0.245]);
oracle 0.700 (36% recovered; the real-text L38 gauge is the limiting
half on pseudonym text). Descriptive rider; no prediction registered.
Full numbers: docs/wikihop_wa_summary.md (rider section).

### WG. Refit the gauge on anonymized text: is the selector's shortfall a distribution mismatch? (registered 2026-09-02, before any data)

**Why.** Across WL/WF/WX/WA the write is stable and the selector is
the variable half: the L38 gauge (fit once on real-entity W0 text,
natural gate 0.776) recovers 93% → 63% → 46% → 43% → 36% of the oracle
as the setting moves away from its fit distribution, bottoming out on
pseudonym text where the write is strongest (+0.650). WG refits the
gauge on anonymized rows and re-reads the frozen loop on the SAME
branches. **Registered directional prediction (the program's 13th):
on the 60 WA rows, the loop selected by a gauge fit on anonymized text
repairs more than the loop selected by the real-text gauge (row-paired
difference in gauge-select correct rate, CI > 0), same branches, same
seeds.**

**Stage 1 — capture + fit (~$1).** W0 capture on all 536 anonymized rows
(std prompt; final-token states at L38/43/48/53, float32). Offline fit
(scripts/wikihop_w0_fit.py --exclude-ids-file): logistic on final-token
states vs std-majority labels from the WA stage-1 grades, 5-fold CV,
**the 60 WA test rows excluded from the fit** (and from donors);
primary layer = argmax CV AUC among {38,43,48,53}; NATURAL GATE CV ≥
0.65 (FAIL → the gauge cannot be fit on pseudonym text; reported, stage
2 still runs with the best layer as descriptive). The anonymized-fit
gauge for all four layers is pinned in a new npz before stage 2.

**Stage 2 — the WA rider re-run (two jobs, ~$2.5).** Identical to the
rider (cross-fit frozen direction, seed 20260836, rungs 1×/2×, loop at
2×) so every branch's generations reproduce; each branch's final-token
state is scored by ALL gauges in one forward: the anonymized-fit gauge
at L38/43/48/53 and the real-text W0 gauge at L38 (the rider's
selector). PRIMARY selector = the anonymized-fit gauge at its pinned
layer. Delivery: hook counters (a zero aborts); generation identity
with the rider is checked (same outputs per branch = same seeds).

**Gates.** (a) NATURAL GATE on anonymized text (stage 1). (b) 13th
prediction: paired (new-gauge loop − real-gauge loop) CI > 0 →
**GAUGE-REFIT-HELPS**; else **SELECTOR-LIMIT-IS-NOT-DISTRIBUTION**
(the shortfall is in what final-token states carry, not in the fit).
(c) Descriptive: loop vs baseline / random branch under each gauge;
fraction of the 0.700 oracle recovered; argmax-gold rate; per-layer
comparison; the real-gauge loop must reproduce the rider's 0.254 (a
consistency check on identical branches). MDE (n = 60, paired): ≈
0.07. Readers: scripts/wikihop_wl_gates.py (per gauge key) +
scripts/wikihop_wg_compare.py.
[Stage 1 launched after registration commit 40521b6 / tooling abb8041:
capture job-xjpm3 (context ctx-7e88dcf3 reused — byte-identical files,
mode by env), 2026-09-02 20:43 UTC; queued at submit.]

**Stage 1 landed (job-xjpm3, 536 rows, 207 s).** Anonymized-text gauge,
fit on 474 rows with the 60 WA test rows held out (202 correct-majority):
5-fold CV AUC **L38 0.821 / L43 0.820 / L48 0.827 / L53 0.814**; primary
= **L48** (argmax); NATURAL GATE PASS (real-text W0 gauge: 0.776 at L38).
Pinned in results/loop_screen/wikihop_wg_pinned.npz (all four layers) +
docs/wikihop_wg_pinned.json. Stage 2 launches: the WA rider re-run with
WH_GAUGE_LAYERS=38,43,48,53 from the anonymized pins (primary L48) and
the real-text L38 gauge as the second npz. [Stage 2 launched after pins
commit abb70e0: job A job-n3i7u (test WA shard 1), job B job-yv7ym
(shard 0), context ctx-09bd2965, seed 20260836 (= the rider's),
2026-09-02 20:57 UTC.]

### WG — LANDED: SELECTOR-LIMIT-IS-NOT-DISTRIBUTION (13th prediction NOT confirmed) (job-n3i7u + job-yv7ym, 2026-09-02)

Stage 1: anonymized-fit gauge CV AUC 0.821/0.820/0.827/0.814 (L38/43/48/
53; primary L48; 60 test rows held out) — natural gate PASS. Stage 2:
identical branches (5,834/5,844 outputs byte-identical to the rider;
gold-address write reproduces +0.650); the real-text gauge reproduces
the rider's 0.254. Anonymized-fit L48 loop **0.308** (44% of the 0.700
oracle; gold argmax 0.350) vs real-text L38 loop 0.254 (36%; 0.233):
row-paired **+0.054 [−0.075, +0.192]** — CI includes 0 → the 13th
prediction is NOT confirmed; verdict as pre-named. All anonymized-fit
gauges beat baseline and random branch (CIs > 0); L53 0.317 (45%).
Reading: the selector's shortfall on steered branches is not a
distribution mismatch of the gauge; non-gold writes also produce
confident-looking final states (selector-write interference). Implied
next registration (not launched): a branch gauge fit on steered
branches. Delivery audit valid on all 4,884 fired records; ~$3.5. Full
numbers: docs/wikihop_wg_summary.md. Program tally: 13 registered
directional predictions, 12 confirmed, 1 refuted (this one), plus the
10th refuted at its positive control for the class-mean direction.

### WB. The branch gauge: fit the selector on steered branches (registered 2026-09-02, before any data)

**Why.** WG showed the selector's shortfall is not distribution: a gauge
fit on unsteered anonymized states (CV 0.83) still ranks steered
branches poorly (44% of the 0.700 oracle), because a wrong-candidate
write also yields a confident-looking final state. WB fits the probe on
the selection task itself: steered branch states from donor rows,
labeled by each branch's own correctness. **Registered directional
prediction (the program's 14th): on the 60 WA rows, the branch gauge
selects correct branches more often than the real-text W0 gauge (the
rider's selector), row-paired difference in gauge-select correct rate
CI > 0, on identical branches.** Secondary (descriptive): vs the
anonymized-fit unsteered gauge (WG, 0.308).

**Branches.** Exactly the WA-rider / WG branches: the 60 WA rows, cross-
fit frozen direction (shard 1 written with shard 0's donor vector and
vice versa), every candidate at 2×, seed 20260836. Branch OUTPUTS and
correctness come from the WG records (results/loop_screen/wikihop_wg_
{a,b}.jsonl; 99.8% byte-identical to the rider). Branch STATES come
from a capture-only re-run of the same jobs (WH_CAPTURE_ONLY=1): one
forward per branch under the identical write, final-token states at
L38/43/48/53 (float32) + per-row baseline states; no generation.
Consistency check (pre-named, hard): the real-text L38 gauge score
recomputed from each captured state must match WG's recorded
`second_L38` score to float tolerance — else EXECUTION-INVALID.

**Fit (offline, scripts/wikihop_wb_fit.py).** Cross-fit by shard: the
gauge that scores shard-1 branches is fit on shard-0 branches and vice
versa. Label = branch correct rate ≥ 0.5 (k=4). Recipe as every gauge
in the program: centered logistic, liblinear C=1.0; layer = argmax
5-fold CV AUC on the donor shard among {38,43,48,53} (pre-named);
BRANCH NATURAL GATE: donor-shard CV AUC ≥ 0.65 (FAIL → reported, the
selection still read). Descriptive: leave-one-row-out fit over all 60
rows (more training branches), and per-layer readings.

**Gates.** (a) consistency (above). (b) BRANCH NATURAL GATE. (c) 14th
prediction: paired (branch-gauge loop − real-text-gauge loop) CI > 0 →
**BRANCH-GAUGE-CLOSES-THE-GAP** (with the fraction of the 0.700 oracle
recovered reported); else **SELECTOR-CEILING** (linear final-token
probes cannot rank steered branches; the loop's limit is
representational). Also reported: loop vs baseline / random branch /
SC@8 under the branch gauge; gold-argmax rate. MDE (paired, n = 60) ≈
0.07. Cost ~$1 (two capture jobs). Readers: scripts/wikihop_wb_fit.py.
[Launched after registration commit be72ffa / tooling d3e8649 + f5b0557:
capture job A job-rm433 (WA shard 1 under shard 0's donor vector), job B
job-cydwg (reverse), context ctx-d1034ab3, seed 20260836, 2026-09-02
22:44 UTC.]

### WB — LANDED: SELECTOR-CEILING (14th prediction NOT confirmed) (job-rm433 + job-cydwg, 2026-09-02)

Consistency hard gate PASS (max |diff| 0.0 on 989 branches). BRANCH
NATURAL GATE PASS (donor CV AUC 0.832 at L48 for test A, 0.836 at L53
for test B; 39 / 27 positive branches). Branch-gauge loop **0.183** (26%
of the 0.700 oracle; gold argmax 0.233) vs real-text unsteered 0.254 vs
anonymized-fit unsteered 0.308: paired branch − real **−0.071 [−0.200,
+0.054]**; branch − anon −0.125 [−0.271, +0.017]. Beats baseline (+0.133
[+0.027, +0.248]) but not random branch ([−0.017, +0.160]). Verdict as
pre-named. Reading: three linear final-token probes land at 0.18–0.31
against a 0.70 oracle on identical branches — the selector's ceiling
for this probe family is representational, not a fitting problem; a
different selection signal (answer log-probability under each write, or
a mention-position probe) would be a new design. State files
sha256 90495ad32920… / 292b23638c6b… recorded in docs/wikihop_wb_summary.md; manifests
committed. Tally: 14 predictions, 12 confirmed, 2 not (13th, 14th).

### WO. The output-level selector: does the model's own acceptance of a write pick the branch? (registered 2026-09-02, before any data)

**Why.** Three linear final-token probes cap the selector at 0.18–0.31
against a 0.70 oracle (WG, WB). The branches' own OUTPUTS carry a
different signal: a right-address write makes the model answer that
candidate (WA rider: gold-address correct 0.70), a wrong-address write
mostly leaves the baseline answer (answers-fired 0.05–0.15). Post-hoc on
the existing branches (recorded in docs/wikihop_wb_summary.md, NOT
registered): selecting by answers-fired first, real-text gauge second,
gave 0.350 vs the real-text gauge's 0.254 on the rider (paired +0.096
[+0.013, +0.183]) and 0.271 vs 0.174 on WX (+0.097 [+0.025, +0.182]); the
output signal alone decides 14/60 rows, and on the rest it restricts
the gauge to branches the model accepted at least once. **Registered
directional prediction (the program's 15th): on a fresh anonymized
draw, the OUTPUT-FIRST selector (argmax over branches of the k=4
answers-fired rate, ties broken by the real-text L38 gauge) selects
correct branches more often than the real-text L38 gauge alone — the
chain's registered selector — row-paired CI > 0, on identical
branches.** Secondary (descriptive, no prediction): vs the anonymized-
fit L48 gauge (post-hoc +0.042 [−0.083, +0.167]); coverage (rows decided
by the output signal alone); output selector vs baseline / random
branch / SC@8; fraction of the oracle recovered.

**Frame.** A new 800-row real draw (seed 20260838; disjoint from every
prior frame), anonymized (seed 20260839) → **507 rows** (no leaks).
Stage 1 (vLLM, ~$1.5): std / closed / hint-first k=8, seed 20260840;
contamination check as WA (closed-book ≤ 0.15); WO rows = doc-dependent
∧ hint-repairable, capped at 60 (seed 20260841); UNDERPOWERED if < 20.
Stage 2 (~$2.5): the WA-rider design unchanged — cross-fit frozen
direction (donors = the other shard), gold at 1×/2×, 3 seeded non-gold
at 1× (specificity), every candidate at 2× with k=4 (the branches), W0's
gauges scored per branch (anonymized-fit L48 primary npz; real-text L38
as second), seed 20260842; hook counters (a zero aborts).

**Gates.** (a) contamination + text ceiling + write consistency
(gold-address dP CI > 0 and specificity CI > 0 at 2× — expected from
WA/rider; a FAIL here is EXECUTION-INVALID for the selector reading).
(b) 15th prediction: paired (output-first − real-text gauge) CI > 0 →
**OUTPUT-SELECTOR-BEATS-GAUGE** (the loop on natural text closes with
one frozen vector and no probe as the primary selector); else
**OUTPUT-SELECTOR-NOT-BETTER**. MDE (paired, n = 60) ≈ 0.07. Reader:
scripts/wikihop_wo_gates.py (--tie-key second_L38, the registered
selector).
[Stage 1 launched after registration commit 9dab0ac: job-dp3e2, context
ctx-89dcecbb, 2026-09-02 23:26 UTC.]

**Stage 1 landed (job-dp3e2, 12,168 generations).** Contamination check
PASS: closed-book **0.113** (WA 0.127; real 0.45). std 0.361, hint-first
0.478; DOC-DEPENDENT 293; hint-repairable ∧ doc-dependent **47 (16.0%;
WA 23.6%)**, 43 reading-driven + 4 memory-driven; all 47 pinned (no cap
needed; ≥ 20 → not underpowered) in docs/wikihop_wo_pinned.json,
cross-fit shards 24 + 23 in docs/wikihop_wo_frozen_pinned.json. MDE at
n = 47 ≈ 0.08 (paired). Stage 2 launches. [Stage 2 launched after pins
commit 9df8e0b: job A job-k76n4 (test shard 1, 23 rows), job B job-6fgby
(test shard 0, 24 rows), context ctx-42367914, seed 20260842, 2026-09-02
23:50 UTC.]

### WO — LANDED: OUTPUT-SELECTOR-BEATS-GAUGE (15th prediction CONFIRMED) (job-dp3e2, job-k76n4 + job-6fgby, 2026-09-03)

Third anonymized frame (507 rows; contamination check PASS, closed-book
0.113); 47 hint-repairable rows, all used. Gate (a) PASS: delivery
audit valid (729 branches), text ceiling 0.891, gold-address frozen
write **+0.386 [+0.255, +0.527] at 1×, +0.657 [+0.524, +0.790] at 2×**,
specificity +0.367 / +0.614 (the WA rider's +0.650 replicates on a
disjoint draw). (b) **15th prediction CONFIRMED**: output-first
selection (answers-fired, real-text gauge tie-break) 0.362 vs the
real-text gauge 0.218, row-paired **+0.144 [+0.032, +0.271]**; 53% of
the 0.686 oracle; gold argmax 0.383 (chance ≈ 0.09); vs baseline
+0.33 [+0.20, +0.47], vs random branch [+0.08, +0.34], vs SC@8 [+0.21,
+0.49]. Secondary: vs the anonymized-fit L48 gauge (0.351, 51%) +0.011
[−0.138, +0.160] — a tie; that gauge transfers across anonymized
frames. Verdict as pre-named. Tally: 15 predictions, 13 confirmed.
Full numbers: docs/wikihop_wo_summary.md. WikiHop chain complete.
