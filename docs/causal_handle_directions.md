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
