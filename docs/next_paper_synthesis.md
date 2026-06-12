# Next-Paper Synthesis

Generated after the Qwen robustness, local-dictionary, target/OOD, trajectory, and prompt-margin gated intervention runs.

## Bottom Line

The strongest current framing is:

**InAbHyD correctness is linearly readable, sparsely lossy under public reconstruction dictionaries, recoverable in better local bases, and causally epiphenomenal as a readout direction: a direct multi-layer necessity test (mean-ablating the per-layer probe direction at every position) leaves behavior statistically unchanged while matched control erasures are destructive.**

The word `distributed` should remain a hypothesis, not a conclusion. As of 2026-06-11 the evidence has moved past `causally inaccessible under tested methods`: the necessity test (jobs 456912, 456915-456918; docs/subspace_erasure_27b_property_sampled_k8_summary.md) supports the stronger positive claim that the correctness readout direction is not load-bearing — erase_raw delta-P=-0.016 (0.4 sigma from zero) vs erase_orthogonal -0.367 (3.4 sigma separation) over 512 sampled generations. This predicts and explains the raw-steering, optimized-vector, DAS, and decode-gate nulls: they intervened on a gauge, not a mechanism. Scope caveats: the erased object is the probe axis, not all linearly decodable correctness information (the INLP check in docs/probe_on_erased_activations_27b_property.json shows retrained probes recover 0.877 AUC after axis erasure — directionally epiphenomenal, informationally redundant); Gemma, 5 of 62 layers. The subtype replication (jobs 456963-456966; docs/subspace_erasure_27b_subtype_sampled_k8_summary.md) is complete and cleaner than the property run: erase_raw delta-P=+0.008 (exactly 0.000 on baseline-correct rows) with orthogonal at 3.5 sigma and Gaussian at 3.0 sigma — the verdict now spans both tasks over 1024 sampled generations.

The companion recognition-state cross-prompt patch (job 456913; docs/recognition_state_patch_27b_property_manifest.json) is a controlled repair null with localization value: donors select gold on 14/14 recognition-gap rows, yet their shared ontology-context encodings transplant no gold preference (F->T=0, disruption equal to matched noise).

The Modal behavioral suite (2026-06-11) then localized the failing variable. On the recognition-gap rows: gold among in-context candidates lifts free-form P(strong) 0.036 -> 0.839; gold is absent from the model's own 16-sample proposal distribution on about two-thirds of rows; self-critique recovers only +0.15; and naming just the target concept repairs at 0.955 (hint-first placement: 1.000), while a wrong-concept hint misdirects to 0.000. The bottleneck is choosing which concept to hypothesize about (`target_concept`), not knowledge, selection, or evaluation. Artifacts: docs/candidates_in_context_27b_property_manifest.json, docs/proposal_distribution_27b_property_manifest.json, docs/self_critique_27b_property_manifest.json, docs/proposal_hints_27b_property_manifest{,_v2}.json.

The hint-state interchange (job 456968; docs/hint_state_interchange_27b_property_manifest.json) converted this into the program's first positive activation-level causal result: patching gold-hint-conditioned context encodings (L30/L40/L45) into the unhinted run lifts P(strong) 0.214 -> 0.714 (+0.500, 8/14 rows false-to-true) against matched noise at 0.036 — a controlled causal repair handle. Qualifiers for the next paper: the position-shuffled donor also repairs (+0.250), so the carrier is partly content rather than position structure, and the wrong-concept misdirection arm is null, so bidirectional control transfers through tokens but not (yet) through activations at these sites. Follow-ups: k>=8 batched rerun, position-band sweep, content-vs-position decomposition.

## Current Paper Integration

Qwen should stay in the current paper as cross-model robustness evidence, not as a full second mechanistic replication.

Recommended current-paper use:

- Add the Qwen robustness appendix/table in `docs/qwen_robustness_appendix_table.md`, showing that Qwen3.5-27B also has strong raw correctness readouts and much weaker metadata baselines.
- Include Qwen recognition-vs-generation evidence with careful non-matched wording: Gemma `14/16` property h3/h4 hard-foil rows and Qwen `43/64` subtype h4 hard-foil rows support the same theme but are not direct replications.
- Include the Qwen prefix-conditioned result as supporting trajectory evidence only: Qwen h4 subtype hard-foil selected-vs-gold is nonnegative on `14/14` rows at checkpoint 0, and gold-vs-hard-foil is nonnegative on `0/14` rows.
- Do not use Qwen to claim the Gemma patching asymmetry generalizes. Qwen patching remains weak/null under current artifacts.
- Keep Qwen local dictionaries labeled as local stand-ins, not first-party Qwen Scope artifacts.

A conservative current-paper sentence:

> Qwen3.5-27B reproduces the main predictive and recognition-vs-generation pattern: correctness remains linearly readable beyond metadata, and recognition-gap rows already prefer the free-form or hard-foil hypothesis over gold at prompt-only scoring. We treat this as cross-model robustness evidence, not a matched causal replication of the Gemma patching results.

## Next-Paper Core Claims

### 1. Correctness Is Linearly Readable

Evidence:

- Gemma raw correctness probes remain strong across the original and name-scrambled settings, with measurable but not catastrophic loss.
- Qwen strengthens the activation-over-metadata claim: Qwen L53 S1 raw AUC is `0.940` for property and `0.920` for subtype, while metadata baselines are much weaker and LAP/logit-style margins are near chance.
- Weak correctness and graded quality are also readable, but they do not replace the strong-correctness target. Weak correctness transfers poorly from h1/h2 to h3/h4, and direct graded quality prediction is dominated by strong-correctness baselines.

Interpretation:

The readout is not a trivial metadata artifact, but it is also not automatically a causal handle. It is best described as a robust predictive state tied to answer quality/correctness.

### 2. Public Sparse Dictionaries Lose Behaviorally Relevant Signal

Evidence:

- Public Gemma Scope residual SAEs at L45 leave stronger probes in reconstruction error than in reconstruction.
- Public MLP-out and public transcoder artifacts narrow the gap somewhat but do not match local dictionaries.
- Local Gemma residual dictionaries move most predictive signal into reconstruction: S1 target/reconstruction/error AUCs are property `0.908/0.890/0.768` and subtype `0.930/0.916/0.828`, with energy around `0.998`.
- Local Gemma MLP-out dictionaries and local MLP-in-to-MLP-out transcoders show the same signal-shift pattern. S1 local MLP-out target/reconstruction/error AUCs are property `0.903/0.877/0.777` and subtype `0.924/0.910/0.854`.
- Qwen local MLP/transcoder dictionaries provide the prior: high-fidelity local dictionaries in the right basis can retain the signal, while public residual SAEs leave predictive reconstruction error.

Interpretation:

The dark-matter story should be sharpened. It is not simply that sparse dictionaries cannot preserve the signal; rather, reconstruction objective, capacity, component basis, and local training recipe matter. Public reconstruction artifacts can discard behaviorally predictive directions, but local high-fidelity dictionaries can recover much of them.

### 3. Recognition and Generation Diverge Across Models

Evidence:

- Gemma forced-choice recognition remains intact on `14/16` free-form-wrong hard-foil rows.
- Qwen has `43/64` subtype h4 hard-foil rows where original free-form is strong-wrong but forced choice selects gold.
- Gemma prefix-conditioned trajectory: at prefix 0, selected-vs-gold is nonnegative on `13/13` parsed rows, while gold-vs-hard-foil is nonnegative on only `1/14` rows.
- Qwen prefix-conditioned trajectory: at prefix 0, hard-foil selected-vs-gold is nonnegative on `14/14` rows, while gold-vs-hard-foil is nonnegative on `0/14` rows.

Interpretation:

The key recognition-vs-generation result is not a late decode transition. The wrong/free-form hypothesis is already preferred under prompt-only scoring on the recognition-gap rowsets. This supports a deployment or selection gap: the model can recognize gold under forced choice while free-form generation remains aligned with a wrong hypothesis.

### 4. Better Monitors Still Do Not Yield Raw-Direction Repair

Evidence:

- The old raw-projection decode gate was not calibrated: `z < 0` fired on nearly every decode trajectory and did not separate regenerated-correct from regenerated-wrong outputs.
- The prompt-margin monitor is cleaner: `gold_vs_foil_logprob_margin < -15` triggers `8/11` regenerated-wrong and `0/3` regenerated-correct Gemma manifest rows.
- The prompt-margin gated L45 raw intervention still yields a controlled null: raw `F->T=0`, `T->F=0`; orthogonal `F->T=0`; matched Gaussian `F->T=0` over 14 paired rows.
- Raw and orthogonal interventions remove the one baseline parse failure, but they do not make any row strong-correct.

Interpretation:

The failure is not just bad gating. Even with a cleaner prompt-margin gate, the raw L45 correctness direction does not act as a repair vector. This materially strengthens the predictive-but-not-causal claim.

### 5. DAS-Style and Patch/Localization Evidence Is Still Mostly Negative

Evidence:

- L45 clean-to-corrupt DAS-style interchange does not repair: DAS `F->T=0`, exact source `F->T=0`, matched Gaussian `F->T=0`.
- L45/L50 reverse DAS-style interchange gives weak margin disruption but no discrete DAS true-to-false flips.
- L50 exact reverse patch gives one true-to-false flip and stronger mean disruption than low-rank DAS, but this is not a clean distributed repair handle.
- AtP-style estimates track exact patch deltas at L50, but exact patch validation remains weak.

Interpretation:

This evidence does not justify the positive claim `causally distributed`. It supports the more conservative claim that the tested 1D, low-rank, and localized interventions do not find a reliable causal handle.

## Revised Thesis

Use this as the next-paper thesis (updated 2026-06-11 after the necessity test):

**Correctness in ontology reasoning is robustly decodable but causally epiphenomenal as a readout. It is linearly readable across Gemma and Qwen, lossy under public sparse reconstruction dictionaries, partly recoverable in local high-fidelity bases, and removable without behavioral cost: multi-layer erasure of the readout direction leaves task behavior statistically unchanged while matched control erasures are destructive. The causally potent variable is hypothesis selection, computed during prompt processing, whose state is not carried by context-token encodings.**

Shorter version:

**Correctness is readable everywhere and needed nowhere: the probe reads an epiphenomenal evaluation of a selection process that the tested interventions never touched.**

Avoid:

- `Correctness is causally distributed` unless DAS or another distributed intervention actually repairs or disrupts behavior with controls.
- `SAEs are useless`; the local dictionary results show the basis/objective story is more nuanced.
- `Qwen replicates Gemma causally`; Qwen currently supports predictive and recognition-vs-generation robustness, not the Gemma causal asymmetry.

## What Is Done Enough

These tracks are mature enough to stop spending GPU time unless a concrete manuscript gap appears:

- Qwen robustness against Gemma.
- Broad prefix-conditioned trajectory measurement.
- Raw-direction decode gates, including the calibrated prompt-margin gate.
- Local Gemma dictionary/dark-matter contrast.
- Weak correctness, height extrapolation, and graded quality-score target checks.

## Remaining Work Worth Doing

### Highest Priority: Write the Story

Turn this synthesis into paper edits:

1. Fold `docs/qwen_robustness_appendix_table.md` into the current paper as a Qwen robustness appendix/table.
2. Next-paper outline around causal abstraction and the predictive-causal gap.
3. A claims table with columns: claim, models, rowset, representation, predictive evidence, causal evidence, caveat.
4. A negative-results table separating CAA/raw steering, optimized vectors, DAS, AtP/exact patching, raw-z decode correction, and prompt-margin gated decode correction.

### Optional Experiment: Genuine Trajectory-Level Repair

Only run more A40 jobs if we implement a different intervention family, not another raw-direction gate.

`docs/causal_handle_directions.md` ranks the candidate families and argues the current nulls only establish inaccessibility under single-site, single-direction, additive, low-power interventions on the correctness summary. Its top two candidates — multi-layer subspace erasure (a necessity test with no null outcome) and recognition-state cross-prompt patching (donor activations for repair demonstrably exist on the same rows) — qualify as different families under this rule.

A worthwhile next experiment would need to be qualitatively different:

- A decode-time method that edits trajectory state, rollback/cache state, or candidate continuation policy rather than adding the same raw correctness vector at every step.
- A monitor trained on decode trajectory outcomes rather than prompt residual z or prompt gold-vs-foil margins alone.
- Same controls as before: regenerated baseline, orthogonal, matched Gaussian, positive control, paired false-to-true threshold.

Decision rule:

- If it repairs at least 3 paired false-to-true examples and exceeds matched noise by at least 2 sigma, pivot toward a positive trajectory-level repair story.
- If it fails with controls, keep the final thesis as causal inaccessibility under tested methods.

### Optional Manuscript-Control Work

Run only if the current paper needs it:

- Qwen name-scramble artifacts if cross-model OOD controls become central.
- Residual graded-quality-by-height analysis if we want a parsimony subsection.
- A small capacity-control dictionary run if reviewers ask whether local dictionary success is just overcomplete reconstruction.

## Final Recommendation

Stop broad experimentation for now. The next unit of work should be manuscript-facing: fold the completed Qwen robustness appendix/table into the current paper and draft a concise next-paper outline from this synthesis. The only experimental branch worth reopening is a genuinely new trajectory-level repair method with controls.
