# Next-Paper Causal-Abstraction Dashboard

Purpose: track the program testing whether InAbHyD correctness is linearly readable, sparsely lossy, and possibly causally distributed.

## Shared Causal Variables

`target_concept`, `candidate_hypothesis`, `relation_or_property`, `proof_depth`, `selected_hypothesis`, `gold_vs_foil_margin`, `free_form_correctness`, `recognition_correctness`, `commitment_state`

## Representation Types

`raw_direction`, `sparse_feature`, `reconstruction`, `error_subspace`, `das_subspace`, `patched_residual_state`, `decode_time_correction_state`

## Required Controls

`regenerated_baseline`, `orthogonal_direction`, `matched_gaussian_noise`, `positive_control`, `exact_patch_validation`

## Existing Evidence

| Claim | Variable | Representation | Status | Evidence |
| --- | --- | --- | --- | --- |
| Gemma raw correctness readout survives full name-scrambling with measurable loss. | `free_form_correctness` | `raw_direction` | completed | fixed original L45 probes on scrambled activations: property natural=0.835, property nonce=0.844, subtype natural=0.842, subtype nonce=0.830 |
| Gemma reconstruction-error direction is predictive but not a decode-step repair handle. | `free_form_correctness` | `error_subspace` | completed | error-direction test AUC=0.897; paired strong flips=0 at all tested error-direction strengths |
| Qwen strengthens activation-over-metadata readout and exposes a raw-vs-LAP gap. | `free_form_correctness` | `raw_direction` | completed | Qwen L53 S1 raw AUC property=0.940, subtype=0.920; metadata baselines are much weaker than raw readouts, while LAP margins are near chance |
| Recognition-vs-generation holds across models but is not a matched replication. | `recognition_correctness` | `patched_residual_state` | completed | Gemma: 14/16 property h3/h4 hard-foil rows; Qwen: 43/64 subtype h4 hard-foil rows |

## Track Reports

| Track | Status | Artifact | Note |
| --- | --- | --- | --- |
| Steering-effectiveness diagnostics | artifact_preflight | docs/steering_effectiveness_diagnostics.md | Artifact preflight over existing probe, metadata, and historical steering reports. |
| Intervention preflight | started | docs/intervention_preflight.md | Part 1 gate for regenerated baselines, positive controls, matched noise, paired flips, and parse-failure reporting. |

## Planned Tracks

| Track | Next step | Success condition |
| --- | --- | --- |
| Shared causal-abstraction model | Use the shared variables and report schema for every new experiment. | Every report names target variable, representation, and predictive/causal status. |
| Steering-effectiveness diagnostics | Use completed LAP/logit-lens and metadata-adjusted summaries to choose the next intervention family. | Dashboard records which correctness directions are linearly readable but not directly logit-accessible. |
| Stronger interventions | Add optimized vectors, DAS/distributed interchange, decode-time correction, and AtP* ranking with exact patch validation. | Repairs exceed matched noise by 2 sigma and at least 3 paired false-to-true examples, or nulls have passing positive controls. |
| Dictionary/dark-matter tests | Audit Gemma e2e/KL/Matryoshka/BatchTopK availability, then train local Gemma dictionaries if needed. | Determine whether predictive signal moves from error into reconstruction in the right basis/objective. |
| Commitment and recognition trajectory | Probe and patch selected hypothesis, margin, and commitment variables across decode positions. | Identify whether Gemma reverse disruption localizes to a commitment transition and whether Qwen lacks the same transition. |
| Target and OOD extensions | Run weak/quality-score, name-scramble, and height-extrapolation probe variants. | Separate correctness, parsimony, depth/difficulty, and name-familiarity components. |

## Interpretation Guardrails

- `causally distributed` is a hypothesis, not a settled result.
- If DAS/distributed interventions fail with passing positive controls, use `causally inaccessible under tested methods`.
- Treat `0.03` raw-AUC gap and `0.05` error-AUC reduction as planning heuristics until explicitly approved for manuscript use.
- Keep Qwen local MLP/transcoder/crosscoder dictionaries labeled as local stand-ins, not first-party Qwen Scope artifacts.
- Record model, task, height, row-selection rule, and foil type for every recognition-vs-generation result.
- Verify new 2026 citations from primary sources before adding them to manuscript text.
