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
| Gemma L45 local MLP-out dictionary shifts much of the correctness signal into reconstruction and weakens error probes. | `free_form_correctness` | `reconstruction` | completed | job=456717; local top-k MLP-out dictionary S1 energy property=0.994, subtype=0.995; target/reconstruction/error AUCs: property=0.903/0.877/0.777, subtype=0.924/0.910/0.854; this is the Gemma analogue of the Qwen local-dictionary prior, though error remains nontrivial |
| Gemma local dictionary signal-shift is S3-robust and also appears under a local MLP-in-to-MLP-out transcoder. | `free_form_correctness` | `reconstruction` | completed | jobs=456718,456719; S3 MLP-out target/reconstruction/error AUCs: property=0.900/0.879/0.767, subtype=0.924/0.909/0.869; S1 local transcoder target/reconstruction/error AUCs: property=0.903/0.884/0.772, subtype=0.924/0.908/0.841; reconstruction energy stays about 0.994 in both cases |
| Gemma local residual dictionary closes the public-SAE dark-matter gap, and the signal-shift is S3-robust. | `free_form_correctness` | `reconstruction` | completed | jobs=456721,456725; local top-k residual dictionary S1 target/reconstruction/error AUCs: property=0.908/0.890/0.768, subtype=0.930/0.916/0.828; S3 target/reconstruction/error AUCs: property=0.900/0.883/0.747, subtype=0.930/0.919/0.847; energy stays about 0.998. Public Gemma Scope residual SAEs at L45 leave stronger error probes than reconstruction probes, while local residual, local MLP-out, and local transcoder dictionaries all move most signal into reconstruction. |
| Public Gemma Scope MLP/transcoder artifacts do not reproduce the local-dictionary signal shift. | `free_form_correctness` | `reconstruction` | completed | jobs=456723,456724 plus existing exact transcoder component reports; public MLP-out hook 16k reconstructs only moderately (energy property=0.570, subtype=0.554) and leaves stronger error probes than reconstruction probes: property recon/error=0.815/0.894, subtype=0.879/0.912. Plain `mlp_out` alignment is a mismatch with negative energy and high error probes. Public exact transcoders partially improve full-output AUC but remain below local dictionaries and have much lower full-output energy. |
| Optimized gold-continuation vector is not a Gemma decode-step repair handle. | `free_form_correctness` | `raw_direction` | completed | job=456693; positive control=passed; optimized F->T max=0, optimized changed max=0, control changed max=1 |
| Gemma L45 decode-time correction gate fired but did not repair correctness. | `free_form_correctness` | `decode_time_correction_state` | completed | job=456708; conditional raw +1sd gate z<0: F->T=0, T->F=0, baseline accuracy=1/4; gate fired on 83-94 of 96 forwards, so prompt-trained z<0 is not a calibrated decode-time trigger; raw and Gaussian controls each introduced one parse failure |
| Gemma L45/L53 baseline decode traces confirm prompt-trained z is not a decode-time correctness gate. | `commitment_state` | `raw_direction` | completed | job=456715; 8 h3/h4 rows, baseline accuracy=3/8, parse failures=0; L45 decode z mean=-3.86 with 97.6% below 0, L53 decode z mean=-3.27 with 98.4% below 0; regenerated-correct rows were not separated from regenerated-wrong rows |
| Gemma L45 low-rank DAS-style interchange is not a repair handle in the first preflight. | `gold_vs_foil_margin` | `das_subspace` | completed | job=456696; clean-to-corrupt L45 last_prompt ranks 1/2/4: DAS F->T=0, exact source F->T=0, matched Gaussian F->T=0; mean DAS margin deltas=-0.36/-1.04/-1.03 |
| Gemma L45 low-rank DAS-style reverse interchange gives weak margin disruption but no discrete breakage. | `gold_vs_foil_margin` | `das_subspace` | completed | job=456697; corrupt-to-clean L45 last_prompt ranks 1/2/4: DAS true-to-false=0, mean breakage=0.079/0.074/0.085 vs exact source=0.120 and matched Gaussian=0.016/0.022/-0.023; not a 2-sigma DAS result |
| Gemma L50 reverse exact patch is stronger, but low-rank DAS-style interchange still does not isolate the causal handle. | `gold_vs_foil_margin` | `das_subspace` | completed | job=456698; corrupt-to-clean L50 last_prompt: exact source true-to-false=1, mean breakage=0.177; DAS true-to-false=0 at ranks 1/2/4, mean breakage=0.055/0.055/0.073; only 0.4-0.6 sigma beyond matched Gaussian |
| Gemma L50 AtP-style truncated-gradient estimates track exact patch deltas but the site remains weak. | `gold_vs_foil_margin` | `patched_residual_state` | completed | job=456705; L50 last_prompt 4-pair reverse checkpoint: approx-vs-exact delta correlation r=0.97, exact true-to-false=0, source mean breakage=0.017 vs matched Gaussian=-0.021; L45/L35 tail-gradient checkpoints OOMed on A40 |
| Qwen strengthens activation-over-metadata readout and exposes a raw-vs-LAP gap. | `free_form_correctness` | `raw_direction` | completed | Qwen L53 S1 raw AUC property=0.940, subtype=0.920; metadata baselines are much weaker than raw readouts, while LAP margins are near chance |
| Weak correctness is readable but much less height-stable than strong correctness. | `free_form_correctness` | `raw_direction` | validated | full-C/bootstrap target/OOD pass: weak-correctness S1/S3 AUCs Gemma property=0.842/0.843, Gemma subtype=0.801/0.772, Qwen property=0.806/0.774, Qwen subtype=0.857/0.873; weak h1/h2 -> h3/h4 AUCs are 0.623-0.694, while strong height AUCs are 0.749/0.754 for Gemma and 0.863/0.818 for Qwen |
| Binary perfect-quality is not a distinct target from strong correctness in current 27B artifacts. | `free_form_correctness` | `raw_direction` | diagnostic_completed | target/OOD preflight: `quality_score == 1.0` is exactly identical to strong correctness for Qwen and differs from Gemma strong correctness on only 21 parsed property rows and 36 parsed subtype rows; a real Occam/parsimony test needs graded quality-score modeling |
| Recognition-vs-generation holds across models but is not a matched replication. | `recognition_correctness` | `patched_residual_state` | completed | Gemma: 14/16 property h3/h4 hard-foil rows; Qwen: 43/64 subtype h4 hard-foil rows |

## Track Reports

| Track | Status | Artifact | Note |
| --- | --- | --- | --- |
| Steering-effectiveness diagnostics | artifact_preflight | docs/steering_effectiveness_diagnostics.md | Artifact preflight over existing probe, metadata, and historical steering reports. |
| Intervention preflight | started | docs/intervention_preflight.md | Part 1 gate for regenerated baselines, positive controls, matched noise, paired flips, and parse-failure reporting. |
| Positive-control steering gate | passed | docs/positive_control_format_gemma3_27b_l45.json | Gemma 3 27B L45 casing/output-format control passed and is reused for raw/error/optimized L45 intervention interpretation. Verbosity gate failed by length saturation. |
| DAS-style subspace interchange preflight | completed | docs/das_subspace_27b_l45_property_clean_to_corrupt_pilot.json; docs/das_subspace_27b_l45_property_corrupt_to_clean_pilot.json; docs/das_subspace_27b_l50_last_prompt_property_corrupt_to_clean_pilot.json | Clean-to-corrupt L45 did not repair; reverse L45/L50 showed weak low-rank margin breakage but no DAS true-to-false flips. L50 exact source patch produced one true-to-false flip. |
| Decode-time correction preflight | completed | docs/decode_time_correction_27b_l45_property_pilot.json | Conditional raw-projection injection ran successfully but did not change strong correctness; gate calibration failed because z<0 fired on nearly every decode forward. |
| Decode projection trace calibration | completed | docs/decode_projection_trace_27b_l45_l53_property_pilot.json | Baseline L45/L53 decode trajectories remain mostly below the prompt-trained z<0 threshold for both regenerated-correct and regenerated-wrong outputs, so future decode-time correction needs a decode-trained monitor or a different trigger variable. |
| Dictionary/dark-matter local MLP-out pilot | completed | docs/gemma3_local_mlpout_reconstruction_probe_27b_l45_w4096_k64_s1.json | Gemma L45 local MLP-out dictionary preserves most target AUC in reconstruction and reduces error-probe AUC relative to target, especially for property. |
| Dictionary/dark-matter S3 and local transcoder pilots | completed | docs/gemma3_local_mlpout_reconstruction_probe_27b_l45_w4096_k64_s3.json; docs/gemma3_local_transcoder_component_probe_27b_l45_w4096_k64_s1.json | S3 repeats the MLP-out pattern, and the local MLP-in-weighted to MLP-out transcoder gives similar reconstruction/error AUCs. |
| Dictionary/dark-matter local residual contrast | completed | docs/gemma3_local_resid_reconstruction_probe_27b_l45_w4096_k64_s1.json; docs/gemma3_local_resid_reconstruction_probe_27b_l45_w4096_k64_s3.json | Local residual dictionary also moves most predictive signal into reconstruction on S1 and S3, unlike public residual Gemma Scope SAEs; this points to artifact objective/capacity/training recipe as a major part of the dark-matter gap. |
| Public MLP-out reconstruction diagnostics | completed | docs/sae_reconstruction_probe_27b_l45_mlp_out_hook_16k_s1.json; docs/sae_reconstruction_probe_27b_l45_mlp_out_16k_s1.json | Hook-aligned public MLP-out SAE improves over public residual reconstruction on subtype but still leaves stronger error probes; plain `mlp_out` is an alignment mismatch. |
| Public exact transcoder component diagnostics | completed | docs/transcoder_component_probe_27b_l45_16k_affine_exact_s1.json; docs/transcoder_component_probe_27b_l45_262k_affine_exact_s1.json; docs/transcoder_component_probe_27b_l45_262k_big_affine_exact_top512_s1.json | Public transcoders narrow the gap somewhat, especially the 262k big top-512 run, but full-output AUC/energy remain below the local dictionaries and do not move the signal cleanly out of error. |
| Target/OOD preflight and weak-target probes | validated | docs/target_ood_preflight.md; docs/target_ood_probe_findings.md; docs/target_ood_raw_probe_27b_weak_main_fullc_bootstrap.json; docs/target_ood_raw_probe_27b_strong_height_fullc_bootstrap.json | Full-C/bootstrap validation confirms weak correctness is readable on S1/S3 but generalizes poorly from h1/h2 to h3/h4; strong correctness height extrapolation is materially better, especially for Qwen. Binary perfect-quality mostly collapses to strong correctness. |

## Dictionary Contrast Snapshot

| Artifact | Site / objective | Split | Property AUC target / recon / error | Subtype AUC target / recon / error | Energy | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| Public Gemma Scope residual 16k | L45 residual SAE | S1 | n/a / 0.786 / 0.894 | n/a / 0.877 / 0.916 | 0.948 | predictive signal remains stronger in error than reconstruction |
| Public Gemma Scope residual 262k | L45 residual SAE | S1 | n/a / 0.806 / 0.897 | n/a / 0.870 / 0.915 | 0.954-0.955 | larger public SAE improves energy only slightly and still leaves predictive error |
| Public Gemma Scope MLP-out 16k | L45 `mlp_out_hook` SAE | S1 | n/a / 0.815 / 0.894 | n/a / 0.879 / 0.912 | 0.554-0.570 | hook-aligned MLP-out reconstruction helps, but error remains stronger than reconstruction |
| Public Gemma Scope MLP-out 16k | L45 plain `mlp_out` SAE | S1 | n/a / 0.617 / 0.895 | n/a / 0.740 / 0.915 | negative | alignment mismatch; reconstruction is not meaningful while error stays predictive |
| Public Gemma Scope transcoder 16k | L45 exact affine full output | S1 | n/a / 0.854 / 0.857 | n/a / 0.890 / 0.889 | 0.638-0.639 | full output roughly matches error but does not exceed local dictionaries |
| Public Gemma Scope transcoder 262k | L45 exact affine full output | S1 | n/a / 0.862 / 0.864 | n/a / 0.897 / 0.888 | 0.661-0.672 | larger public transcoder narrows but does not close the local gap |
| Public Gemma Scope transcoder 262k big | L45 exact affine top-512 full output | S1 | n/a / 0.863 / 0.848 | n/a / 0.888 / 0.878 | 0.797-0.802 | best public transcoder energy so far; still below local reconstruction AUCs |
| Local residual top-k | L45 residual local AE | S1 | 0.908 / 0.890 / 0.768 | 0.930 / 0.916 / 0.828 | 0.998 | local recipe captures most correctness signal in reconstruction |
| Local residual top-k | L45 residual local AE | S3 | 0.900 / 0.883 / 0.747 | 0.930 / 0.919 / 0.847 | 0.998 | residual local recipe is split-robust |
| Local MLP-out top-k | L45 MLP-out local AE | S1 | 0.903 / 0.877 / 0.777 | 0.924 / 0.910 / 0.854 | 0.994-0.995 | right component basis also captures most signal |
| Local transcoder top-k | L45 MLP-in-weighted -> MLP-out | S1 | 0.903 / 0.884 / 0.772 | 0.924 / 0.908 / 0.841 | 0.994-0.995 | similar to local MLP-out AE; no decisive transcoder advantage yet |

## Planned Tracks

| Track | Next step | Success condition |
| --- | --- | --- |
| Shared causal-abstraction model | Use the shared variables and report schema for every new experiment. | Every report names target variable, representation, and predictive/causal status. |
| Steering-effectiveness diagnostics | Use completed LAP/logit-lens and metadata-adjusted summaries to choose the next intervention family. | Dashboard records which correctness directions are linearly readable but not directly logit-accessible. |
| Stronger interventions | Optimized-vector, L45/L50 DAS-style, L50 AtP-style, first decode-time correction, and L45/L53 decode-trace calibration passes are complete; next decode-time work should train a decode-trajectory monitor before any new gated intervention. | Repairs exceed matched noise by 2 sigma and at least 3 paired false-to-true examples, or nulls have passing positive controls and exact-patch validation. |
| Dictionary/dark-matter tests | Gemma L45 local MLP-out S1/S3, local residual S1/S3, local MLP-in-weighted -> MLP-out transcoder S1, and public residual/MLP-out/transcoder comparisons are complete; next either close this track for now or test a denser/local-public hybrid if we want a final capacity-control. | Determine whether predictive signal moves from error into reconstruction in the right basis/objective. |
| Commitment and recognition trajectory | Probe and patch selected hypothesis, margin, and commitment variables across decode positions. | Identify whether Gemma reverse disruption localizes to a commitment transition and whether Qwen lacks the same transition. |
| Target and OOD extensions | Full-C/bootstrap weak-correctness and height-extrapolation validation is complete; next add a graded quality-score regression/ordinal probe only if we want a true parsimony target, or generate Qwen name-scramble artifacts if cross-model OOD controls become central. | Separate correctness, parsimony, depth/difficulty, and name-familiarity components. |

## Interpretation Guardrails

- `causally distributed` is a hypothesis, not a settled result.
- If DAS/distributed interventions fail with passing positive controls, use `causally inaccessible under tested methods`.
- Treat AtP-style rankings as localization hypotheses until exact patch validation passes; full 27B gradients before L50 are not feasible on current A40 jobs without further memory work.
- Treat prompt-residual probe z-scores as uncalibrated for decode-time gating until baseline decode traces separate regenerated correct and incorrect trajectories.
- Treat `0.03` raw-AUC gap and `0.05` error-AUC reduction as planning heuristics until explicitly approved for manuscript use.
- Use full-C/bootstrap target/OOD reports for numerical claims; keep fixed-`C=1.0` target/OOD reports as historical planning artifacts only.
- Keep Qwen local MLP/transcoder/crosscoder dictionaries labeled as local stand-ins, not first-party Qwen Scope artifacts.
- Keep Gemma local dictionaries labeled as local stand-ins for basis/objective tests, not as Gemma Scope 2 first-party artifacts.
- Record model, task, height, row-selection rule, and foil type for every recognition-vs-generation result.
- Verify new 2026 citations from primary sources before adding them to manuscript text.
