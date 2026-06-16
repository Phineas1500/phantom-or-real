# Qwen Robustness Appendix Table

Purpose: manuscript-facing summary of the Qwen3.5-27B results relative to the Gemma 3 results. This should be used as a current-paper appendix/table, not as a full second mechanistic replication.

## Recommended Current-Paper Claim

Qwen3.5-27B supports the cross-model robustness of the predictive, recognition-vs-generation, and raw-axis non-necessity findings:

- Correctness is strongly linearly readable from activations.
- Metadata baselines are much weaker than the activation readout.
- Public residual Qwen Scope dictionaries leave predictive reconstruction error.
- Local Qwen MLP/transcoder stand-ins move much more signal into reconstruction.
- Recognition-vs-generation rows show the same theme as Gemma, but the rowsets are not matched.
- Qwen property raw-axis erasure does not degrade behavior over 512 generations.
- Qwen steering and full-residual patching remain weak/null under current artifacts, and Qwen does not replicate Gemma's destructive-control erasure pattern.

Use Qwen as robustness evidence for the predictive-causal gap and for raw-axis non-necessity. Do not claim Qwen replicates the Gemma reverse-patching asymmetry or the Gemma destructive-control erasure profile.

## Table A: Predictive Readout and Metadata Controls

| Result | Gemma comparison | Qwen3.5-27B result | Manuscript use |
| --- | --- | --- | --- |
| Raw correctness readout | Gemma 3 27B is the main mechanistic model with strong L45 readouts. | Best Qwen raw layer is L53. S1 AUC: property `0.940`, subtype `0.920`. S3 AUC: property `0.933`, subtype `0.915`. | Cross-model support that correctness is linearly readable. |
| Metadata baselines | Gemma subtype metadata baselines were the weakest part of the activation-over-metadata story. | Qwen metadata baselines are much weaker than raw. Best B0 values: property S1 `0.602`, property S3 `0.601`, subtype S1 `0.675`, subtype S3 `0.677`. | Qwen strengthens the claim that activation signal is not reducible to metadata. |
| Metadata-adjusted raw signal | Gemma raw probes survive metadata controls. | Qwen L45 property raw test AUC `0.918`; metadata-plus-raw AUC `0.918`; raw adds about `+0.317` over name-frequency B0. | Cite as a robustness check, not as a new primary probe layer. |
| LAP / logit-style accessibility | Gemma diagnostics separate readout strength from steerability. | Qwen LAP first-diff margins are near chance: peak AUC about `0.452` for property and `0.514` for subtype. | Supports the raw-vs-accessibility gap: readable does not mean logit-accessible or steerable. |

## Table B: Sparse Dictionary and Dark-Matter Evidence

| Artifact | Qwen site / objective | Qwen result | Interpretation |
| --- | --- | --- | --- |
| Public Qwen Scope residual SAE, L53 L0_100 | Residual reconstruction | S1 sparse AUC property/subtype `0.869`/`0.858`; reconstruction AUC `0.876`/`0.864`; error AUC `0.934`/`0.918`; energy about `0.731`/`0.728`. | Public residual SAE leaves behaviorally predictive signal in reconstruction error, matching the Gemma public-SAE dark-matter pattern. |
| Public Qwen Scope residual SAE, L53 L0_50 | Residual reconstruction | S1 reconstruction AUC `0.840`/`0.843`; error AUC `0.937`/`0.921`; energy about `0.676`/`0.674`. | Lower-energy residual dictionary makes the reconstruction-error gap even clearer. |
| Local Qwen MLP-out dictionary | Local stand-in, not first-party Qwen Scope | S1 target/reconstruction/error AUC property `0.937`/`0.913`/`0.818`; subtype `0.916`/`0.896`/`0.816`; energy about `0.987`. | High-fidelity local component dictionary moves much of the signal from error into reconstruction. |
| Local Qwen transcoder | Local MLP-in-weighted to MLP-out stand-in | S1 reconstruction/error AUC property `0.914`/`0.795`; subtype `0.895`/`0.811`; energy about `0.987`. | Confirms the basis/objective story without relying on public residual artifacts. |
| Local Qwen crosscoder | Local cross-layer stand-in | S1 sparse AUC property/subtype `0.891`/`0.836`; raw concat same layers `0.937`/`0.913`. | Cross-layer sparse stand-in trails raw concat; useful as a limitation, not as a stronger result. |

Manuscript wording should say that Qwen public residual SAEs reproduce the predictive-error pattern, while local stand-ins show that the failure is not inherent to all dictionaries. The right basis, objective, and reconstruction fidelity matter.

## Table C: Recognition-vs-Generation and Trajectory Evidence

| Result | Gemma 3 27B | Qwen3.5-27B | Caveat |
| --- | --- | --- | --- |
| Forced-choice recovery on free-form-wrong rows | `14/16` property h3/h4 hard-foil rows select gold under forced choice. | `43/64` subtype h4 hard-foil rows select gold under forced choice; MCQ accuracy `0.672`, parse failure `0`. | Same theme, not a matched replication. Tasks, heights, row-selection rules, and foil definitions differ. |
| Prompt-only selected-vs-gold margin | Gemma property h3/h4 regenerated selected hypothesis: selected >= gold on `13/13` parsed rows; mean margin `17.318`. | Qwen subtype h4 Stage 1 hard foil: selected >= gold on `14/14` rows; mean margin `12.864`. | Predictive trajectory evidence only. |
| Prompt-only gold-vs-foil margin | Gemma gold >= hard foil on `1/14` rows; mean margin `-11.650`. | Qwen gold >= hard foil on `0/14` rows; mean margin `-12.864`. | Supports early wrong-hypothesis preference, not a causal transition. |
| Final-prefix margin | Gemma gold-vs-foil becomes positive on `8/14`; selected-vs-gold remains positive on `9/13`. | Qwen hard foil remains above gold on `14/14`; gold-vs-foil remains `0/14`. | Qwen selected is the hard foil by construction, so later-prefix comparisons are less diagnostic. |

## Table D: Qwen Intervention and Patching Checks

| Method | Qwen result | Manuscript use |
| --- | --- | --- |
| Raw free-form steering | L53 property probe AUC `0.940`; 4-row smoke had baseline/raw/orthogonal strong accuracy `0.5`, weak accuracy `1.0`, parse failure `0`, and zero paired correctness changes. | Supports a Qwen predictive-causal gap, but sample is small. |
| Answer-property steering | Gold-polarity validation/test AUC `1.000`/`1.000`; 8-row smoke produced no answer-content changes and no strong-correctness flips. | Strong readout still failed as a simple content steering handle. |
| Sparse bundle steering | L45 sparse probe AUC `0.802`; L53 sparse probe AUC `0.869`; 8-row smoke produced zero changes across bundle, shuffled, random, and orthogonal conditions. | Sparse features are predictive but not causal handles under tested steering. |
| Single-feature steering | L53 features `7169`, `23296`, and `4212`, plus random controls, produced zero correctness changes. | Single-feature Qwen result mirrors the null steering theme. |
| Multi-layer raw-direction erasure | Property HF-hook chain over 16 balanced h3/h4 rows and 512 generations: baseline P(strong)=`0.352`, erase_raw=`0.422`, dP=`+0.070` CI [`-0.031`,`+0.188`]; orthogonal dP=`+0.047`, Gaussian dP=`+0.016`; L53 probe AUC `0.940`. | Supports cross-model non-necessity of the raw readout axis; controls are also non-destructive, so the Gemma control-separation profile does not replicate here. |
| Full-residual patching | Clean-to-corrupt and corrupt-to-clean subtype patching were weak/null at tested sites. | Do not claim Qwen replicates the Gemma reverse-patching asymmetry. |

## Suggested Appendix Caption

Qwen3.5-27B reproduces the main predictive pattern from Gemma 3 27B: correctness is linearly readable from activations and remains well above metadata baselines, while public residual sparse dictionaries leave predictive reconstruction error. Local Qwen dictionary stand-ins show that high-fidelity component dictionaries can retain much of the signal, so the sparse-dictionary result is basis- and objective-sensitive rather than a blanket SAE failure. Qwen also supports the recognition-vs-generation theme, but the forced-choice and trajectory rowsets are not matched to Gemma. The Qwen property erasure chain supports raw-axis non-necessity, while steering and patching checks remain weak/null. Qwen is therefore robustness evidence for the predictive-causal gap and readout-axis non-necessity, not a causal replication of the Gemma patching asymmetry or destructive-control erasure profile.

## Conservative Body Text

As a cross-model robustness check, we repeated the main predictive and recognition analyses on Qwen3.5-27B using Qwen Scope residual dictionaries and local dictionary stand-ins. Qwen shows strong raw correctness readouts at L53 (S1 AUC `0.940` for property and `0.920` for subtype), well above metadata baselines. Public residual Qwen Scope reconstructions leave near-raw probes in reconstruction error, while local high-fidelity MLP/transcoder dictionaries shift much of the signal into reconstruction. Qwen also exhibits a recognition-vs-generation gap: on subtype h4 free-form-wrong rows, forced choice selects the gold answer in `43/64` cases, and a 14-row trajectory subset already prefers the hard-foil hypothesis over gold at prompt-only scoring. Finally, Qwen property raw-axis erasure did not degrade behavior over 512 generations (dP=`+0.070`, CI [`-0.031`,`+0.188`]), though matched controls were also non-destructive. These Qwen results support the cross-model predictive pattern and raw-axis non-necessity, but should not be treated as a matched causal replication of the Gemma patching asymmetry or destructive-control erasure profile.

## Do Not Claim

- Do not say Qwen proves the Gemma causal mechanism generalizes.
- Do not say Qwen replicates Gemma destructive controls; it supports raw-axis non-necessity without the same control-separation profile.
- Do not say the Qwen forced-choice result is a direct replication of Gemma `14/16`.
- Do not call local Qwen MLP/transcoder/crosscoder dictionaries first-party Qwen Scope artifacts.
- Do not claim `causally distributed` from current Qwen evidence.
- Do not cite unverified 2026 repair or steering literature in the manuscript text until primary sources are checked.

## Source Artifacts

- `docs/qwen_gemma_experiment_coverage.md`
- `docs/qwen_27b_completion_audit.md`
- `docs/qwen35_27b_b0_summary.json`
- `docs/qwen_scope_raw_probe_27b_l53_s1.json`
- `docs/qwen_scope_raw_probe_27b_l53_s3.json`
- `docs/qwen_scope_raw_probe_metadata_residualization_27b_l45.json`
- `docs/lap_qwen35_27b_infer_property_s1.json`
- `docs/lap_qwen35_27b_infer_subtype_s1.json`
- `docs/qwen35_27b_infer_subtype_h4_hardfoil_forced_choice.json`
- `docs/qwen_prefix_conditioned_margin_trajectory_h4_subset.json`
- `docs/prefix_conditioned_margin_trajectory_comparison_gemma_qwen.json`
- `docs/qwen35_subspace_erasure_27b_property_sampled_k8_summary.md`
- `docs/qwen35_subspace_erasure_27b_property_sampled_k8.json`
