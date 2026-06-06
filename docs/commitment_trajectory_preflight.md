# Commitment/Recognition Trajectory Preflight

Generated: `2026-06-06T13:13:28.725848+00:00`

Purpose: prepare the commitment/recognition track before launching new GPU jobs. This preflight inventories existing recognition, patching, DAS/AtP, and decode-time artifacts and identifies the missing row/trajectory data needed for the next experiment.

## Current Interpretation

Existing evidence supports a recognition-vs-generation theme and a Gemma-specific forward-null/reverse-disruption asymmetry. It does not yet localize a clean `commitment_state` transition or a causal repair handle. The next work should unify row selection and trajectory variables before more intervention jobs.

## Recognition Evidence

| model | task | height | n | foil | recognition acc. | orig margin | MCQ margin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemma3_27b | infer_property | h3/h4 | 16 | stage1_model_output_hard_foil | 87.5% | -12.045 | 9.930 |
| qwen35_27b | infer_subtype | h4 | 64 | stage1_model_output_hard_foil | 67.2% | -13.344 | 0.996 |

Note: Gemma `14/16` and Qwen `43/64` support the same theme but are not matched replications.

## Natural Patching Snapshot

| model | task | direction | layer | mode | n | recovery | breakage | margin delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma3_27b | infer_property | h1_to_h4 | L35 | clean | 8 | 0.108 | NA | 2.836 |
| gemma3_27b | infer_property | h1_to_h4 | L35 | noise | 8 | 0.036 | NA | 0.948 |
| gemma3_27b | infer_property | h4_to_h1 | L35 | corrupt | 8 | -0.100 | 0.100 | -2.508 |
| gemma3_27b | infer_property | h4_to_h1 | L35 | noise | 8 | 0.015 | -0.015 | 1.200 |
| gemma3_27b | infer_property | h1_to_h4 | L40 | clean | 8 | 0.079 | NA | 1.526 |
| gemma3_27b | infer_property | h1_to_h4 | L40 | noise | 8 | -0.001 | NA | -0.223 |
| gemma3_27b | infer_property | h4_to_h1 | L40 | corrupt | 8 | -0.136 | 0.136 | -2.842 |
| gemma3_27b | infer_property | h4_to_h1 | L40 | noise | 8 | -0.018 | 0.018 | -0.711 |
| gemma3_27b | infer_property | h1_to_h4 | L45 | clean | 8 | 0.046 | NA | 0.161 |
| gemma3_27b | infer_property | h1_to_h4 | L45 | noise | 8 | 0.085 | NA | 3.497 |
| gemma3_27b | infer_property | h4_to_h1 | L45 | corrupt | 8 | -0.120 | 0.120 | -2.214 |
| gemma3_27b | infer_property | h4_to_h1 | L45 | noise | 8 | 0.065 | -0.065 | 2.097 |
| gemma3_27b | infer_property | h1_to_h4 | L50 | clean | 8 | 0.071 | NA | 1.088 |
| gemma3_27b | infer_property | h1_to_h4 | L50 | noise | 8 | 0.115 | NA | 4.345 |
| gemma3_27b | infer_property | h4_to_h1 | L50 | corrupt | 8 | -0.177 | 0.177 | -4.704 |
| gemma3_27b | infer_property | h4_to_h1 | L50 | noise | 8 | -0.023 | 0.023 | 0.227 |
| qwen35_27b | infer_subtype | h1_to_h4 | L35 | clean | 7 | -0.008 | NA | -0.102 |
| qwen35_27b | infer_subtype | h1_to_h4 | L35 | noise | 7 | 0.002 | NA | -0.030 |
| qwen35_27b | infer_subtype | h4_to_h1 | L35 | corrupt | 7 | 0.003 | -0.003 | 0.069 |
| qwen35_27b | infer_subtype | h4_to_h1 | L35 | noise | 7 | 0.000 | -0.000 | 0.005 |
| qwen35_27b | infer_subtype | h1_to_h4 | L40 | clean | 7 | -0.014 | NA | -0.219 |
| qwen35_27b | infer_subtype | h1_to_h4 | L40 | noise | 7 | 0.008 | NA | 0.154 |
| qwen35_27b | infer_subtype | h4_to_h1 | L40 | corrupt | 7 | 0.008 | -0.008 | 0.176 |
| qwen35_27b | infer_subtype | h4_to_h1 | L40 | noise | 7 | -0.000 | 0.000 | 0.028 |
| qwen35_27b | infer_subtype | h1_to_h4 | L45 | clean | 7 | -0.008 | NA | -0.117 |
| qwen35_27b | infer_subtype | h1_to_h4 | L45 | noise | 7 | 0.001 | NA | 0.017 |
| qwen35_27b | infer_subtype | h4_to_h1 | L45 | corrupt | 7 | 0.001 | -0.001 | -0.021 |
| qwen35_27b | infer_subtype | h4_to_h1 | L45 | noise | 7 | -0.005 | 0.005 | 0.050 |

## DAS/AtP Snapshot

| artifact | method | n | margin delta | breakage | vs noise sigma | F->T | T->F |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemma_l45_das_forward | DAS-style low-rank interchange | 8 | -1.029 | NA | -0.604 | 0 | 0 |
| gemma_l45_das_reverse | DAS-style low-rank interchange | 8 | -2.069 | 0.085 | -1.037 | 0 | 0 |
| gemma_l50_das_reverse | DAS-style low-rank interchange | 8 | -1.967 | 0.073 | -0.375 | 0 | 0 |
| gemma_l50_atp_reverse | AtP-style ranking plus exact patch validation | 4 | -0.976 | 0.017 | -0.647 | 0 | 0 |

## Decode Trajectory Snapshot

| artifact | layer | n | baseline acc. | decode z mean | z<0 | prefill/flip summary | claim |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemma_decode_trace | L45 | 8 | 37.5% | -3.859 | 97.6% | -0.186 | prompt-trained z threshold does not separate regenerated-correct and regenerated-wrong decode trajectories |
| gemma_decode_trace | L53 | 8 | 37.5% | -3.269 | 98.4% | -0.238 | prompt-trained z threshold does not separate regenerated-correct and regenerated-wrong decode trajectories |
| gemma_decode_time_correction:conditional_gaussian_pos1sd_zlt0 | L45 | 4 | NA | NA | NA | F->T 0, T->F 0, changed 0 | conditional decode-time injection fired but did not repair correctness |
| gemma_decode_time_correction:conditional_orthogonal_pos1sd_zlt0 | L45 | 4 | NA | NA | NA | F->T 0, T->F 0, changed 0 | conditional decode-time injection fired but did not repair correctness |
| gemma_decode_time_correction:conditional_raw_pos1sd_zlt0 | L45 | 4 | NA | NA | NA | F->T 0, T->F 0, changed 0 | conditional decode-time injection fired but did not repair correctness |

## Missing Pieces

- Existing Gemma and Qwen recognition runs support the same recognition-vs-generation theme but are not matched by model, task, height, or foil distribution.
- Existing decode traces only cover Gemma property, 8 rows, L45/L53, and a prompt-trained correctness projection; there is no Qwen decode trace yet.
- Existing patching is margin-first, not free-form generation repair; discrete false-to-true generation effects remain untested for localized commitment sites.
- Gemma forward/reverse patching uses strict natural h1/h4 pairs sharing full gold hypotheses, not exact same-ontology cross-height pairs.
- DAS-style and AtP-style runs localize weak margin effects but do not isolate a successful low-rank causal repair handle.
- No single canonical row set currently ties together free-form wrong rows, forced-choice recognition, decode trajectories, and patching pairs.

## Recommended Next Steps

1. Build a canonical commitment row-set manifest. Record model, task, height, source row, foil type, free-form correctness, recognition result, and patch-pair membership for Gemma and Qwen.
2. Extend decode trajectory measurement to gold-vs-foil and selected-hypothesis margins. Use the canonical manifest; run Gemma property first, then Qwen subtype if the measurement is informative.
3. Only then consider new patching/generation jobs. A new GPU job should target a specific commitment transition, not repeat broad patching scans.

## Artifact Inventory

| artifact | category | model | task | report n | jsonl rows | path |
| --- | --- | --- | --- | --- | --- | --- |
| gemma_decode_correction | decode_correction | gemma3_27b | infer_property | 16 | 16 | docs/decode_time_correction_27b_l45_property_pilot.json |
| gemma_decode_trace | decode_trace | gemma3_27b | infer_property | 8 | 8 | docs/decode_projection_trace_27b_l45_l53_property_pilot.json |
| gemma_forward_patching | patching | gemma3_27b | infer_property | None | 320 | docs/clean_to_corrupt_patching_27b_property_margin_pilot.json |
| gemma_hardfoil_forced_choice | recognition | gemma3_27b | infer_property | None | 208 | docs/answer_property_margins_27b_l45_polarity_hardfoil.json |
| gemma_l45_das_forward | das | gemma3_27b | infer_property | 8 | 168 | docs/das_subspace_27b_l45_property_clean_to_corrupt_pilot.json |
| gemma_l45_das_reverse | das | gemma3_27b | infer_property | 8 | 168 | docs/das_subspace_27b_l45_property_corrupt_to_clean_pilot.json |
| gemma_l50_atp_reverse | atp | gemma3_27b | infer_property | 4 | 20 | docs/atp_rank_validate_27b_l50_last_prompt_property_corrupt_to_clean_pilot.json |
| gemma_l50_das_reverse | das | gemma3_27b | infer_property | 8 | 168 | docs/das_subspace_27b_l50_last_prompt_property_corrupt_to_clean_pilot.json |
| gemma_reverse_patching | patching | gemma3_27b | infer_property | None | 64 | docs/corrupt_to_clean_patching_27b_property_margin_pilot.json |
| qwen_forward_patching | patching | qwen35_27b | infer_subtype | None | 42 | docs/qwen35_27b_infer_subtype_clean_to_corrupt_patching_margin_pilot.json |
| qwen_h4_hardfoil_forced_choice | recognition | qwen35_27b | infer_subtype | None | 64 | docs/qwen35_27b_infer_subtype_h4_hardfoil_forced_choice.json |
| qwen_reverse_patching | patching | qwen35_27b | infer_subtype | None | 42 | docs/qwen35_27b_infer_subtype_corrupt_to_clean_patching_margin_pilot.json |

## Causal-Abstraction Claim

This preflight is diagnostic only. It organizes existing evidence for `selected_hypothesis`, `gold_vs_foil_margin`, `recognition_correctness`, and `commitment_state`; it does not add a new causal intervention result.
