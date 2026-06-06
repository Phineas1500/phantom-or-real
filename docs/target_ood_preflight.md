# Target/OOD Preflight

Generated: `2026-06-06T04:01:18.329361+00:00`

This preflight audits the low-cost target and OOD extensions for the causal-abstraction program. It uses parsed rows for probe feasibility and does not load large activation tensors.

## Label Feasibility

| source | parsed n | strong+ | weak+ | quality=1+ | strong!=weak | strong!=quality=1 | quality unique |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemma3_27b/infer_property | 10025 | 4610/10025 (46.0%) | 6413/10025 (64.0%) | 4631/10025 (46.2%) | 1803 | 21 | 82 |
| gemma3_27b/infer_subtype | 10737 | 2243/10737 (20.9%) | 4486/10737 (41.8%) | 2279/10737 (21.2%) | 2243 | 36 | 133 |
| qwen35_27b/infer_property | 10997 | 6148/10997 (55.9%) | 10559/10997 (96.0%) | 6148/10997 (55.9%) | 4411 | 0 | 20 |
| qwen35_27b/infer_subtype | 10999 | 4533/10999 (41.2%) | 9352/10999 (85.0%) | 4533/10999 (41.2%) | 4819 | 0 | 18 |

Initial read: `is_correct_weak` is the informative alternate binary target. `quality_score_perfect` is identical to strong correctness for Qwen and nearly identical for Gemma, so it is mostly a sanity check unless we model the graded score directly.

## Height Extrapolation

| source | target | h1/h2 train + | h3/h4 test + | runnable |
| --- | --- | --- | --- | --- |
| gemma3_27b/infer_property | is_correct_strong | 2113/2806 (75.3%) | 2497/7219 (34.6%) | yes |
| gemma3_27b/infer_property | is_correct_weak | 2491/2806 (88.8%) | 3922/7219 (54.3%) | yes |
| gemma3_27b/infer_property | quality_score_perfect | 2117/2806 (75.4%) | 2514/7219 (34.8%) | yes |
| gemma3_27b/infer_subtype | is_correct_strong | 1623/2865 (56.6%) | 620/7872 (7.9%) | yes |
| gemma3_27b/infer_subtype | is_correct_weak | 2201/2865 (76.8%) | 2285/7872 (29.0%) | yes |
| gemma3_27b/infer_subtype | quality_score_perfect | 1634/2865 (57.0%) | 645/7872 (8.2%) | yes |
| qwen35_27b/infer_property | is_correct_strong | 2085/3000 (69.5%) | 4063/7997 (50.8%) | yes |
| qwen35_27b/infer_property | is_correct_weak | 2943/3000 (98.1%) | 7616/7997 (95.2%) | yes |
| qwen35_27b/infer_property | quality_score_perfect | 2085/3000 (69.5%) | 4063/7997 (50.8%) | yes |
| qwen35_27b/infer_subtype | is_correct_strong | 1886/3000 (62.9%) | 2647/7999 (33.1%) | yes |
| qwen35_27b/infer_subtype | is_correct_weak | 2849/3000 (95.0%) | 6503/7999 (81.3%) | yes |
| qwen35_27b/infer_subtype | quality_score_perfect | 1886/3000 (62.9%) | 2647/7999 (33.1%) | yes |

## Main Activation Alignment

| site | available | shape | sidecar rows | source matches | strong mismatches |
| --- | --- | --- | --- | --- | --- |
| gemma3_27b/infer_property/L45/resid_post | yes | [11000, 5376] | 11000 | yes | 0 |
| gemma3_27b/infer_subtype/L45/resid_post | yes | [11000, 5376] | 11000 | yes | 0 |
| qwen35_27b/infer_property/L53/resid_post | yes | [10997, 5120] | 10997 | yes | 0 |
| qwen35_27b/infer_subtype/L53/resid_post | yes | [10999, 5120] | 10999 | yes | 0 |

## Name-Scramble OOD

| model | available | summary |
| --- | --- | --- |
| gemma3_27b | yes | infer_property/natural: auc 0.835, drop 0.141; infer_property/nonce: auc 0.844, drop 0.132; infer_subtype/natural: auc 0.842, drop 0.143; infer_subtype/nonce: auc 0.830, drop 0.152 |
| qwen35_27b | no | No Qwen name-scramble activations were found in the current repo. |

## Recommended Next Runs

| priority | source | split | target | reason |
| --- | --- | --- | --- | --- |
| high | gemma3_27b/infer_property | s1 | is_correct_weak | Weak correctness differs substantially from strong correctness and is class-balanced enough for a raw probe. |
| high | gemma3_27b/infer_property | s3 | is_correct_weak | Weak correctness differs substantially from strong correctness and is class-balanced enough for a raw probe. |
| medium | gemma3_27b/infer_property | height_h12_to_h34 | is_correct_strong | Height extrapolation is class-balanced and directly tests whether the correctness direction is a depth/difficulty proxy. |
| medium | gemma3_27b/infer_property | height_h12_to_h34 | is_correct_weak | Weak-correctness height extrapolation checks whether relaxed validity behaves like strong correctness across depth. |
| high | gemma3_27b/infer_subtype | s1 | is_correct_weak | Weak correctness differs substantially from strong correctness and is class-balanced enough for a raw probe. |
| high | gemma3_27b/infer_subtype | s3 | is_correct_weak | Weak correctness differs substantially from strong correctness and is class-balanced enough for a raw probe. |
| medium | gemma3_27b/infer_subtype | height_h12_to_h34 | is_correct_strong | Height extrapolation is class-balanced and directly tests whether the correctness direction is a depth/difficulty proxy. |
| medium | gemma3_27b/infer_subtype | height_h12_to_h34 | is_correct_weak | Weak-correctness height extrapolation checks whether relaxed validity behaves like strong correctness across depth. |
| high | qwen35_27b/infer_property | s1 | is_correct_weak | Weak correctness differs substantially from strong correctness and is class-balanced enough for a raw probe. |
| high | qwen35_27b/infer_property | s3 | is_correct_weak | Weak correctness differs substantially from strong correctness and is class-balanced enough for a raw probe. |
| medium | qwen35_27b/infer_property | height_h12_to_h34 | is_correct_strong | Height extrapolation is class-balanced and directly tests whether the correctness direction is a depth/difficulty proxy. |
| medium | qwen35_27b/infer_property | height_h12_to_h34 | is_correct_weak | Weak-correctness height extrapolation checks whether relaxed validity behaves like strong correctness across depth. |
| high | qwen35_27b/infer_subtype | s1 | is_correct_weak | Weak correctness differs substantially from strong correctness and is class-balanced enough for a raw probe. |
| high | qwen35_27b/infer_subtype | s3 | is_correct_weak | Weak correctness differs substantially from strong correctness and is class-balanced enough for a raw probe. |
| medium | qwen35_27b/infer_subtype | height_h12_to_h34 | is_correct_strong | Height extrapolation is class-balanced and directly tests whether the correctness direction is a depth/difficulty proxy. |
| medium | qwen35_27b/infer_subtype | height_h12_to_h34 | is_correct_weak | Weak-correctness height extrapolation checks whether relaxed validity behaves like strong correctness across depth. |

## Causal-Abstraction Claim

This artifact is predictive/diagnostic only. It identifies which target variables and OOD splits are runnable before causal or intervention claims are made.
