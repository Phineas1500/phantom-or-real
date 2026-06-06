# Commitment Row-Set Manifest

Generated: `2026-06-06T13:20:45.020551+00:00`

Purpose: canonicalize the rows used by the commitment/recognition track before launching new GPU jobs. The manifest links source rows, hard-foil forced-choice recognition, decode-trace coverage, and natural patch-pair membership.

## Current Interpretation

The existing row sets are mostly disjoint. Recognition-vs-generation is well supported, but the current artifacts do not yet identify a shared row set where recognition, decode trajectory, and patch-pair evidence can be interpreted together. The next GPU job should extend decode trajectory measurement on manifest-selected recognition rows before more intervention scans.

## Coverage Summary

| model | task | rows | recognition | decode trace | patch rows | recog+decode | recog+patch | decode+patch | all three |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma3_27b | infer_property | 39 | 16 | 8 | 16 | 0 | 0 | 1 | 0 |
| qwen35_27b | infer_subtype | 77 | 64 | 0 | 14 | 0 | 1 | 0 | 0 |

## Recognition Rows

| model | task | heights | n | MCQ correct | MCQ acc. | parse fail | orig margin | MCQ margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma3_27b | infer_property | 3,4 | 16 | 14 | 87.5% | 0.0% | -12.045 | 9.930 |
| qwen35_27b | infer_subtype | 4 | 64 | 43 | 67.2% | 0.0% | -13.344 | 0.996 |

Gemma and Qwen support the same recognition-vs-generation theme, but remain non-matched: Gemma is property h3/h4 with balanced polarity, while Qwen is subtype h4.

## Decode Trace Candidates

Rows below are free-form strong-incorrect, hard-foil MCQ-correct, and not already covered by the decode-trace pilot. They are the preferred next decode-trajectory batch.

| model | task | row | example | h | orig margin | MCQ margin | patch pair? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemma3_27b | infer_property | 3073 | property_h3_00073 | 3 | -20.501 | 13.750 | False |
| gemma3_27b | infer_property | 3290 | property_h3_00290 | 3 | 64.931 | 12.750 | False |
| gemma3_27b | infer_property | 3415 | property_h3_00415 | 3 | -4.034 | 10.000 | False |
| gemma3_27b | infer_property | 4322 | property_h3_01322 | 3 | -14.906 | 14.250 | False |
| gemma3_27b | infer_property | 4675 | property_h3_01675 | 3 | -19.346 | 14.750 | False |
| gemma3_27b | infer_property | 5292 | property_h3_02292 | 3 | -18.613 | 15.250 | False |
| gemma3_27b | infer_property | 6188 | property_h4_00188 | 4 | -28.312 | 11.000 | False |
| gemma3_27b | infer_property | 6327 | property_h4_00327 | 4 | -4.445 | 14.625 | False |
| gemma3_27b | infer_property | 8035 | property_h4_02035 | 4 | -22.326 | 11.500 | False |
| gemma3_27b | infer_property | 8298 | property_h4_02298 | 4 | -24.994 | 11.250 | False |
| gemma3_27b | infer_property | 8874 | property_h4_02874 | 4 | -18.807 | 13.500 | False |
| gemma3_27b | infer_property | 9549 | property_h4_03549 | 4 | -26.689 | 8.000 | False |
| gemma3_27b | infer_property | 10079 | property_h4_04079 | 4 | -12.238 | 12.000 | False |
| gemma3_27b | infer_property | 10714 | property_h4_04714 | 4 | -12.828 | 13.500 | False |
| qwen35_27b | infer_subtype | 6085 | ontology_h4_00085 | 4 | -15.393 | 0.375 | False |
| qwen35_27b | infer_subtype | 6100 | ontology_h4_00100 | 4 | -12.859 | 3.250 | False |
| qwen35_27b | infer_subtype | 6184 | ontology_h4_00184 | 4 | -12.068 | 3.000 | True |
| qwen35_27b | infer_subtype | 6293 | ontology_h4_00293 | 4 | -11.822 | 2.375 | False |
| qwen35_27b | infer_subtype | 6322 | ontology_h4_00322 | 4 | -12.984 | 3.375 | False |
| qwen35_27b | infer_subtype | 6610 | ontology_h4_00610 | 4 | -13.317 | 1.875 | False |
| qwen35_27b | infer_subtype | 6676 | ontology_h4_00676 | 4 | -11.730 | 2.625 | False |
| qwen35_27b | infer_subtype | 6757 | ontology_h4_00757 | 4 | -12.322 | 1.125 | False |
| qwen35_27b | infer_subtype | 6925 | ontology_h4_00925 | 4 | -12.044 | 3.000 | False |
| qwen35_27b | infer_subtype | 6932 | ontology_h4_00932 | 4 | -14.583 | 3.125 | False |
| qwen35_27b | infer_subtype | 6971 | ontology_h4_00971 | 4 | -13.383 | 3.500 | False |
| qwen35_27b | infer_subtype | 7145 | ontology_h4_01145 | 4 | -13.692 | 0.375 | False |
| qwen35_27b | infer_subtype | 7306 | ontology_h4_01306 | 4 | -12.251 | 0.750 | False |
| qwen35_27b | infer_subtype | 7452 | ontology_h4_01452 | 4 | -11.646 | 2.250 | False |
| qwen35_27b | infer_subtype | 7513 | ontology_h4_01513 | 4 | -14.150 | 3.500 | False |
| qwen35_27b | infer_subtype | 7537 | ontology_h4_01537 | 4 | -11.842 | 3.250 | False |
| qwen35_27b | infer_subtype | 7618 | ontology_h4_01618 | 4 | -13.460 | 0.625 | False |
| qwen35_27b | infer_subtype | 7733 | ontology_h4_01733 | 4 | -11.933 | 3.000 | False |
| qwen35_27b | infer_subtype | 8142 | ontology_h4_02142 | 4 | -12.768 | 1.000 | False |
| qwen35_27b | infer_subtype | 8148 | ontology_h4_02148 | 4 | -13.084 | 2.875 | False |
| qwen35_27b | infer_subtype | 8194 | ontology_h4_02194 | 4 | -12.550 | 0.625 | False |
| qwen35_27b | infer_subtype | 8204 | ontology_h4_02204 | 4 | -11.604 | 3.000 | False |
| qwen35_27b | infer_subtype | 8270 | ontology_h4_02270 | 4 | -12.962 | 2.750 | False |
| qwen35_27b | infer_subtype | 8289 | ontology_h4_02289 | 4 | -13.921 | 0.875 | False |
| qwen35_27b | infer_subtype | 8351 | ontology_h4_02351 | 4 | -12.606 | 0.125 | False |
| qwen35_27b | infer_subtype | 8361 | ontology_h4_02361 | 4 | -12.368 | 1.125 | False |
| ... | ... | ... | ... | ... | ... | ... | 17 additional candidates omitted from Markdown; see JSON. |

## Patch Pair Coverage

| model | task | artifact | direction | pair | clean row | clean h | corrupt row | corrupt h | clean recog/trace | corrupt recog/trace |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma3_27b | infer_property | gemma_forward_patching | h1_to_h4 | 0 | 113 | 1 | 6604 | 4 | False/False | False/True |
| gemma3_27b | infer_property | gemma_forward_patching | h1_to_h4 | 1 | 116 | 1 | 6649 | 4 | False/False | False/False |
| gemma3_27b | infer_property | gemma_forward_patching | h1_to_h4 | 2 | 979 | 1 | 8198 | 4 | False/False | False/False |
| gemma3_27b | infer_property | gemma_forward_patching | h1_to_h4 | 3 | 364 | 1 | 9950 | 4 | False/False | False/False |
| gemma3_27b | infer_property | gemma_forward_patching | h1_to_h4 | 4 | 980 | 1 | 7312 | 4 | False/False | False/False |
| gemma3_27b | infer_property | gemma_forward_patching | h1_to_h4 | 5 | 521 | 1 | 6298 | 4 | False/False | False/False |
| gemma3_27b | infer_property | gemma_forward_patching | h1_to_h4 | 6 | 693 | 1 | 9560 | 4 | False/False | False/False |
| gemma3_27b | infer_property | gemma_forward_patching | h1_to_h4 | 7 | 692 | 1 | 6211 | 4 | False/False | False/False |
| gemma3_27b | infer_property | gemma_reverse_patching | h4_to_h1 | 0 | 113 | 1 | 6604 | 4 | False/False | False/True |
| gemma3_27b | infer_property | gemma_reverse_patching | h4_to_h1 | 1 | 116 | 1 | 6649 | 4 | False/False | False/False |
| gemma3_27b | infer_property | gemma_reverse_patching | h4_to_h1 | 2 | 979 | 1 | 8198 | 4 | False/False | False/False |
| gemma3_27b | infer_property | gemma_reverse_patching | h4_to_h1 | 3 | 364 | 1 | 9950 | 4 | False/False | False/False |
| gemma3_27b | infer_property | gemma_reverse_patching | h4_to_h1 | 4 | 980 | 1 | 7312 | 4 | False/False | False/False |
| gemma3_27b | infer_property | gemma_reverse_patching | h4_to_h1 | 5 | 521 | 1 | 6298 | 4 | False/False | False/False |
| gemma3_27b | infer_property | gemma_reverse_patching | h4_to_h1 | 6 | 693 | 1 | 9560 | 4 | False/False | False/False |
| gemma3_27b | infer_property | gemma_reverse_patching | h4_to_h1 | 7 | 692 | 1 | 6211 | 4 | False/False | False/False |
| qwen35_27b | infer_subtype | qwen_forward_patching | h1_to_h4 | 0 | 934 | 1 | 8245 | 4 | False/False | False/False |
| qwen35_27b | infer_subtype | qwen_forward_patching | h1_to_h4 | 1 | 599 | 1 | 6184 | 4 | False/False | True/False |
| qwen35_27b | infer_subtype | qwen_forward_patching | h1_to_h4 | 2 | 947 | 1 | 9747 | 4 | False/False | False/False |
| qwen35_27b | infer_subtype | qwen_forward_patching | h1_to_h4 | 3 | 806 | 1 | 8807 | 4 | False/False | False/False |
| qwen35_27b | infer_subtype | qwen_forward_patching | h1_to_h4 | 4 | 426 | 1 | 6140 | 4 | False/False | False/False |
| qwen35_27b | infer_subtype | qwen_forward_patching | h1_to_h4 | 5 | 959 | 1 | 6526 | 4 | False/False | False/False |
| qwen35_27b | infer_subtype | qwen_forward_patching | h1_to_h4 | 6 | 662 | 1 | 7904 | 4 | False/False | False/False |
| qwen35_27b | infer_subtype | qwen_reverse_patching | h4_to_h1 | 0 | 934 | 1 | 8245 | 4 | False/False | False/False |
| qwen35_27b | infer_subtype | qwen_reverse_patching | h4_to_h1 | 1 | 599 | 1 | 6184 | 4 | False/False | True/False |
| qwen35_27b | infer_subtype | qwen_reverse_patching | h4_to_h1 | 2 | 947 | 1 | 9747 | 4 | False/False | False/False |
| qwen35_27b | infer_subtype | qwen_reverse_patching | h4_to_h1 | 3 | 806 | 1 | 8807 | 4 | False/False | False/False |
| qwen35_27b | infer_subtype | qwen_reverse_patching | h4_to_h1 | 4 | 426 | 1 | 6140 | 4 | False/False | False/False |
| qwen35_27b | infer_subtype | qwen_reverse_patching | h4_to_h1 | 5 | 959 | 1 | 6526 | 4 | False/False | False/False |
| qwen35_27b | infer_subtype | qwen_reverse_patching | h4_to_h1 | 6 | 662 | 1 | 7904 | 4 | False/False | False/False |

## Recommended Next Job

1. Run a Gemma decode-trajectory margin job over the 14 Gemma recognition-gap candidates in this manifest, tracking `gold_vs_foil_margin`, `selected_hypothesis`, and the existing prompt-trained correctness projection at L45/L53.
2. If the Gemma measurement separates regenerated-correct from regenerated-wrong trajectories, run the same measurement on a balanced Qwen subset drawn from the 43 Qwen recognition-gap candidates.
3. Keep patching jobs paused until decode trajectories identify a candidate commitment transition or until the row set is expanded so recognition and patch-pair rows overlap.

## Causal-Abstraction Claim

Diagnostic row manifest only. It links free-form source rows, hard-foil recognition, decode trace status, and patch-pair membership before new commitment-state jobs.

