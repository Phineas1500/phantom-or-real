# Subtype Capture-Ladder Discriminator - Job 458387

Output JSON: `docs/subtype_discriminator_27b_l35rep2_seedA.json`
Rows: 16 prepared from 16 manifest rows.

## Capture ladder

| layer | old trio? | row-mean delta norm | position mean | position rms |
| --- | --- | ---: | ---: | ---: |
| 30 | yes | 4837.992 | 4806.605 | 5319.738 |
| 35 | no | 6012.595 | 6036.044 | 6581.614 |
| 40 | yes | 6602.563 | 6659.200 | 7169.049 |
| 45 | yes | 8221.980 | 8259.903 | 8859.919 |

Selected off-trio layers: `35`.

## Causal arms

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| L35_concept_replace | 0.320 | +0.133 [+0.000, +0.297] | baseline |
| L35_random_replace | 0.219 | +0.031 [+0.000, +0.078] | baseline |
| L35_rank4_loo_add | 0.227 | +0.039 [+0.000, +0.086] | baseline |
| baseline | 0.188 | - | none |
| old_trio_full_replace_L30_40_45 | 0.234 | +0.047 [-0.172, +0.266] | baseline |

Reading rule: off-trio concept replacement repair supports layer mismatch; old-trio null plus off-trio null, especially with large capture norms and null random controls, supports insufficiency of this residual-state route rather than a splice bug.
