# Subtype Capture-Ladder Discriminator - Job 458377

Output JSON: `docs/subtype_discriminator_27b_l35rep_seedB.json`
Rows: 16 prepared from 16 manifest rows.

## Capture ladder

| layer | old trio? | row-mean delta norm | position mean | position rms |
| --- | --- | ---: | ---: | ---: |
| 30 | yes | 4837.992 | 4806.605 | 5319.738 |
| 40 | yes | 6602.563 | 6659.200 | 7169.049 |
| 45 | yes | 8221.980 | 8259.903 | 8859.919 |

Selected off-trio layers: ``.

## Causal arms

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| baseline | 0.195 | - | none |
| old_trio_full_replace_L30_40_45 | 0.195 | +0.000 [-0.227, +0.227] | baseline |

Reading rule: off-trio concept replacement repair supports layer mismatch; old-trio null plus off-trio null, especially with large capture norms and null random controls, supports insufficiency of this residual-state route rather than a splice bug.
