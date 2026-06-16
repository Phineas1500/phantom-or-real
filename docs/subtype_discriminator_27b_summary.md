# Subtype Capture-Ladder Discriminator - Job 457170

Output JSON: `docs/subtype_discriminator_27b.json`
Rows: 16 prepared from 16 manifest rows.

## Capture ladder

| layer | old trio? | row-mean delta norm | position mean | position rms |
| --- | --- | ---: | ---: | ---: |
| 15 | no | 253.957 | 255.344 | 302.946 |
| 20 | no | 861.103 | 859.031 | 1057.628 |
| 25 | no | 2934.598 | 2931.868 | 3577.058 |
| 30 | yes | 4837.992 | 4806.605 | 5319.738 |
| 35 | no | 6012.595 | 6036.044 | 6581.614 |
| 40 | yes | 6602.563 | 6659.200 | 7169.049 |
| 45 | yes | 8221.980 | 8259.903 | 8859.919 |
| 50 | no | 11954.304 | 11945.366 | 12848.308 |
| 53 | no | 15385.652 | 15380.041 | 16549.474 |

Selected off-trio layers: `53,50,35`.

## Causal arms

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| L35_concept_replace | 0.297 | +0.102 [-0.008, +0.258] | baseline |
| L35_random_replace | 0.188 | -0.008 [-0.023, +0.000] | baseline |
| L50_concept_replace | 0.234 | +0.039 [-0.023, +0.141] | baseline |
| L50_random_replace | 0.195 | +0.000 [-0.023, +0.023] | baseline |
| L53_concept_replace | 0.227 | +0.031 [-0.023, +0.117] | baseline |
| L53_random_replace | 0.188 | -0.008 [-0.031, +0.016] | baseline |
| L53_rank4_loo_add | 0.195 | +0.000 [+0.000, +0.000] | baseline |
| baseline | 0.195 | - | none |
| old_trio_full_replace_L30_40_45 | 0.211 | +0.016 [-0.211, +0.227] | baseline |

## Verdict

The capture ladder did its triage job: subtype hint writes are present and very large late in the network, with `L53,L50,L35` selected as off-trio peaks. The causal arms do not land a layer-mismatch repair. `L35_concept_replace` is the only suggestive bump (+0.102 vs baseline; matched concept-minus-random +0.109), but both intervals touch or cross zero; `L50`/`L53` are smaller, the old-trio replay remains null-ish, and `L53_rank4_loo_add` is exactly baseline.

Reading rule outcome: not a splice bug, but also not a clean off-trio layer-mismatch win. The subtype carrier is unresolved outside the tested residual-state concept-position replacement route; treat L35 as a replication target, not a landed claim.
