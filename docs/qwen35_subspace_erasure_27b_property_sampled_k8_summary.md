# Qwen 3.5 27B Property Subspace-Erasure Summary

Aggregate of shards `457191`-`457194` (`512` generations; 16 balanced S1 test h3/h4 rows; k=8).

## Arms

| condition | P(strong) | P(weak) | parse fail | dP vs baseline (CI95) |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.352 | 0.906 | 0.008 | - |
| erase_raw | 0.422 | 0.883 | 0.000 | +0.070 [-0.031, +0.188] |
| erase_orthogonal | 0.398 | 0.891 | 0.000 | +0.047 [-0.023, +0.141] |
| erase_gaussian | 0.367 | 0.906 | 0.000 | +0.016 [-0.039, +0.086] |

## Probe Directions

| layer | val AUC | test AUC | train projection std |
| --- | ---: | ---: | ---: |
| L16 | 0.710 | 0.731 | 0.0128 |
| L31 | 0.850 | 0.863 | 0.0381 |
| L40 | 0.896 | 0.907 | 0.0947 |
| L45 | 0.906 | 0.918 | 0.1520 |
| L53 | 0.933 | 0.940 | 0.5555 |

## Verdict

Qwen property replicates the non-necessity side of the Gemma erasure story: ablating the readable raw correctness direction does not degrade behavior (`erase_raw` is +0.070 with CI crossing zero). The matched controls are also non-destructive (`erase_orthogonal` +0.047, `erase_gaussian` +0.016), so this is not the full Gemma control-separation pattern. The safe claim is cross-model support that the raw readout axis is not load-bearing, not that Qwen shows the same perturbation sensitivity profile as Gemma.

Artifacts: `docs/qwen35_subspace_erasure_27b_property_sampled_k8.json` and shard reports `docs/qwen35_subspace_erasure_27b_property_sampled_k8_shard*of4.json`.
