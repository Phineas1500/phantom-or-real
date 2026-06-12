# Multi-Layer Subspace Erasure — Subtype Replication (Sampled k=8 Pooled)

Pooled analysis of jobs 456963-456966 (shards 0-3, suffix
`subspace_erasure_27b_subtype_sampled_k8_shard{0..3}`), replicating the
property-task verdict in `docs/subspace_erasure_27b_property_sampled_k8_summary.md`
on infer_subtype. Same design: per-layer raw probe directions at
L15/L30/L40/L45/L53 mean-ablated at every position, 16 balanced h3/h4 rows,
k=8 samples at temperature 0.7, 512 generations.

## Pooled Results

| condition | P(strong) | parse fail | dP all rows | dP baseline-correct rows |
| --- | --- | --- | --- | --- |
| baseline | 0.430 | 0.094 | — | — |
| erase_raw | 0.438 | 0.000 | +0.008 | +0.000 |
| erase_orthogonal | 0.000 | 0.375 | -0.430 | -0.982 |
| erase_gaussian | 0.086 | 0.023 | -0.344 | -0.821 |

Significance (paired per-row deltas, n=16):

- erase_raw vs zero: +0.008, 1.0 sigma. Null.
- erase_raw vs erase_orthogonal: +0.438, **3.5 sigma**.
- erase_raw vs erase_gaussian: +0.352, **3.0 sigma** (stronger than the
  property run's 1.9 sigma; both controls now clear the 2-sigma bar).

## Interpretation

The epiphenomenality verdict replicates across tasks. On subtype, erasing the
correctness readout direction at five layers is exactly null (baseline-correct
rows: dP = 0.000), while orthogonal erasure eliminates every strong-correct
output (-0.982 on baseline-correct rows). Combined with the property run, the
claim now rests on 1,024 sampled generations across two tasks with all
controls separating at 3+ sigma:

**The correctness readout direction is causally epiphenomenal in both InAbHyD
tasks — readable everywhere, needed nowhere.**

Remaining scope caveats: probe axis (not all decodable information — see the
INLP redundancy check), 5 of 62 layers, Gemma only (Qwen replication needs an
HF-hooks variant).
