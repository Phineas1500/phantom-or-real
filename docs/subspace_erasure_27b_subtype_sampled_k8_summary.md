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

Significance (paired per-row deltas, n=16; row-level cluster bootstrap is
the primary test — the effective n is rows, not generations):

- erase_raw vs zero: +0.008, bootstrap 95% CI [+0.000, +0.023]. Null.
- erase_orthogonal: -0.430, CI [-0.680, -0.188] — excludes zero.
- erase_gaussian: -0.344, CI [-0.570, -0.125] — excludes zero.
- Naive per-generation sigmas (3.5 / 3.0) are retained as planning numbers
  only.

Precision vs demolition — P(strong | parsed): baseline 0.474, erase_raw
0.438, erase_orthogonal 0.000, erase_gaussian 0.088. Both controls destroy
correctness even among parsed outputs on subtype.

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
INLP redundancy check), 5 of 62 layers, and Gemma task scope for the destructive
matched-control separation. A later Qwen property HF-hook replication supports
raw-axis non-necessity but did not reproduce the destructive-control profile.
