# Multi-Layer Subspace Erasure — Sampled k=8 Pooled Results

Pooled analysis of jobs 456915-456918 (shards 0-3 of
`scripts/stage2_subspace_erasure.py`, suffix
`subspace_erasure_27b_property_sampled_k8_shard{0..3}`), confirming the
smoke result in `docs/subspace_erasure_27b_property_smoke.json` (job 456912).

## Design

- Gemma 3 27B, infer_property, S1 test split, 16 balanced h3/h4 rows
  (8 originally strong-correct, 8 strong-incorrect), disjoint 4-row shards.
- Per-layer raw correctness probe directions at L15/L30/L40/L45/L53
  (test AUC 0.786 / 0.856 / 0.890 / 0.897 / 0.902), mean-ablated at every
  position during prompt processing and decode.
- Conditions: regenerated baseline, erase_raw, erase_orthogonal (per-layer
  orthogonal to the probe direction), erase_gaussian (per-layer random unit).
- k=8 samples per row per condition at temperature 0.7, deterministic
  per-sample seeds; 512 generations total.

## Pooled Results

| condition | P(strong) | parse fail | dP all rows | dP baseline-correct rows | rows degraded / improved |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.391 | 0.047 | — | — | — |
| erase_raw | 0.375 | 0.062 | -0.016 | -0.054 | 5 / 3 |
| erase_orthogonal | 0.023 | 0.305 | -0.367 | -0.750 | 10 / 0 |
| erase_gaussian | 0.164 | 0.477 | -0.227 | -0.571 | 9 / 1 |

Baseline-correct rows = 7 rows with baseline P(strong) >= 0.5.

Significance (paired per-row deltas, n=16; the effective n is rows, not
generations, so row-level cluster bootstrap is the primary test):

- erase_raw vs zero: dP = -0.016, bootstrap 95% CI [-0.102, +0.070]. Null;
  at n=16 rows this rules out causal effects larger than about 0.1, not
  smaller.
- erase_orthogonal: dP = -0.367, CI [-0.562, -0.188] — excludes zero.
- erase_gaussian: dP = -0.227, CI [-0.422, -0.039] — excludes zero.
- Naive per-generation SEs (raw-vs-orthogonal 3.4 sigma) overstate
  precision and are retained only as historical planning numbers.

Precision vs demolition — P(strong | parsed):

- baseline 0.410, erase_raw 0.400, erase_orthogonal 0.034, erase_gaussian
  0.313. Orthogonal erasure destroys correctness even among outputs that
  parse, so its damage is not mere format destruction; the Gaussian
  condition's damage is substantially format (48% parse failures).

## Interpretation

Mean-ablating the correctness readout direction at five layers
simultaneously, at every position, leaves task behavior statistically
unchanged, while the same operation on matched control directions is
destructive (orthogonal erasure eliminates nearly all strong-correct
outputs). The intervention machinery is demonstrably potent; the
correctness axis specifically is not load-bearing.

This is the first direct necessity test in the program and it fails in the
informative direction: **the linearly readable correctness state is causally
epiphenomenal as a direction — readable everywhere, needed nowhere.** It
predicts and explains the raw-steering, optimized-vector, DAS, and
decode-gate nulls: interventions on a gauge do not move the machine.

## Caveats

- Direction vs information: this erases the probe's 1D axis per layer, not
  all linearly decodable correctness information. The probe-on-erased check
  (docs/probe_on_erased_activations_27b_property.json) quantifies this: a
  retrained probe recovers AUC 0.877 at L45 (baseline 0.897) after the first
  mean-ablation, and eight successive INLP rounds only reach 0.858 (L30:
  0.856 -> 0.841 -> plateau near 0.82). Correctness information is
  high-dimensionally redundant, so the scoped claim is that the canonical
  readout axis is not load-bearing even while abundant correctness
  information remains decodable. A full information-level necessity test
  would need LEACE-style subspace erasure of many directions.
- Five of 62 layers; the readout could in principle be re-derived between
  erased layers (though it stays clamped at the erased ones through decode).
- Control magnitudes are matched in per-direction train-projection sd units,
  not absolute residual L2; the orthogonal/gaussian projection means come
  from last-prompt-position statistics applied at all positions.
- Historical scope at the time of this Gemma property run was Gemma-only. The
  subtype replication later confirmed the Gemma verdict with both controls above
  3 sigma — see docs/subspace_erasure_27b_subtype_sampled_k8_summary.md. A later
  Qwen property HF-hook replication supports raw-axis non-necessity without the
  same destructive-control profile — see docs/qwen35_subspace_erasure_27b_property_sampled_k8_summary.md.

## Status

Closes the experiment-1 branch of `docs/causal_handle_directions.md` with
the epiphenomenality outcome. Decision rule satisfied: report as a positive
claim (the readout direction is not necessary for task behavior), with the
direction-vs-information caveat scoped explicitly.
