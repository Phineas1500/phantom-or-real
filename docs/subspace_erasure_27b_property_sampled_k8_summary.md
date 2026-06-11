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

Significance (paired per-row deltas, n=16):

- erase_raw vs zero: mean dP = -0.016, SE = 0.044 -> 0.4 sigma. Null.
- erase_raw vs erase_orthogonal: mean diff = +0.352, SE = 0.104 -> 3.4 sigma.
- erase_raw vs erase_gaussian: mean diff = +0.211, SE = 0.109 -> 1.9 sigma
  (the Gaussian condition's damage is partly format destruction: 48% parse
  failures, so its correctness deltas are noisier).

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
  all linearly decodable correctness information. The dictionary results
  show the signal is recoverable in other bases, so the model may re-read it
  elsewhere. The stronger claim needs full linear concept erasure
  (LEACE-style) plus a probe-on-erased-activations check.
- Five of 62 layers; the readout could in principle be re-derived between
  erased layers (though it stays clamped at the erased ones through decode).
- Control magnitudes are matched in per-direction train-projection sd units,
  not absolute residual L2; the orthogonal/gaussian projection means come
  from last-prompt-position statistics applied at all positions.
- Property task, Gemma only. Subtype and Qwen replications are cheap with
  the same script.

## Status

Closes the experiment-1 branch of `docs/causal_handle_directions.md` with
the epiphenomenality outcome. Decision rule satisfied: report as a positive
claim (the readout direction is not necessary for task behavior), with the
direction-vs-information caveat scoped explicitly.
