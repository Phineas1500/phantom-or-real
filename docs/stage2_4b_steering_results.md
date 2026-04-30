# Gemma 3 4B Steering Results

Date: 2026-04-30

Plan reference: `docs/stage2_4b_steering_plan.md`

## Summary

All three planned `infer_property` steering comparisons completed locally on
Gemma 3 4B layer 22:

1. Raw L22 dense correctness direction.
2. Residual-SAE reconstruction-error direction.
3. Big-affine sparse-probe decoder bundle.

For those three runs, the steering protocol was stable: every generation was
parseable, every hook application count matched the intended decode-time
intervention, and all runs completed under the expanded WSL memory limit.
However, none of the directions produced a strong-correctness flip relative to
the run-local baseline.

Main result: the probed directions are predictive, but under this small
decode-time sweep they did not show causal leverage over strong
`infer_property` correctness.

A later answer-property follow-up used a more Cox-style target: raw L22
gold-polarity answer content rather than generic correctness. That probe was
perfectly predictive (`val_auc=test_auc=1.000`), but steering still produced no
polarity flips, no predicate flips toward gold, and no strong false-to-true
repairs.

## Local Execution Notes For First Three Runs

- Machine: single RTX 4090-class local WSL environment.
- WSL memory after restart: about 23 GiB RAM plus 32 GiB swap.
- Model: `google/gemma-3-4b-it`.
- Layer: 22.
- Task: `infer_property`.
- Heights: `3,4`.
- Steering subset: 16 rows, balanced by height and cached strong correctness.
- Generation: `temperature=0`, `max_new_tokens=96`, `n_ctx=2048`,
  `dtype=bfloat16`, `load_mode=no-processing`.
- Intervention scope: `last_token_each_forward`.
- All generations reached `max_new_tokens=96`.

## Result Table

| Direction | Probe source | Probe val AUC | Probe test AUC | Scale std | Rows | Parse failures | Strong flips | Weak flips | Text changes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw L22 | `hook_resid_post` activations | 0.9130 | 0.9036 | 11.8741 | 144 | 0 | 0 | 0 | 16 |
| Reconstruction error | L22 residual-SAE raw minus reconstruction | 0.9120 | 0.9005 | 26.3208 | 144 | 0 | 0 | 2 | 12 |
| Big-affine bundle | 50-feature sparse-probe bundle | 0.8762 | 0.8602 | 123.4487 | 272 | 0 | 0 | 5 | 31 |

Separate answer-property result:

| Direction | Probe source | Probe val AUC | Probe test AUC | Scale std | Rows | Parse failures | Polarity flips toward gold | Predicate flips toward gold | Strong false-to-true |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw L22 answer polarity | `hook_resid_post` gold polarity labels | 1.0000 | 1.0000 | 101.7908 | 320 | 2 | 0 | 0 | 0 |

Baseline behavior was identical across the first three runs:

| Metric | Value |
|---|---:|
| Strong accuracy | 0.5625 |
| Weak accuracy | 0.7500 |
| Parse-failure rate | 0.0000 |
| Mean generated tokens | 96.0 |

## Experiment Details

### Experiment 1: Raw L22 Direction

Artifacts:

- Report: `docs/raw_steering_4b_l22_property_decode_sweep.json`
- Rows: `results/stage2/steering/raw_4b_l22_property_decode_sweep.jsonl`
- Direction: `results/stage2/steering/raw_4b_l22_property_decode_sweep_direction.npz`

The raw dense logistic direction had strong predictive performance
(`test_auc=0.9036`) and a train projection standard deviation of `11.8741`.
Steering at `blocks.22.hook_resid_post` with strengths `-0.5,-1,0.5,1` SD
caused some textual changes, but no weak or strong correctness changes.
Orthogonal controls behaved similarly.

Outcome: negative causal result for strong correctness at this sample size and
strength sweep.

### Experiment 2: Reconstruction-Error Direction

Artifacts:

- Report: `docs/error_steering_4b_l22_16k_property_decode_sweep.json`
- Rows: `results/stage2/steering/error_4b_l22_16k_property_decode_sweep.jsonl`
- Direction: `results/stage2/steering/error_4b_l22_16k_property_decode_sweep_direction.npz`
- Reconstruction diagnostic:
  `docs/sae_reconstruction_probe_4b_l22_s1_property16k_local.json`

The residual-SAE reconstruction-error probe nearly matched the raw direction
(`test_auc=0.9005`). Steering this error direction was also parse-clean and
produced no strong correctness flips. There were two weak changes: one in the
error-direction condition and one in an orthogonal control condition.

Outcome: negative causal result for strong correctness. The error direction
remains predictive, but the steering effect was not distinguishable from
control behavior in this run.

### Experiment 3: Big-Affine Sparse-Probe Bundle

Artifacts:

- Report: `docs/bundle_steering_4b_l22_big_affine_property_decode_sweep.json`
- Rows:
  `results/stage2/steering/bundle_4b_l22_big_affine_property_decode_sweep.jsonl`
- Direction:
  `results/stage2/steering/bundle_4b_l22_big_affine_property_decode_sweep_direction.npz`
- Local sparse-probe report:
  `docs/transcoder_probe_4b_l22_262k_big_affine_exact_top512_s1_local.json`

Preprocessing generated the missing exact L22 transcoder-hook activations:

- `results/stage2/activations/gemma3_4b_infer_property_L22_mlp_in_weighted.safetensors`
- `results/stage2/activations/gemma3_4b_infer_property_L22_mlp_out_hook.safetensors`

It then encoded the big-affine top-512 feature cache:

- `results/stage2/sae_features/gemma3_4b_infer_property_L22_mlp_in_weighted_layer_22_width_262k_l0_big_affine_top512.safetensors`

The sparse probe was predictive (`test_auc=0.8602`). The bundle used the top
25 positive and top 25 negative standardized coefficients after train-density
filtering (`0.02 <= density <= 0.50`). The decoder-space bundle was normalized
and scaled using the train projection standard deviation of the exact
`mlp_out_hook` activations.

Controls:

- Random feature bundle with the same feature count.
- Shuffled-coefficient bundle over the selected features.
- Norm-matched orthogonal vector at `blocks.22.hook_mlp_out`.

The bundle run produced no strong correctness flips. It produced five weak
changes, but those appeared across bundle and control conditions rather than
concentrating in the learned bundle.

Outcome: negative causal result for strong correctness. The learned dictionary
contains a distributed predictive signal, but this bundle did not provide
specific causal leverage under the tested intervention protocol.

### Experiment 4: Raw L22 Answer-Property Direction

Artifacts:

- Report: `docs/answer_property_steering_4b_l22_polarity_decode_sweep.json`
- Smoke report: `docs/answer_property_steering_4b_l22_polarity_smoke.json`
- Rows:
  `results/stage2/steering/answer_property_4b_l22_polarity_decode_sweep.jsonl`
- Direction:
  `results/stage2/steering/answer_property_4b_l22_polarity_direction.npz`

This follow-up trained a raw L22 probe on gold answer polarity instead of
correctness. The probe was perfectly predictive on the S1 split
(`val_auc=1.0000`, `test_auc=1.0000`) over 10,721 parseable rows. The main
sweep used 32 h3/h4 test rows balanced by height and cached strong correctness,
`max_new_tokens=160`, and strengths `0.5`, `1`, and `2` projection SD for
`toward_gold`, `away_gold`, and orthogonal conditions.

The full sweep produced no polarity flips in either direction, no predicate
flips toward gold, and no strong false-to-true flips. The few answer-content
changes were invalid predicate changes or also appeared under orthogonal
controls. Per the plan, sparse answer-property steering was not run because the
raw answer-property direction did not move answer content above controls.

Outcome: negative causal result for answer-content steering, despite perfect
linear decodability.

## Interpretation

The three-way comparison supports a conservative conclusion:

- Raw activations, reconstruction error, and big-affine sparse features all
  encode predictive information about `infer_property` correctness.
- Raw answer polarity is also perfectly decodable from raw L22 activations.
- Decode-time additive steering at the tested layer/sites, strengths, and row
  samples did not convert that predictive information into reliable causal
  changes in strong correctness or emitted answer content.
- The absence of parse failures is useful: the negative result is not explained
  by obvious off-manifold degeneration.
- Weak/text changes occurred, but they were also present in controls and did
  not align with strong correctness.

This is currently stronger evidence for a predictive-versus-causal gap than for
a successful learned-feature steering direction.

## Caveats

- The steering subset is small: 16 examples, selected from h3/h4 test rows.
- The answer-property follow-up is larger but still bounded: 32 h3/h4 examples
  with 320 generated rows across baseline, directed, and orthogonal conditions.
- In the first three runs, all generations hit `max_new_tokens=96`, so some
  answers may be truncated or budget-limited. The answer-property follow-up
  separately used `max_new_tokens=160`.
- Strong flips are computed relative to the run-local baseline generation, not
  the cached original Stage 1 label. Some run-local baselines differ from the
  cached correctness label.
- Only `infer_property` was tested. The plan intentionally deferred
  `infer_subtype` because 4B subtype behavior is near-collapsed at h3/h4.
- The tested strengths were `+/-0.5` and `+/-1.0` projection SD. Larger strengths
  or different intervention sites could behave differently, but should be
  treated as a new experiment with fresh controls.

## Code Changes Supporting These Runs

- `scripts/stage2_steer_raw_direction.py`: added `--activation-prefix` so the
  same raw-direction steering code can train on reconstruction-error artifacts.
- `scripts/stage2_sae_reconstruction_diagnostics.py`: stripped private sklearn
  artifacts before JSON serialization.
- `scripts/stage2_extract_exact_transcoder_hooks.py`: added `--tasks` so local
  exact-hook extraction can run `infer_property` only.
- `scripts/stage2_probe_sae.py`: stripped private sklearn artifacts before JSON
  serialization.
- `scripts/stage2_steer_sparse_probe_bundle.py`: added sparse-probe bundle
  steering with random, shuffled, and orthogonal controls.

Validation command:

```bash
python -m py_compile \
  scripts/stage2_extract_exact_transcoder_hooks.py \
  scripts/stage2_probe_sae.py \
  scripts/stage2_steer_sparse_probe_bundle.py \
  scripts/stage2_steer_raw_direction.py \
  scripts/stage2_sae_reconstruction_diagnostics.py
```
