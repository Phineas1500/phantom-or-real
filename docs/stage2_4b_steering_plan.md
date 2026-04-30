# Gemma 3 4B Steering Plan

Date: 2026-04-30

Purpose: define a small, reproducible 4B steering handoff that can run on a
single RTX 4090-class local machine. The goal is not to prove a monosemantic
feature; it is to compare whether dense raw directions, reconstruction-error
directions, and learned-dictionary directions have causal leverage over
`infer_property` success/failure.

## Recommendation

Start with `infer_property` only. The 4B subtype task has near-collapsed
behavior at h3/h4 and very small positive counts, so flips are hard to
interpret even though subtype probe AUCs are high.

Run experiments in this order:

1. Raw L22 correctness direction, decode-time sweep.
2. Reconstruction-error direction, if the reconstruction-error activation file
   is available or easy to regenerate.
3. Big-affine transcoder feature-bundle direction.
4. Optional single-feature or residual-SAE feature tests only after the first
   three comparisons.

The main comparison should be:

`raw direction` vs `reconstruction-error direction` vs `big-affine sparse-probe bundle` vs `orthogonal/random controls`.

## Required Inputs

These files must exist locally before steering can run:

- `results/full/with_errortype/gemma3_4b_infer_property.jsonl`
- `results/stage2/splits.jsonl`
- `results/stage2/activations/gemma3_4b_infer_property_L22.safetensors`
- `results/stage2/activations/gemma3_4b_infer_property_L22.example_ids.jsonl`

For reconstruction-error steering:

- Residual-SAE error activations from the L22 reconstruction diagnostic, if
  retained locally. If absent, regenerate them using the existing 4B
  reconstruction script before attempting error-direction steering.

For big-affine learned-dictionary steering:

- `results/stage2/sae_features/gemma3_4b_infer_property_L22_mlp_in_weighted_layer_22_width_262k_l0_big_affine_top512.safetensors`
- `results/stage2/sae_features/gemma3_4b_infer_property_L22_mlp_in_weighted_layer_22_width_262k_l0_big_affine_top512.meta.json`
- `results/stage2/sae_features/gemma3_4b_infer_property_L22_mlp_in_weighted_layer_22_width_262k_l0_big_affine_top512.example_ids.jsonl`
- Hugging Face artifact:
  `google/gemma-scope-2-4b-it/transcoder_all/layer_22_width_262k_l0_big_affine`

If the large `.safetensors` caches are not in the local clone, copy them from
Scholar or rerun the relevant Phase A/Phase D extraction jobs locally. The PRs
mostly add scripts and JSON summaries; the large activation/feature tensors may
not be tracked by git.

## Existing 27B Code To Reuse

Raw direction steering:

- `scripts/stage2_steer_raw_direction.py`
- `scripts/stage2_steer_raw_27b_L45_property_decode_sweep.sbatch`

Learned-feature steering:

- `scripts/stage2_steer_transcoder_features.py`
- `scripts/stage2_steer_big_l0_features_27b_L45_property_pilot.sbatch`

The raw script is already model/layer/task parameterized. For 4B, use
`--model google/gemma-3-4b-it`, `--model-key gemma3_4b`, and `--layer 22`.

The learned-feature script is built for affine transcoders. For 4B big-affine,
use `--hook-name blocks.22.hook_mlp_out`,
`--activation-site mlp_in_weighted`, and
`--sae-id layer_22_width_262k_l0_big_affine`.

## Local 4090 Settings

Use one GPU:

- `--n-devices 1`
- `--dtype bfloat16`
- `--load-mode no-processing`
- `--n-ctx 2048` initially
- `--max-new-tokens 96`
- `--temperature 0`
- `--intervention-scope last_token_each_forward`

Environment:

```bash
export HF_HOME=$PWD/.hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BD_PATH=/path/to/beyond-deduction
```

If WSL has only about 24 GB active RAM, avoid code paths that load full 262K
parameter tensors into CPU memory unnecessarily. The existing learned-feature
script uses `safe_open` for decoder rows; keep that pattern for bundle
steering.

If CUDA OOM occurs, reduce in this order:

1. `--n-ctx 1024`
2. `--max-new-tokens 64`
3. `--per-height-label 2`
4. run raw steering before any big-affine steering

## Experiment 1: Raw L22 Decode-Time Sweep

This is the first experiment to run. It tests whether the strong raw L22
property probe is causally useful under Cox-style decode-time steering.

Recommended command:

```bash
python -u scripts/stage2_steer_raw_direction.py \
  --jsonl results/full/with_errortype/gemma3_4b_infer_property.jsonl \
  --model google/gemma-3-4b-it \
  --model-key gemma3_4b \
  --task infer_property \
  --layer 22 \
  --activation-dir results/stage2/activations \
  --splits results/stage2/splits.jsonl \
  --split-family s1 \
  --heights 3,4 \
  --per-height-label 4 \
  --conditions baseline,raw,orthogonal \
  --strengths=-0.5,-1,0.5,1 \
  --intervention-scope last_token_each_forward \
  --max-new-tokens 96 \
  --temperature 0 \
  --n-devices 1 \
  --n-ctx 2048 \
  --dtype bfloat16 \
  --load-mode no-processing \
  --out-jsonl results/stage2/steering/raw_4b_l22_property_decode_sweep.jsonl \
  --direction-output results/stage2/steering/raw_4b_l22_property_decode_sweep_direction.npz \
  --output docs/raw_steering_4b_l22_property_decode_sweep.json
```

Initial sample size is 16 rows: h3/h4 x baseline-correct/baseline-incorrect x
4 examples. If the run is stable and there are direction-specific flips, rerun
with `--per-height-label 8` or `16`.

Interpretation:

- Positive result: raw steering produces more false-to-true or directional
  answer changes than orthogonal steering at comparable parse-failure rates.
- Negative result: raw and orthogonal controls behave similarly, or changes are
  mostly parse failures/off-manifold degeneration.

## Experiment 2: Reconstruction-Error Direction

This is high value because both 27B and 4B reconstruction diagnostics show that
raw-minus-reconstruction error recovers near-raw property AUC.

Preferred implementation:

1. Add an optional `--activation-prefix` argument to
   `scripts/stage2_steer_raw_direction.py`.
2. If `--activation-prefix` is supplied, use it directly instead of constructing
   `results/stage2/activations/{model_key}_{task}_L{layer}`.
3. Train the same logistic direction on the reconstruction-error activation
   file and steer at the matching residual site.

Run the same conditions and strengths as Experiment 1, changing only output
paths and the activation prefix. Keep orthogonal controls.

Interpretation:

- If raw and error directions both steer while sparse features do not, that
  directly supports the raw-vs-SAE discrepancy story.
- If error steers more cleanly than raw, it suggests the causally relevant
  direction is concentrated in the small residual component missed by the SAE.

## Experiment 3: Big-Affine Sparse-Probe Bundle

Do not start with single big-affine features. The strongest 4B learned-dictionary
property result is distributed:

- Big-affine L22 262K top512 property AUC: S1 `0.855`, S3 `0.874`
- Raw L22 property AUC: S1 `0.903`, S3 `0.906`

A single feature is unlikely to carry enough causal mass. The right test is a
bundle direction derived from the sparse probe.

Implementation target:

Use `scripts/stage2_steer_sparse_probe_bundle.py`, which was added after the
27B steering-null checkpoint. It combines patterns from:

- `scripts/stage2_probe_sae.py`
- `scripts/stage2_analyze_sae_features.py`
- `scripts/stage2_steer_transcoder_features.py`

Bundle construction:

1. Load the big-affine top-k feature matrix:
   `gemma3_4b_infer_property_L22_mlp_in_weighted_layer_22_width_262k_l0_big_affine_top512`.
2. Fit the same split-aware logistic probe used in the sparse-probe reports.
3. Select train-active features by standardized coefficient.
4. Start with top `K=25` positive and top `K=25` negative features, excluding
   features with train density outside roughly `0.02-0.50`.
5. Build a decoder-space vector:
   `bundle = sum_i standardized_coef_i * w_dec[i]`.
6. Normalize the bundle to unit L2 norm.
7. Scale interventions by the train-set projection standard deviation of
   `blocks.22.hook_mlp_out` activations onto that bundle, if exact
   `mlp_out_hook` activations are present. If not, use L2-matched deltas
   against the raw steering direction.

Steer at:

- `blocks.22.hook_mlp_out`
- `last_token_each_forward`

Use the same row selection, output schema, scorer, and summary logic as
`scripts/stage2_steer_transcoder_features.py`.

Controls:

- Random feature bundle with the same number of features.
- Shuffled-coefficient bundle over the same selected features.
- Norm-matched orthogonal vector at `blocks.22.hook_mlp_out`.

Interpretation:

- If the bundle works and random/shuffled controls do not, the learned
  dictionary contains a distributed causal direction.
- If raw works but the bundle does not, this supports the conclusion that the
  tested sparse dictionary exposes predictive but not causally sufficient
  structure.
- If neither raw nor bundle works, the issue is likely the steering protocol or
  task format, not just the dictionary.

## Optional Experiment 4: Single Features

Single-feature steering is useful only as a falsification or sanity check. It
should not be the first 4B steering claim.

Potential residual SAE candidates are listed in:

- `docs/feature_candidate_shortlist_4b.md`

Those candidates are residual SAE features, not big-affine transcoder features.
Do not pass them blindly to `stage2_steer_transcoder_features.py` unless the
script is parameterized to load the matching residual SAE and steer the matching
residual hook.

If single-feature tests are run:

- Use one positive and one negative candidate first.
- Compare against random same-width features.
- Treat any result as exploratory unless it survives bundle and control tests.

## Reporting Schema

For every steering run, report:

- selected source rows and heights
- baseline strong/weak accuracy
- condition strong/weak accuracy
- parse-failure rate
- changed outputs vs baseline
- false-to-true flips
- true-to-false flips
- examples of changed outputs
- orthogonal/random-control comparison
- whether outputs hit `max_new_tokens`

Avoid reporting only aggregate accuracy. A useful steering result needs
direction-specific flips without a matching rise in parse failures or generic
orthogonal-control damage.

## Decision Table

| Raw | Error | Sparse bundle | Interpretation |
| --- | --- | --- | --- |
| works | works | works | learned dictionary captures a causal part of the raw correctness direction |
| works | works | fails | causally relevant signal is likely in SAE/transcoder-missed residual structure |
| works | fails | fails | raw direction may use structure not captured by the residual-error file or sparse dictionary |
| fails | fails | fails | current steering protocol is too weak/misaligned for this task |
| fails | works | fails | strongest evidence for reconstruction-error-centered mechanism |

## Notes For Local Codex

First implement/run only Experiment 1. Do not edit the feature-bundle code until
the raw 4B decode-time sweep is known to run cleanly on the 4090.

After Experiment 1, inspect:

- `docs/raw_steering_4b_l22_property_decode_sweep.json`
- `results/stage2/steering/raw_4b_l22_property_decode_sweep.jsonl`

Only proceed to Experiment 2/3 if baseline generation parses correctly and raw
or orthogonal steering does not immediately collapse outputs.
