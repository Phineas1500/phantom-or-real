# Gemma 3 4B Answer/Property Steering Plan

Date: 2026-04-30

Status: raw answer-property steering completed; see
`docs/stage2_4b_answer_property_steering_results.md`. The raw L22 gold-polarity
probe reached `val_auc=test_auc=1.000`, but decode-step steering produced no
polarity flips, no predicate flips toward gold, and no strong false-to-true
repairs. The sparse follow-up remains intentionally gated off because the raw
answer-content direction did not move emitted answers above controls.

Purpose: run a Cox-style steering follow-up on Gemma 3 4B using answer-content
directions instead of generic correctness directions. This is the next causal
test after the completed 4B raw, reconstruction-error, and big-affine bundle
correctness-steering nulls in `docs/stage2_4b_steering_results.md`.

## Why This Is Different

The completed steering runs trained directions for `is_correct_strong`. Those
directions are predictive, but "be correct" does not specify which ontology
hypothesis the model should output. Cox-style answer steering is more concrete:
train a direction for answer content, then steer toward a target answer class or
away from it during decoding.

For our `infer_property` task, reusable answer-content labels are:

- Polarity: affirmative hypothesis vs negated hypothesis.
- Predicate: the property word, such as `small`, `large`, `opaque`, or
  `transparent`.
- Predicate-pair labels: for high-support opposite pairs such as
  `small`/`large`, `opaque`/`transparent`, `hot`/`cold`, and `fast`/`slow`.

Do not train this first pass on `is_correct_strong`. That would repeat the
previous null result.

## Expected Scientific Value

This is still in scope for the report. It answers whether the steering protocol
can move concrete answer representations at all.

- If answer/property steering works but correctness steering does not, the
  conclusion is clean: the model has steerable answer-content directions, while
  our success/failure directions are diagnostic rather than command-like.
- If raw answer/property steering works but sparse answer/property bundles do
  not, that strengthens the raw-vs-learned-dictionary gap.
- If raw answer/property steering also fails, the current decode-time steering
  setup is probably not a reliable causal test for this generative ontology
  task, and the report should keep steering as negative/inconclusive.

## Run Order

Run only `infer_property` first. The 4B `infer_subtype` task is too collapsed at
h3/h4 for clean steering interpretation.

1. Raw L22 polarity direction, using gold polarity labels.
2. Raw L22 polarity direction, using cached model-output polarity labels if the
   parser can recover enough labels.
3. Raw L22 predicate-pair directions for one or two high-support pairs.
4. Sparse big-affine answer/property bundle only if a raw answer/property
   direction produces directional answer changes above orthogonal controls.

The first run should be a small smoke test. Expand only if it is parse-clean and
the hook counts are correct.

## Required Local Inputs

These should already be present after pulling `main` and copying or regenerating
the local 4B tensors:

- `results/full/with_errortype/gemma3_4b_infer_property.jsonl`
- `results/stage2/splits_4b_property.jsonl`
- `results/stage2/activations/gemma3_4b_infer_property_L22.safetensors`
- `results/stage2/activations/gemma3_4b_infer_property_L22.example_ids.jsonl`

For sparse follow-up only:

- `results/stage2/sae_features/gemma3_4b_infer_property_L22_mlp_in_weighted_layer_22_width_262k_l0_big_affine_top512.safetensors`
- `results/stage2/sae_features/gemma3_4b_infer_property_L22_mlp_in_weighted_layer_22_width_262k_l0_big_affine_top512.example_ids.jsonl`
- `results/stage2/activations/gemma3_4b_infer_property_L22_mlp_out_hook.safetensors`
- `results/stage2/activations/gemma3_4b_infer_property_L22_mlp_out_hook.example_ids.jsonl`
- Hugging Face artifact:
  `google/gemma-scope-2-4b-it/transcoder_all/layer_22_width_262k_l0_big_affine`

## Local 4090 Settings

Use the same stable settings as the completed 4B steering runs, except increase
the generation budget because every previous generation hit 96 tokens.

```bash
export HF_HOME=$PWD/.hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BD_PATH=/path/to/beyond-deduction
```

Recommended generation settings:

- `--model google/gemma-3-4b-it`
- `--model-key gemma3_4b`
- `--task infer_property`
- `--layer 22`
- `--n-devices 1`
- `--dtype bfloat16`
- `--load-mode no-processing`
- `--n-ctx 2048`
- `--max-new-tokens 160`
- `--temperature 0`
- `--intervention-scope last_token_each_forward`

If memory gets tight, reduce `--max-new-tokens` to `128` before reducing
`--n-ctx`.

## Implementation Target

Create a new script:

`scripts/stage2_steer_answer_property_direction.py`

Start by copying the structure of `scripts/stage2_steer_raw_direction.py`.
Reuse these helpers where possible:

- `load_probe_dataset`, `read_split_assignments`, and
  `split_indices_from_assignments` from `src.stage2_probes`
- `select_balanced_stage1_rows`, `score_reply`, `summarize_steering_rows`,
  `make_orthogonal_unit_direction`, and generation hooks from
  `src.stage2_steering`
- `parse_hypotheses` from `src.gemma3_parse`
- `parse_hypothesis_structure` from the Beyond Deduction parser if needed

The new script should differ from `stage2_steer_raw_direction.py` in four ways.

### 1. Build Answer Labels Instead Of Correctness Labels

Add:

```text
--answer-label-source gold|stage1_model_output
--answer-target polarity|predicate_pair|predicate_one_vs_rest
--positive-predicate <name>
--negative-predicate <name>
```

For the first run use:

```text
--answer-label-source gold
--answer-target polarity
```

Polarity label convention:

- `1` means the gold hypothesis is negated, for example `X are not angry`.
- `0` means the gold hypothesis is affirmative, for example `X are angry`.

For `gold`, read labels from:

```python
row["ontology_fol_structured"]["hypothesis"]["negated"]
row["ontology_fol_structured"]["hypothesis"]["predicate"]
```

For `stage1_model_output`, parse `row["model_output"]`, drop unparseable rows,
and label from the parsed hypothesis. This is closest to Cox et al. because it
trains on the model's own cached answer rather than the gold answer. Run this
only after the gold-label smoke test is working.

### 2. Train The Probe On Those Labels

Keep the same split-aware logistic probe pattern as the raw correctness script:

- train on S1 train rows
- choose `C` by S1 validation AUC
- report held-out S1 test AUC
- undo standardization to recover a raw-space direction
- compute the train projection standard deviation

The sidecar gives the row indices for the activation rows. Join each activation
row back to the source JSONL row and build the answer label from that row.

Fail fast if any split has only one class. For polarity, this should not happen:
the 4B property file has about `5.4k` negated and `5.6k` affirmative gold
hypotheses.

### 3. Use Per-Example Steering Signs

The old script uses fixed conditions like `raw_pos1sd` and `raw_neg1sd`. For
answer steering, add dynamic conditions:

```text
baseline
toward_gold_0p5sd
toward_gold_1sd
toward_gold_2sd
away_gold_1sd
orthogonal_1sd
```

For polarity with the convention above:

```python
sign_to_gold = +1 if gold_negated else -1
delta = sign_to_gold * strength_sd * projection_std
```

For `away_gold`, multiply by `-1`.

For orthogonal controls, use the same per-example sign and strength, but apply
the norm-matched orthogonal direction instead of the answer direction.

Keep optional fixed-sign diagnostics (`toward_negated`, `toward_affirmative`)
out of the first run unless needed for debugging.

### 4. Record Answer-Content Metrics

The output JSONL should include the existing correctness fields plus:

- `gold_predicate`
- `gold_negated`
- `baseline_predicate`
- `baseline_negated`
- `parsed_predicate`
- `parsed_negated`
- `polarity_matches_gold`
- `predicate_matches_gold`
- `answer_content_changed_vs_baseline`

The compact JSON summary should include, by condition:

- strong accuracy
- weak accuracy
- parse-failure rate
- polarity match rate
- predicate match rate
- polarity flips toward gold
- polarity flips away from gold
- predicate flips toward gold
- mean generated tokens

The key metric for the first polarity run is not just strong correctness. It is
whether `toward_gold_*` changes wrong-polarity outputs toward the gold polarity
more often than orthogonal controls.

## First Smoke Test Command

After implementing the script, run a small parse/hook smoke test:

```bash
python -u scripts/stage2_steer_answer_property_direction.py \
  --jsonl results/full/with_errortype/gemma3_4b_infer_property.jsonl \
  --model google/gemma-3-4b-it \
  --model-key gemma3_4b \
  --task infer_property \
  --layer 22 \
  --activation-dir results/stage2/activations \
  --splits results/stage2/splits_4b_property.jsonl \
  --split-family s1 \
  --answer-label-source gold \
  --answer-target polarity \
  --heights 3,4 \
  --per-height-label 2 \
  --conditions baseline,toward_gold,away_gold,orthogonal \
  --strengths=0.5,1 \
  --intervention-scope last_token_each_forward \
  --max-new-tokens 160 \
  --temperature 0 \
  --n-devices 1 \
  --n-ctx 2048 \
  --dtype bfloat16 \
  --load-mode no-processing \
  --out-jsonl results/stage2/steering/answer_property_4b_l22_polarity_smoke.jsonl \
  --direction-output results/stage2/steering/answer_property_4b_l22_polarity_direction.npz \
  --output docs/answer_property_steering_4b_l22_polarity_smoke.json
```

Expected row count:

- `8` source rows for `--per-height-label 2`
- baseline plus four steering conditions if `0.5,1` are both used for
  `toward_gold` and `away_gold`, plus two orthogonal controls
- exact count depends on how the local script expands dynamic conditions

Smoke-test success criteria:

- all JSON outputs validate with `jq empty`
- no hook-count mismatch
- no unexpected parse-failure spike
- probe test AUC is meaningfully above chance
- output summary includes answer-content flip metrics

## Main Polarity Run

If the smoke test is clean, run:

```bash
python -u scripts/stage2_steer_answer_property_direction.py \
  --jsonl results/full/with_errortype/gemma3_4b_infer_property.jsonl \
  --model google/gemma-3-4b-it \
  --model-key gemma3_4b \
  --task infer_property \
  --layer 22 \
  --activation-dir results/stage2/activations \
  --splits results/stage2/splits_4b_property.jsonl \
  --split-family s1 \
  --answer-label-source gold \
  --answer-target polarity \
  --heights 3,4 \
  --per-height-label 8 \
  --conditions baseline,toward_gold,away_gold,orthogonal \
  --strengths=0.5,1,2 \
  --intervention-scope last_token_each_forward \
  --max-new-tokens 160 \
  --temperature 0 \
  --n-devices 1 \
  --n-ctx 2048 \
  --dtype bfloat16 \
  --load-mode no-processing \
  --out-jsonl results/stage2/steering/answer_property_4b_l22_polarity_decode_sweep.jsonl \
  --direction-output results/stage2/steering/answer_property_4b_l22_polarity_direction.npz \
  --output docs/answer_property_steering_4b_l22_polarity_decode_sweep.json
```

This gives 32 source rows before conditions:

- h3 correct
- h3 incorrect
- h4 correct
- h4 incorrect

It should still be feasible on a 4090 given the completed 16-row 4B steering
runs.

## Predicate-Pair Follow-Up

Only do this after the polarity run. Predicate-pair steering is closer to
"which concrete answer content does the model emit?" than polarity alone.

Use high-support pairs first:

- `small` vs `large`
- `opaque` vs `transparent`
- `hot` vs `cold`
- `fast` vs `slow`

Example:

```bash
python -u scripts/stage2_steer_answer_property_direction.py \
  --jsonl results/full/with_errortype/gemma3_4b_infer_property.jsonl \
  --model google/gemma-3-4b-it \
  --model-key gemma3_4b \
  --task infer_property \
  --layer 22 \
  --activation-dir results/stage2/activations \
  --splits results/stage2/splits_4b_property.jsonl \
  --split-family s1 \
  --answer-label-source gold \
  --answer-target predicate_pair \
  --positive-predicate small \
  --negative-predicate large \
  --heights 3,4 \
  --per-height-label 4 \
  --conditions baseline,toward_gold,away_gold,orthogonal \
  --strengths=0.5,1,2 \
  --intervention-scope last_token_each_forward \
  --max-new-tokens 160 \
  --temperature 0 \
  --n-devices 1 \
  --n-ctx 2048 \
  --dtype bfloat16 \
  --load-mode no-processing \
  --out-jsonl results/stage2/steering/answer_property_4b_l22_small_large_decode_sweep.jsonl \
  --direction-output results/stage2/steering/answer_property_4b_l22_small_large_direction.npz \
  --output docs/answer_property_steering_4b_l22_small_large_decode_sweep.json
```

For predicate pairs, filter the probe dataset and steering subset to rows whose
gold predicate is one of the two predicates. Use this label convention:

- `1` means the positive predicate, e.g. `small`
- `0` means the negative predicate, e.g. `large`

For steering signs, use `+1` for the positive predicate and `-1` for the
negative predicate. For `toward_gold`, choose the sign per example from the
gold predicate.

## Sparse Follow-Up If Raw Works

If raw answer/property steering moves answer content above controls, implement
the sparse analogue.

Use the same labels and dynamic signs, but fit the probe on the 4B L22
big-affine top512 sparse feature matrix:

`results/stage2/sae_features/gemma3_4b_infer_property_L22_mlp_in_weighted_layer_22_width_262k_l0_big_affine_top512.safetensors`

Then build a decoder bundle exactly like
`scripts/stage2_steer_sparse_probe_bundle.py`:

- select top 25 positive and top 25 negative standardized coefficients
- use density filter `0.02 <= train_density <= 0.50`
- combine decoder rows into one unit direction
- steer at `blocks.22.hook_mlp_out`
- scale by projection SD on exact `mlp_out_hook` activations
- include random, shuffled-coefficient, and orthogonal controls

Do not run the sparse version if raw answer/property steering is completely
flat. It would be expensive and unlikely to clarify the report.

## How To Interpret Outcomes

Positive raw, negative sparse:

- Good report result. Raw activations contain steerable answer content, while
  the tested learned dictionary exposes predictive but not causally sufficient
  structure.

Positive raw and positive sparse:

- Best case for an interpretable causal story. Inspect selected sparse features
  and run falsification checks before claiming mechanism.

Negative raw and negative sparse:

- Steering protocol is probably not the right causal test for this task. Keep
  all steering as negative/inconclusive and focus the report on predictive
  localization plus reconstruction-error diagnostics.

Positive away-gold degradation, no toward-gold repair:

- Useful but weaker. It means the direction can damage answer content, but does
  not reliably repair it. Treat as causal sensitivity, not successful steering.

## Validation Before Committing Results

Run:

```bash
python -m py_compile scripts/stage2_steer_answer_property_direction.py
jq empty docs/answer_property_steering_4b_l22_polarity_decode_sweep.json
git diff --check
```

Also inspect:

```bash
jq '.probe_direction, .summary' docs/answer_property_steering_4b_l22_polarity_decode_sweep.json
```

The final summary should explicitly state:

- probe validation/test AUC
- condition row counts
- parse-failure rates
- answer-content flip rates
- whether any effect exceeds orthogonal controls
