# Positive-Control Steering Spec

Purpose: define the gate that must pass before interpreting new correctness-steering nulls.

## Decision

Use a simple output-style control on Gemma 3 27B at L45, not the historical answer-property direction. The historical answer-property smoke had a perfect offline polarity probe but did not produce target-directed free-form answer movement, so it is evidence for another predictive-but-not-causal result, not a positive control.

## Proposed Gate

Task: verbosity or output-format steering on neutral prompts.

Preferred first version: verbosity. Build paired prompts with the same neutral question under two instructions:

- short condition: answer in one word or one short phrase.
- long condition: answer in at least three sentences with explanation.

Representation: raw L45 residual direction trained as long-vs-short using activations at the final prompt token.

Intervention site: `blocks.45.hook_resid_post` on `google/gemma-3-27b-it`.

Conditions:

- regenerated baseline
- toward long
- toward short
- orthogonal direction
- matched Gaussian/noise direction

Primary metrics:

- generated token count
- sentence count
- heuristic short/long classification
- paired direction-of-change versus regenerated baseline
- parse/format failure rate if a structured-format variant is used

Pass criterion for using this as a positive control:

- toward-long increases generated length above matched Gaussian/noise by at least `2σ`;
- toward-short decreases generated length above matched Gaussian/noise by at least `2σ`;
- at least 70% of paired rows move in the intended direction for one of the two steering directions;
- orthogonal and matched-Gaussian controls do not show comparable directional movement.

These thresholds are an engineering gate, not a manuscript claim.

## Why This Gate

A positive control only needs to show that the same model, hook site, generation path, and reporting stack can produce a real intervention effect when the target behavior is known to be steerable. It does not need to be an InAbHyD reasoning variable. In fact, it is cleaner if it is not: a style control tests the intervention machinery without reusing the disputed correctness mechanism.

## Non-Goals

- Do not treat this as evidence that correctness is steerable.
- Do not cite this as a reasoning result.
- Do not proceed to optimized/DAS/decode-time null interpretation unless this gate or an equivalent positive control passes.

## Implementation Notes

The positive-control job should write the same causal-abstraction report fields as the correctness-steering jobs: `model`, `task`, `target_variable`, `representation_type`, `controls`, `baseline_metrics`, `intervention_metrics`, `paired_flips`, `parse_fail_rate`, `matched_noise_summary`, and `causal_abstraction_claim`. For this control, use `target_variable=positive_control_behavior`; keep the causal-abstraction claim explicit that this is an auxiliary machinery check, not an InAbHyD reasoning variable.

## Runnable Implementation

Script: `scripts/stage2_positive_control_verbosity.py`

Slurm launcher: `scripts/stage2_positive_control_verbosity_27b_L45.sbatch`

Submit:

```bash
sbatch scripts/stage2_positive_control_verbosity_27b_L45.sbatch
```

Expected artifacts:

- `docs/positive_control_verbosity_gemma3_27b_l45.json`
- `results/stage2/positive_control/verbosity_gemma3_27b_l45.jsonl`
- `results/stage2/positive_control/verbosity_gemma3_27b_l45_direction.npz`

The preflight report reads `docs/positive_control_verbosity_gemma3_27b_l45.json` and treats `summary.matched_noise_summary.passed_positive_control_gate=true` as the gate pass.

## Verbosity Gate Result

Artifact: `docs/positive_control_verbosity_gemma3_27b_l45.json`

Status: failed. The Gemma 3 27B L45 verbosity run completed cleanly, but all baseline, target, orthogonal, and Gaussian conditions saturated at the generation token cap. The matched-noise token-delta standard deviation was `0.0`, so the length metric cannot validate the steering stack.

Interpretation: keep the artifact as a failed positive-control candidate. Do not use it as the gate for interpreting correctness-steering nulls.

## Active Format Gate

Use a casing/output-format control next: train an upper-vs-lowercase raw L45 direction and evaluate whether steering increases alphabetic uppercase fraction above orthogonal and matched-Gaussian controls.

Script: `scripts/stage2_positive_control_format.py`

Slurm launcher: `scripts/stage2_positive_control_format_27b_L45.sbatch`

Submit:

```bash
sbatch scripts/stage2_positive_control_format_27b_L45.sbatch
```

Expected artifacts:

- `docs/positive_control_format_gemma3_27b_l45.json`
- `results/stage2/positive_control/format_gemma3_27b_l45.jsonl`
- `results/stage2/positive_control/format_gemma3_27b_l45_direction.npz`

The preflight report now prefers `docs/positive_control_format_gemma3_27b_l45.json`; the gate passes only if `summary.matched_noise_summary.passed_positive_control_gate=true`.

## Format Gate Result

Artifact: `docs/positive_control_format_gemma3_27b_l45.json`

Status: passed. The Gemma 3 27B L45 casing/output-format run completed cleanly on Slurm job `456660` with 8 eval prompts and 136 generated rows. The best toward-upper condition moved mean uppercase fraction by `0.835` over baseline, converted all 8 paired rows into uppercase style, and exceeded matched orthogonal/Gaussian controls by about `47.9σ`.

Interpretation: this is the accepted positive-control gate for the current intervention stack. Later correctness-steering nulls may be interpreted only if they keep their own regenerated baselines, orthogonal controls, matched-Gaussian controls, and paired reporting.
