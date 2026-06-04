# Intervention Preflight

Purpose: Part 1 gate for the next causal-intervention experiments.

Status: `started`

## Objective

Freeze current Gemma/Qwen evidence and build the preflight gate for interpretable causal-intervention tests.

## Gates

| Gate | Status | Required | Passes For Future Jobs | Next Action |
| --- | --- | --- | --- | --- |
| `current_evidence_freeze` | branch pushed pending merge or pr | yes | no | Merge or open a PR for codex/name-scramble-error-steering before treating the current evidence as frozen on main. |
| `regenerated_balanced_baseline` | historical small n available refresh required | yes | no | Regenerate balanced h3/h4 Gemma 3 27B infer_property baselines in the same job as each new intervention family; use paired regenerated rows, not historical labels, for flips. |
| `positive_control_steering` | passed | yes | yes | Use the format positive-control artifact as the intervention-stack gate for later correctness-steering nulls. |
| `orthogonal_direction_control` | implemented and historical available | yes | yes | Keep orthogonal controls in every optimized, DAS, decode-time, and patch-validation intervention report where the method permits a direction control. |
| `matched_gaussian_noise_control` | raw direction scaffolded pending gpu rerun | yes | no | Rerun the relevant Gemma 27B steering job with `gaussian` conditions before claiming repair exceeds matched noise. |
| `paired_flip_and_parse_reporting` | schema available needs enforcement per method | yes | no | Every new JSON report must include paired false-to-true/true-to-false flips, parse-fail rate, matched-noise summary, and a causal_abstraction_claim. |
| `exact_patch_validation` | pending for attribution rankings | method-specific | no | When AtP*/attribution rankings are added, validate top-ranked sites with exact patching before treating localization as causal evidence. |

## Next Jobs

| Priority | Purpose | Command |
| --- | --- | --- |
| 1 | Verify A40 availability before long jobs. | `srun -A gpu --constraint=J --gres=gpu:1 --time=00:03:00 --ntasks=1 --cpus-per-task=1 --mem=12G --immediate=60 bash -lc 'hostname; nvidia-smi -L'` |
| 2 | Run the active format positive-control steering gate; answer-property and verbosity candidates did not pass. | `sbatch scripts/stage2_positive_control_format_27b_L45.sbatch` |
| 3 | Refresh raw-direction correctness steering with regenerated baseline, orthogonal, and matched-Gaussian controls. | `sbatch scripts/stage2_steer_raw_27b_L45_property_decode_sweep.sbatch` |
| 4 | Refresh reconstruction-error steering with regenerated baseline, orthogonal, and matched-Gaussian controls if the raw preflight passes. | `sbatch scripts/stage2_steer_error_27b_L45_property_decode_sweep.sbatch` |

## Interpretation Rule

Do not interpret optimized vectors, DAS, decode-time correction, or AtP* nulls unless regenerated baselines, positive controls, orthogonal controls, and matched-noise controls are present for the relevant method.

## Notes

- Historical steering reports remain useful context but do not pass the full preflight because matched Gaussian/noise controls and a declared positive-control gate were not yet in place.
- The answer-property steering artifact is a failed positive-control candidate, not a gate: it did not produce target-directed free-form answer movement.
- The failed verbosity gate is `docs/positive_control_verbosity_gemma3_27b_l45.json`; the active format gate is `scripts/stage2_positive_control_format_27b_L45.sbatch` and writes `docs/positive_control_format_gemma3_27b_l45.json`.
- Future Qwen comparisons should still label local dictionaries as local stand-ins, not first-party Qwen Scope artifacts.
