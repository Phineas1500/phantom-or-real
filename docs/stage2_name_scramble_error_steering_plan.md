# Stage 2 Name-Scramble And Reconstruction-Error Steering Results

This note records the two highest-leverage discussion/limitations follow-ups:

1. name-scrambled regeneration for Gemma 3 27B; and
2. Gemma 3 27B reconstruction-error direction steering.

Both are now completed for the scoped 27B target.

## Name-Scrambled Regeneration

Primary job:

```bash
sbatch scripts/stage2_gemma3_27b_namescramble.sbatch
```

Completed as Slurm job `456306` on 2026-05-24.

Scope:

- model: `google/gemma-3-27b-it`
- source rows: existing `results/full/with_errortype/gemma3_27b_{task}.jsonl`
- tasks: `infer_property`, `infer_subtype`
- heights: `1 2 3 4`
- conditions: `nonce`, `natural`
- sample: `PER_HEIGHT=250` per task/height/condition
- raw activation layer: L45

Outputs:

- prepared/rescored rows: `data/stage2/namescramble_27b/{condition}/{task}_h{height}.jsonl`
- regenerated scored rows: `results/stage2/namescramble_infer_27b/{condition}/{task}_h{height}.jsonl`
- combined scored rows: `results/stage2/namescramble_infer_27b/{condition}/{task}.jsonl`
- scrambled L45 activations: `results/stage2/activations_namescramble_27b/gemma3_27b_{condition}_{task}_L45.*`
- fixed original-probe evaluation: `docs/namescramble_27b_l45_raw_probe_s1.json`

Implementation note: the completed generations were rescored after fixing simple
plural replacement in name-scrambled reference fields, e.g. `Bempins` now maps to
`Madelines` rather than leaving a stale source symbol in `ground_truth`. The
nonce-pool builder now sorts symbols for future deterministic preparation. The
current report uses the completed generations' own stored mappings, so no 27B
inference rerun was needed.

### Corrected Behavioral Accuracy

| Condition | Task | h1 strong | h2 strong | h3 strong | h4 strong | all strong | parse fail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| natural | `infer_property` | 0.836 | 0.236 | 0.112 | 0.128 | 0.328 | 0.124 |
| natural | `infer_subtype` | 0.812 | 0.188 | 0.124 | 0.104 | 0.307 | 0.165 |
| nonce | `infer_property` | 0.808 | 0.172 | 0.108 | 0.048 | 0.284 | 0.130 |
| nonce | `infer_subtype` | 0.704 | 0.164 | 0.052 | 0.052 | 0.243 | 0.123 |

Name scrambling makes the task behaviorally harder, especially for nonce names
and deeper rows. This supports keeping the semantic-shortcut limitation in the
paper: natural lexical familiarity and/or source naming conventions do matter.
It does not, by itself, explain away the activation result.

### Fixed Original Raw-Probe Transfer

The original Gemma 3 27B L45 raw probe was trained on the original S1 split and
then evaluated on scrambled activations. The matched-original subset is the same
source-row subset before scrambling, after parse-clean filtering and activation
availability.

| Task | Condition | Scrambled AUC | Matched-original AUC | AUC drop | Scrambled strong | Matched-original strong | Strong delta | Kept rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `infer_property` | natural | 0.835 | 0.976 | 0.141 | 0.374 | 0.598 | -0.223 | 876 |
| `infer_property` | nonce | 0.844 | 0.977 | 0.132 | 0.326 | 0.615 | -0.289 | 870 |
| `infer_subtype` | natural | 0.842 | 0.985 | 0.143 | 0.368 | 0.407 | -0.039 | 835 |
| `infer_subtype` | nonce | 0.830 | 0.982 | 0.152 | 0.277 | 0.393 | -0.116 | 877 |

Interpretation: the raw correctness direction transfers under full
name-scrambling, but with a real AUC loss of about `0.13-0.15` against matched
original rows. That is a useful middle result. It rules out the strongest
critique that the raw probe is only reading fixed target-name semantics, while
also showing that symbol naming and prompt surface do carry part of the
predictive structure.

For the report, this should strengthen rather than replace the S3 heldout-target
claim: S3 showed the signal survives heldout target symbols; full regeneration
shows it also survives fresh names, with measurable degradation.

## Reconstruction-Error Steering

Primary job:

```bash
sbatch scripts/stage2_steer_error_27b_L45_property_decode_sweep.sbatch
```

Completed as Slurm job `456305` on 2026-05-24.

Scope:

- model: `google/gemma-3-27b-it`
- task: `infer_property`
- steering site: `blocks.45.hook_resid_post`
- direction source: `results/stage2/sae_reconstructions/gemma3_27b_infer_property_L45_layer_45_width_262k_l0_small_top128_error.safetensors`
- rows: S1 h3/h4, `PER_HEIGHT_LABEL=2`
- strengths: `-0.25,-0.5,-1,0.25,0.5,1` train-projection SD
- controls: regenerated baseline and norm-matched orthogonal direction

Outputs:

- rows: `results/stage2/steering/error_l45_layer_45_width_262k_l0_small_top128_property_decode_sweep.jsonl`
- direction: `results/stage2/steering/error_l45_layer_45_width_262k_l0_small_top128_property_decode_sweep_direction.npz`
- summary: `docs/error_l45_layer_45_width_262k_l0_small_top128_property_decode_sweep.json`

The error-direction probe itself recovered the expected near-raw offline signal:
`test_auc=0.8973`, `val_auc=0.8802`, and train-projection SD `32.16` over
`10,025` parse-clean rows.

Steering result:

| Direction family | Strengths | Paired strong false-to-true | Paired true-to-false | Net accuracy delta |
| --- | --- | ---: | ---: | ---: |
| reconstruction-error direction | +/-0.25, +/-0.5, +/-1 SD | 0 at every strength | 0 at every strength | 0.000 at every strength |
| orthogonal control | -0.25, -0.5 SD | 0 | 1 each | -0.125 each |
| orthogonal control | remaining strengths | 0 | 0 | 0.000 |

The regenerated baseline was `3/8` strong-correct and `5/8` weak-correct with no
parse failures. Historical labels wobbled under regeneration, so intervention
claims should be made against the paired regenerated baseline, not against the
old source labels alone.

Interpretation: this closes the focused causal test of the Section 5.4
reconstruction-error pivot. The reconstruction-error subspace remains a strong
predictive diagnostic, but calibrated decode-step steering along that direction
did not repair or systematically break free-form answers above controls.

## Paper-Level Meaning

These follow-ups sharpen the discussion section in two ways.

First, the name-scramble result weakens a purely lexical shortcut critique but
does not eliminate it. The signal survives regenerated names at AUC `0.83-0.84`,
so pre-generation activations are still tracking something relevant to whether
the model will solve the regenerated problem. The matched-original AUC drop of
about `0.13-0.15` says surface naming and semantic familiarity are part of the
readout. The right language is therefore: the correctness probe is not merely a
fixed-name detector, but it is also not name-invariant.

Second, reconstruction-error steering strengthens the predictive-versus-causal
thesis. Section 5.4 identified a striking predictive residual left outside the
SAE reconstruction. The new steering run tested the most direct causal version
of that result and found no beneficial control. That supports treating the
reconstruction-error direction as a diagnostic of missing or distorted
information in the dictionary representation, not as an established control
axis for generation.

## Remaining Low-Cost Extensions

The two explicit high-leverage limitations are now addressed. The remaining
useful additions are analysis/documentation rather than broad new GPU sweeps:

1. Reanalyze weak accuracy or Occam/quality-style scores, since the paper uses a
   binary strong label rather than Sun & Saparov's graded parsimony metric.
2. Add a baseline-regeneration stability paragraph: deterministic regeneration
   can flip some historical strong labels, so causal intervention claims should
   use paired regenerated baselines.
3. Only expand patching pairs if the report needs more statistical support for
   the h4-to-h1 asymmetric disruption effect.

I would not run another generic variation unless one of these analyses exposes a
specific failure mode. The current evidence is already coherent: the signal is
real, partly name/surface-sensitive, and still not converted into causal control
by the tested steering interventions.
