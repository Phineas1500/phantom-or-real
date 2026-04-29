# Stage 2 Results Pack

This is the compact 27B results pack for the final report. Gemma 3 4B is
deferred until teammate results are available.

## Behavioral Context

Stage 1 used 11,000 rows per 27B task, split across heights 1-4.

| Task | h1 strong | h2 strong | h3 strong | h4 strong | h4 weak | h4 parse fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `infer_property` | 0.960 | 0.577 | 0.392 | 0.264 | 0.424 | 0.104 |
| `infer_subtype` | 0.973 | 0.325 | 0.114 | 0.055 | 0.240 | 0.014 |

Use this as Figure/Table 1: accuracy collapses monotonically with ontology
height, giving a behavioral success/failure label with real depth structure.

## Metadata Baselines

| Split | Task | Best B0 feature set | Test AUC |
| --- | --- | --- | ---: |
| S1 random | `infer_property` | `b0_prompt` | 0.743 |
| S1 random | `infer_subtype` | `b0_height` | 0.841 |
| S3 target-symbol heldout | `infer_property` | `b0_namefreq` | 0.711 |
| S3 target-symbol heldout | `infer_subtype` | `b0_prompt` | 0.859 |

Activation claims should be stated as deltas over these baselines, not only
absolute AUCs.

## Raw Residual Probes

Best layer is L45 for both 27B tasks.

| Split | Task | Raw L45 test AUC | 95% bootstrap CI | Delta vs B0 |
| --- | --- | ---: | --- | ---: |
| S1 random | `infer_property` | 0.897 | [0.881, 0.912] | +0.153 |
| S1 random | `infer_subtype` | 0.914 | [0.896, 0.932] | +0.073 |
| S3 target-symbol heldout | `infer_property` | 0.884 | [0.868, 0.901] | +0.173 |
| S3 target-symbol heldout | `infer_subtype` | 0.917 | [0.898, 0.934] | +0.058 |

Label-shuffle control stayed near chance on S1: property 0.493 and subtype
0.481. The main raw-probe claim is therefore robust to metadata baselines,
label shuffle, and heldout target symbols.

## Cross-Task Transfer

| Split | Direction | Source test AUC | Target test AUC |
| --- | --- | ---: | ---: |
| S1 random | property -> subtype | 0.897 | 0.862 |
| S1 random | subtype -> property | 0.914 | 0.786 |
| S3 target-symbol heldout | property -> subtype | 0.884 | 0.846 |
| S3 target-symbol heldout | subtype -> property | 0.917 | 0.788 |

Interpretation: there is shared success/failure signal across tasks, but it is
not fully task invariant.

## Residual SAE Probes

All rows use L45 Gemma Scope 2 residual SAEs with top-128 features. Top-128
already captured every nonzero active feature in the tested files, so the
top-512 diagnostic did not change metrics.

| Split | Task | Width 16K AUC | Width 262K AUC | Raw L45 AUC |
| --- | --- | ---: | ---: | ---: |
| S1 random | `infer_property` | 0.786 | 0.806 | 0.897 |
| S1 random | `infer_subtype` | 0.876 | 0.870 | 0.914 |
| S3 target-symbol heldout | `infer_property` | 0.799 | 0.779 | 0.884 |
| S3 target-symbol heldout | `infer_subtype` | 0.865 | 0.867 | 0.917 |

Interpretation: residual SAE features retain some predictive signal, but they
trail raw residual probes. On S3 subtype they only barely clear the stronger B0
baseline.

## MLP-Output Site Pilot

The L45 post-MLP site was extracted at
`blocks.45.ln2_post.hook_normalized` and probed with the same S1/S3 splits. The
raw MLP-output activations carry essentially the same signal as raw residuals,
but the Gemma Scope 2 `mlp_out_all` width-16K SAE exposes little of it.

| Split | Task | Raw `mlp_out` AUC | MLP-out SAE 16K AUC | Raw residual L45 AUC |
| --- | --- | ---: | ---: | ---: |
| S1 random | `infer_property` | 0.895 | 0.577 | 0.897 |
| S1 random | `infer_subtype` | 0.916 | 0.674 | 0.914 |
| S3 target-symbol heldout | `infer_property` | 0.892 | 0.550 | 0.884 |
| S3 target-symbol heldout | `infer_subtype` | 0.915 | 0.702 | 0.917 |

Interpretation: moving from residual stream to MLP-output activations does not
rescue the tested sparse dictionary story. The behaviorally relevant signal is
visible in raw activations at multiple sites, but not well exposed by the
tested Gemma Scope residual or MLP-output sparse features.

## Skip-Transcoder Pilot

The L45 pre-MLP normalized site was extracted at
`blocks.45.ln2.hook_normalized` and encoded with the Gemma Scope 2 affine
skip-transcoder `transcoder_all/layer_45_width_16k_l0_small_affine`. The raw
`mlp_in` activations again carry raw-residual-level signal, while
skip-transcoder features are intermediate: stronger than the MLP-output SAE,
but still below raw activations and not a clean rescue of the sparse-feature
story.

| Split | Task | Raw `mlp_in` AUC | Skip-transcoder 16K AUC | Raw residual L45 AUC |
| --- | --- | ---: | ---: | ---: |
| S1 random | `infer_property` | 0.897 | 0.722 | 0.897 |
| S1 random | `infer_subtype` | 0.915 | 0.821 | 0.914 |
| S3 target-symbol heldout | `infer_property` | 0.885 | 0.722 | 0.884 |
| S3 target-symbol heldout | `infer_subtype` | 0.914 | 0.841 | 0.917 |

Interpretation: computation-oriented sparse features expose some correctness
signal, but the tested skip-transcoder still misses a large part of the
raw-activation signal. A crosscoder pilot should be treated as optional future
work unless the report needs one more explicit multi-layer check.

## Reconstruction/Error Diagnostic

| Split | Task | SAE width | Energy explained | Reconstruction AUC | Error AUC | Raw L45 AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| S1 random | `infer_property` | 16K | 0.948 | 0.786 | 0.894 | 0.897 |
| S1 random | `infer_property` | 262K | 0.955 | 0.806 | 0.897 | 0.897 |
| S1 random | `infer_subtype` | 16K | 0.948 | 0.877 | 0.916 | 0.914 |
| S1 random | `infer_subtype` | 262K | 0.954 | 0.870 | 0.915 | 0.914 |
| S3 target-symbol heldout | `infer_property` | 16K | 0.948 | 0.799 | 0.881 | 0.884 |
| S3 target-symbol heldout | `infer_property` | 262K | 0.955 | 0.788 | 0.886 | 0.884 |
| S3 target-symbol heldout | `infer_subtype` | 16K | 0.948 | 0.865 | 0.916 | 0.917 |
| S3 target-symbol heldout | `infer_subtype` | 262K | 0.954 | 0.867 | 0.914 | 0.917 |

Main mechanistic result: the SAEs reconstruct about 95% of residual energy, but
the raw-minus-reconstruction error recovers almost the entire raw-probe signal.
The strongest correctness direction is therefore mostly outside the tested
decoded residual-SAE subspace.

## Steering Pilot

The bounded 27B `infer_property` steering pilot used 8 balanced S1 test rows,
prompt-only L45 raw-direction interventions at +/-2 SD, and matched orthogonal
controls.

| Condition | Strong accuracy | Weak accuracy | Parse fail rate | Flips vs baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.375 | 0.625 | 0.000 | NA |
| raw +/-2 SD | 0.375 | 0.625 | 0.125 | 0 |
| orthogonal +/-2 SD | 0.375 | 0.625 | 0.125 | 0 |

Interpretation: this is a useful causal/plumbing check, but it is not positive
causal evidence. All generations hit the token cap, and the one output change
appeared under both raw and orthogonal steering.

## Current Report Claim

Gemma 3 27B pre-generation residuals contain a robust signal for
reasoning-task success/failure beyond metadata baselines. The tested Gemma
Scope 2 residual SAEs partially expose this signal, but reconstruction/error
diagnostics show the strongest correctness-predictive component is concentrated
in the small residual subspace that those SAEs fail to reconstruct. Causal
steering is currently inconclusive.
