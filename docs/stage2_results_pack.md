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

## MLP-Output Site

The L45 post-MLP site was extracted at
`blocks.45.ln2_post.hook_normalized` and probed with the same S1/S3 splits. The
raw MLP-output activations carry essentially the same signal as raw residuals,
but the Gemma Scope 2 `mlp_out_all` width-16K SAE exposes little of it.
Later exact-hook auditing for the 262K transcoder showed this bare normalized
site is missing the learned RMSNorm weight relative to `blocks.45.hook_mlp_out`,
so that first MLP-output SAE row is superseded.

The exact-hook rerun encodes `blocks.45.hook_mlp_out` with the same
`mlp_out_all/layer_45_width_16k_l0_small` SAE. Its feature density is no longer
pathological: mean L0 is about `23.6` for property and `22.5` for subtype,
compared with exactly `2.0` under the old bare-normalized site.

| Split | Task | Raw exact `mlp_out` AUC | Exact MLP-out SAE 16K AUC | Old MLP-out SAE pilot AUC | Raw residual L45 AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| S1 random | `infer_property` | 0.896 | 0.811 | 0.577 | 0.897 |
| S1 random | `infer_subtype` | 0.916 | 0.878 | 0.674 | 0.914 |
| S3 target-symbol heldout | `infer_property` | 0.892 | 0.807 | 0.550 | 0.884 |
| S3 target-symbol heldout | `infer_subtype` | 0.915 | 0.879 | 0.702 | 0.917 |

Interpretation: the behaviorally relevant signal is visible in raw activations
at multiple sites. Exact-hook MLP-output SAE features are meaningful and roughly
residual-SAE-like, but still trail raw exact `hook_mlp_out`.

## Skip-Transcoder Pilot

The first L45 16K skip-transcoder pilot encoded the bare normalized site
`blocks.45.ln2.hook_normalized` with
`transcoder_all/layer_45_width_16k_l0_small_affine`. Exact-hook audits later
showed that the fair input is learned-weighted
`ln2.hook_normalized * ln2.w`, with component diagnostics compared against
exact `blocks.45.hook_mlp_out`. We therefore reran the 16K affine
skip-transcoder with the corrected hook convention.

We also ran the Neuronpedia-visible L45 262K affine transcoder
`transcoder_all/layer_45_width_262k_l0_small_affine`. The first run reused the
bare `ln2.hook_normalized` input and is now superseded: a hook audit showed the
Gemma Scope 2 target is better matched by learned-weighted
`ln2.hook_normalized * ln2.w` and `blocks.45.hook_mlp_out`, with the affine
skip applied as `x @ W_skip`. The corrected exact-hook rerun is Scholar job
`451226`.

The corrected transcoders are no longer weak. Exact 16K improves substantially
over the old bare-normalized 16K pilot, and exact 262K is slightly stronger
than exact 16K on most sparse-latent comparisons. Both still trail raw same-site
activations.

| Split | Task | Raw exact `mlp_in` AUC | Old 16K bare-norm AUC | Exact 16K AUC | Exact 262K AUC | Raw residual L45 AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S1 random | `infer_property` | 0.897 | 0.722 | 0.787 | 0.795 | 0.897 |
| S1 random | `infer_subtype` | 0.916 | 0.821 | 0.868 | 0.873 | 0.914 |
| S3 target-symbol heldout | `infer_property` | 0.885 | 0.722 | 0.785 | 0.802 | 0.884 |
| S3 target-symbol heldout | `infer_subtype` | 0.914 | 0.841 | 0.880 | 0.885 | 0.917 |

Interpretation: computation-oriented sparse features expose some correctness
signal, and exact hook/scale alignment materially improves both tested
transcoder widths. The main conclusion is still unchanged: even corrected
transcoder features do not close the gap to raw activations.

## 262K Transcoder Component Diagnostic

We split the corrected exact-hook L45 262K affine transcoder into decoded
sparse latents, the affine skip path, the full latent+skip output, and
`target - full` against the exact `blocks.45.hook_mlp_out` target. This was a
quick diagnostic run with `C=1.0`, `liblinear`, and no bootstrap resampling.
The full component now has interpretable reconstruction statistics: energy
explained is `0.672` for property and `0.661` for subtype, with global cosine
`0.821/0.814`.

| Split | Task | Latent AUC | Affine skip AUC | Full AUC | Error AUC | Raw exact `mlp_in` AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S1 random | `infer_property` | 0.791 | 0.856 | 0.862 | 0.864 | 0.897 |
| S1 random | `infer_subtype` | 0.867 | 0.890 | 0.897 | 0.888 | 0.916 |
| S3 target-symbol heldout | `infer_property` | 0.796 | 0.844 | 0.851 | 0.861 | 0.885 |
| S3 target-symbol heldout | `infer_subtype` | 0.883 | 0.882 | 0.889 | 0.886 | 0.914 |

Interpretation: the old negative-energy diagnostic was a target-site/scale
mismatch, not a property of the artifact. With exact hooks, the 262K
transcoder reconstructs a substantial part of the MLP output and exposes
moderate-to-strong correctness signal. It still does not reach the raw exact
input/output probes, so it is supporting evidence for partial sparse
localization rather than a complete sparse mechanism.

For the exact-hook 16K skip-transcoder, the same component split is
interpretable but reconstructs less target energy than 262K: full energy
explained is `0.639/0.638` for property/subtype. Exact 16K latent/skip/full/error
AUCs are S1 property `0.782/0.854/0.854/0.857`, S1 subtype
`0.868/0.888/0.890/0.889`, S3 property `0.781/0.841/0.848/0.855`, and S3
subtype `0.883/0.884/0.889/0.891`. This supports a small width benefit but not
a qualitative bridge to raw activations.

## Sparse Feature-Family Concat

We tested whether residual sparse features and corrected exact transcoder
features are complementary by concatenating sparse top-k matrices and training
the same split-aware logistic probes.

| Split | Task | Residual 262K + exact TC 262K | L45 four-block low-C | L45 five-block | L30+L45 all sparse | Raw exact `mlp_out` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S1 random | `infer_property` | 0.815 | 0.830 | 0.832 | 0.839 | 0.896 |
| S1 random | `infer_subtype` | 0.870 | 0.888 | 0.883 | 0.887 | 0.916 |
| S3 target-symbol heldout | `infer_property` | 0.800 | 0.828 | 0.829 | 0.834 | 0.892 |
| S3 target-symbol heldout | `infer_subtype` | 0.881 | 0.888 | 0.889 | 0.892 | 0.915 |

`L45 four-block low-C` means residual SAE 16K + residual SAE 262K + exact 262K
transcoder + exact MLP-output SAE 16K, with validation-selected `C` from the
low regularization grid. `L45 five-block` adds the corrected exact-hook 16K
transcoder. `L30+L45 all sparse` adds L30 residual SAE 16K/262K on top of the
five L45 blocks and uses the broader C grid
`0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10`.
Its 95% bootstrap CIs are S1 property/subtype
`0.819-0.858`/`0.862-0.909` and S3 property/subtype
`0.813-0.856`/`0.869-0.913`.

The L30-only residual checks explain the multi-layer gain. Standalone L30
features are weak for property but carry useful complementary signal: L30 16K
is S1/S3 `0.752/0.860` and `0.748/0.860`, L30 262K is
`0.770/0.867` and `0.771/0.866`, and L30 residual 16K+262K concat is
`0.786/0.872` and `0.788/0.864`.

Interpretation: sparse feature families are complementary, especially for
property. Adding L30 residual features to the corrected L45 sparse family gives
the strongest sparse-only property result so far and the strongest S3 subtype
result, but the combined sparse representation still trails raw exact
activations. This narrows the gap; it does not erase it.

## Crosscoder Pilot

Scholar job `451181` completed a bounded 27B crosscoder pilot over residual
layers `{16,31,40,53}` with
`crosscoder/layer_16_31_40_53_width_65k_l0_medium`. The fair comparison is the
raw-concat baseline over exactly those same layers.

| Split | Task | Raw concat AUC | Crosscoder 65K AUC | Raw residual L45 AUC |
| --- | --- | ---: | ---: | ---: |
| S1 random | `infer_property` | 0.893 | 0.787 | 0.897 |
| S1 random | `infer_subtype` | 0.904 | 0.868 | 0.914 |
| S3 target-symbol heldout | `infer_property` | 0.883 | 0.724 | 0.884 |
| S3 target-symbol heldout | `infer_subtype` | 0.903 | 0.853 | 0.917 |

Interpretation: the multi-layer crosscoder does not rescue the sparse-feature
localization story. Raw concat over the crosscoder layers nearly matches raw
L45, but the 65K crosscoder features trail raw concat on every task/split and
are especially weak for S3 property. This strengthens the conclusion that the
tested Gemma Scope sparse dictionaries expose only part of the correctness
signal.

## Cross-Method Sparse Feature Comparison

This table is the clearest compact view of where crosscoders sit relative to
every feature source tried so far.

| Method | S1 property | S1 subtype | S3 property | S3 subtype |
| --- | ---: | ---: | ---: | ---: |
| Metadata B0 | 0.743 | 0.841 | 0.711 | 0.859 |
| Raw L45 residual | 0.897 | 0.914 | 0.884 | 0.917 |
| Raw concat L16/31/40/53 | 0.893 | 0.904 | 0.883 | 0.903 |
| Residual SAE 16K | 0.786 | 0.876 | 0.799 | 0.865 |
| Residual SAE 262K | 0.806 | 0.870 | 0.779 | 0.867 |
| MLP-out SAE 16K old bare-norm | 0.577 | 0.674 | 0.550 | 0.702 |
| Exact MLP-out SAE 16K | 0.811 | 0.878 | 0.807 | 0.879 |
| Skip-transcoder 16K | 0.722 | 0.821 | 0.722 | 0.841 |
| Exact skip-transcoder 16K | 0.787 | 0.868 | 0.785 | 0.880 |
| Exact transcoder 262K | 0.795 | 0.873 | 0.802 | 0.885 |
| L30 residual SAE 16K | 0.752 | 0.860 | 0.748 | 0.860 |
| L30 residual SAE 262K | 0.770 | 0.867 | 0.771 | 0.866 |
| L30 residual concat | 0.786 | 0.872 | 0.788 | 0.864 |
| All L45 sparse concat low-C | 0.830 | 0.888 | 0.828 | 0.888 |
| All L45 sparse concat + exact TC16K | 0.832 | 0.883 | 0.829 | 0.889 |
| L30+L45 residual concat | 0.827 | 0.889 | 0.816 | 0.874 |
| L30+L45 all sparse concat | 0.839 | 0.887 | 0.834 | 0.892 |
| Crosscoder 65K | 0.787 | 0.868 | 0.724 | 0.853 |

Interpretation: the L30+L45 all-sparse concat is now the strongest sparse-only
result, especially for property, but it still trails raw activations. The main
ordering is therefore raw activations/reconstruction-error probes first,
multi-block sparse concat next, the best individual sparse dictionaries after
that, crosscoders lower on S3, and the old bare-normalized MLP-output SAE last.

## Dense Active-Feature Probe Check

To test whether sparse CSR scaling was causing the raw-vs-sparse gap, we
reran the sparse artifact probes after selecting train-active feature columns,
materializing them as dense matrices, and using ordinary centered scaling.

| Method | S1 property | S1 subtype | S3 property | S3 subtype |
| --- | ---: | ---: | ---: | ---: |
| Residual SAE 16K dense-active | 0.787 | 0.877 | 0.799 | 0.865 |
| Residual SAE 262K dense-active | 0.806 | 0.870 | 0.779 | 0.868 |
| MLP-out SAE 16K dense-active | 0.617 | 0.740 | 0.611 | 0.763 |
| Skip-transcoder 16K dense-active | 0.722 | 0.821 | 0.724 | 0.841 |
| Transcoder 262K old bare-norm dense-active | 0.654 | 0.811 | 0.653 | 0.831 |
| Crosscoder 65K dense-active | 0.786 | 0.868 | 0.725 | 0.854 |
| Exact MLP-out SAE 16K dense-active | 0.814 | 0.880 | 0.805 | 0.879 |
| Exact transcoder 262K dense-active | 0.800 | 0.878 | 0.805 | 0.883 |
| Four-block concat low-C dense-active | 0.831 | 0.888 | 0.828 | 0.887 |

Interpretation: dense centering/scaling does not remedy the disparity. Residual
SAE, old bare-normalized transcoder, and crosscoder AUCs are essentially
unchanged from the standard sparse probes. MLP-out SAE improves from extremely
weak to still weak, but remains far below raw `mlp_out`. This rules out
sparse-matrix scaling as the main explanation for the raw-vs-sparse gap. For
the 262K transcoder specifically, the later exact-hook rerun shows the dominant
issue was hook/scale alignment, not sparse matrix scaling. Repeating the
dense-active check after exact-hook correction gives only tiny individual
artifact gains and leaves the four-block concat essentially tied with the
low-C sparse concat (`0.830/0.888` S1, `0.828/0.888` S3).

## Dtype And Feature-Audit Sanity Checks

Scholar job `451218` compared the existing bfloat16 sparse encodings against
float32 re-encodings of the first 512 rows for the main sparse artifacts. The
comparison checks L0 stability, active-feature overlap, and top feature ID
agreement.

| Method | Task | bf16 L0 | fp32 L0 | Active Jaccard | Top-1 match |
| --- | --- | ---: | ---: | ---: | ---: |
| Residual SAE 16K | `infer_property` | 17.570 | 17.629 | 0.995 | 0.998 |
| Residual SAE 16K | `infer_subtype` | 17.963 | 18.012 | 0.995 | 0.988 |
| MLP-out SAE 16K | `infer_property` | 2.000 | 2.000 | 1.000 | 1.000 |
| MLP-out SAE 16K | `infer_subtype` | 2.000 | 2.000 | 1.000 | 1.000 |
| Skip-transcoder 16K | `infer_property` | 5.332 | 5.379 | 0.991 | 1.000 |
| Skip-transcoder 16K | `infer_subtype` | 6.043 | 6.102 | 0.990 | 1.000 |
| Crosscoder 65K | `infer_property` | 43.988 | 44.201 | 0.994 | 0.984 |
| Crosscoder 65K | `infer_subtype` | 41.678 | 41.930 | 0.992 | 0.994 |

Interpretation: bfloat16 encoding is not causing the sparse/raw gap. Active
sets are nearly identical under fp32 re-encoding, and the weak MLP-output/skip
L0 values persist exactly or nearly exactly in fp32.

Neuronpedia is not a direct audit route for the present top L45 residual SAE
features. Its public `gemma-3-27b-it` residual dashboards currently cover layers
`16`, `31`, `40`, and `53`; our main residual SAE feature probes are layer `45`.
We therefore used the public all-layer `gemmascope-2-transcoder-262k` source for
an L45 Neuronpedia audit, and reran that audit after the exact-hook correction.
The corrected top features differ from the old bare-normalized audit, but still
look mostly generic or lexical rather than ontology-reasoning-specific:

- Property top explanations include "exhibit", "for all", "okay/affirmation",
  "technical contexts", "abstract concepts", "assuming", "exactly", and
  "model".
- Subtype top explanations include "code and structure", "explanatory
  fragments", "Code keywords", "pronouns and scope", "entity classification",
  "technical contexts", and "American".

Interpretation: the Neuronpedia-facing 262K transcoder audit reinforces the
same cautionary story. It provides inspectable L45 dashboard links, but the
features look like broad lexical/style/code correlates, not clean sparse
ontology-reasoning features.

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
in the small residual subspace that those SAEs fail to reconstruct. A corrected
exact-hook 262K transcoder partially improves the sparse-feature picture but
still trails raw activations, and Neuronpedia-facing top features remain
generic rather than clean ontology-reasoning mechanisms. Causal steering is
currently inconclusive.
