# 27B Feature Candidate Shortlist

Date: 2026-04-29

Purpose: identify a small number of learned-dictionary features worth auditing
or steering. This is not evidence of causality by itself; it is a triage pass
from probe coefficients, task overlap, activation density, and source
interpretability.

## Source Compatibility

| Source | Local artifact | Predictive status | Neuronpedia status | Use |
| --- | --- | --- | --- | --- |
| L45 262K big-L0 exact transcoder | `transcoder_all/layer_45_width_262k_l0_big_affine` | Best single learned dictionary: S1 `0.853/0.893`, S3 `0.854/0.894` | No confirmed matching public dashboard. Do not reuse `45-gemmascope-2-transcoder-262k`, which points to small-L0. | Main local steering candidate source. |
| L45 262K small-L0 exact transcoder | `transcoder_all/layer_45_width_262k_l0_small_affine` | Weaker: S1 `0.795/0.873`, S3 `0.802/0.885` | Matching public source: `45-gemmascope-2-transcoder-262k`. Audit already done. | Neuronpedia-facing interpretability/falsification source. |
| L40 residual SAE small-L0 | `resid_post_all/layer_40_width_{16k,262k}_l0_small` | Weak standalone; small concat gain on S3 subtype | Public L40 residual pages appear to use `l0_medium`, not our cached `l0_small`. | Do not attach Neuronpedia explanations unless we run the matching medium artifact. |

## Big-L0 Local Candidates

These come from
`docs/transcoder_feature_stability_27b_l45_262k_big_affine_exact_top512_s1.json`.
The feature report refits the S1 probe at the saved best C values and ranks
standardized logistic coefficients over train-active features. It reproduces
the sparse probe AUCs: property `0.853`, subtype `0.893`.

| Priority | Feature | Assoc. | Property rank/density | Subtype rank/density | Why it is on the list |
| --- | ---: | --- | ---: | ---: | --- |
| 1 | 72374 | correct | 24 / 0.258 | 5 / 0.153 | Best balanced candidate: positive for both tasks, moderately sparse, not always-on. |
| 2 | 35036 | incorrect | 20 / 0.189 | 16 / 0.392 | Best negative/incorrect candidate with same-sign task overlap and nontrivial but not global density. |
| 3 | 4892 | correct | 17 / 0.273 | 28 / 0.156 | Cross-task positive candidate with moderate density. |
| 4 | 75345 | correct | 4 / 0.227 | not top-100 | Strong property-specific positive feature. |
| 5 | 187589 | correct | 93 / 0.097 | 6 / 0.156 | Subtype-heavy positive feature, fairly sparse. |
| 6 | 45599 | incorrect | not top-100 | 15 / 0.012 | Very sparse subtype-specific negative feature; useful for falsification but may be too rare for steering. |
| 7 | 102057 | correct | 40 / 0.313 | 8 / 0.333 | Stable positive candidate, but denser and lower property rank than the top choices. |

Predictive but less attractive as first steering targets:

| Feature | Assoc. | Reason to deprioritize |
| ---: | --- | --- |
| 18130 | incorrect | Rank 1 property and rank 25 subtype, but density is about 0.98 on both tasks. |
| 8016 | incorrect | Rank 2 property and rank 14 subtype, but density is about 0.93-0.99. |
| 18564 | incorrect | Top-5 on both tasks, but effectively always active. |
| 11145 | correct | Top-10 on both tasks, but effectively always active. |

## Local Mini-Dashboard Audit

The local Neuronpedia substitute is:

- `scripts/stage2_feature_mini_dashboard.py`
- `docs/feature_mini_dashboard_27b_l45_262k_big_affine_top512.json`
- `docs/feature_mini_dashboard_27b_l45_262k_big_affine_top512.md`

It joins top activating rows back to prompts, model outputs, correctness, height,
and error types. It also records one GPT-5.5 qualitative explanation per
shortlisted feature. The GPT pass used six calls and 48,197 total tokens.

Important interpretation detail: coefficient rank comes from the trained sparse
probe, while the AUC below is the univariate score from the feature activation
alone. A feature can therefore have a high probe rank but a weak single-feature
AUC if it is useful through interactions or after the probe combines it with
other features.

| Feature | Probe assoc. | Property AUC | Subtype AUC | Audit interpretation | Steering priority |
| ---: | --- | ---: | ---: | --- | --- |
| 72374 | correct | 0.641 | 0.759 | Direct/simple universal generalization; strongest top examples are height-1, so this is heavily height/template-confounded. | Low as a reasoning target; useful as a surface-confound control. |
| 35036 | incorrect | 0.400 | 0.343 | Error-associated complex common-supertype/fan-in prompts; often wrong-direction or over-enumerated generations. | Medium; best negative/error candidate. |
| 4892 | correct | 0.524 | 0.488 | Common-superclass hypothesis candidate, but weak univariate correctness signal and height confounding. | Medium-low. |
| 75345 | correct | 0.548 | 0.516 | Moderate-depth common-superclass pattern; property coefficient rank is strong but correctness evidence is mixed. | Medium for property-specific steering. |
| 187589 | correct | 0.574 | 0.754 | Simple universal generalization, especially subtype; substantial low-height confounding. | Medium for subtype, but report confound clearly. |
| 45599 | incorrect | 0.488 | 0.497 | Sparse fan-in/exhaustive-hypothesis-risk feature; too rare and weak alone. | Low; useful mainly as falsification. |

## Neuronpedia-Facing Candidates

The existing exact small-L0 L45 262K audit is:

- `docs/neuronpedia_transcoder_audit_27b_l45_262k_exact.json`
- `docs/neuronpedia_transcoder_audit_27b_l45_262k_exact.md`

Those top features mostly look generic, lexical, or code/style related rather
than ontology-reasoning specific. The only small-L0 feature worth keeping as a
possible falsification example is property feature `53191` ("for all"), because
it is at least semantically adjacent to logic/quantification. It is not strong
enough to be the main steering target.

## Steering Implication

If we run learned-feature steering next, use the mini-dashboard result to keep
the pilot small and interpretable:

- Primary negative/error feature: `35036`.
- Primary property-specific positive feature: `75345`.
- Primary subtype-positive feature: `187589`.
- Surface-confound control: `72374`, because it has high subtype AUC but appears
  dominated by height-1 direct-generalization examples.

For each feature, use both amplify and suppress conditions, compare against
random same-density or coefficient-matched controls, and report this as
candidate causal validation rather than proof of a discovered reasoning feature.

Prepared steering scripts:

- `scripts/stage2_steer_transcoder_features.py`: generic learned-feature
  steering driver for affine transcoders. It adds selected decoder rows to
  `blocks.45.hook_mlp_out`, scaled by cached feature activations.
- `scripts/stage2_steer_big_l0_features_27b_L45_property_pilot.sbatch`: property
  pilot for `35036`, `75345`, and `72374`.
- `scripts/stage2_steer_big_l0_features_27b_L45_subtype_pilot.sbatch`: subtype
  pilot for `35036`, `187589`, and `72374`.
- `scripts/stage2_steer_raw_27b_L45_property_decode_sweep.sbatch`: Cox-style
  raw correctness-direction comparator. This is not a learned feature; it tests
  whether the dense raw probe direction is causally useful under a stronger
  decode-step protocol.
