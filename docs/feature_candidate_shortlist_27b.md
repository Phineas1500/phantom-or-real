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

If we run learned-feature steering next, start with the big-L0 local candidates:
`72374`, `35036`, `4892`, and `75345`. Treat `187589` and `45599` as subtype
follow-ups. For each feature, use both amplify and suppress conditions, compare
against random same-density or coefficient-matched controls, and report this as
candidate causal validation rather than proof of a discovered reasoning feature.

