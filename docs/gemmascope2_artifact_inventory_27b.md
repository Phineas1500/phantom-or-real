# Gemma Scope 2 27B Artifact Inventory

- Repository: `google/gemma-scope-2-27b-it`
- Generated: `2026-04-29T23:13:26.960649+00:00`
- Total repo files listed: `3778`

## Family Counts

| Family | Artifacts | Locally cached |
| --- | ---: | ---: |
| `crosscoder` | 6 | 2 |
| `mlp_out_all` | 248 | 1 |
| `resid_post_all` | 248 | 8 |
| `transcoder` | 72 | 0 |
| `transcoder_all` | 506 | 3 |

## Candidate Artifacts

| Family | Artifact | Layers | Width | L0 | Affine | Cached | Recommendation |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| `crosscoder` | `layer_16_31_40_53_width_65k_l0_big` | 16,31,40,53 | 65k | big | no | yes | optional multi-layer sparse candidate; compare to raw concat |
| `crosscoder` | `layer_16_31_40_53_width_65k_l0_medium` | 16,31,40,53 | 65k | medium | no | yes | optional multi-layer sparse candidate; compare to raw concat |
| `crosscoder` | `layer_16_31_40_53_width_262k_l0_big` | 16,31,40,53 | 262k | big | no | no | optional multi-layer sparse candidate; compare to raw concat |
| `crosscoder` | `layer_16_31_40_53_width_262k_l0_medium` | 16,31,40,53 | 262k | medium | no | no | optional multi-layer sparse candidate; compare to raw concat |
| `crosscoder` | `layer_16_31_40_53_width_524k_l0_big` | 16,31,40,53 | 524k | big | no | no | optional multi-layer sparse candidate; compare to raw concat |
| `crosscoder` | `layer_16_31_40_53_width_524k_l0_medium` | 16,31,40,53 | 524k | medium | no | no | optional multi-layer sparse candidate; compare to raw concat |
| `mlp_out_all` | `layer_45_width_16k_l0_big` | 45 | 16k | big | no | no | optional MLP-output SAE follow-up |
| `mlp_out_all` | `layer_45_width_16k_l0_small` | 45 | 16k | small | no | yes | optional MLP-output SAE follow-up |
| `mlp_out_all` | `layer_45_width_262k_l0_big` | 45 | 262k | big | no | no | optional MLP-output SAE follow-up |
| `mlp_out_all` | `layer_45_width_262k_l0_small` | 45 | 262k | small | no | no | optional MLP-output SAE follow-up |
| `transcoder` | `layer_40_width_16k_l0_big_affine` | 40 | 16k | big | yes | no | selected-layer higher-L0 exact transcoder candidate |
| `transcoder` | `layer_40_width_16k_l0_medium_affine` | 40 | 16k | medium | yes | no | selected-layer higher-L0 exact transcoder candidate |
| `transcoder` | `layer_40_width_16k_l0_small_affine` | 40 | 16k | small | yes | no | selected-layer exact transcoder candidate |
| `transcoder` | `layer_40_width_65k_l0_big_affine` | 40 | 65k | big | yes | no | selected-layer higher-L0 exact transcoder candidate |
| `transcoder` | `layer_40_width_65k_l0_medium_affine` | 40 | 65k | medium | yes | no | selected-layer higher-L0 exact transcoder candidate |
| `transcoder` | `layer_40_width_65k_l0_small_affine` | 40 | 65k | small | yes | no | selected-layer exact transcoder candidate |
| `transcoder` | `layer_40_width_262k_l0_big_affine` | 40 | 262k | big | yes | no | selected-layer higher-L0 exact transcoder candidate |
| `transcoder` | `layer_40_width_262k_l0_medium_affine` | 40 | 262k | medium | yes | no | selected-layer higher-L0 exact transcoder candidate |
| `transcoder` | `layer_40_width_262k_l0_small_affine` | 40 | 262k | small | yes | no | selected-layer exact transcoder candidate |
| `transcoder` | `layer_53_width_16k_l0_big_affine` | 53 | 16k | big | yes | no | selected-layer higher-L0 exact transcoder candidate |
| `transcoder` | `layer_53_width_16k_l0_medium_affine` | 53 | 16k | medium | yes | no | selected-layer higher-L0 exact transcoder candidate |
| `transcoder` | `layer_53_width_16k_l0_small_affine` | 53 | 16k | small | yes | no | selected-layer exact transcoder candidate |
| `transcoder` | `layer_53_width_65k_l0_big_affine` | 53 | 65k | big | yes | no | selected-layer higher-L0 exact transcoder candidate |
| `transcoder` | `layer_53_width_65k_l0_medium_affine` | 53 | 65k | medium | yes | no | selected-layer higher-L0 exact transcoder candidate |
| `transcoder` | `layer_53_width_65k_l0_small_affine` | 53 | 65k | small | yes | no | selected-layer exact transcoder candidate |
| `transcoder` | `layer_53_width_262k_l0_big_affine` | 53 | 262k | big | yes | no | selected-layer higher-L0 exact transcoder candidate |
| `transcoder` | `layer_53_width_262k_l0_medium_affine` | 53 | 262k | medium | yes | no | selected-layer higher-L0 exact transcoder candidate |
| `transcoder` | `layer_53_width_262k_l0_small_affine` | 53 | 262k | small | yes | no | selected-layer exact transcoder candidate |
| `transcoder_all` | `layer_45_width_16k_l0_big_affine` | 45 | 16k | big | yes | no | highest-priority exact transcoder candidate |
| `transcoder_all` | `layer_45_width_16k_l0_small_affine` | 45 | 16k | small | yes | yes | already run |
| `transcoder_all` | `layer_45_width_262k_l0_big_affine` | 45 | 262k | big | yes | no | highest-priority exact transcoder candidate |
| `transcoder_all` | `layer_45_width_262k_l0_small_affine` | 45 | 262k | small | yes | yes | already run |

## Interpretation

- Found `2` unrun higher-L0 or denser single-layer affine L45 transcoder candidate(s).
- Run the smallest such exact candidate first, with the existing hook-audit check before probing.
