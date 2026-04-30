# 4B Feature Candidate Shortlist

Generated: 2026-04-30T04:33:43.452708+00:00

Source: `docs/sae_feature_stability_4b_l22_s1.json`.

Positive weights are associated with correct model outputs; negative weights are associated with incorrect outputs. Density is measured on the S1 train split. Target density is roughly 0.05-0.30; denser features are kept only when they are strong cross-task candidates but flagged as less ideal steering targets.

## Correct-Direction Candidates

| Rank | SAE ID | Feature | Density | Property weight | Subtype weight | Cross-width corr | Note |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `layer_22_width_16k_l0_small` | 456 | 0.127 | 0.2546 | 0.2952 | 0.722 | target-density |
| 2 | `layer_22_width_262k_l0_small` | 46005 | 0.116 | 0.1326 | 0.1255 | NA | target-density |
| 3 | `layer_22_width_262k_l0_small` | 18586 | 0.142 | 0.1155 | 0.1801 | NA | target-density |
| 4 | `layer_22_width_262k_l0_small` | 26141 | 0.062 | 0.1824 | 0.1130 | NA | target-density |
| 5 | `layer_22_width_16k_l0_small` | 10 | 0.232 | 0.0751 | 0.2261 | NA | target-density |
| 6 | `layer_22_width_262k_l0_small` | 36427 | 0.256 | 0.1922 | 0.0721 | NA | target-density |

## Incorrect-Direction Candidates

| Rank | SAE ID | Feature | Density | Property weight | Subtype weight | Cross-width corr | Note |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `layer_22_width_16k_l0_small` | 195 | 0.070 | -0.1688 | -0.1968 | NA | target-density |
| 2 | `layer_22_width_16k_l0_small` | 14976 | 0.050 | -0.1157 | -0.1198 | NA | target-density |
| 3 | `layer_22_width_16k_l0_small` | 3317 | 0.290 | -0.2215 | -0.1058 | NA | target-density |
| 4 | `layer_22_width_16k_l0_small` | 3513 | 0.113 | -0.1055 | -0.1552 | NA | target-density |

## Handoff Notes

- Use this as a candidate list for Phase E discussion, not as completed 4B steering work.
- 4B steering itself remains out of scope; the intended handoff is to compare stable 4B sparse candidates against the teammate 27B steering setup.
- D.8 big-affine completed and is strongest among 4B sparse/transcoder probes on property, but this shortlist remains based on stable residual SAE features because those have the available cross-width stability analysis.
- Avoid near-always-active features as first steering targets even when their coefficients are large.
- Subtype h3/h4 labels have tiny positive counts, so prioritize candidates that are same-sign across both tasks and have reasonable activation density.
