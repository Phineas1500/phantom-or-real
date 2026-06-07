# Prompt-Margin Gated Decode Correction

Generated: `2026-06-07T00:31:52.169244+00:00`

Purpose: test whether the calibrated prompt-margin commitment gate makes raw L45 decode-time injection a repair handle on the Gemma recognition-gap manifest.

## Summary

- Slurm job: `456741`.
- Rows: `14`; generated rows in report: `56`.
- Gate: `gold_vs_foil_logprob_margin < -15.0` at prefix checkpoint `0`.
- Gate fired on `8` rows: `8` regenerated-wrong and `0` regenerated-correct under the calibration baseline.
- Baseline strong accuracy: `0.214` (`3/14`); parse-fail rate: `0.071` (`1/14`).

## Paired Flips

| condition | false->true | true->false | changed | paired n | parse fail rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prompt_margin_gaussian_pos1sd_gvfltneg15` | 0 | 0 | 0 | 14 | 0.071 |
| `prompt_margin_orthogonal_pos1sd_gvfltneg15` | 0 | 0 | 0 | 14 | 0.000 |
| `prompt_margin_raw_pos1sd_gvfltneg15` | 0 | 0 | 0 | 14 | 0.000 |

## Interpretation

- The calibrated prompt-margin gate is cleaner than the old raw-z gate as a row selector, but raw L45 decode-time injection still does not repair strong correctness.
- Raw and orthogonal injections remove the one baseline parse failure, but neither changes any row from strong-wrong to strong-correct.
- Gaussian matched noise also has zero strong flips, so the result is a controlled null rather than an implementation failure.
- This supports keeping the commitment/trajectory evidence as predictive unless a different intervention family, such as a genuinely trajectory-level decode correction method, is implemented.

## Causal-Abstraction Claim

Tests whether a conservative prompt-margin `gold_vs_foil_margin` gate can make a raw residual direction act as a decode-time correction state for `commitment_state` / free-form correctness. A repair claim requires false-to-true repairs above orthogonal and matched-Gaussian controls.
