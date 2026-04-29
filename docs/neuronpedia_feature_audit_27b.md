# Neuronpedia Feature Audit Notes

Date: 2026-04-29

## Availability Check

Neuronpedia's public docs define feature pages as:

`https://neuronpedia.org/[MODEL_ID]/[SAE_ID]/[FEATURE_INDEX]`

and JSON feature access as:

`https://neuronpedia.org/api/feature/[MODEL_ID]/[SAE_ID]/[FEATURE_INDEX]`

For `gemma-3-27b-it`, the public resource list currently exposes residual SAE dashboards only at layers `16`, `31`, `40`, and `53` for widths `16k`, `65k`, and `262k`. It also exposes `gemmascope-2-transcoder-262k` dashboards for every layer, including `45-gemmascope-2-transcoder-262k`. I did not find public crosscoder sources, and the exact layer-45 residual sources used by our main SAE probes are not available.

Concrete check: querying `https://www.neuronpedia.org/api/feature/gemma-3-27b-it/45-gemmascope-2-res-16k/180` returned "The model, source, or feature you specified is not available." This means Neuronpedia cannot directly interpret our current top L45 residual SAE probe features.

## Implication

Neuronpedia is useful for a residual-SAE audit only if we run a same-source
residual SAE probe on one of its public residual layers, most naturally layer
`40` or `53` because those overlap the crosscoder layer set.

We also ran/probed the public L45 `262k` transcoder source and audited the top
transcoder features on Neuronpedia. See:

- `docs/transcoder_probe_27b_l45_262k_affine_s1.json`
- `docs/transcoder_probe_27b_l45_262k_affine_s3_target_symbol.json`
- `docs/neuronpedia_transcoder_audit_27b_l45_262k.md`

It is not a clean drop-in audit tool for the present L45 residual SAE result.
It also does not directly audit our completed L45 16K skip-transcoder pilot,
because that pilot used the HF `transcoder_all/layer_45_width_16k_l0_small_affine`
artifact, not Neuronpedia's public `45-gemmascope-2-transcoder-262k` source.

The L45 feature-stability output is still useful locally. A notable pattern is that several top-weight residual SAE features are nearly always active, so they are probably not satisfying "sparse reasoning feature" candidates even when they help the linear probe.

## Current L45 Top Features

From `docs/sae_feature_stability_27b_l45_s1.json`:

| SAE | Task | Rank | Feature | Assoc. | Density | Note |
| --- | --- | ---: | ---: | --- | ---: | --- |
| 16k | property | 1 | 180 | correct | 0.005 | sparse high-weight feature |
| 16k | property | 2 | 1096 | correct | 1.000 | always-active probe feature |
| 16k | property | 3 | 12292 | correct | 1.000 | almost always-active probe feature |
| 16k | subtype | 1 | 4329 | correct | 0.133 | moderately sparse |
| 16k | subtype | 2 | 19 | incorrect | 1.000 | always-active negative feature |
| 262k | property | 1 | 368 | correct | 0.099 | moderately sparse |
| 262k | property | 2 | 160112 | correct | 1.000 | always-active probe feature |
| 262k | subtype | 1 | 368 | correct | 0.090 | shared top feature across tasks |
| 262k | subtype | 2 | 160112 | correct | 1.000 | always-active shared feature |
| 262k | subtype | 4 | 46747 | incorrect | 0.002 | sparse negative feature |

## Completed L45 262K Transcoder Audit

The L45 262K transcoder is a useful auxiliary audit but not a rescue result.
It scored S1 property/subtype AUCs of 0.654/0.811 and S3 property/subtype AUCs
of 0.653/0.831, below the raw `mlp_in` probes and mostly below the L45 16K
skip-transcoder.

The top Neuronpedia explanations are mostly broad lexical/style/code features:
property includes "accessories", "titles and labels", "code syntax", "numbers",
and "code and symbols"; subtype includes "open", "to be", "titles and labels",
"code elements", "code syntax", "quotes", and "months".

If we want another Neuronpedia result in the final report, the remaining clean
option is a layer-40 or layer-53 residual SAE probe/audit. That would be an
auxiliary residual-feature check, not a replacement for the main L45 residual
result.
