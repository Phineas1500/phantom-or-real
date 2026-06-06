# Gemma/Qwen Prefix-Conditioned Margin Comparison

Generated: `2026-06-06T20:53:20.429282+00:00`

Purpose: compare the completed prefix-conditioned margin diagnostics for Gemma and Qwen on recognition-gap rows.

## Bottom Line

The prefix-conditioned diagnostics support a cross-model predictive pattern: on recognition-gap rows, the wrong/free-form hypothesis is already more likely than gold at the prompt-only checkpoint. This is evidence for a recognition-vs-generation deployment gap, not evidence that we have found a causal commitment-transition handle.

## Prompt-Only Checkpoint

| model | task / heights | selected definition | selected>=gold | selected-vs-gold mean | gold>=foil | gold-vs-foil mean |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 3 27B | infer_property h=3,4 | regenerated selected hypothesis | 13/13 | 17.318 | 1/14 | -11.650 |
| Qwen3.5 27B | infer_subtype h=4 | Stage 1 hard foil | 14/14 | 12.864 | 0/14 | -12.864 |

## Final Prefix Checkpoint

| model | selected>=gold | selected-vs-gold mean | gold>=foil | gold-vs-foil mean |
| --- | --- | --- | --- | --- |
| Gemma 3 27B | 9/13 | 2.693 | 8/14 | 3.288 |
| Qwen3.5 27B | 14/14 | 8.611 | 0/14 | -8.611 |

## Interpretation

- Both models show the key prompt-only signature: the free-form selected or hard-foil hypothesis is already preferred to gold before any generated prefix is added.
- This makes the prefix result a stronger recognition-vs-generation diagnostic than a clean commitment-transition localization. The wrong-hypothesis preference is present at checkpoint 0.
- Gemma and Qwen are not matched replications. Gemma is property h3/h4 with regenerated decode traces and one parse failure; Qwen is subtype h4 with original Stage 1 output prefixes and selected defined as the hard foil.
- Later-prefix behavior should be interpreted cautiously. In Qwen, selected is identical to the hard foil, so selected-vs-gold is the negative of gold-vs-foil by construction.

## Recommendation

Close the prefix-conditioned trajectory measurement as predictive evidence for now. Do not run another broad trajectory batch unless it is tied to a specific intervention. If we continue this track causally, the next experiment should first train or calibrate a decode-trajectory monitor on selected-vs-gold or gold-vs-foil margin state, then test a gated intervention against matched-noise and positive controls.

## Causal-Abstraction Claim

Predictive only. The reports test `selected_hypothesis`, `gold_vs_foil_margin`, and `commitment_state` as readouts under prefix-conditioned contexts; they do not intervene on the model state.

## Inputs

- Gemma report: `docs/prefix_conditioned_margin_trajectory_gemma_manifest.json`
- Qwen report: `docs/qwen_prefix_conditioned_margin_trajectory_h4_subset.json`
