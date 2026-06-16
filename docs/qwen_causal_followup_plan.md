# Qwen Causal Follow-Up

This is the Qwen-side causal follow-up to the Gemma forced-choice and patching
branch. The default target is `infer_subtype` height 4, because Qwen3.5-27B has
many parsed-but-wrong rows there while avoiding Gemma's large parse-failure
confound.

## Completed Results

Completed on 2026-05-23 with `Qwen/Qwen3.5-27B`, Qwen thinking disabled, and
Scholar J-node A40 jobs.

| Experiment | Slurm job | N | Main result | Interpretation |
|---|---:|---:|---|---|
| Hard-foil forced choice | `456255` | 64 h4 failures | MCQ choice accuracy `43/64 = 0.671875`; parse fail rate `0.0`; mean original margin `-13.344`; mean MCQ margin `+0.996` | Many free-form h4 subtype failures still contain recoverable answer knowledge under a constrained gold-vs-own-foil format. |
| Corrupt-to-clean patching | `456254` | 7 strict h1/h4 pairs | Largest mean normalized effect was L40 `last_prompt` at about `0.0079`; no condition had any examples above `0.25` recovery or breakage | Wrong h4 residual states do not strongly disrupt clean h1 contexts at these sites. |
| Clean-to-corrupt patching | `456256` | 7 strict h1/h4 pairs | Strongest mean margin delta was L40 `last_prompt` at `-0.219`; no condition had any examples above `0.25` breakage/recovery | Clean h1 residual states are not a simple causal switch whose replacement reliably breaks or repairs subtype answers. |
| Raw property steering, L45 | `456261` | 4 balanced h3/h4 rows | Raw direction test AUC `0.9179`; baseline, raw +/-1sd, and orthogonal +/-1sd all stayed at strong accuracy `0.5`, weak accuracy `1.0`, parse fail `0.0`; zero paired flips | A predictive correctness direction at the original Qwen Scope site is not a free-form repair knob. |
| Raw property steering, L53 | `456266` | 4 balanced h3/h4 rows | Raw direction test AUC `0.9400`; projection std `0.5555`; baseline, raw +/-1sd, and orthogonal +/-1sd all stayed at strong accuracy `0.5`, weak accuracy `1.0`, parse fail `0.0`; zero paired flips | Moving to Qwen's strongest raw layer improves prediction but not control. |
| Property multi-layer erasure | `457191`-`457194` | 16 balanced h3/h4 property rows, k=8 | Baseline P(strong)=`0.352`; raw-direction erasure P(strong)=`0.422`, dP=`+0.070` CI [`-0.031`,`+0.188`]; orthogonal dP=`+0.047`; Gaussian dP=`+0.016`; L53 probe test AUC `0.940`. | Cross-model support that the raw readout axis is not load-bearing. Controls are also non-destructive, so this is not a replication of Gemma's destructive-control separation. |
| Answer-property steering, L45 | `456262` | 8 balanced h3/h4 rows | Gold-polarity direction val/test AUC `1.0000`/`1.0000`; no answer-content changes and no strong-accuracy flips under toward-gold, away-gold, or orthogonal conditions | Even a concrete answer-content direction does not loosen the free-form output. |
| Sparse-probe bundle steering, L45 | `456263` | 8 balanced h3/h4 rows | W80K L0_100 sparse probe test AUC `0.8019`; bundle +/-0.5sd plus shuffled, random, and orthogonal controls all had zero output-correctness changes | Distributed residual-SAE decoder bundles reproduce the Gemma steering null at Qwen L45. |
| Sparse-probe bundle steering, L53 | `456267` | 8 balanced h3/h4 rows | W80K L0_100 sparse probe test AUC `0.8687`; bundle +/-0.5sd plus shuffled, random, and orthogonal controls all had zero output-correctness changes | The stronger L53 sparse readout still does not become a steering handle. |
| Single-feature steering, L53 | `456293` | 8 balanced h3/h4 rows | W80K L0_100 features `7169`, `23296`, and `4212` at +/-0.25 mean-nonzero scale plus random feature controls `54098`, `54479`, and `75388`; baseline and all interventions stayed at strong accuracy `0.5`, weak accuracy `1.0`, parse fail `0.0`; zero paired changes | Individual L53 residual-SAE decoder columns are not repair or breakage handles at this scale. |

Detailed artifacts:

- `docs/qwen35_27b_infer_subtype_h4_hardfoil_forced_choice.json`
- `results/stage2/qwen_causal/qwen35_27b_infer_subtype_h4_hardfoil_forced_choice.jsonl`
- `docs/qwen35_27b_infer_subtype_corrupt_to_clean_patching_margin_pilot.json`
- `results/stage2/patching/qwen35_27b_infer_subtype_corrupt_to_clean_margin_pilot.jsonl`
- `docs/qwen35_27b_infer_subtype_clean_to_corrupt_patching_margin_pilot.json`
- `results/stage2/patching/qwen35_27b_infer_subtype_clean_to_corrupt_margin_pilot.jsonl`
- `docs/qwen35_raw_steering_27b_l45_property_pilot.json`
- `results/stage2/steering/qwen35_raw_l45_property_steering_pilot.jsonl`
- `docs/qwen35_raw_steering_27b_l53_property_pilot.json`
- `results/stage2/steering/qwen35_raw_l53_property_steering_pilot.jsonl`
- `docs/qwen35_subspace_erasure_27b_property_sampled_k8_summary.md`
- `docs/qwen35_subspace_erasure_27b_property_sampled_k8.json`
- `docs/qwen35_answer_property_steering_27b_l45_polarity_smoke.json`
- `results/stage2/steering/qwen35_answer_property_l45_polarity_smoke.jsonl`
- `docs/qwen35_scope_sparse_bundle_steering_27b_l45_l0_100_property_smoke.json`
- `results/stage2/steering/qwen35_scope_l45_l0_100_sparse_bundle_property_smoke.jsonl`
- `docs/qwen35_scope_sparse_bundle_steering_27b_l53_l0_100_property_smoke.json`
- `results/stage2/steering/qwen35_scope_l53_l0_100_sparse_bundle_property_smoke.jsonl`
- `docs/qwen35_scope_single_feature_steering_27b_l53_l0_100_7169_23296_4212_property_smoke.json`
- `results/stage2/steering/qwen35_scope_l53_l0_100_single_features_7169_23296_4212_property_smoke.jsonl`

Bottom line: Qwen shows a strong format effect, strong pre-generation
correctness readouts, and no strong single-site full-residual patching or
decode-step steering effect, now including individual L53 Qwen Scope residual-SAE
decoder columns. The multi-layer property erasure now adds a causal non-necessity
result: ablating the raw correctness direction did not degrade behavior over 512
generations. The forced-choice result says that a large share of the selected h4
subtype errors are not pure absence of the right answer. The patching and
steering pilots say that the tested L35/L40/L45/L53 residual states and
probe-derived directions are not enough, by themselves, to act as localized
causal repair or breakage handles. This mirrors the Gemma-side
predictive-versus-causal theme, with a clearer Qwen recognition-vs-generation
split, but Qwen does not show Gemma's destructive-control erasure profile.

## 1. Hard-Foil Forced Choice

Run:

```bash
sbatch scripts/stage2_qwen35_27b_hardfoil_subtype.sbatch
```

This loads `Qwen/Qwen3.5-27B` through Hugging Face, selects S1 test h4 rows that
were wrong under free-form generation, uses the model's own emitted wrong
hypothesis as the foil, and measures whether Qwen chooses the gold hypothesis
in a two-option MCQ prompt.

Outputs:

- `results/stage2/qwen_causal/qwen35_27b_infer_subtype_h4_hardfoil_forced_choice.jsonl`
- `docs/qwen35_27b_infer_subtype_h4_hardfoil_forced_choice.json`

## 2. Forward Patching

Run:

```bash
sbatch scripts/stage2_qwen35_27b_patch_subtype_clean_to_corrupt.sbatch
```

This patches h1-correct residual states into matched h4-incorrect prompts at
`model.model.layers.{35,40,45}.output`, using the final prompt token by
default. It scores the gold-vs-own-foil logprob margin and compares against a
matched noise patch with the same vector norm.

Outputs:

- `results/stage2/patching/qwen35_27b_infer_subtype_clean_to_corrupt_margin_pilot.jsonl`
- `docs/qwen35_27b_infer_subtype_clean_to_corrupt_patching_margin_pilot.json`

## 3. Reverse Patching

Run:

```bash
sbatch scripts/stage2_qwen35_27b_patch_subtype_corrupt_to_clean.sbatch
```

This patches h4-incorrect residual states into matched h1-correct prompts at
the same layers and reports margin breakage against the noise control.

Outputs:

- `results/stage2/patching/qwen35_27b_infer_subtype_corrupt_to_clean_margin_pilot.jsonl`
- `docs/qwen35_27b_infer_subtype_corrupt_to_clean_patching_margin_pilot.json`

## Notes

- Set `PAIR_LIMIT`, `LAYERS`, `LANDMARKS`, `SPLIT_FAMILY`, or `TASK` in the
  Slurm environment to scale or redirect the pilot.
- The HF patching hook matches the Qwen-Scope extraction site:
  `model.model.layers[L]` output.
- `--disable-thinking` is enabled in all Qwen jobs so chat rendering stays
  aligned with the Stage 1 and activation-extraction runs.
