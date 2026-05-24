# Qwen Scope Replication Plan

This is the Qwen-side mirror of the Gemma Scope Stage 2 experiments in this
repo. The closest one-for-one model is `Qwen/Qwen3.5-27B` with Qwen Scope
residual SAEs:

- `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50`
- `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100`

Qwen Scope does not currently publish a 4B SAE tier in the HF collection. The
available smaller Qwen Scope tiers are 2B/9B for Qwen3.5 and 1.7B/8B for Qwen3,
so the first replication target is the paper's main 27B story.

## Artifact Names

- Coverage audit: `docs/qwen_gemma_experiment_coverage.md`
- 27B completion audit: `docs/qwen_27b_completion_audit.md`
- Stage 1 Qwen inference rows:
  - `results/full/with_errortype/qwen35_27b_infer_property.jsonl`
  - `results/full/with_errortype/qwen35_27b_infer_subtype.jsonl`
- Raw residual activations:
  - `results/stage2/activations/qwen35_27b_{task}_L45.safetensors`
- Probe/control summaries:
  - `docs/qwen35_27b_b0_summary.json`
  - `docs/qwen_scope_raw_probe_27b_l45_s1_label_shuffle.json`
  - `docs/qwen_scope_raw_probe_transfer_27b_l45_s1.json`
  - `docs/qwen_scope_raw_probe_transfer_27b_l45_s3.json`
  - `docs/qwen_scope_raw_probe_metadata_residualization_27b_l45.json`
  - `docs/qwen_scope_dense_active_sae_probe_27b_l45_s1.json`
  - `docs/qwen_scope_dense_active_sae_probe_27b_l45_s3.json`
  - `docs/qwen_scope_feature_stability_27b_l45_w80k_l0_50_s1.json`
  - `docs/qwen_scope_feature_stability_27b_l45_w80k_l0_100_s1.json`
  - `docs/qwen_scope_feature_mini_dashboard_27b_l45_w80k_l0_100_top8.json`
  - `docs/qwen_scope_feature_mini_dashboard_27b_l45_w80k_l0_100_top8.md`
  - `docs/qwen_scope_feature_stability_27b_l53_w80k_l0_50_s1.json`
  - `docs/qwen_scope_feature_stability_27b_l53_w80k_l0_100_s1.json`
  - `docs/qwen_scope_feature_mini_dashboard_27b_l53_w80k_l0_100_top8.json`
  - `docs/qwen_scope_feature_mini_dashboard_27b_l53_w80k_l0_100_top8.md`
  - `docs/qwen35_scope_sparse_bundle_steering_27b_l45_l0_100_property_smoke.json`
  - `docs/qwen35_scope_sparse_bundle_steering_27b_l53_l0_100_property_smoke.json`
  - `docs/qwen35_scope_single_feature_steering_27b_l53_l0_100_7169_23296_4212_property_smoke.json`
- Raw layer-scan summaries:
  - `docs/qwen_scope_raw_probe_27b_layers_16_31_40_53_s1.json`
  - `docs/qwen_scope_raw_probe_27b_layers_16_31_40_53_s3.json`
- Raw MLP-site summaries:
  - `docs/qwen_scope_raw_probe_27b_l45_53_mlp_in_weighted_s1.json`
  - `docs/qwen_scope_raw_probe_27b_l45_53_mlp_in_weighted_s3.json`
  - `docs/qwen_scope_raw_probe_27b_l45_53_mlp_out_s1.json`
  - `docs/qwen_scope_raw_probe_27b_l45_53_mlp_out_s3.json`
- Sparse concat summaries:
  - `docs/qwen_scope_sparse_concat_probe_27b_l45_w80k_l0_50_l0_100_s1.json`
  - `docs/qwen_scope_sparse_concat_probe_27b_l45_w80k_l0_50_l0_100_s3.json`
  - `docs/qwen_scope_sparse_concat_probe_27b_l53_w80k_l0_50_l0_100_s1.json`
  - `docs/qwen_scope_sparse_concat_probe_27b_l53_w80k_l0_50_l0_100_s3.json`
  - `docs/qwen_scope_sparse_concat_probe_27b_l45_l53_w80k_l0_50_l0_100_s1.json`
  - `docs/qwen_scope_sparse_concat_probe_27b_l45_l53_w80k_l0_50_l0_100_s3.json`
  - `docs/qwen_scope_sparse_concat_probe_27b_l53_residual_local_mlpout_localtc_s1.json`
  - `docs/qwen_scope_sparse_concat_probe_27b_l53_residual_local_mlpout_localtc_s3.json`
  - `docs/qwen_scope_sparse_concat_ablation_27b_l53_residual_local_mlpout_localtc_summary.json`
  - `docs/qwen_scope_sparse_concat_ablation_27b_l53_residual_local_mlpout_localtc_minus_{block}_{s1,s3}.json`
  - `docs/qwen_scope_dense_active_local_sparse_probe_27b_s1.json`
  - `docs/qwen_scope_dense_active_local_sparse_probe_27b_s3.json`
  - `docs/qwen_scope_dense_active_sparse_concat_probe_27b_l53_residual_local_mlpout_localtc_s1.json`
  - `docs/qwen_scope_dense_active_sparse_concat_probe_27b_l53_residual_local_mlpout_localtc_s3.json`
- Local learned-dictionary pilot summaries:
  - `docs/qwen_scope_local_mlpout_topk_probe_27b_l53_w4096_k64_s1.json`
  - `docs/qwen_scope_local_mlpout_topk_probe_27b_l53_w4096_k64_s3.json`
  - `docs/qwen_scope_local_transcoder_probe_27b_l53_w4096_k64_s1.json`
  - `docs/qwen_scope_local_transcoder_probe_27b_l53_w4096_k64_s3.json`
  - `docs/qwen_scope_local_mlpout_reconstruction_probe_27b_l53_w4096_k64_s1.json`
  - `docs/qwen_scope_local_mlpout_reconstruction_probe_27b_l53_w4096_k64_s3.json`
  - `docs/qwen_scope_local_transcoder_component_probe_27b_l53_w4096_k64_s1.json`
  - `docs/qwen_scope_local_transcoder_component_probe_27b_l53_w4096_k64_s3.json`
  - `docs/qwen_scope_local_crosscoder_probe_27b_layers_16_31_40_53_w4096_k64_s1.json`
  - `docs/qwen_scope_local_crosscoder_probe_27b_layers_16_31_40_53_w4096_k64_s3.json`
  - `docs/qwen_scope_raw_concat_probe_27b_layers_16_31_40_53_s1.json`
  - `docs/qwen_scope_raw_concat_probe_27b_layers_16_31_40_53_s3.json`
- Qwen Scope feature files:
  - `results/stage2/sae_features/qwen35_27b_{task}_L45_qwenscope_qwen35_27b_w80k_l0_50_top50.safetensors`
  - `results/stage2/sae_features/qwen35_27b_{task}_L45_qwenscope_qwen35_27b_w80k_l0_100_top100.safetensors`
  - `results/stage2/sae_features/qwen35_27b_{task}_L53_qwenscope_qwen35_27b_w80k_l0_50_top50.safetensors`
  - `results/stage2/sae_features/qwen35_27b_{task}_L53_qwenscope_qwen35_27b_w80k_l0_100_top100.safetensors`

## Completed Findings

The Qwen Scope replication and Qwen causal follow-up are complete enough to
record the current interpretation. See `docs/qwen_causal_followup_plan.md` for
the detailed causal runs.

Stage 1 behavior:

- Qwen3.5-27B outputs are complete for both tasks: `11000` rows each for
  `infer_property` and `infer_subtype`.
- `infer_property` strong accuracy by height was h1 `1.0000`, h2 `0.5425`,
  h3 `0.5007`, h4 `0.5122`; parse failures were effectively absent.
- `infer_subtype` strong accuracy by height was h1 `1.0000`, h2 `0.4430`,
  h3 `0.3567`, h4 `0.3154`; parse failures were effectively absent.

Qwen Scope artifact support:

- A live Hugging Face inventory check of Qwen-owned SAE repos found residual SAE
  families only: Qwen3.5 2B/9B/27B/35B-A3B and Qwen3 1.7B/8B/30B-A3B residual
  SAEs. A 2026-05-24 check of the official Qwen-Scope collection likewise lists
  `SAE-Res-*` artifacts across those tiers, and the Qwen-Scope paper describes
  the release as 14 groups of SAEs across 7 Qwen3/Qwen3.5 variants. Searches
  under Qwen for `transcoder`, `crosscoder`, `MLP`, and `SAE-MLP` did not return
  matching first-party artifact repos. The Qwen replication can therefore mirror
  Gemma residual-SAE experiments directly, but Gemma's exact MLP-output SAE,
  skip-transcoder, affine-transcoder, and crosscoder runs remain artifact-limited
  unless Qwen publishes those families or we train local equivalents.

Qwen Scope / activation readouts:

- Raw L45 residual probes remained strong: S1 test AUCs were `0.9179` for
  `infer_property` and `0.9012` for `infer_subtype`; S3 test AUCs were
  `0.9074` and `0.9043`.
- The Gemma-style raw layer scan over Qwen layers `16`, `31`, `40`, and `53`
  is complete for S1 and S3, and L53 is strongest for both tasks on both
  splits. S1 property test AUCs were L16 `0.7314`, L31 `0.8634`, L40 `0.9069`,
  L53 `0.9400`; S1 subtype AUCs were L16 `0.7356`, L31 `0.8462`, L40
  `0.8870`, L53 `0.9205`. S3 property AUCs were L16 `0.6941`, L31 `0.8429`,
  L40 `0.8900`, L53 `0.9325`; S3 subtype AUCs were L16 `0.7427`, L31
  `0.8488`, L40 `0.8960`, L53 `0.9148`.
- Metadata baselines are much weaker than raw Qwen activations. The best B0
  test AUCs were property S1/S3 `0.6015`/`0.6009` and subtype S1/S3
  `0.6748`/`0.6767`, leaving raw-vs-B0 gaps of about `+0.316`, `+0.306`,
  `+0.226`, and `+0.228` AUC respectively.
- Raw Qwen MLP-site probes over layers `45` and `53` are complete with the
  canonical split file. L53 is best in every cell. `mlp_in_weighted` reached
  S1 property/subtype test AUC `0.9390`/`0.9197` and S3 `0.9323`/`0.9140`;
  `mlp_out` reached S1 `0.9374`/`0.9157` and S3 `0.9333`/`0.9119`. These
  same-site raw baselines nearly match raw residual L53 and exceed Gemma's
  raw L45 MLP-site baselines, but they are not Qwen MLP SAE/transcoder
  artifact mirrors.
- A local split-trained Qwen L53 `mlp_out` top-k dictionary pilot now covers the
  Gemma exact MLP-output SAE experimental slot as far as local artifacts allow.
  `scripts/stage2_train_local_topk_dictionary.py` trains top-k autoencoders on
  train rows only for each split family and writes standard sparse feature
  files; `scripts/stage2_qwen35_27b_local_mlpout_dictionary.sbatch` ran it as
  job `456280`. The W4096/K64 dictionaries explained about `0.913`
  standardized train energy. Sparse probes reached S1 property/subtype
  `0.9056`/`0.8889` and S3 `0.8965`/`0.8691`. This beats the available
  Qwen Scope residual SAEs in several cells and trails raw `mlp_out`, so it
  preserves the raw-vs-learned-sparse gap. It is not a first-party Qwen Scope
  MLP SAE and should be labeled as a local pilot. The follow-up local
  reconstruction/error diagnostic in `scripts/stage2_local_dictionary_reconstruction_diagnostics.py`
  ran as job `456290` and wrote `docs/qwen_scope_local_mlpout_reconstruction_probe_27b_l53_w4096_k64_{s1,s3}.json`.
  It found near-perfect target reconstruction by the local dictionary: S1
  energy/cosine were `0.9872`/`0.9936` for property and `0.9876`/`0.9938` for
  subtype; S3 energy/cosine were `0.9870`/`0.9935` and `0.9875`/`0.9937`.
  Reconstruction probes reached S1 `0.9134`/`0.8956` and S3 `0.9088`/`0.8797`,
  while error probes were much weaker at S1 `0.8181`/`0.8161` and S3
  `0.8002`/`0.8326`.
- A local split-trained Qwen L53 transcoder-style pilot now covers the Gemma
  exact skip/affine-transcoder slot as far as local artifacts allow.
  `scripts/stage2_train_local_topk_transcoder.py` trains W4096/K64 top-k
  encoders from `mlp_in_weighted` to reconstruct `mlp_out`;
  `scripts/stage2_qwen35_27b_local_transcoder.sbatch` ran it as job `456281`.
  The dictionaries explained about `0.911` standardized target energy. Sparse
  probes reached S1 `0.9032`/`0.8807` and S3 `0.8963`/`0.8736` for
  property/subtype. This is close to the local MLP-out dictionary but still
  well below raw L53 MLP-site probes. The component diagnostic from job `456290`
  wrote `docs/qwen_scope_local_transcoder_component_probe_27b_l53_w4096_k64_{s1,s3}.json`.
  Its reconstruction probes reached S1 `0.9142`/`0.8954` and S3
  `0.9107`/`0.8852`, with energy explained around `0.987`. Error probes were
  weaker: S1 `0.7949`/`0.8111` and S3 `0.7904`/`0.8232`. As with the local
  MLP-output dictionary, the learned local decoder reconstructs the same-site
  target very well, so the Qwen correctness signal is concentrated in the
  reconstruction rather than the reconstruction error.
- A local multi-layer crosscoder-style pilot now covers the Gemma crosscoder
  slot as far as local artifacts allow. `scripts/stage2_train_local_topk_crosscoder.py`
  trains W4096/K64 top-k dictionaries on concatenated Qwen layers
  `16,31,40,53`; `scripts/stage2_qwen35_27b_local_crosscoder.sbatch` produced
  local sparse reports, and `scripts/stage2_qwen35_27b_raw_concat_crosscoder_layers.sbatch`
  produced the fair raw-concat baseline. The local crosscoder reached S1
  `0.8914`/`0.8359` and S3 `0.8731`/`0.8372`; raw concat over the same layers
  reached S1 `0.9368`/`0.9129` and S3 `0.9275`/`0.9089`. The local
  crosscoder therefore preserves the Gemma-style sparse-vs-raw gap.
- The S1 raw label-shuffle control stayed near chance: property `0.4908` and
  subtype `0.4899` test AUC.
- Raw L45 cross-task transfer is strong but not task-invariant. Property ->
  subtype transfer reached S1/S3 `0.8704`/`0.8654`; subtype -> property reached
  S1/S3 `0.8582`/`0.8236`.
- Metadata residualization keeps substantial Qwen activation signal. After
  residualizing the raw score against `b0_namefreq`, test AUCs were property
  S1/S3 `0.8715`/`0.8609` and subtype S1/S3 `0.8154`/`0.7953`; adding the raw
  score back to metadata essentially recovered the raw AUCs.
- Qwen Scope sparse probes kept useful but weaker signal at L45, then improved
  substantially at L53. The L45 W80K L0_100 S1 test AUCs were `0.8019` for
  `infer_property` and `0.8137` for `infer_subtype`; L45 W80K L0_50 S1 test
  AUCs were `0.7416` and `0.7824`. At L53, W80K L0_100 reached S1
  `0.8687`/`0.8576` and S3 `0.8356`/`0.8537`; W80K L0_50 reached S1
  `0.8363`/`0.8397` and S3 `0.7846`/`0.8322`.
- Feature-stability and mini-dashboard audits now cover the L45 and L53 residual
  Qwen Scope SAEs on S1. L45 L0_100 reproduced the sparse-probe AUCs
  `0.8019`/`0.8137`; its top-25 property/subtype coefficient overlap was only
  5 features, and its top-50 overlap was 14 features. The strongest shared
  L45 L0_100 candidates (`28479`, `69171`, `5666`, `80956`, `68759`) were
  mostly dense directions with `0.9`-`1.0` activation density, while sparse
  task-specific candidates such as `77473`, `68363`, and `67802` fired mostly
  on h1/h2 rows. L53 improves the sparse local fit: L53 L0_100 reached S1
  property/subtype test AUC `0.8687`/`0.8576`, and L53 L0_50 reached
  `0.8363`/`0.8397`. L53 L0_100 also has stronger cross-task overlap, with 5
  shared top-10 features and 25 shared top-50 features, all same-sign. Its top
  shared candidates still mix dense correctness/depth features (`28984`,
  `32398`, `51800`, `68475`) with lower-height sparse features (`4212`), so it
  improves predictive readout more than interpretability.
- Dense-active sparse controls do not reveal a hidden sparse-probe gain. After
  materializing train-active Qwen Scope columns and using centered dense
  scaling, L0_100 test AUCs were S1 `0.8036`/`0.8150` and S3
  `0.8035`/`0.8067` for property/subtype, essentially matching the standard
  sparse probes. The same control now covers the Qwen local learned sparse
  pilots. Local MLP-out dense-active reached S1 `0.9056`/`0.8887` and S3
  `0.8957`/`0.8691`; local transcoder reached S1 `0.9034`/`0.8807` and S3
  `0.8958`/`0.8738`; local crosscoder reached S1 `0.8916`/`0.8364` and S3
  `0.8730`/`0.8371`. Dense-active residual+local concat reached S1
  `0.9146`/`0.8949` and S3 `0.9024`/`0.8854`, essentially tied with the
  ordinary sparse concat. This mirrors the Gemma dense-active conclusion:
  sparse matrix scaling is not the bottleneck.
- Residual-only sparse concat is complete for the available Qwen Scope L45/L53
  residual SAEs. L45 W80K L0_50+L0_100 reached S1 `0.8007`/`0.8178` and S3
  `0.8022`/`0.8221`; L53 W80K L0_50+L0_100 reached S1 `0.8806`/`0.8617` and
  S3 `0.8409`/`0.8607`; L45+L53 reached S1 `0.8818`/`0.8692` and S3
  `0.8536`/`0.8688` for property/subtype. Adding the local L53 MLP-output
  dictionary and local L53 transcoder to the L53 residual SAEs gives the
  current strongest learned-sparse Qwen concat: S1 `0.9148`/`0.8950` and S3
  `0.9033`/`0.8851`. This shows complementarity across local sparse families,
  but the combined sparse representation remains below raw residual, raw
  MLP-site, and raw multi-layer concat probes. The matching leave-one-block-out
  ablation completed as job `456291`. Removing local MLP-out caused the largest
  S1 drop, `-0.0067`/`-0.0087`, and the largest S3 property drop, `-0.0120`;
  removing local transcoder caused S1 `-0.0062`/`-0.0061` and S3
  `-0.0097`/`-0.0067`. Removing either residual SAE block changed S1 by under
  `0.0012`, and on S3 property slightly improved AUC, so the marginal Qwen
  concat gain comes mostly from the local MLP-site dictionaries rather than the
  residual SAE pair.
- Reconstruction diagnostics show lower residual energy capture than the Gemma
  branch. At L45, W80K L0_50 explained about `0.744`/`0.737` energy for
  property/subtype, while W80K L0_100 explained about `0.781`/`0.774`.
  Reconstruction-error probes were still strong, with S1 error-probe AUCs near
  `0.915`/`0.901` for property/subtype under L0_100. L53 pushes energy lower:
  W80K L0_50 explains `0.6761`/`0.6738`, with error-probe AUC
  `0.9374`/`0.9213` and reconstruction-probe AUC `0.8397`/`0.8428`; W80K
  L0_100 explains `0.7313`/`0.7284`, with error-probe AUC `0.9337`/`0.9183`
  and reconstruction-probe AUC `0.8760`/`0.8641`. On S3, L53 W80K L0_50
  reconstruction/error reached reconstruction-probe AUC `0.7866`/`0.8330`,
  error-probe AUC `0.9269`/`0.9125`, and the same full-dataset energy
  `0.6761`/`0.6738`. S3 W80K L0_100 reached reconstruction-probe AUC
  `0.8462`/`0.8550`, error-probe AUC `0.9260`/`0.9127`, and energy
  `0.7313`/`0.7284`.

Causal follow-up:

- Hard-foil forced choice recovered `43/64 = 0.671875` h4 subtype failures that
  were originally free-form wrong. Mean original gold-vs-foil margin was
  `-13.344`; mean MCQ margin became `+0.996`.
- Full-residual patching at L35/L40/L45 `last_prompt` was weak in both
  directions. The reverse corrupt-to-clean run's largest mean normalized
  effect was only about `0.0079`, and clean-to-corrupt's strongest mean margin
  delta was only `-0.219`; no tested patch condition had any examples above
  `0.25` recovery or breakage.
- Decode-step Qwen steering remains null across raw correctness directions,
  answer-polarity direction, sparse-probe bundles, and the L53 single-feature
  residual-SAE smoke. The single-feature run tested Qwen Scope W80K L0_100
  features `7169`, `23296`, and `4212` at +/-0.25 mean-nonzero scale, plus
  random feature controls, and every paired condition had zero output-correctness
  changes.

Interpretation: Qwen's Qwen Scope readouts are predictive, and the forced-choice
result shows that many free-form subtype failures retain answer recognition.
However, the tested single-site residual state patches do not expose a compact
causal repair/breakage handle. The Qwen branch therefore supports the same broad
predictive-versus-causal caution as the Gemma branch, with an especially clear
output-format dependence.

Recent follow-ups as of 2026-05-24:

- Raw layer scan over Qwen layers `16`, `31`, `40`, and `53` completed as
  Slurm job `456259` via `scripts/stage2_qwen35_27b_raw_layers_scan.sbatch`;
  both S1 and S3 reports landed with L53 best for both tasks.
- bf16-vs-fp32 Qwen Scope sparse encoding sanity completed as Slurm job
  `456260` via `scripts/stage2_qwen_scope_dtype_sanity.sbatch`. The first-512
  fp32 re-encoding exactly matched L0 counts, had active-set Jaccard about
  `0.991`-`0.995`, and had top-1 feature match rates `0.994`-`1.000`.
- Raw free-form Qwen steering pilot completed as Slurm job `456261` via
  `scripts/stage2_qwen35_27b_raw_steering_property_pilot.sbatch`. The L45
  property direction was predictive (S1 test AUC `0.9179`), but the 4-row
  smoke had no accuracy flips: baseline, raw +/-1sd, and orthogonal +/-1sd all
  stayed at strong accuracy `0.5` with parse-fail rate `0.0`. The layer-aware
  L53 follow-up completed as Slurm job `456266`; its property direction was
  stronger (S1 test AUC `0.9400`, projection std `0.5555`) but remained
  behaviorally inert on the same 4-row smoke, with zero raw or orthogonal paired
  flips and the same strong/weak/parse-fail rates as baseline.
- Answer-property Qwen steering smoke completed as Slurm job `456262` via
  `scripts/stage2_qwen35_27b_answer_property_steering_smoke.sbatch`. The L45
  gold-polarity direction had val/test AUC `1.0000`/`1.0000`, but the 8-row
  smoke was behaviorally inert: baseline, toward-gold, away-gold, and
  orthogonal conditions all stayed at strong accuracy `0.5`, weak accuracy
  `1.0`, parse-fail rate `0.0`, polarity/predicate match rate `1.0`, and
  answer-content change rate `0.0`.
- Qwen Scope sparse-probe bundle steering smoke completed at L45 as Slurm job
  `456263` via `scripts/stage2_qwen35_27b_sparse_bundle_property_smoke.sbatch`.
  The run trained the L45 W80K L0_100 property sparse probe with test AUC
  `0.8019`, selected 50 decoder features, loaded cached layer-45 decoder rows,
  and tested 8 balanced h3/h4 rows. Baseline, bundle +/-0.5sd, shuffled,
  random, and orthogonal controls all stayed at strong accuracy `0.5`, weak
  accuracy `1.0`, and parse-fail rate `0.0`; every paired condition had zero
  output correctness changes versus baseline. The sbatch is layer/task-aware,
  and the L53 property L0_100 sparse-bundle follow-up completed as Slurm job
  `456267`. Its fitted L53 sparse probe had test AUC `0.8687`, but the 8-row
  generation smoke was also inert: baseline, bundle +/-0.5sd, shuffled, random,
  and orthogonal controls all stayed at strong accuracy `0.5`, weak accuracy
  `1.0`, parse-fail rate `0.0`, with zero output-correctness changes.
- Qwen Scope single-feature steering completed as Slurm job `456293` via
  `scripts/stage2_qwen35_27b_single_feature_property_smoke.sbatch`. The new HF
  port is `scripts/stage2_qwen_steer_single_features.py`. It tested L53 W80K
  L0_100 features `7169`, `23296`, and `4212`, selected from the L53
  mini-dashboard, at +/-0.25 mean-nonzero scale with random feature controls
  `54098`, `54479`, and `75388`. On 8 balanced h3/h4 rows, baseline and all
  feature/control conditions stayed at strong accuracy `0.5`, weak accuracy
  `1.0`, parse-fail rate `0.0`, with zero paired output-correctness changes.
- Because the raw layer scan made L53 the strongest Qwen raw layer, Qwen Scope
  L53 residual-SAE follow-up jobs have completed. Feature extraction
  completed as job `456264` for both W80K L0_50 and L0_100 on property/subtype;
  local feature-stability audits, the L53 L0_100 top-8 mini-dashboard, and the
  full S1/S3 raw/sparse/reconstruction reports have landed and job `456265`
  completed. L53 raw steering completed as `456266`, L53 sparse-bundle steering
  completed as `456267`, and L53 single-feature steering completed as `456293`.
- Raw Qwen MLP-site extraction completed as job `456272` using
  `scripts/stage2_qwen35_27b_mlp_site_probes.sbatch`, which now supports
  `SKIP_EXTRACTION=1` and passes `results/stage2/splits.jsonl` explicitly for
  S1/S3 probes. Corrected split-aware probe reports completed as job `456277`:
  `docs/qwen_scope_raw_probe_27b_l45_53_mlp_in_weighted_{s1,s3}.json` and
  `docs/qwen_scope_raw_probe_27b_l45_53_mlp_out_{s1,s3}.json`.
- Residual-only Qwen Scope sparse concat completed as job `456279` via
  `scripts/stage2_qwen_scope_sparse_concat_probes.sbatch`. The job exposed and
  fixed a sparse-concat serialization bug by removing the non-JSON
  `_artifact_model` from `scripts/stage2_probe_sparse_concat.py` reports.
- Local Qwen L53 `mlp_out` top-k dictionary training/probing completed as job
  `456280` via `scripts/stage2_qwen35_27b_local_mlpout_dictionary.sbatch`.
  The new reusable trainer is `scripts/stage2_train_local_topk_dictionary.py`;
  reports landed at
  `docs/qwen_scope_local_mlpout_topk_probe_27b_l53_w4096_k64_{s1,s3}.json`.
- Local Qwen L53 `mlp_in_weighted` -> `mlp_out` top-k transcoder
  training/probing completed as job `456281` via
  `scripts/stage2_qwen35_27b_local_transcoder.sbatch`; reports landed at
  `docs/qwen_scope_local_transcoder_probe_27b_l53_w4096_k64_{s1,s3}.json`.
- Residual+local MLP/transcoder sparse concat completed as job `456282` via
  `scripts/stage2_qwen_scope_local_sparse_concat.sbatch`; reports landed at
  `docs/qwen_scope_sparse_concat_probe_27b_l53_residual_local_mlpout_localtc_{s1,s3}.json`.
- Leave-one-block-out ablation for the Qwen residual+local MLP/transcoder concat
  completed as job `456291` via `scripts/stage2_qwen_scope_local_sparse_ablation.sbatch`.
  The reusable validator is `scripts/stage2_validate_qwen_local_sparse_ablation.py`;
  reports landed at
  `docs/qwen_scope_sparse_concat_ablation_27b_l53_residual_local_mlpout_localtc_summary.json`
  and the matching `minus_{block}_{s1,s3}.json` files.
- Local learned-sparse dense-active controls completed as job `456292` via
  `scripts/stage2_qwen_scope_local_dense_active_controls.sbatch`. The reports
  `docs/qwen_scope_dense_active_local_sparse_probe_27b_{s1,s3}.json` and
  `docs/qwen_scope_dense_active_sparse_concat_probe_27b_l53_residual_local_mlpout_localtc_{s1,s3}.json`
  show dense centering/scaling leaves the local dictionary, local transcoder,
  local crosscoder, and residual+local concat results essentially unchanged.
- Local Qwen multi-layer top-k crosscoder training/probing completed after a
  serialization fix to `scripts/stage2_probe_crosscoder.py`; the successful
  rerun was job `456286`, with reports at
  `docs/qwen_scope_local_crosscoder_probe_27b_layers_16_31_40_53_w4096_k64_{s1,s3}.json`.
  The comparable raw-concat baseline needed a matching serialization fix to
  `scripts/stage2_probe_raw_concat.py` and completed as job `456287`, writing
  `docs/qwen_scope_raw_concat_probe_27b_layers_16_31_40_53_{s1,s3}.json`.
- Local learned-dictionary reconstruction/component diagnostics completed as
  job `456290` via `scripts/stage2_qwen35_27b_local_dictionary_reconstruction.sbatch`.
  The reusable diagnostic script is
  `scripts/stage2_local_dictionary_reconstruction_diagnostics.py`; reports
  landed at `docs/qwen_scope_local_mlpout_reconstruction_probe_27b_l53_w4096_k64_{s1,s3}.json`
  and `docs/qwen_scope_local_transcoder_component_probe_27b_l53_w4096_k64_{s1,s3}.json`.

## Run Order

Optional local inference, if no OpenAI-compatible Qwen endpoint is already
available and a single long job is acceptable:

```bash
sbatch scripts/stage2_qwen35_27b_infer_vllm.sbatch
```

For the full Scholar-local run, prefer the sharded array. It runs 22 shards
across the two tasks/heights, writes shard JSONLs under
`results/full/with_errortype/qwen35_27b_shards`, and keeps split regeneration
off until the merge succeeds:

```bash
sbatch scripts/stage2_qwen35_27b_infer_array.sbatch
sbatch --dependency=afterok:<array_jobid> scripts/stage2_qwen35_27b_merge_inference.sbatch
```

For a smoke run without touching the fixed output files:

```bash
sbatch --export=ALL,TASKS=property,HEIGHTS=1,LIMIT=50,OUTPUT_DIR=/scratch/scholar/$USER/tmp/qwen35_smoke,MODEL_SLUG=qwen35_27b_smoke,MAKE_SPLITS=0 scripts/stage2_qwen35_27b_infer_vllm.sbatch
```

After merging, `scripts/stage2_qwen35_27b_merge_inference.sbatch` validates that
Qwen outputs do not contain the old thinking preamble and regenerates
`results/stage2/splits.jsonl`.

Extract the Qwen residual stream at the Qwen Scope residual hook:

```bash
sbatch scripts/stage2_qwen35_27b_extract_L45.sbatch
```

Encode L45 residuals through both Qwen Scope 27B SAEs:

```bash
sbatch scripts/stage2_qwen_scope_27b_L45_features.sbatch
```

Run raw probes, sparse probes, and reconstruction/error probes for S1 and S3:

```bash
sbatch scripts/stage2_qwen_scope_27b_L45_probes.sbatch
```

Run the Gemma-style Qwen raw-layer scan over layers 16/31/40/53. Qwen3.5-27B
has 64 text layers, so all four Gemma comparison layers are valid Qwen hooks:

```bash
sbatch scripts/stage2_qwen35_27b_raw_layers_scan.sbatch
```

Run the non-GPU raw-probe controls after the L45 activation/probe artifacts are
available. These commands mirror the Gemma metadata, shuffle, transfer, and
metadata-residualization checks:

```bash
module load conda/2024.09
conda activate /scratch/scholar/$USER/conda-envs/phantom
export HF_HOME=/scratch/scholar/$USER/hf-cache

python scripts/stage2_train_b0.py \
  --jsonl-dir results/full/with_errortype \
  --splits results/stage2/splits.jsonl \
  --output results/stage2/baselines/qwen35_27b_b0_metadata.json \
  --summary docs/qwen35_27b_b0_summary.json \
  --hf-cache /scratch/scholar/$USER/hf-cache \
  --split-families s1 s3 \
  --models Qwen/Qwen3.5-27B \
  --tasks infer_property infer_subtype

python scripts/stage2_probe_raw.py \
  --activation-dir results/stage2/activations \
  --model-key qwen35_27b \
  --tasks infer_property infer_subtype \
  --layers 45 \
  --splits results/stage2/splits.jsonl \
  --split-family s1 \
  --shuffle-labels \
  --output docs/qwen_scope_raw_probe_27b_l45_s1_label_shuffle.json \
  --bootstrap-samples 1000

python scripts/stage2_probe_transfer.py \
  --activation-dir results/stage2/activations \
  --model-key qwen35_27b \
  --tasks infer_property infer_subtype \
  --layers 45 \
  --splits results/stage2/splits.jsonl \
  --split-family s1 \
  --output docs/qwen_scope_raw_probe_transfer_27b_l45_s1.json \
  --bootstrap-samples 1000

python scripts/stage2_probe_transfer.py \
  --activation-dir results/stage2/activations \
  --model-key qwen35_27b \
  --tasks infer_property infer_subtype \
  --layers 45 \
  --splits results/stage2/splits.jsonl \
  --split-family s3 \
  --output docs/qwen_scope_raw_probe_transfer_27b_l45_s3.json \
  --bootstrap-samples 1000

python scripts/stage2_probe_metadata_residualization.py \
  --activation-dir results/stage2/activations \
  --splits results/stage2/splits.jsonl \
  --model-key qwen35_27b \
  --tasks infer_property infer_subtype \
  --layer 45 \
  --split-families s1 s3 \
  --raw-report-template 'qwen_scope_raw_probe_27b_l45_{split_family}.json' \
  --output docs/qwen_scope_raw_probe_metadata_residualization_27b_l45.json
```

## Notes

- Qwen3.5 thinking mode is disabled in the vLLM inference wrapper and in
  `scripts/stage2_extract_hf.py` prompt rendering, keeping Stage 1 outputs
  aligned with the final-hypothesis-only Gemma contract.
- `scripts/stage2_extract_hf.py` uses Hugging Face hooks on
  `model.model.layers[LAYER]`, matching the Qwen Scope model-card example.
- `scripts/stage2_extract_qwen_scope_features.py` loads `layer{L}.sae.pt`
  directly and writes the same top-k sparse artifact schema as the Gemma Scope
  extractor.
- `scripts/stage2_qwen_scope_reconstruction_diagnostics.py` decodes the Qwen
  feature files with Qwen's `(d_model, d_sae)` decoder convention, then reuses
  the existing raw-probe machinery for reconstruction and error probes.
- The existing steering and patching scripts are still Gemma/TransformerLens
  oriented. After the predictive/sparse/reconstruction pass lands, the next
  Qwen-specific causal port should use the same Hugging Face hook backend added
  here.

Primary references:

- Qwen blog: https://qwen.ai/blog?id=qwen-scope
- Qwen Scope collection: https://huggingface.co/collections/Qwen/qwen-scope
- 27B L0_50 SAE card: https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50
