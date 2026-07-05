# Novelty Check — New Findings (F-series / hint-free thread), 2026-07-05

Deep-research sweep (103 agents, 21 sources fetched, 25 claims → 21
confirmed / 4 refuted by 3-vote adversarial verification). Scope: the five
NEW findings from 2026-07-04/05, distinct from the July-4 sweep that
positioned claims 1–12. Full JSON in the session task output; verdicts and
consequences below. All five closest rivals are unreviewed preprints; two
(AdaRAS, trajectory-steering) are recent enough for concurrent-work
framing, but all must be cited.

## Verdicts

| finding | verdict | closest prior work |
| --- | --- | --- |
| 1. Class-mean repair of reasoning correctness | **PARTIALLY ANTICIPATED — material scoop risk on the phenomenon** | AdaRAS (2601.19847); Valentino et al. (2505.12189); ALS (2509.18116) |
| 2. Raw diff-in-means null / subspace-projected potent | **NOVEL** | SAE-RSV (2509.23799) — adjacent premise only |
| 3. Shuffled-label + sign-flip control battery | **NOVEL** (medium conf.) | none found |
| 4. Causal alignment without decodable alignment | **NOVEL** (medium conf.) | none found |
| 5. Answer-free lift on hardest failures | **PARTIALLY ANTICIPATED**; +40pt magnitude distinctive | AdaRAS; ALS; 2604.05655 |
| Gauge-gated deployment framing | **SCOOPED as a framing** | AdaRAS (probe AUROC 0.8347 gates steering); 2604.05655 (mid-reasoning predictor gates on 12.3% of examples) |

## The five papers to cite (verified quotes in the sweep output)

- **AdaRAS (arXiv:2601.19847, Jan 2026)** — closest overall. Polarity-aware
  mean-difference over naturally-correct vs incorrect trajectories selects
  "Reasoning-Critical Neurons" (sparse MLP neurons, Qwen3-1.7B/4B);
  answer-free steering lifts AIME-24 47.8→60.9, HumanEval 57.7→79.2; a
  failure-prediction probe (AUROC 0.8347) gates the intervention.
  Differences from us: MLP neurons not a causally-identified residual
  subspace; no amplitude matching; no raw-vs-projected dissociation; no
  label-shuffle/sign controls; no decodability analysis; math/code not
  abduction; 1.7–4B not 27B.
- **Valentino et al. (arXiv:2505.12189)** — diff-in-means from correct vs
  content-bias-incorrect syllogism runs, residual stream, up to +15%
  (K-CAST variant; static per-model numbers smaller, sometimes negative —
  Gemma 2 9B −1.7). Last-token intervention, binary validity task.
- **ALS (arXiv:2509.18116)** — RAW full-dim class-mean from ~1,500 graded
  generations, per-token at penultimate layer; MATH-500 76→91
  (Qwen-2.5-7B), 3.5→68.5 under a degraded prompt. NOTE: their raw vector
  WORKS where ours nulls — different layer/application; they never test
  the raw-vs-projected contrast. Cosine-threshold gating (crude gauge).
- **Trajectory steering (arXiv:2604.05655)** — mid-reasoning correctness
  predictor (ROC-AUC 0.87) gates rank-32 PCA steering on 12.3% of GSM8K;
  gains concentrate on hardest slices (6-step 75.4→83.0) but are ~8pts vs
  our +40; contrast is termination-vs-step states, not natural
  correct-vs-incorrect.
- **SAE-RSV (arXiv:2509.23799)** — raw diff-in-means vectors are "93.6%
  noisy features"; SAE filtering improves steering 45.6→56.4% on
  behavioral concepts. Filtering HELPS but is never shown NECESSARY; the
  necessity dissociation (Finding 2) remains ours.

Also adjacent (answer-label-free prompt-contrast lineage, cite in related
work): PDS (2510.05498, GSM8K 68→74 via CoT-prototype steering);
Fractional Reasoning (2506.15882, PC1 of prompt contrasts, needs 20-sample
ensembles). SEA (2405.09719) verified NOT prior art (0-3 refuted twice —
hurts GSM8K).

## Consequences for positioning

1. **The phenomenon "class-mean steering raises reasoning accuracy" is a
   2025–26 lineage, not our discovery.** The next-paper framing must lead
   with what IS ours: the mechanistic account — causal-subspace NECESSITY
   (raw null at matched norm: the direct opposite of ALS's working raw
   vector), the label-specificity battery, causal-without-decodable, and
   the gauge/lever dissociation feeding it. Our +0.40 on a 12% baseline
   also remains the largest hardest-slice repair in the verified set.
2. **Gauge-gated deployment cannot be claimed as a novel framing** —
   AdaRAS and 2604.05655 already run probe-gated steering. Cite both;
   differentiate on what is gated (a causally-identified subspace
   class-mean vs neuron/PCA steering) and on the F(ii)-c collateral-slice
   discipline.
3. **Sweep's open question #4 is already answered in our data**: whether
   the subspace projection is load-bearing for the headline lift IS
   F(ii)'s raw-vs-proj contrast (+0.043 vs +0.341). Feature this pairing
   prominently — it is simultaneously our novelty moat and the direct
   reply to the AdaRAS/ALS lineage.
4. **Thin-coverage caveats**: the LEACE/amnesic-probing theory lineage
   (does theory predict causal-without-decodable?) and steering-rigor
   critiques returned no confirmed claims — Finding 3/4 novelty is
   medium-confidence. The outsourced disconfirmation sweep (Codex
   prompts, 2026-07-05) covers part of this; fold its results in here.
5. **Re-run before submission** — three of the five closest papers
   appeared within nine months.
