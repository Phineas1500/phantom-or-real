# Bibliography — Gauge, Not Lever (living doc, started 2026-07-06)

One entry per source: ID · what it shows · where we cite it · verification
provenance (which sweep verified quotes: L1 = July-4 litreview, L2 = July-5
novelty check, L3 = Codex disconfirmation sweep; UNV = not yet
quote-verified — verify before the camera-ready). Keep sorted within
sections; add new sources here FIRST, then cite.

## A. Co-closest rivals (readable-but-unsteerable; §1, §4 block 1)

- **2605.05715** — "Decodable but Not Corrected": failure decodable 71.6%,
  29 linear-steering configs null, LEACE erasure destructive −3.6pp →
  entanglement account. Our item D discriminates it. [L1]
- **2604.13068** — detection-without-correction in 7/7 models incl.
  Qwen-2.5; clean negative. [L1]
- **2605.23315** — coins "epiphenomenal correctness" (cross-model CKA
  convergence; top-PC ablation 1.5–5.5%). Cite at first use of the word.
  [L1]

## B. Diff-in-means / steering-for-reasoning lineage (§4 block 2)

- **2601.19847 AdaRAS** — mean-diff Reasoning-Critical MLP neurons,
  Qwen3-1.7B/4B; AIME-24 +13, HumanEval +21; failure-probe gating (AUROC
  0.8347). Closest overall; gauge-gating precedent. [L2]
- **2509.18116 ALS** — raw full-dim class-mean, penultimate layer,
  per-token; MATH-500 76→91 (3.5→68.5 degraded prompt). Their raw vector
  works where ours nulls. [L2]
- **2505.12189 Valentino et al.** — diff-in-means from correct vs
  content-bias-incorrect syllogism runs; +15% (K-CAST variant); static
  per-model results mixed. [L2]
- **2604.05655** — trajectory steering; mid-reasoning correctness
  predictor (ROC-AUC 0.87) gates rank-32 PCA steering on 12.3% of GSM8K;
  hardest-slice gains ~8pts. Also correlational-probing background. [L2]
- **2509.23799 SAE-RSV** — raw diff-in-means vectors 93.6% noise; SAE
  filtering improves 45.6→56.4%. Filtering helps ≠ necessary. [L2]
- **2510.05498 PDS** — CoT-prototype prompt-contrast steering, GSM8K
  68→74; no outcome labels. [L2]
- **2506.15882 Fractional Reasoning** — PC1 of prompt contrasts,
  20-strength ensembles. [L2]
- **2405.09719 SEA** — spectral editing for truthfulness; HURTS GSM8K
  (−1.52). Verified NOT prior art (refuted 0-3 twice). [L2]
- **2306.03341 ITI (Li et al.)** — truthfulness intervention lineage
  root; truthfulness/helpfulness tradeoff. [L3]

## C. Steering-rigor critiques (§4 block 4; appendix checklist)

- **2407.12404 Tan et al.** — per-input anti-steerability under aggregate
  gains → our per-row deltas (16↑/0↓). [L3]
- **2410.17245 Pres et al.** — evaluation-protocol critique → our
  parse-conditional + weak-match endpoints. [L3]
- **2501.17148 AxBench** — prompting outperforms steering → our
  self-consistency 0.000 / best-of-8 0.192 baselines. [L3]
- **2505.22637 Braun et al.** — item-level sign/coherence diagnostics.
  [L3]
- **2507.11771 Ali et al.** — CAA shrinks with model size; our 27B result
  runs against the trend. [L3]
- **2411.07213 Brumley et al.** — ICV unpredictability/brittleness →
  limitations. [L3]
- **2406.15518 Stickland et al.**; **2411.11296 O'Brien et al.** — side
  effects on general capabilities → limitations. [L3]
- **2312.06681 CAA (Panickssery)**; **2308.10248 ActAdd (Turner)** —
  technique roots + token-separation confound notes. [L3]
- **2406.11717 Arditi et al.** — refusal single direction; add-direction
  side effects. [L3]
- **2602.06801** — steering-vector non-identifiability → "an identified
  sufficient channel" wording. [L3]
- **2605.05973 Xu et al.** — winner's-curse in layer/scale search → our
  provenance discipline. [L3]
- **Gelman & Carlin 2014** — Type-S/M errors at small n → stats sidebar.
  [L3]

## D. Methods bar & precedents (§3, §4 block 3)

- **2507.08802** — causal-abstraction vacuity (NeurIPS'25 spotlight) →
  state encoding assumptions explicitly. [L1]
- **2511.04638** — harmless vs pernicious divergence; behavioral
  null-space vocabulary (ICLR'26 oral). [L1]
- **2506.11673** — mean-projection/LEACE over INLP (ACL'25 Findings) →
  item-D operation choice. [L1]
- **2311.17030 Makelov, Lange & Nanda** — subspace-patching illusions;
  **2401.12631 Wu et al.** — counterpoint. [L1]
- **2504.05419**, **2602.06022 CORAL** — correlational correctness
  probing background. [L1]
- **2509.06608**, **2505.15634**, **2506.18167** — low-rank causal
  carrier precedents. [L1]
- **2604.03867** — per-input optimal steering layers; **2605.03907** —
  token-position-dependent steering. [L1]

## E. Concurrent convergence (§4 block 5)

- **Anthropic, transformer-circuits.pub/2026/workspace** — "Verbalizable
  Representations Form a Global Workspace": J-space carries 6-7% of
  variance yet is causally dominant (59% vs 5% swaps; non-J → 0 under
  clamping); decodable-but-inert cases; "ignition" commitment at relative
  depth 0.38–0.92. Cite as convergence; differentiate on variable/repair/
  controls. Also methodologically load-bearing for the Qwen mystery
  (§6/next paper): Tuned Lens (linear regression on states) substantially
  underperforms J-Lens — linear probes under-read consumption-staged
  content (see qwen_jspace_connection.md). [WebFetch summary 2026-07-06;
  UNV — verify quotes against the page before camera-ready]
- **Nanda, "A review of Anthropic's Global Workspace paper"
  (lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S)** — independent skeptical
  review endorsing the core claim ("overwhelming amount of evidence for
  the existence of this cognitive space") with a **Qwen 3.6 27B
  replication** (Jacobians to penultimate layer, 25 Pile prompts, 128
  tokens; most probing/causal effects reproduce, weaker but positive;
  poetry/arithmetic fail on capability grounds). Cite: (i) as
  third-party validation of the workspace machinery in the Qwen family;
  (ii) for the explicit probing-vs-Jacobian reading distinction; (iii)
  for the cheap replication recipe our J-aware F(i)-analog would use.
  [WebFetch summary 2026-07-07; UNV — verify quotes before camera-ready]

## F. Task, tooling, and infrastructure

- **InAbHyD / beyond-deduction (upstream v2 paper)** — task generator,
  scoring conventions, system prompt. [repo]
- **TransformerLens** — Gemma intervention harness. [repo]
- **Gemma Scope** — public SAEs for the claim-9 geometry rider. [repo]
- **INLP (Ravfogel et al.)**; **LEACE (Belrose et al.)** — erasure
  operations lineage. [UNV — pull canonical IDs]
- **Gemma 3 / Qwen3.5 model reports**. [UNV]
- **Global workspace theory primary sources (Baars; Dehaene)** — only if
  we lean on the workspace framing; otherwise cite via the Anthropic
  paper. [UNV]

## Maintenance rules

1. New source → entry here first (with sweep provenance), then cite in
   drafts.
2. Before submission: every UNV entry gets a quote-verification pass; the
   scoop re-check appends anything new to §A/§B.
3. Numbers quoted in the paper must match this file's one-liners; if they
   diverge, the pooled verdict docs win and this file gets corrected.
