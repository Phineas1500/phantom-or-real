# Literature Review & Next-Experiment Ranking (2026-07-04)

Deep-research sweep of arXiv (early 2025 – mid 2026) across four angles:
probing vs. causal relevance, low-rank activation steering, causal
abstraction / interchange interventions, and interp of reasoning
correctness. 18 primary sources fetched; 25 load-bearing claims verified
against abstracts/full text by 3-vote adversarial check (25 confirmed, 0
refuted). Companion to `docs/next_paper_skeleton.md` §4 (related work) and
`docs/causal_handle_directions.md` (experiment designs).

## Goal 1 — Scoop check: NOT scooped, but the niche is crowding fast

No published work combines both halves of the core dissociation. Three
preprints from March–May 2026 independently report the **negative half** —
correctness/failure signals linearly decodable yet causally null under
probe-direction steering or ablation — but all three stop at the null: none
localizes the causal variable to a low-rank subspace outside the probe
directions, and none demonstrates a positive behavioral repair. The rank-8
carrier (+0.23 fresh-row repair, 91% of concept-replace) remains the
differentiating contribution.

### The three co-closest 2026 preprints

1. **arXiv 2605.05715** — "Decodable but Not Corrected by Fixed
   Residual-Stream Linear Steering" (May 2026, medical Overthinking regime).
   Linearly decodable at 71.6% balanced accuracy; 29 configs across five
   families of fixed linear steering all ≈0; replicated cross-architecture
   (Qwen2.5-7B) and cross-domain. **Critical difference: its explanation is
   representational ENTANGLEMENT** — LEACE-erasing the decodable direction
   *damages* accuracy (−3.6pp, p=0.01) while 10 random-direction erasures
   are null. That is the *opposite* causal status from our harmless raw-axis
   erasure. This paper is simultaneously (a) the first citation to
   differentiate, (b) the direct methodological template for our LEACE
   experiment, and (c) the rival account our erasure results discriminate
   against. Single-author, under review, numbers may move in revision.
2. **arXiv 2604.13068** — hallucination signals detectable by linear probes
   but steering along the probe direction fails to correct in 7/7 models
   (117M–7B; GPT-2, Pythia, **Qwen-2.5**; 0% correction across 42
   model×magnitude configs). Self-describes as "a clean negative result."
   Consequence: the detection-without-correction asymmetry is already
   documented as architecture-general *including Qwen* — a Qwen null
   replication adds little; only porting the positive carrier is novel.
3. **arXiv 2605.23315** — "Convergence Without Understanding" (May 2026).
   **Terminology collision: already publishes the phrase "epiphenomenal
   correctness."** Correctness decodable across 16 models (66% cross-model
   probe transfer) with only 1.5–5.5% prediction-flip under ablation of top
   probe PCs. But it is a cross-model CKA convergence result, not a
   within-model dissection: no causal-variable localization, no repair, no
   Gemma 3, no ontology reasoning. Our title/abstract wording ("Correctness
   Readouts Are Epiphenomenal") must cite and differentiate immediately —
   or adjust phrasing.

### Precedent that bounds the novelty claim

- **Makelov, Lange & Nanda, arXiv 2311.17030**: subspace patching can look
  causal while the effect flows through a dormant parallel pathway —
  readable-vs-causal subspace mismatch articulated in 2023. Wu, Geiger et
  al. (arXiv 2401.12631) dispute the normative "illusion" framing but not
  the phenomenon. Our novelty is the specific form: high-AUC correctness
  readability that is causally null, causal variable in an *identified*
  rank-8 subspace outside the probe directions, working repair, on a
  reasoning task.
- Correctness decodability itself is established, correlational-only prior
  art: arXiv 2604.05655 (GSM8K correctness AUC 0.87 at L29,
  Llama-3.1-8B; steering never uses the probe direction), arXiv 2504.05419
  (intermediate-answer correctness AUC >0.9 on R1-Distill-Qwen-32B; zero
  intervention vocabulary in the full text), arXiv 2602.06022 (CORAL;
  probe used only for post-hoc answer re-weighting). The causal test is the
  gap we fill.
- Low-rank causal carriers of reasoning are precedented (plausibility
  support, not scoops): arXiv 2509.06608 (trained per-layer steering
  vectors recover 95.3%/87.8% of full fine-tuning's reasoning gains),
  arXiv 2505.15634 (PCA-like contrastive-delta steering basis,
  methodologically parallel to our hinted-minus-unhinted PCA), arXiv
  2506.18167 (reasoning behaviors readable AND steerable — the opposite
  regime, process-style behaviors not correctness).

### Positioning consequences for the draft

- The readable-but-not-steerable gap is now **established background**, not
  a headline. Foreground the positive identification of the causal carrier
  + the erasure-based inertness demonstration.
- Frame the erasure result explicitly as discriminating between two
  published accounts of steering nulls: entanglement (2605.05715 — erasure
  destructive) vs. readable-stack non-necessity (ours — erasing the full
  rank-9 readable subspace at all five layers is behaviorally free,
  +0.047 [−0.070,+0.203], while matched-rank random stacks are
  catastrophic, −0.38; wording per the control-matching verdict: the
  harmlessness tracks the readout's low within-run variance, so we claim
  non-necessity, not special inertness).
- Free theory framing: our causally-null readout maps onto the "behavioral
  null-space" formalism of arXiv 2511.04638 (ICLR 2026 oral); the rank-8
  add must be shown non-"pernicious" in their sense (see experiment 2).
- Three independent groups hit the negative half in a 3-month window — the
  topic is hot and the moat is the positive result. **Re-run the scoop
  check over the June–July 2026 arXiv window immediately before
  submission.**

## Goal 2 — Ranked next experiments (hardening-first)

The two peer-reviewed methodological critiques set the reviewer bar:

- **arXiv 2507.08802** (NeurIPS 2025 spotlight): unrestricted causal
  abstraction is vacuous — expressive nonlinear alignment maps reach 100%
  IIA on randomly initialized models. Cuts in our favor (the rank-8 basis
  is a fixed linear construction) but encoding assumptions must be stated
  explicitly.
- **arXiv 2511.04638** (ICLR 2026 oral): patching/DAS/mean-difference
  interventions frequently push activations off-distribution; formalizes
  harmless (behavioral-null-space) vs. pernicious (dormant-pathway)
  divergences; mean-difference patching specifically "can activate hidden
  pathways."

### 1. Random-basis specificity control + on-distribution diagnostics for the rank-8 core — RUN FIRST

The flagship positive claim (8) currently has no random-BASIS control: the
guard arms include `L30_random_replace` (random positions) but no matched
random rank-8 *add* basis at concept positions. 2507.08802/2511.04638 make
this the first thing a reviewer asks for; if random matched-norm bases also
repair, the specificity claim collapses — existential, so it runs before
anything else.

- Design: guard-v2 harness, one new arm family — 10 random rank-8 bases,
  matched per-component norms to the LOO PCA basis, added at concept
  positions at the same scale; same 26 fresh rows or the 13 manifest rows,
  k=8, row-cluster bootstrap. Offline riders on saved states (no GPU):
  logit-lens/unembedding projection of the 8 components, principal angles
  vs the INLP probe stack per layer (sharpens claim 9's subspace fractions
  into angles), rank sweep 1–16 from the saved concept-position submatrix,
  nearest-neighbor distance of steered L30 states to natural hinted states
  + dormant-unit activation check (2511.04638 diagnostics).
- Cost: one 4h SLURM slot + offline analysis.
- Branches: random bases null → specificity guarded, claim 8 hardened, and
  the on-distribution numbers become a methods sidebar. Random bases repair
  → the repair is generic-perturbation at that norm; claim 8 must be
  re-scoped before drafting (this is exactly why it runs first).

### 2. LEACE full-readable-subspace erasure + layer-exhaustive ablation

Claim 3's listed future work, upgraded to must-do: it is the unique
discriminator against 2605.05715's entanglement account, and it closes
claim 7's remaining necessity question (single-fixed-layer nulls are
unreliable per arXiv 2604.03867: per-input optimal steering layers beat the
best fixed layer by 55–86%, oracle bound). Methodological blueprint: arXiv
2506.11673 (ACL 2025 Findings — mean-projection/LEACE cause far less
collateral damage than INLP in amnesic probing).

- Design: LEACE (or mean-projection) of the full INLP readable stack, all
  readable layers simultaneously, prompt positions; controls per house
  style (orthogonal, Gaussian, height direction, dose-response,
  P(strong|parsed) split) PLUS the 10-random-direction erasure control
  mirroring 2605.05715 so the cross-paper comparison is direct.
- Cost: LEACE fit is CPU on saved activations; one to two 4h slots for the
  behavioral arms (property; subtype optional).
- Branches (no null outcome — same structure as the original erasure
  decision rule): accuracy intact → epiphenomenality upgrades from "the
  readout axis" to "the entire readable subspace," the entanglement rival
  is excluded in our setting, claim 3's caveat is deleted. Accuracy drops →
  the readable subspace is entangled and only the axis is inert; claims
  2–3 re-scope, and the paper aligns with 2605.05715's account at subspace
  granularity — still publishable, framed as reconciliation.

### 3. Subtype carrier: layer × position sweep with token-conditional coefficients

The row-sparseness of the L35 result (3/13 addressable rows) may be
per-row layer mismatch, not a weak carrier: 2604.03867 shows per-input
optimal layers deviate 3.8–6.5 layers from any fixed choice, and arXiv
2605.03907 shows prompt-induced activation shifts are strongly
token-position-dependent (convergent with our concept-mention finding;
motivates token-conditional rather than constant coefficients).

- Design: extend the discriminator ladder to a dense layer grid with
  per-row best-layer readout (report both fixed-layer pooled and
  per-row-oracle numbers, labeled as such), concept-mention-gated
  coefficients; same 16 manifest rows + fresh seeds, k=8.
- Cost: one to two 4h slots.
- Branches: per-row layer selection recovers a dense repair → claim 12
  upgrades toward matched replication (with the oracle/fixed distinction
  stated). Still sparse → current wording stands and the hedge closes
  harder; the property/subtype asymmetry becomes a real task-structure
  finding (open question 4 of the survey).

### 4. Qwen positive-carrier port — DEMOTED to next paper

The null half is already architecture-general including Qwen-2.5
(2604.13068), so a Qwen null replication adds nothing; only porting the
rank-8 carrier is novel, and that is a full pipeline (hint suite, delta
extraction, PCA, guard) — not a hardening item. Keep claim 11's current
scoping sentence. Scope to Qwen2.5-7B/14B (fits 2 GPUs) if reviewers demand
a second architecture; otherwise it is the opening experiment of the next
paper (and the survey's open question 3).

## Writing to-dos (zero compute)

- Related work §4 restructure: co-closest prior = 2605.05715 + 2604.13068
  (cite both; a verifier split 2-1 on any "closest" superlative); the
  terminology-collision paragraph for 2605.23315; Makelov 2311.17030 +
  Wu 2401.12631 as the readable-vs-causal precedent pair; correlational
  correctness-probing block (2604.05655, 2504.05419, 2602.06022); low-rank
  carrier precedent block (2509.06608, 2505.15634, 2506.18167); methods bar
  (2507.08802, 2511.04638, 2506.11673). Existing planned citations (Cox et
  al., ITI, hydra-effect, Gemma Scope) stand.
- State encoding assumptions for the rank-8 construction explicitly
  (fixed linear PCA basis, no trained alignment map) — the 2507.08802
  defense.
- Adopt/engage the behavioral-null-space vocabulary of 2511.04638 for the
  probe-readout result.

## Caveats

- The three closest analogues are non-peer-reviewed preprints from
  March–May 2026; numbers may change in revision. The not-scooped verdict
  is bounded by arXiv coverage; June–July 2026 postings not exhaustively
  swept — re-check at submission.
- 2605.03907 / 2604.03867 study persona/alignment (CAA) steering on
  Llama-2 / Qwen-1.5 / Gemma-2 — their token-position and layer-selection
  lessons are directional analogies for Gemma 3 27B ontology reasoning,
  not demonstrated transfers; the 55%/86% figures are post-hoc oracle
  bounds.
- 2509.06608's vectors are per-layer dense biases (~100–230K params), a
  low-parameter but not literally rank-8 intervention; its gain-recovery
  figures are 95.3%/87.8% (verifier-corrected from the abstract's rounding).
