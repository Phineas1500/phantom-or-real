# Comparison Matrix — Our Work vs the Closest Papers (2026-07-06)

Companion to `contribution_ledger_2026-07-05.md` (what we claim) and
`bibliography.md` (verification provenance). One row per capability the
paper touches; ✓ = demonstrated, ~ = partial/adjacent, — = absent.
Sources: L1/L2/L3 verified sweeps + WebFetch of the Anthropic paper.

## The matrix

| capability | **ours** | 2605.05715 "Decodable ≠ Corrected" | 2604.13068 detect-w/o-correct | 2605.23315 "epiphenom." | AdaRAS 2601.19847 | ALS 2509.18116 | Valentino 2505.12189 | 2604.05655 trajectory | SAE-RSV 2509.23799 | **Anthropic workspace 2026** |
|---|---|---|---|---|---|---|---|---|---|---|
| correctness/error readable | ✓ 0.90–0.94, 2 models | ✓ 71.6% | ✓ 7/7 models | ✓ | ~ (failure probe 0.83) | — | — | ✓ 0.87 | — | — (concepts, not correctness) |
| steering the readout fails | ✓ (~10 attempts) | ✓ 29 configs | ✓ | ✓ | — | — | — | — | — | — |
| erasure of readout tested | ✓ axis AND full rank-9 stack, harmless; destructive matched-rank controls | ✓ LEACE, HURT −3.6pp → entanglement | — | ~ top-PC ablation ±1.5–5.5% | — | — | — | — | — | ~ J-ablation impairs reasoning (different variable) |
| repair of failing runs (accuracy ↑) | ✓ +0.25 patch → +0.40 answer-free on 0.12 baseline | — | — | — | ✓ +13 AIME (neuron steering) | ✓ MATH-500 76→91 | ~ ±, model-dependent | ~ +8 hardest slices | ~ behavioral concepts, not accuracy | — (content redirection, not repair) |
| answer-free CONTENT at inference | ✓ (class-mean, pooled scale) | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | n/a |
| answer-free ADDRESSING | ✗ measured & bounded (item H: gold positions load-bearing; policies fail) | — | — | — | ~ global neuron set | ~ every token | ~ last token | ~ rank-32 every-step | ~ | n/a |
| **necessity dissociation** (raw null / subspace potent, matched norm) | **✓ +0.043 vs +0.341** | — | — | — | — | — (raw works) | — (raw works) | — | ~ raw = 93.6% noise; filtering helps, never shown necessary | **~ convergent: non-J inert (→0 clamped), J potent — different variable/method** |
| shuffled-label control | **✓ −0.043 (4 draws)** | — | — | — | — | — | — | — | — | — |
| sign-flip control | **✓ −0.120 (harmful)** | — | — | — | — | — | — | — | — | ~ bidirectional swaps (opposite property!) |
| causal-without-decodable | **✓ F(i)+F(ii)** | — (opposite: decodable w/o causal) | — | — | — | — | — | — | — | ~ both directions shown for verbalizable concepts |
| sampling-baseline comparison | ✓ self-consistency 0.000; best-of-8 0.192 | — | — | — | ~ | ~ | — | — | — | — |
| collateral cost on correct inputs | ✓ beneficial +0.266 | — | — | — | — | — | — | — | — | — |
| probe-gated deployment | ~ tested, gating OPTIONAL (collateral benign) | — | — | — | ✓ AUROC 0.8347 gate | ~ cosine gate | — | ✓ 12.3% gated | — | — |
| model scale | 27B (+27B: carrier replicates, rank-8 core directional-only) | mid | 7 models | multi | 1.7–4B | 7B | 1–9B | 8B | 8B | frontier (Claude 4.5/4.6) |
| pre-registration / branch-complete rules | ✓ 10+ items, 2 predictions confirmed | — | — | — | — | — | — | — | — | — |

## Narrative: the three comparison frontiers

1. **vs the readable-but-unsteerable trio** (05715/13068/23315): they
   establish the wall; we walk through it. The decisive divergence is
   05715's destructive LEACE (→ entanglement account) vs our harmless
   full-stack erasure with destructive matched-rank controls — item D
   directly adjudicates, in our setting, in favor of non-necessity.
2. **vs the diff-in-means repair lineage** (AdaRAS/ALS/Valentino/
   trajectory/SAE-RSV): they own the phenomenon; we own the mechanism
   and the rigor. Every one of their raw vectors works, so none can say
   where the causal content lives; none runs label/sign controls; none
   reports sampling baselines or collateral. Our +0.40 on a 0.12 slice
   (where self-consistency = 0.000) is ~2–5× their hardest-slice gains,
   at a scale where the literature says such effects shrink.
3. **vs Anthropic's workspace paper**: convergence on structure
   (tiny causally-privileged subspace; inert majority; decodable ≠
   causal; ignition-like commitment; depth band containing our L30),
   disjoint on substance (no correctness, no repair, no controls, no
   accuracy). One sharpening CONTRAST: their J-swaps steer
   bidirectionally; our channel repairs but never misdirects (sign-flip
   harms; item H: commitment verb, address = object) — the lever may be
   a decision mechanism within (or beside) their content workspace.
   Bridging test reserved for the next paper: J-lens ↔ rank-8 overlap.

## One-line summary

Nine papers hold pieces — the wall, the recipe family, the noise
account, the workspace geometry; ours is the only one holding the
mechanism (where the causal content lives), the proof it's the labels
(shuffled/sign battery), both dissociation directions in one model, and
the deployment ledger — under pre-registration, at 27B.
