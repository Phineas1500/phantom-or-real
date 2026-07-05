# Steering-Artifact Checklist Triage (2026-07-05)

Source: outsourced disconfirmation sweep (Codex, 5 parallel agents +
synthesis; bibliography of 18 verified sources incl. Tan 2407.12404,
Pres 2410.17245, AxBench 2501.17148, Braun 2505.22637, Ali 2507.11771,
Xu 2605.05973). Their ranked 10-item artifact checklist for our +0.341
class-mean-proj result, triaged against our data. Offline diagnostics run
same day on the F(ii) row-level JSONLs (26 rows × 8 samples/arm).

| # | artifact class | status |
| --- | --- | --- |
| 1 | failure-selection + small-n inflation | **CLEARED (design)**: row-cluster bootstrap is the only sanctioned sigma (26 clusters, never 208 samples); LOO band [+0.315,+0.355]; in-job regenerated baselines handle regression-to-mean. Residual: a third fresh failure draw would strengthen further (candidate arm). |
| 2 | parser/format artifact | **MOSTLY CLEARED**: parse rates flat (proj 0.817 vs baseline 0.904 — slightly LOWER); P(strong\|parsed) 0.565 vs 0.133; P(weak) tracks. Residual: no logprob/blinded-adjudication route; sample-level manual read of repaired generations pending. |
| 3 | weak baselines | **CLEARED (new, decisive)**: on the same 26 rows, majority-vote self-consistency over the 8 baseline samples scores **0/26 (0.000)** — the model's modal answer on hard rows is systematically wrong; best-of-8 ceiling is 0.192. The proj arm's 0.462 exceeds the best-of-n CEILING by 2.4×, so no sampling-based baseline explains it. Residual: few-shot/CoT prompting arms not run (AxBench defense: our claim is causal control, not test-time-accuracy SOTA — wording W7). |
| 4 | per-item anti-steering (Tan/Braun) | **CLEARED (new)**: per-row proj deltas — 16/26 improved, 10 unchanged, **0 harmed** (min delta +0.00). No anti-steered items; effect not outlier-driven. |
| 5 | layer/position/scale multiplicity (Xu winner's curse) | **CLEARED (provenance)**: L30 and concept positions were fixed on development rows months before the fresh guard set existed; scale is norm-matched, not tuned; pre-registrations sequence everything. Disclose the historical layer search in methods. |
| 6 | entropy/answer-collapse | **MOSTLY CLEARED (new)**: no global-attractor collapse — targets_gold_concept rises only 0.380→0.490 (gold differs per row, so a fixed-vector attractor would show up here) while accuracy rises 0.120→0.462; notably, precision GIVEN gold-targeting jumps 32%→94%, i.e., the intervention fixes the whole hypothesis, not just subject choice. Residual: token-entropy/KL not computable from stored fields (logits not saved) — note as limitation or add telemetry to a future arm. |
| 7 | prompt brittleness (Brumley) | **OPEN**: no paraphrase/reorder/template variants run. Next-paper scope; list in limitations. |
| 8 | refusal/abstention/length | **CLEARED (new + design)**: generation length is fixed at the 96-token cap in every arm (length artifact impossible); zero empty outputs; parse-fail (abstention proxy) flat. Unrelated-task side effects (O'Brien/Stickland) untested — limitations. |
| 9 | leakage/holdout | **CLEARED (design)**: capture rows (vector), composite rows (subspace), and guard rows (evaluation) are pairwise disjoint; parser frozen before stage 2; rules pre-registered before unblinding. Document the provenance chain in methods. |
| 10 | mechanistic overclaim (non-identifiability, 2602.06801) | **MOSTLY CLEARED**: items C + F(ii)-b are exactly the demanded battery (random subspaces, shuffled labels, sign-flip, noise). Residual: "8-dim subspace is more specific than behaviorally equivalent alternatives" — rotations-within-subspace and mediation analysis not run; soften wording to "an identified sufficient low-rank channel" rather than unique. |

## New numbers from today's diagnostics (cite in appendix)

- Self-consistency (maj-of-8) on hard rows: 0.000; best-of-8: 0.192;
  proj arm single-sample: 0.462.
- Per-row proj deltas: 16 improved / 10 flat / 0 harmed.
- targets_gold_concept: 0.380 → 0.490; P(strong | targets gold):
  0.32 → 0.94.
- Length: constant 96 (cap) all arms; empty outputs: 0.

## Defenses to prepare (Codex's "most dangerous citations")

- **Tan et al. 2407.12404** → per-item deltas reported (0 harmed).
- **Pres et al. 2410.17245** → exact-match is not sole endpoint
  (parsed-conditional + weak-match reported; logprob route acknowledged
  as future work).
- **AxBench 2501.17148** → prompting baselines: self-consistency now
  reported (0.000); claim scoped to causal control, not SOTA test-time
  accuracy.
- **Ali et al. 2507.11771** (CAA weakens with model size) → our effect is
  AT 27B, which runs against the shrinkage trend — worth one sentence.

## Remaining to-do fold-in

- W7 limitations gains items 6 (entropy telemetry), 7 (brittleness), 8
  (unrelated-task side effects), 10 (identifiability wording).
- Bibliography merged into next_paper_skeleton §4 references at drafting.
