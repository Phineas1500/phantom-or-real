# Next-Paper Section Mapping (Inverted Pyramid)

Structural decision (2026-06-12): the dashboard is a detective story —
chronological, each null earning the next experiment. The paper inverts the
pyramid: lead with the verdict, present evidence as support, not as a hunt.

## Lead (abstract + intro thesis)

Readable everywhere, needed nowhere: correctness probes read an epiphenomenal
evaluation. The causal variable is concept commitment, written in place at
mention sites during prompt processing, multiply realized across pathways,
and invisible to the readable subspace.

## Claims-table -> section mapping

| Section | Claims | Figures (from dashboard SVGs) |
| --- | --- | --- |
| 1 Intro + thesis | 1, 10 | Part 6 causal diagram |
| 2 The readout is epiphenomenal (necessity) | 2, 3 | Part 2 erasure bars; INLP curve (Part 3) |
| 3 The failure is concept focus (behavioral) | 4 | Part 5 candidates bars + follow-up table |
| 4 The focus state is causally accessible (interchange) | 5 | Part 7 interchange bars |
| 5 Localization and structure (hint-delta + composite) | 6, 7, 8 | Part 8 arm bars; Part 9 composite figure |
| 6 Geometry: lever outside the gauge's subspace | 9 | quantified projection plot |
| 7 Discussion: multiply realized variables | 7, 8 + redundancy symmetry | — |
| 8 Robustness | 11, 12 | subtype/Qwen tables |
| Appendix | negative-results table | steering decision table |

## Reviewer-fire sections — write first, carefully

- **Erasure control asymmetry**: write now with an explicit placeholder for
  the between-run control (job pending); cite Cox et al.; include
  P(correct|parsed) and the dose-response when it lands. Defensibility
  hinges on the between-run control.
- **Necessity at ceiling**: the ceiling caveat appears in the same sentence
  as the null, never in a footnote. Wording fixed in the claims table.

## Limitations paragraph (verbatim draft)

> Our causal results characterize controllability, not training-time
> mechanism. Interventions were evaluated on recognition-gap rows — items
> where free-form generation reliably fails while forced-choice recognition
> succeeds — which is the population of interest for the deployment-gap
> question but not a random sample of task items. Mechanistic claims are
> established on Gemma 3 27B (cross-model replication of the predictive
> results, and of the epiphenomenality result pending the Qwen erasure) and
> on two InAbHyD tasks (the localization replication on the subtype task
> reported in §8). Patching interventions target three residual layers
> (L30/L40/L45); the necessity analysis shows pathways outside these layers
> and in the token stream remain candidates, which the KV/hint-span
> experiment addresses. InAbHyD is a synthetic ontology benchmark; the
> recognition-generation dissociation and the gauge/lever geometry should
> be tested on naturalistic reasoning tasks before generalizing.

## Scope rule

Sections map 1:1 to landed claims; the KV job fills §5's mechanism
subsection and the robustness jobs fill §8. Nothing else enters the draft.
