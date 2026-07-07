# Item J2 — Fresh-Row Confirmation + Staging-Site Steering: A-PRIMARY FAILS (0.806 → 0.664); gauge-not-lever HOLDS at Qwen's staging site

Job 458525 (Scholar, 2× A40, bf16), completed 2026-07-07, 49 min.
`scripts/stage2_qwen_j2_hf.py` per the J2 registration. Full battery:
`docs/qwen_j2_probe.json`; steering rows:
`results/stage2/erasure/qwen_j2_steering.jsonl`.

## Part A — fresh-row position sweep: A-PRIMARY FAILS

96 fresh balanced sources (seed 20260710, disjoint from J1/G6 sources,
test rows, G0 — verified in-job).

| cell | J1 (G6 sources) | J2 (fresh) |
|---|---:|---:|
| concept positions, L43 | 0.504 | 0.483 |
| concept positions, L53 | — | 0.450 |
| **final token, L43** | **0.806** | **0.664** |
| final token, L53 | 0.808 | 0.709 |
| random positions, L43 (2 draws) | — | 0.339 / 0.396 |
| final_L43 label-shuffle p95 | 0.660 | 0.635 |

Registered rule: fresh final_L43 ≥ 0.70 AND above shuffle p95. **0.664
< 0.70 → FAIL.** J1's 0.806 sat at the lucky end of its sampling
distribution (AUC SE ≈ 0.06 at n=96). What DOES replicate is the
pattern: final token above chance and above its shuffle null in both
draws; concept positions at chance in both; random positions at or
below chance. Defensible sentence: "Qwen's outcome information is
weakly-to-moderately decodable at the final prompt token (0.66–0.81
across two independent 96-row draws) and absent at concept positions —
position-dependence is replicable, its strength is not yet pinned."
The "found the shelf" phrasing from the J1 discussion is WITHDRAWN in
favor of the above. No claim lands.

## Part B — staging-site steering: registered prediction HOLDS

Gate: unhinted baseline verbatim vs G6′ on the 24 shared rows (0/192
mismatches; P 0.177 exactly). Rows 3153/3548 (G6-filtered) ride along;
registered pooling on the 24, all-26 descriptive (numbers agree).

| arm | dP vs baseline | 95% CI |
|---|---:|---|
| finaltok class-mean, 0.5× state norm | +0.062 | [−0.010, +0.141] |
| finaltok class-mean, 1.0× | +0.042 | [−0.031, +0.130] † |
| finaltok shuffled, 1.0× | +0.036 | [−0.036, +0.120] |
| classmean − shuffled (paired) | **+0.005** | **[−0.078, +0.078]** |

† parse-gate breach: the 1.0× arm's parse-fail is 9.1% (> the 5%
gate; 0.5× and shuffled arms 1.0–1.4%) — the full-amplitude write at a
single position begins disrupting output format. P(strong|parsed)
preserves the ordering (0.259 vs baseline 0.183), so the breach is
flagged but does not alter the call.

**Registered prediction (i) CONFIRMED — the 6th confirmed registered
prediction of the program**: writing the outcome class-mean AT the
staging site does not repair (both dose CIs straddle zero), and the
texture is decisive — the label-shuffled control matches the real
vector exactly (paired +0.005, dead zero). The small uniform lift
(~+0.04–0.06 in every arm including shuffled) is a label-independent
perturbation artifact at the final token, not steering.

## Combined reading

**Gauge, not lever — now shown at Qwen's staging site by position.**
Where Qwen's outcome information is (weakly) readable — the final
token — writing it does nothing label-specific. Where writing it works
— the concept positions — it is not linearly readable. Gemma separates
gauge from lever by SUBSPACE at one address; Qwen separates them by
POSITION at one layer. The thesis sentence survives its second model
in a form nobody designed: the readable thing and the causal thing are
different objects in both architectures, along different axes.

## Consequences

- Current paper §6 line (hypothesis-flagged, one sentence): "in Qwen,
  preliminary evidence places the readable outcome signal at a
  different POSITION (final token) from the causal write-port (concept
  mentions) at the same layer; writing at the readable site does not
  repair — the gauge/lever separation may be positional there."
- Explainer's "open hunt" phrasing stands (correctly hedged; do NOT
  upgrade to "found the shelf").
- J-series closes for the current paper. Next-paper items, in order:
  (a) pin the final-token decodability with a properly powered probe
  (train on the full stage-1 dataset splits, not 96-row CV — the
  stage-1 L53 probe at 0.94 suggests the ceiling is high); (b) the
  attention-routing test (block final-token→concept attention; trace
  where the concept-position repair write flows); (c) Q1 full map.
- Registered-prediction ledger: 6 confirmed (D branch, H mechanism,
  G4′ sign-flip, G6 sign-flip, G6′ sign-flip-raw, J2 no-repair).
