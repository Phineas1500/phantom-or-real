# Items WZ + WC — the literature's label-free detectors, and the reasoning / decoding baselines (2026-09-05)

Registered before data (docs/causal_handle_directions.md items WZ, WC).
Jobs: WZ scoring **job-u82kn** (NQ-Swap), **job-hqd3e** (counterfactual
SQuAD), **job-z9jdx** (WikiHop states; the first WikiHop attempt
job-meq6r hit its 90-minute cap after a preemption), **job-kikqy**
(WikiHop per-candidate riders); WC chain-of-thought **job-d5dpu /
job-bmsjk / job-h54v4**. Readers `scripts/wikihop_wz_gates.py`,
`scripts/wikihop_wc_reader.py`.

## Detectors (probes fit on each frame's non-blind rows, scored on its blind rows)
| detector | NQ-Swap (240 blind) | counterfactual SQuAD (120) | WikiHop real (219 WE rows) |
|---|---|---|---|
| answer-token probe, L30 mean (Orgad et al.) | **0.998** · recall 0.98 @ FPR 0.01 | **0.995** · 0.93 @ 0.03 | 0.850 · 0.51 @ 0.07 |
| answer-token probe, best pooling | 1.000 (L43 last) | 0.998 (L43 mean) | **0.872** (L30 last) · 0.44 @ 0.05 |
| final-prompt-token probe (the gauge's position) | 0.86–0.95 | 0.83–0.94 | 0.76–0.79 |
| P(True), the model's own verdict (Kadavath et al.) | 0.653 | 0.658 | 0.521 |
| NQ-Swap-fit probe transferred | — | 0.97 (FPR 0.20) | **0.57** (FPR 0.92) |

**The calibration correction, disclosed.** The first reading set the
≤ 5%-FPR operating point on in-sample training scores; a near-separable
probe inflates those and the test FPR came out 0.17–0.35. The registered
quantity is estimated out-of-fold; with five-fold OOF scores the test
FPR is 0.01–0.03 on the conflict frames. Both readings are in the
registry.

**The swap artifact, pre-registered and borne out.** On both conflict
frames the correct answer is by construction the swapped, out-of-place
entity. The probe's near-perfect separation there does not transfer to
WikiHop (AUROC 0.57), and on WikiHop's own rows it reaches 0.87. Probing
the answer tokens instead of the final prompt token is a real gain on
natural text (+0.1 AUROC) and still far from a deployable detector.

## Deployment: the two-stage rule with flag = groundedness ∨ probe, output-vote selector
| frame | groundedness alone | ∨ answer-token probe | oracle detector |
|---|---|---|---|
| NQ-Swap (240) | +0.056 [+0.029, +0.087] · 14 / 1 | **+0.080 [+0.040, +0.123]** · 23 / 5 | +0.080 |
| counterfactual SQuAD (120) | +0.033 [+0.008, +0.067] · 4 / 0 | **+0.070 [+0.030, +0.117]** · 9 / 0 | +0.078 |
| WikiHop real (WE rows, stratum-weighted) | +0.001 | **+0.010 [−0.009, +0.026]** · 12 / 1 | +0.029 [+0.018, +0.041] |

**33rd (A-LITERATURE-DETECTOR-CROSSES-THE-BAR): CONFIRMED** on the
pooled conflict rows (AUROC 0.997; paired gain over the groundedness
rule +0.028 [+0.007, +0.051]), with the two qualifications above.
**34th (THE BLIND LOOP REACHES WIKIHOP): NOT CONFIRMED.** P(True) as a
branch selector is worse than the output vote everywhere.

## Rerankers without the loop (same blind rows)
| reranker | NQ-Swap | counterfactual SQuAD | WikiHop |
|---|---|---|---|
| argmax P(True) over candidates | −0.007 [−0.046, +0.034] | −0.002 [−0.052, +0.048] | (job-kikqy pending) |
| **context-aware decoding, α = 0.5** (Shi et al.) | **+0.106 [+0.067, +0.148]** | **+0.115 [+0.062, +0.174]** | (pending) |

CAD as a plain candidate reranker is the strongest frame-level method
on the conflict regime, ahead of the instruction (+0.077 / +0.051), the
probe-driven loop (+0.080 / +0.070), the grounded loop (+0.056 /
+0.033) and chain-of-thought.

## WC — chain-of-thought (k = 4, 384 tokens, answer parsed from the last "Answer:" line)
| | NQ-Swap | counterfactual SQuAD | WikiHop real |
|---|---|---|---|
| accuracy vs document: std → CoT | 0.578 → 0.585 (+0.7) | 0.674 → 0.703 (+2.8) | 0.466 → 0.475 (+0.8) |
| repair share of failures | 0.150 [0.106, 0.195] | 0.243 [0.165, 0.330] | 0.076 [0.047, 0.109] |
| … of hint-repairable rows | 0.338 | 0.564 | 0.220 |
| … of grounded wrong answers | **0.066** | 0.167 | **0.073** |
| correct-majority rows broken | 8.9% | 5.0% | 8.9% |

Reasoning fixes a third to a half of the attention lapses on the
conflict frames but breaks 5–9% of correct rows and barely touches the
grounded lapses, netting less than the instruction and about the same
as or less than the blind grounded loop, which is the only method with
near-zero collateral.

## The comparison table the paper needs (frame-level accuracy change, blind, Gemma-3-27B)
| method | NQ-Swap | counterfactual SQuAD | WikiHop real | collateral |
|---|---|---|---|---|
| context-aware decoding reranker | **+0.106** | **+0.115** | pending | in the net |
| instruction alone (WI) | +0.077 | +0.051 | — | 1–2% |
| grounded loop ∨ answer-token probe (WZ) | +0.080 | +0.070 | +0.010 n.s. | 5 / 0 rows |
| grounded loop (WK′ / WS) | +0.050 | +0.033 | 0 (inert) | 0 rows |
| chain-of-thought (WC) | +0.007 | +0.028 | +0.008 | 5–9% |
| always-answer loop | −0.10 | −0.03 | — | −0.11 to −0.27 |

## Verdict
The grounded-lapse gap is not closed. The best literature detector on
natural text is a 0.87-AUROC probe; the model's own verdict is at
chance; the conflict-frame "solution" is a swap artifact. On the
conflict regime the deployment ranking is CAD reranking, then the
instruction, then the probe-driven or grounded loop, then reasoning.
The loop's standing claim is unchanged: a mechanism result with a
distinct, collateral-free reach, not the best product.

## Program tally after WZ + WC
34 registered predictions: **26 confirmed**, 7 not (13th, 14th, 19th,
28th, 31st, 32nd, 34th), 1 intermediate (27th). ≈ $160 across 149 H100
jobs (WZ ≈ $10 + job-kikqy; WC $3.10).
