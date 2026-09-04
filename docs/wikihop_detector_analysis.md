# Blind failure detectors on the WD and WE frames — the detector cannot close the gap (descriptive, unregistered; 2026-09-04)

Question: how much of the oracle two-stage ceiling (+0.104 anonymized,
+0.029 real text; docs/wikihop_loop_descriptives.md §D) can a blind
detector recover? Script `scripts/wikihop_detector_analysis.py` →
`docs/wikihop_detector_analysis.json`. Rows: WD 207 (47 repairable from
WO, 100 unrepairable, 60 correct-majority), WE 219 (59 repairable from
WX, 100 unrepairable, 60 correct). Every feature is available before any
label: the unsteered gauge score on the baseline state, std
self-consistency (modal share, distinct answers), closed-book agreement
with the std answer, whether the std answer is a whole-word span of the
documents / a listed candidate, candidate-list size, and the loop's own
branch acceptance profile (how often the baseline's own branch fires
versus the best alternative). Logistic detectors are weighted to frame
stratum proportions and scored two honest ways: fit on the other frame
with the threshold also chosen there, or within-frame five-fold
out-of-fold. Frame net follows the two-stage rule (flagged rows run the
loop with the baseline removed from ties; unflagged rows keep the
baseline).

## Single features (weighted AUROC, failure vs correct)
| feature | WD | WE | note |
|---|---|---|---|
| native gauge on baseline state | 0.81 | 0.79 | the strongest single signal on both frames |
| candidate-list size | 0.75 | 0.66 | a frame artifact (harder WikiHop rows list more candidates) |
| closed-book agreement | 0.56 | 0.60 | correct rows agree with memory more often |
| std self-consistency | 0.53 | 0.53 | useless by pool construction: failures are 0/8, correct rows ≥ 5/8 |
| answer in documents / in candidates | 0.51–0.55 | 0.50–0.55 | nearly always true in both strata |
| loop acceptance margin (base branch − best other) | 0.62 | 0.58 | see below |

Acceptance profile by stratum (mean branch acceptance at 2×):
| | WD repairable / unrepairable / correct | WE repairable / unrepairable / correct |
|---|---|---|
| baseline's own branch fires | 0.78 / 0.90 / 0.97 | 0.71 / 0.84 / 0.97 |
| best alternative branch fires | **0.81** / 0.44 / 0.30 | **0.55** / 0.24 / 0.25 |

Correct rows resist the write (the natural-text analog of the sandbox's
stabilization finding): alternatives fire a quarter of the time while the
baseline's branch fires at 0.97. Repairable rows are the stratum where
an alternative fires as often as the baseline. Unrepairable rows look
like correct rows — a firm belief resists the write whether it is right
or wrong — so the acceptance profile marks repairability, not failure.

## Detectors driving the two-stage rule (answer variant; frame net = whole-frame accuracy change)
| frame | detector | flags repairable / unrepairable / correct | collateral | frame net [95% CI] |
|---|---|---|---|---|
| WD | gauge < 0 (the §D rule) | 57% / 69% / 22% | −0.100 | +0.023 [−0.017, +0.061] |
| WD | **baseline features, fit on WE, τ chosen on WE** | 23% / 12% / 0% | 0.000 | **+0.028 [+0.012, +0.049]** |
| WD | baseline features, within-frame 5-fold, τ = 0.5 | 79% / 84% / 37% | −0.133 | +0.038 [−0.005, +0.083] |
| WD | + loop acceptance, within-frame, best τ (optimistic) | 72% / 79% / 30% | −0.117 | +0.044 [+0.001, +0.086] |
| WD | oracle: flag repairable rows only | 100% / 0% / 0% | 0.000 | +0.051 [+0.037, +0.064] |
| WD | oracle: flag all failures | 100% / 100% / 0% | 0.000 | +0.104 |
| WE | gauge < 0 | 80% / 81% / 33% | −0.098 | −0.023 [−0.061, +0.011] |
| WE | baseline features, fit on WD, τ chosen on WD | 78% / 83% / 35% | −0.115 | −0.029 [−0.070, +0.006] |
| WE | + loop acceptance, within-frame, best τ (optimistic) | 34% / 24% / 2% | 0.000 | +0.011 [+0.004, +0.020] |
| WE | oracle: flag all failures | 100% / 100% / 0% | 0.000 | +0.029 |

Detector AUROCs are 0.80–0.86 on both frames, and that is the practical
limit with blind features. At that discrimination the operating point
that avoids collateral flags only a fifth of the failures, and the
operating point that catches most failures flags a third of the correct
rows, each of which costs the full collateral. The honest cross-frame
detector lifts the anonymized frame from +0.023 to +0.028; the
optimistic within-frame ceiling is +0.044 against an oracle of +0.104.
On real text nothing blind exceeds +0.011 against an oracle of +0.029.

## Reading
The detector is not where the blind loop is lost. Even a perfect
repairability flag is worth +0.051 on the anonymized frame and +0.023 on
real WikiHop, because the fixable rows are a quarter of the failures and
the failures are half the frame. A blind loop that helps at the frame
level, as it did on InAbHyD, needs a frame where most failures are
reading failures — the knowledge-conflict regime (item WK) — not a
better detector on this one. The acceptance-profile finding carries over
as a free signal there: on the sandbox-like regime, correct rows should
resist the write and conflict rows should accept the document's answer.
