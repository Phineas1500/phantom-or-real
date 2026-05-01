# Stage 2 Steering Decision Table

Date: 2026-04-30

Purpose: keep the causal steering experiments organized by target type, output
format, and decision gate. The key distinction is whether a probe reads generic
success/failure, free-form answer content, or a Cox-style forced-choice answer.

## Current Results

| Experiment | Model/site | Target | Output format | Probe result | Steering result | Interpretation |
|---|---|---|---|---|---|---|
| Raw correctness direction | 27B L45 residual | `is_correct_strong` | Free-form hypothesis | Test AUC `0.8965` | 0 false-to-true flips on 8 balanced rows | Predictive correctness direction is not a clean repair knob. |
| Single big-L0 features | 27B L45 262K big-affine transcoder | Shortlisted sparse features | Free-form hypothesis | Feature candidates from sparse probe/dashboard | 0 false-to-true flips; 2 true-to-false changes | Individual features not established as causal repair handles. |
| Sparse-probe bundle | 27B L45 262K big-affine transcoder | Sparse correctness probe coefficients | Free-form hypothesis | Test AUC `0.853` | 0 false-to-true flips on 8 balanced rows | Distributed sparse direction still did not repair answers. |
| Raw correctness / error / sparse bundle | 4B L22 | `is_correct_strong` | Free-form hypothesis | Raw/error/sparse probes all predictive | 0 strong flips in local 4B sweeps | Same predictive-versus-causal gap appears at 4B scale. |
| Raw answer-property direction | 4B L22 residual | Gold answer polarity | Free-form hypothesis | `val_auc=test_auc=1.000` | 0 polarity flips, 0 predicate flips toward gold, 0 strong repairs | Even concrete free-form answer content did not steer. |
| Raw answer-property direction | 27B L45 residual | Gold answer polarity | Free-form hypothesis | `val_auc=test_auc=1.000` | 0 useful `toward_gold` flips; one wrong-direction repair under `away_gold` | Free-form answer-property steering also fails at 27B scale. |
| Raw answer-property margin | 27B L45 residual | Gold answer polarity | Original prompt + MCQ margins | `val_auc=test_auc=1.000` | Small MCQ margin shift; 0 MCQ choice flips; original margins noisy | Weak constrained-margin sensitivity, not answer repair. |
| Raw answer-property hard-foil margin | 27B L45 residual | Gold answer polarity | Original prompt + MCQ vs model-emitted wrong foil | `val_auc=test_auc=1.000` | 0 choice flips, 0 false-to-true MCQ flips through 2 SD; tiny MCQ deltas | Cox-style forced-choice branch is also negative for this probe direction. |
| Clean-to-corrupt full-state patching | 27B L30/35/40/45/50 residual landmarks | h1-correct state transplanted into h4-incorrect prompt | Gold-vs-model-foil logprob margin | N/A | Late `last_prompt` clean patches weakly improve margins, but matched noise is comparable or stronger | Not a missing-state failure; looks like late wrong-answer commitment that generic perturbation can slightly loosen. |
| Corrupt-to-clean full-state patching | 27B L35/40/45/50 `last_prompt` | h4-incorrect state transplanted into h1-correct prompt | Gold-vs-model-foil logprob margin | N/A | Corrupt patches reduce h1 margins more than noise overall, strongest at L50; high-headroom pairs are mixed even in absolute deltas | Asymmetric patching result: h4->h1 disrupts more than noise, while h1->h4 does not repair above noise. |

## Active Gates

| Gate | Status | Stop/continue rule |
|---|---|---|
| 27B raw answer-property smoke | Completed as Slurm job `452301`; null for controlled `toward_gold` steering | Stop free-form answer-property steering. |
| Margin + forced-choice smoke | Completed as Slurm job `452338` | Do not scale the easy opposite-polarity foil. If continuing, use gold vs model-emitted wrong hypothesis or pivot to patching/reconstruction-error steering. |
| Hard-foil forced-choice refinement | Completed as Slurm job `452362` | Close probe-direction steering and pivot to activation patching/interchange. |
| Clean-to-corrupt patching pilot | Completed as Slurm job `452478` | Do not claim a repair/localization result. Optional final check is reverse h4-to-h1 patching at late `last_prompt` sites; otherwise move to report assembly. |
| Corrupt-to-clean patching pilot | Completed as Slurm job `452492` | Use as a calibrated asymmetry result; no broader patch grid unless the report specifically needs stronger localization. |

## Recommended Next Branches

1. Treat the free-form and forced-choice answer-property branches as closed for
   this probe-derived direction.
2. Treat the clean-to-corrupt patching pilot as evidence against a simple
   missing-state repair story at the tested residual sites.
3. Treat reverse patching as the final asymmetry check: it shows aggregate
   corrupt-state disruption above noise, driven by lower/mid-headroom pairs.
4. Shift effort to report assembly plus feature falsification.

## Current Interpretation

The project should not claim a successful causal steering mechanism yet. The
consistent result is that raw and learned-dictionary probes can read predictive
information from pre-generation activations, but additive decode-time steering
has not reliably converted that information into correct emitted ontology
answers. The forced-choice and patching branches sharpen that conclusion:
recognition can be intact under constrained MCQ formatting, but free-form
generation appears to enter a committed wrong-answer state that is not repaired
by either probe-direction steering or clean h1 residual transplantation.
Reverse patching adds an asymmetry: h4 incorrect states can partially disrupt
h1 correct margins more than noise, but this does not hold robustly for
high-headroom examples and should be presented as an asymmetry result before a
mechanistic commitment claim.
