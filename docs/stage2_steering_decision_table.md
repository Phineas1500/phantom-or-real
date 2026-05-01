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

## Active Gates

| Gate | Status | Stop/continue rule |
|---|---|---|
| 27B raw answer-property smoke | Completed as Slurm job `452301`; null for controlled `toward_gold` steering | Stop free-form answer-property steering. |
| Margin + forced-choice smoke | Completed as Slurm job `452338` | Do not scale the easy opposite-polarity foil. If continuing, use gold vs model-emitted wrong hypothesis or pivot to patching/reconstruction-error steering. |
| Hard-foil forced-choice refinement | Completed as Slurm job `452362` | Close probe-direction steering and pivot to activation patching/interchange. |

## Recommended Next Branches

1. Treat the free-form and forced-choice answer-property branches as closed for
   this probe-derived direction.
2. Pivot to activation patching/interchange to test whether any compact hidden
   state can causally repair hard rows.
3. If patching finds a site, try CAA-style content vectors there rather than
   correctness or polarity directions.
4. If patching is null, treat steering as a
   negative/inconclusive causal check and shift effort to report assembly plus
   feature falsification.

## Current Interpretation

The project should not claim a successful causal steering mechanism yet. The
consistent result is that raw and learned-dictionary probes can read predictive
information from pre-generation activations, but additive decode-time steering
has not reliably converted that information into correct emitted ontology
answers. The forced-choice branch tests whether that null is partly due to the
free-form output channel.
