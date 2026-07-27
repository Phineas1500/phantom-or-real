# Phantom or Real? → Gauge, Not Lever

Mechanistic-interpretability study of whether a language model's internal
"I am about to get this right/wrong" signal is causally load-bearing — or
just a gauge on the dashboard. Models: **Gemma 3 27B-IT** (primary) and
**Qwen3.5-27B** (cross-model port), on depth-controlled ontology-reasoning
tasks (InAbHyD). The project began as a stage-1 behavioral/probing pipeline
(see `BEHAVIORAL_DATA_PLAN.md`) and grew into a pre-registered causal
program: **19 registered verdicts, 6 confirmed registered predictions,
two external adversarial review rounds survived**.

**Status (2026-07-07): the experimental program is complete. The current
milestone is drafting the paper** from `docs/paper_outline_final.md` and
`docs/contribution_ledger_2026-07-05.md`.

## The thesis in one paragraph

In a model that visibly predicts its own failures, the readable
correctness signal and the causal repair channel are **different
objects**: the entire readable subspace can be deleted without behavioral
cost, while repair flows exclusively through a low-rank channel that
carries the label-specific content of natural success yet is itself no
more decodable than chance — gauge and lever, anatomically separated.

## Headline findings (all pre-registered, row-cluster bootstrap CIs)

**Gemma 3 27B (the main battery):**

- **The gauge**: correctness is linearly readable pre-generation
  (AUC 0.90; Qwen 0.94), robust to name-scrambling, beats metadata.
- **The gauge is not necessary**: erasing everything linearly readable —
  the full rank-9 INLP stack, five layers, every token — costs nothing
  (+0.047 [−0.070, +0.203]), while matched-rank random erasures are
  catastrophic (−0.38). (Item D)
- **The lever is elsewhere**: a rank-8 channel at concept-mention tokens
  (L30) repairs held-out failures (+0.245) and passes the full
  specificity ladder — random subspaces null, matched noise destructive.
  (Item C)
- **The necessity dissociation**: a natural correct-minus-incorrect
  class-mean is null at full dimension (+0.043) and potent only when
  concentrated into the channel (+0.341) — and only with real labels
  (shuffled ≈ 0; sign-flipped actively harmful). (F(ii), F(ii)-b)
- **The inversion**: the channel that causally controls success is not
  itself decodably privileged (probe inside the random-subspace null) —
  the mirror image of the field's decodable-but-not-causal refrain. (F(i))
- **Answer-free repair**: a fully donor-free variant (direction from
  natural outcomes, amplitude from pooled scale) reaches **+0.399** on a
  0.12 baseline where majority-vote self-consistency scores 0.000;
  firing it on already-correct rows *helps* (+0.266). Position
  addressing remains the one answer-adjacent ingredient (item H: the
  vector is a positional commitment command). (F(ii)-b/c, H)

**Qwen3.5 27B (the cross-model port, G/J series):**

- The **motif transfers, the coordinates don't**: same recognition-gap
  failure mode, same concept-position carrier — but at rel depth 0.67
  (vs 0.48), channel rank ~16 (vs 8; 95% recovery needs rank 64), zero
  mean share (vs 35%).
- **The dissociation inverts**: in Qwen the *raw* class-mean repairs
  (+0.120 [+0.016, +0.224]) and projection into the compact channel
  *dilutes* it — Gemma concentrates, Qwen distributes. Label battery
  passes (real-vs-shuffled +0.182 [+0.070, +0.299]; sign-flip harmful,
  as registered-predicted in every design). **The recipe is
  cross-model; the compression is not.** (G6/G6′)
- **Gauge/lever separation by position**: Qwen's outcome information is
  weakly-to-moderately decodable at the final prompt token (0.66–0.81
  across two 96-row draws) and absent at the concept positions where
  the repair channel operates; writing the class-mean at the readable
  site does not repair (shuffled matches real). Gemma separates gauge
  from lever by subspace at one position; Qwen by position at one
  layer. (J1/J2)

## Where to look

| what | where |
|---|---|
| Paper outline + assembly order | `docs/paper_outline_final.md` |
| What we claim / cite / share | `docs/contribution_ledger_2026-07-05.md` |
| Claims × evidence × caveats | `docs/next_paper_claims_table.md` |
| Us vs the 10 closest papers | `docs/comparison_matrix_2026-07-06.md` |
| All pre-registrations (decision rules before unblinding) | `docs/causal_handle_directions.md` |
| Per-item verdicts | `docs/*_summary.md` (C, D, E, F-series, H, G-series, J-series) |
| Statistical hardening (BCa/LOO/MDE) | `docs/stat_hardening_2026-07.md` |
| Running lab notebook | `docs/REPORT_NOTES.md` |
| Visual status overview | `docs/causal_status_overview.html` |
| Bibliography (verification-graded) | `docs/bibliography.md` |
| Artifact page sources | `docs/artifacts/` |

## Methodology notes

Every causal item was **pre-registered with branch-complete decision
rules before unblinding**; six registered predictions confirmed. All
effect sizes are pooled P(strong-correct) deltas vs in-job baselines
with row-cluster bootstrap CIs (10k, fixed seed); claim-bearing
contrasts near zero additionally get BCa + leave-one-row-out. Job
determinism is seed-exact: replication gate arms reproduce prior jobs
token-for-token (verified 576/576 generations across three jobs).
Controls used throughout: matched-norm noise, random subspaces/stacks,
shuffled-label class-means, sign-flips, dose ladders, sampling
baselines (self-consistency, best-of-N), collateral slices.

## Stage 1 — behavioral/probing pipeline (the dataset everything runs on)

Generates InAbHyD reasoning examples, runs Gemma 3 4B/27B (+ Qwen)
inference, scores correctness, annotates structure, ships labeled JSONL
(44k rows/model). Full spec: `BEHAVIORAL_DATA_PLAN.md`; operating
layout, credentials, and common commands: `CLAUDE.md`.

```bash
# One-time setup
conda env create -f environment.yml            # or environment.lock.yml for exact pins
conda activate phantom

# You need a clone of beyond-deduction somewhere — options:
#   - set BD_PATH=/path/to/beyond-deduction
#   - or ln -s /path/to/beyond-deduction third_party_beyond_deduction
#   - or clone it to ~/beyond-deduction

# Generate the full dataset (~44k examples)
python -m src.generate_examples --counts full

# Stage-2 causal harnesses live in scripts/stage2_*.py; each registered
# item has a matching sbatch file and a verdict doc under docs/.
```

## What's next

1. **The paper** (in progress): headline sections §5.2/§5.5 first —
   every number frozen in pooled verdict docs.
2. Next-paper queue (registered or specified): powered final-token
   probe + attention-routing test in Qwen (J-series follow-up), Q1
   layer×position information map, projection-headroom transplant
   (now regime-conditional), position-policy v2, Qwen3.6 bridge.
