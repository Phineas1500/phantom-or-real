# Deployment Riders (item F(ii)-c) — Verdict, Job 458431

Single job (26 failing guard rows + 16 fresh naturally-correct rows, 42
prepared of 48; k=8; row-cluster bootstrap 10k). Rules pre-registered in
`causal_handle_directions.md` F(ii)-c before any data. Exploratory: no
current-paper claim moves; this completes the deployment profile of the
donor-free repair (ledger §1.5).

## Arms

| arm | dP (CI95) |
| --- | ---: |
| fixednorm_proj (gold positions, failing rows) | **+0.447 [+0.293,+0.611]** |
| fixednorm_allpos (ALL concept positions, failing rows) | +0.029 [−0.053,+0.120] |
| correct_fixednorm (gold positions, naturally-correct rows) | **+0.266 [+0.086,+0.469]** vs correct baseline 0.727 |

## Rules → outcomes

- **POSITION-FREE: FAILS.** The all-positions arm is null. Writing the
  same vector at every taxonomy-concept mention — gold included —
  produces nothing; interference from wrong-concept positions cancels the
  effect rather than merely diluting it (consistent with the class vector
  acting as a positional selection signal: mark every candidate and
  nothing is selected). Deployment consequence: gold-concept POSITION
  knowledge is load-bearing. The honest caveat stands in the ledger: the
  intervention is answer-free in CONTENT (direction + amplitude) but not
  yet in ADDRESSING; a position-selection policy is the remaining gap.
- **COLLATERAL-SAFE: PASSES — and the result is beneficial, not merely
  safe.** On fresh naturally-correct rows the intervention IMPROVES
  P(strong) from 0.727 to 0.992 (+0.266, CI floor +0.086). Correct-row
  selection was by a single stage-1 sample; under k=8 resampling those
  rows are only ~73% stable, and the intervention stabilizes them to
  near-ceiling. Deployment consequence: gauge-gating is nice-to-have,
  not load-bearing — false-positive firings (with correct addressing)
  help rather than harm. Misfires with WRONG addressing are bounded by
  the allpos result (~null) rather than harmful, though that composite
  case was not directly tested.
- **Bonus replication**: fixednorm_proj at +0.447 is an independent
  same-row/new-seed resample of F(ii)-b's +0.399 (new arm index → new
  sampling seeds) — labeled as internal consistency, not replication.

## Ledger §1.5 deployment profile (final wording)

Donor-free in direction and amplitude; +0.40–0.45 on hardest failures;
beneficial (+0.27) when fired on already-correct rows at correct
addresses; null when addressing is unknown (all-positions). The remaining
research gap for a deployable method is position selection, not content.
