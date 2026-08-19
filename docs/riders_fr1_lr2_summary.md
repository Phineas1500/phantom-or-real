# Riders F-R1 + L-R2 — both land: the channel is NOT the outlier dims (rank-6 remainder repairs at 109% of the anchor), and frozen-beats-fresh resolves to donor-count (ladder saturates at ≥20 donors at the frozen level)

Jobs 462362 (F-R1, 1h34) + 462363 (L-R2, 1h14), Aug 18–19.
Registered 2026-08-18 (review round 2); L-R2 pre-data amendment
(12 test rows for the 3h wall).

## F-R1 — REPAIR-SURVIVES (the outlier concern is closed)
Guard frame, n=26, baseline 0.120:
- rank-8 anchor (re-run): **+0.447 [+0.293, +0.606]** — reproducing
  the registered +0.447 anchor exactly.
- rank-6 remainder (outlier-aligned pair removed, matched norm):
  **+0.486 [+0.322, +0.649]** — 109% of the anchor; paired
  remainder−anchor +0.038 [−0.048, +0.135].
The two components that are near-axis-aligned with the model's
outlier dimensions contribute nothing detectable to the repair. The
channel is not the high-bandwidth broadcast dims; combined with the
shuffled-label control, the outlier account is fully closed.

## L-R2 — the ladder resolves frozen-beats-fresh as donor-count
n=12 test rows, baseline 0.177 (within-ladder comparisons row-matched
across arms):
| donors | dP |
|---|---|
| 5 | +0.302 [+0.094, +0.531] |
| 10 | +0.333 [+0.135, +0.552] |
| 20 | **+0.479 [+0.260, +0.708]** |
| 40 | +0.479 [+0.240, +0.719] |
Monotone rise, saturation at ≥20 donors at a level equal to the
frozen construction's point value (+0.498/+0.513 on its frames).
Reading: **the frozen-vs-fresh gap was donor-count** — the in-job
LOO refits sat low on this curve; no strong stationarity claim is
needed (the month-stability observation stands as descriptive).

**Disclosed deviation (caught at verdict):** the ladder's selection
ran the default exclusion stack, not the L-series stack, so its 12
rows are NOT the registered L′ rows — the row-paired frozen contrast
is empty and the frozen comparison above is point-value descriptive.
Provenance checked: zero overlap with the 96-row class-mean capture
or FIIC donor rows (no leakage). Within-ladder monotonicity, the
claim-bearing content, is unaffected (same rows across arms).

With these, every registered item and rider on the program's board is
closed: 34 registered verdicts, 9 confirmed directional predictions.
