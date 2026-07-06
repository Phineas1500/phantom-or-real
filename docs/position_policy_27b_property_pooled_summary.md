# Position-Selection Policy (item H) — Pooled Verdict, Jobs 458435 + 458440

26 rows stitched (main + 8-row remainder after a 4h TIMEOUT; per-candidate
seeds layout-independent). Rules pre-registered in
`causal_handle_directions.md` item H. Row-cluster bootstrap, 10k.

| readout | result |
| --- | ---: |
| baseline / gold-position k=8 | 0.120 / 0.510 |
| GATE: gold-candidate fires (k=4) | **+0.380 [+0.226,+0.543] — PASSES** |
| wrong-concept fires (pooled) | −0.036 [−0.079,−0.004] |
| targets_fired rate, gold vs wrong fires | 0.519 vs **0.472** |
| P1 self-ratification | +0.053 [−0.005,+0.144]; picked gold 12% — **FAILS** |
| P2 global majority | +0.072 [−0.005,+0.173] — fails |

## Outcome: POLICY-FAILS (pre-registered branch)

Both pre-named answer-free selection policies fail. The addressing caveat
("answer-free in content, not in addressing") is FINAL for this paper.

## The mechanism readout (the important part)

Firing at ANY concept pulls hypotheses toward the fired concept about half
the time — wrong concepts are ratified nearly as often as gold (0.472 vs
0.519). The vector is therefore a positional COMMITMENT COMMAND ("select
whatever is marked"), not a gold-detector: content supplies the verb,
address supplies the object. This explains, in one stroke: the
all-positions cancellation (mark everything → conflicting commands), the
wrong-fire harm (−0.036: committing to wrong concepts yields wrong
answers), the sign-flip harm, and why self-ratification cannot work (the
command self-ratifies everywhere). Read jointly with F(ii)-b (shuffled-
label vectors do nothing even at gold positions): the label-specific
class-mean content IS the commit verb; position selection is a genuinely
separate problem requiring either a trained policy or gauge-scored reruns
(future work).

## Slots filled

- Outline §5.5 / contribution (vi): deployment profile final — collateral
  beneficial, addressing load-bearing, policies P1/P2 fail, verb/object
  mechanism stated.
- Ledger §1.5: append the verb/object sentence; §4 unchanged.
