# Items WK′ + WS — the grounded two-stage rule, registered and replicated: a fresh NQ-Swap draw (29th prediction) and counterfactual SQuAD (30th) (2026-09-04)

Registered: docs/causal_handle_directions.md items WK′ and WS (before
any data). Both pin the same deployment recipe: **one frozen direction
(the WikiHop WX hint-delta, fit once on 59 WikiHop rows), written at L30
× 2× at every candidate's mentions; a label-free groundedness check
(is the model's answer a whole-word span of the passage?); on flagged
rows, the output-first loop with the baseline removed from ties; run
blind over a uniform draw.** WK′: XA′ **job-rtda7** / XB′ **job-ejkyx**
(registered arm) + A′ **job-z3eqz** / B′ **job-d5jgz** (own-donor
rider); WS: stage 1 **job-sgtjp**, stage 2 XA **job-xehsu** / XB
**job-rf6wz**. Reader `scripts/wikihop_wk_gates.py --prediction
grounded-wikihop`.

## WK′ — fresh NQ-Swap draw (120 rows disjoint from WK's; baseline 0.617)
| blind rule | WikiHop vector (registered) | own-frame donors (rider) |
|---|---|---|
| **grounded two-stage** | **+0.050 [+0.017, +0.092]** · 6 up / 0 down · acts on 9% | +0.008 [0.000, +0.025] · 1 / 0 |
| abstention (WK's 28th) | 0.000 · 0 / 0 | +0.004 [−0.021, +0.029] · 2 / 1 |
| always answer | −0.100 [−0.183, −0.017] · 8 / 20 | −0.029 [−0.092, +0.033] |
| oracle two-stage | +0.083 [+0.033, +0.142] · 11 / 1 | +0.025 [−0.008, +0.067] |

**29th prediction CONFIRMED — GROUNDED-TWO-STAGE-HELPS-BLIND.** Frame
accuracy 0.617 → 0.667; none of the 73 correct rows is touched. Strata
(WikiHop vector): repairable conflict failures +0.250, unrepairable
+0.150, other failures and correct rows 0. The write on the 12
repairable rows: WikiHop vector 2× **+0.750 [+0.500, +1.000]**,
specificity +0.697; the 25-donor own-frame direction +0.229 — the
cross-task vector is not just better, it is the only one that carries
the rider on this draw.

### Pooled WK + WK′ (240 blind rows, descriptive)
| rule | WikiHop vector | own donors |
|---|---|---|
| grounded two-stage | **+0.056 [+0.029, +0.087]** · 14 up / 1 down | +0.023 [+0.006, +0.042] · 6 / 0 |
| abstention | −0.002 | −0.003 |
| always answer | −0.099 [−0.156, −0.043] | −0.054 [−0.095, −0.014] |
| oracle two-stage | +0.080 [+0.043, +0.121] | +0.022 |

The grounded rule with the frozen WikiHop vector recovers 70% of the
oracle-detector ceiling on the pooled draw, with one row harmed in 240.
