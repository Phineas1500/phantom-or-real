# WikiHop loop — descriptive addendum after WD (post-hoc, unregistered; 2026-09-03)

Analyses of data already on disk, no new GPU jobs. Nothing here is a
verdict; every number is a hypothesis for a later registration. Scripts:
`scripts/wikihop_loop_descriptives.py` (→ `docs/wikihop_loop_descriptives.json`),
`scripts/wikihop_two_stage_selector.py` (→ `docs/wikihop_two_stage_selector.json`),
`scripts/wikihop_judge_relations.py` + `scripts/wikihop_judge_relation_base_rate.py`
(gpt-5.4-mini judge; → `docs/wikihop_wrong_selection_relations.json`,
`docs/wikihop_nongold_relation_base_rate.json`). Rows: WO (47 hint-repairable),
WD Y1+Y2 (100 unrepairable), WD C (60 correct-majority), plus WX (59 real
text) and the WA rider (60 anonymized) for the selector decomposition. All
frozen write at 2×, k=4 per branch, output-first selection.

## A. The abstention threshold does not matter; the tie does
| rule | yield (weighted) | 95% CI | repairable dP / abstained | unrepairable dP / abstained | collateral | 95% CI | abstained | frame net |
|---|---|---|---|---|---|---|---|---|
| always answer | +0.123 | [+0.075, +0.173] | +0.332 / 0% | +0.083 / 0% | −0.133 | [−0.217, −0.050] | 0% | +0.023 |
| unique top ≥ 0.25 | +0.039 | [+0.014, +0.070] | +0.128 / 66% | +0.022 / 40% | 0.000 | [0, 0] | 28% | +0.023 |
| unique top ≥ 0.5 (WD's rule) | +0.039 | [+0.015, +0.069] | +0.128 / 66% | +0.022 / 40% | 0.000 | [0, 0] | 28% | +0.023 |
| unique top ≥ 0.75 | +0.039 | [+0.015, +0.069] | +0.128 / 66% | +0.022 / 40% | 0.000 | [0, 0] | 28% | +0.023 |
| unique top = 1.0 | +0.037 | [+0.014, +0.068] | +0.128 / 68% | +0.020 / 41% | 0.000 | [0, 0] | 28% | +0.022 |
| margin ≥ 0.25 | +0.039 | [+0.015, +0.070] | +0.128 / 66% | +0.022 / 40% | 0.000 | [0, 0] | 28% | +0.023 |
| margin ≥ 0.5 | +0.036 | [+0.011, +0.066] | +0.106 / 70% | +0.022 / 43% | 0.000 | [0, 0] | 28% | +0.021 |
| margin = 1.0 | +0.030 | [+0.007, +0.061] | +0.085 / 77% | +0.020 / 47% | 0.000 | [0, 0] | 32% | +0.018 |

Whenever the top branch is unique it is accepted 3–4 times out of 4, so
every threshold from 0.25 to 1.0 gives the same answer set; margin rules
only cost yield. The abstention rule is not tunable: what it abstains on
is a tie at full acceptance, and that tie is the whole story below.

## B. Where the selector loses to the oracle
Per row: was the gold branch selected, tied at the top and lost on the
gauge tie-break, beaten by a wrong branch that fired more, or never fired?

| setting | rows | oracle | loop | gap | gold selected | tied, lost (share of gap) | beaten (share) | never fired |
|---|---|---|---|---|---|---|---|---|
| WX real text | 59 | 0.377 | 0.271 | 0.106 | 16 | 6 (96%) | 1 (4%) | 36 |
| WA rider anonymized | 60 | 0.700 | 0.350 | 0.350 | 21 | 19 (90%) | 5 (10%) | 15 |
| WO anonymized | 47 | 0.686 | 0.362 | 0.324 | 17 | 14 (92%) | 2 (8%) | 14 |
| WD unrepairable | 100 | 0.180 | 0.083 | 0.098 | 8 | 9 (92%) | 3 (10%) | 80 |

**90–96% of the oracle gap is a tie at full acceptance**, lost on the
gauge. A wrong branch out-firing gold is rare. On the unrepairable
stratum the write itself is the limit (gold never fires on 80 of 100
rows); on the repairable strata the write works and the tie is the limit.

**A lower dose does not break the tie.** Among tied pairs with a 1×
branch for both members: gold 1× acceptance vs the tied wrong branch's —
WX 0.15 vs 0.12 (83% equal), WA 0.43 vs 0.56 (gold higher 19%, lower 36%),
WO 0.40 vs 0.55 (gold higher 9%, lower 23%). The tied wrong branches are
ones the model is at least as eager to accept.

**The tied wrong branch is usually the model's own answer.** Rows where
gold tied and lost: the selected wrong candidate IS the baseline modal
answer in 5/12 (WX), 8/21 (WA), 12/15 (WO), 5/12 (WD); the baseline
answer is among the tied in 8/12, 18/21, 15/15, 10/12. Nudging the model
toward what it already believes trivially "confirms" it, so acceptance
carries no information for that branch. Removing the baseline answer
from a tie (with oracle knowledge that the row is a failure) lifts the
loop: WX 0.271 → 0.322 (+0.051 [0, +0.119]), WA 0.350 → 0.417 (+0.067
[−0.017, +0.167]), WO 0.362 → 0.574 (+0.213 [+0.085, +0.340]), WD
unrepairable 0.083 → 0.113 (+0.030 [0, +0.070]) — and, applied to correct
rows, removes the right answer: 0.850 → 0.717. It is not a blind rule.

## C. What the wrong picks are (gpt-5.4-mini judge, original strings)
| setting | n wrong picks | PARENT | CHILD | ALIAS | SIBLING | UNRELATED |
|---|---|---|---|---|---|---|
| WX real text | 43 | 19 (44%) | 2 (5%) | 0 | 10 (23%) | 12 (28%) |
| WA rider anonymized | 39 | 12 (31%) | 1 (3%) | 0 | 12 (31%) | 14 (36%) |
| WO anonymized | 29 | 12 (41%) | 2 (7%) | 0 | 7 (24%) | 8 (28%) |
| WD unrepairable | 92 | 45 (49%) | 6 (7%) | 5 (5%) | 23 (25%) | 13 (14%) |
| **all wrong picks** | 203 | **88 (43%)** | 11 (5%) | 5 (2%) | 52 (26%) | 47 (23%) |
| tied-at-top wrong picks | 48 | 18 (38%) | 3 (6%) | 0 | 14 (29%) | 13 (27%) |
| **base rate, 200 random non-gold candidates** (pool 4,827) | 200 | **18 (9%)** | 5 (2%) | 0 | 42 (21%) | 135 (68%) |

PARENT = a broader entity containing the gold (Ukraine for Kiev, Austria
for Vorarlberg, Loricariidae for Hypostominae, the United Kingdom for the
London Borough of Lambeth). Parents are 9% of candidates and 43% of wrong
picks — a five-fold enrichment; siblings are at their base rate;
unrelated candidates are depleted three-fold. The model does not accept
arbitrary nudges: it accepts nudges toward answers the documents also
support, and the containing entity is always supported. String cousins
(containment / token overlap) catch only 5–14% of wrong picks; the
cousin pattern is semantic. This is the sandbox's close-cousin scope in
natural-text form.

## D. A blind two-stage rule, and the ceiling it exposes
Stage 1: the unsteered gauge score on the baseline state (< 0 → failure).
Stage 2: on judged failures, run the loop with the baseline answer
removed from the tie; judged-correct rows keep the baseline.

| detector | variant | flags as failure: repairable / unrepairable / correct rows | repairable dP | unrepairable dP | yield weighted | collateral | frame net |
|---|---|---|---|---|---|---|---|
| anonymized-fit L48 | answer | 57% / 69% / 22% | +0.269 | +0.070 | +0.102 [+0.057, +0.151] | −0.100 [−0.183, −0.033] | +0.023 |
| anonymized-fit L48 | abstain | 57% / 69% / 22% | +0.269 | +0.050 | +0.085 [+0.048, +0.130] | −0.067 [−0.133, −0.017] | +0.025 |
| real-text L38 | answer | 77% / 71% / 35% | +0.439 | +0.070 | +0.129 [+0.085, +0.179] | −0.117 [−0.200, −0.033] | +0.033 |
| real-text L38 | abstain | 77% / 71% / 35% | +0.396 | +0.040 | +0.097 [+0.060, +0.139] | −0.083 [−0.167, −0.017] | +0.026 |
| **oracle detector** | answer | 100% / 100% / 0% | +0.545 | +0.110 | **+0.180 [+0.125, +0.240]** | **0.000** | **+0.104** |
| oracle detector | abstain | 100% / 100% / 0% | +0.460 | +0.070 | +0.133 [+0.086, +0.183] | 0.000 | +0.077 |

With perfect failure detection the blind loop would add about ten
points to the whole frame with zero collateral; the gauge as detector
flags 22–35% of correct rows as failures and gives most of it back
(frame net +0.02–0.03, the same as plain abstention). **The deployment
ceiling is now the failure detector, not the write and not the branch
selector.**

## Implied registrations (not launched)
1. A specificity tie-break: among tied accepted branches prefer the
   narrower entity when one contains another (the parent finding), on a
   fresh frame with the rule pinned before data.
2. A better failure detector than the single-layer gauge (e.g. the gauge
   plus baseline self-consistency, or a detector fit on anonymized
   frames), scored by the two-stage rule's frame net against the +0.104
   oracle ceiling, on a fresh frame.
Both are cheap ($3–5) and both use the existing job code.

## E. The specificity tie-break, estimated before registering it (2026-09-03, later)
Blind rule: among branches tied at the top acceptance rate, drop any
candidate the judge calls a PARENT of another tied candidate; pick the
unique survivor, else the gauge tie-break among survivors. 603 tied
pairs judged (gpt-5.4-mini; `docs/wikihop_tied_pair_relations.json`;
script `scripts/wikihop_specificity_tiebreak.py`).

| setting | rows | tied rows | rows changed | plain output-first | specificity tie-break | paired Δ [95% CI] |
|---|---|---|---|---|---|---|
| WX real (59) | 59 | 24 | 3 | 0.271 | 0.254 | −0.017 [−0.051, 0.000] |
| WA rider anonymized (60) | 60 | 44 | 13 | 0.350 | 0.300 | −0.050 [−0.133, +0.017] |
| WO anonymized (47) | 47 | 29 | 14 | 0.362 | 0.426 | +0.064 [−0.064, +0.191] |
| WD unrepairable (100) | 100 | 37 | 13 | 0.083 | 0.083 | 0.000 [−0.040, +0.040] |
| WE unrepairable, real (100) | 100 | 18 | 6 | 0.013 | 0.022 | +0.010 [0.000, +0.030] |
| WD correct rows (60) | 60 | 16 | 2 | 0.850 | 0.850 | 0.000 [−0.050, +0.050] |
| WE correct rows (60) | 60 | 11 | 3 | 0.900 | 0.883 | −0.017 [−0.067, +0.033] |

No consistent gain: the rule helps on one setting, hurts on two, and
is null on the rest. The parent enrichment among wrong picks (section
C) is real, but preferring the narrower tied candidate also drops gold
when gold is the broader one. **Not registered.** The implied
registration list is now down to the failure detector, whose ceiling on
real text is +0.029 (WE); nothing on the deployment side is worth a job.
