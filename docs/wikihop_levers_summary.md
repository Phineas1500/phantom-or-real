# Items WI, WJ, WU, WI′ (+ the detector addendum) — can the blind gain be made bigger? (2026-09-04)

Four levers on the whole-frame number, each registered before data
(docs/causal_handle_directions.md). Cost $11.25 over 12 jobs; all
readers in `scripts/` (`wikihop_wk_detector_sweep.py`,
`wikihop_wi_reader.py`, `wikihop_wj_gates.py`, `wikihop_wk_gates.py`).

## 1. A wider detector — closed (free, 360 blind rows already run)
| rule (pooled 360 rows) | flags correct rows | frame net [CI] | up / down |
|---|---|---|---|
| groundedness check (as deployed) | 1% | +0.049 [+0.026, +0.072] | 18 / 1 |
| ∨ branch acceptance ≥ τ (any τ) | 30–34% | −0.093 [−0.146, −0.040] | 33 / 68 |
| ∨ alternative out-fires the baseline | 1% | +0.053 [+0.031, +0.078] | 20 / 1 |
| oracle (all failures) | 0% | +0.080 [+0.051, +0.110] | 32 / 4 |
The loop's own acceptance signal flags a third of correct rows; the
remaining gap to the oracle is grounded wrong answers, which no
label-free signal separates from correct ones.

## 2. The prompting baseline (WI, jobs job-g3kw5 / job-cehc7; descriptive)
"Answer using only the documents below. If the documents disagree with
what you remember, follow the documents."
| | NQ-Swap (800) | counterfactual SQuAD (476) |
|---|---|---|
| accuracy vs document: std → instruction | 0.578 → **0.655** (+7.7) | 0.674 → **0.725** (+5.1) |
| collateral (correct-majority rows broken) | 10/462 (2.2%) | 3/321 (0.9%) |
| conflict failures repaired: instruction / hint / both | **0.210** / 0.323 / 0.459 | **0.202** / 0.365 / 0.452 |
| overlap: both / hint-only / instruction-only / neither | 34 / **40** / 14 / 141 | 17 / **21** / 4 / 62 |
The instruction is the stronger frame-level mitigation and costs
nothing; attention reaches a fifth of conflict failures the instruction
does not. Both responses are bimodal.

## 3. Depth × dose on conflict rows (WJ, jobs job-b55u4 / job-s9fu3 / job-gjr8y; descriptive)
40 repairable + 40 unrepairable conflict rows, WikiHop XA donors.
| layer × dose | repairable gold rate (dP) | reach on unrepairable | paired vs L30 × 2× |
|---|---|---|---|
| L20 × 2× | 0.412 (+0.325) | 0.069 | −0.225 [−0.400, −0.050] |
| L25 × 2× | 0.544 (+0.456) | 0.081 | −0.094 [−0.250, +0.044] |
| **L30 × 1×** | **0.675 (+0.588)** | 0.138 | +0.037 [−0.056, +0.131] |
| **L30 × 2× (the recipe)** | 0.637 (+0.550) | **0.256 [0.138, 0.388]** | reference |
| L30 × 3× | 0.487 (+0.400) | 0.163 | −0.150 [−0.256, −0.056] |
L30 wins at every dose on the conflict regime (WikiHop peaked at L25);
2× has the highest reach into hint-unrepairable rows. Recipe confirmed.

## 4. The grounded rule on Qwen3.5-27B (WU, jobs job-qu4h8 / job-teem6 / job-6mzf7; 31st prediction)
Stage 1: std 0.604, memory rate 0.756, 136 conflict failures,
hint-repairable 0.272 [0.199, 0.353]. Blind draw of 120: 7 repairable.
| rule, Qwen, WY WikiHop donors at L31, probe L48 | frame net [CI] | up / down |
|---|---|---|
| **grounded + probe select (REGISTERED, 31st)** | **+0.015 [−0.004, +0.036]** | 5 / 4 |
| grounded + output vote (rider) | **+0.045 [+0.013, +0.083]** | 7 / 2 |
| always answer | −0.072 [−0.131, −0.017] | 12 / 26 |
| oracle two-stage | +0.026 [+0.003, +0.057] | 4 / 2 |
**31st NOT CONFIRMED.** The effect is present on Qwen with the output
vote (the registration bet on the probe, following WY, and lost); the
write on the 7 repairable rows +0.679 [+0.482, +0.839].

## 5. Stacking on the instruction (WI′, jobs job-eb56t / job-sbr5w; 32nd prediction)
Third fresh draw of 120 NQ-Swap rows, every test prompt under the
instruction, the frozen WikiHop vector unchanged.
| on the same 120 rows | net [CI] | up / down |
|---|---|---|
| instruction alone (WI grades) | +0.125 [+0.074, +0.182] | |
| **grounded loop over the instructed baseline (REGISTERED, 32nd)** | **+0.025 [0.000, +0.058]** | 3 / 0 |
| instruction + loop vs plain std | +0.146 [+0.085, +0.213] | |
| always-answer under the instruction | −0.159 [−0.247, −0.074] | 7 / 27 |
**32nd NOT CONFIRMED** (the lower bound sits on zero). The loop adds a
little on top of the instruction, not measurably at this size.

## Verdict
The whole-frame number is set by the regime, not by the tooling: no
detector, layer, dose, or stacking raises it, and it replicates on
Qwen only with the output vote. For the paper: the instruction is the
deployment tool; the frozen write is the mechanism, with a distinct
reach the instruction lacks (a fifth of conflict failures; a quarter
of hint-unrepairable rows), and as a blind tool it is worth 3–6 points
where the model ignores the passage.

## Program tally
32 registered predictions: **25 confirmed**, 6 not (13th, 14th, 19th,
28th, 31st, 32nd), 1 intermediate (27th). ≈ $145 across 139 H100 jobs.
