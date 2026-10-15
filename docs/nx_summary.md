# Item NX (+ rider NX-h) — is the WikiHop lever a few legible neurons? (2026-10-15)

Prompted by Cammarata, "Neural Networks Are Already Clean" (Goodfire,
2026-10-15): most MLP neurons have one job, so a mechanism should be
statable in a few legible neurons. Our lever is a residual direction:
the frozen hint-delta u at L30 × 2× repairs WikiHop failures at the gold
candidate's mentions (WX +0.360; WV: flipped/random vectors inert; WT:
it routes L31–33 attention onto the span). Registered:
docs/causal_handle_directions.md item NX (predictions 43–45) and rider
NX-h (46th, registered after the stage-1 data that motivated it).
Gemma-3-27B; the 59 WX rows, cross-fit; 8 H100 jobs, ≈ $3.2.

## Verdict
| | prediction | result |
|---|---|---|
| 43rd | the top 64 MLP neurons' write repairs ≥ +0.18 and beats 64 random | **NOT** — +0.013 [0.000, +0.038] (DISTRIBUTED-SOURCE) |
| 44th | clamping the top 64 downstream neurons removes the repair | not read — gate **QK-DIRECT**: freezing every span MLP in L31–36 keeps 104% |
| 45th | the source neurons are legible (win ≥ 0.75, > random) | **NOT, reversed** — 3/32 vs random 20/32 |
| 46th (rider) | the top 8 heads' write repairs ≥ +0.18 and beats 8 random heads | **CONFIRMED** — +0.479 [+0.360, +0.597]; random +0.03 |

## Where u comes from (stage 1, exact decomposition; reconstruction 0.985 / 1.010)
The hinted run's change at the gold mentions, projected on u, is written
by attention heads (+1.6 × the net) against the MLPs (−0.6 ×). Eight
heads in L23–L29 carry 100% of the net, and the SAME eight in both
independent donor shards (L27 h3 ≈ 0.27, L25 h22 0.19, L29 h18 0.14,
L23 h4, L27 h24, L29 h0, L27 h29, L29 h30). MLP mass is broad: the top
64 positive neurons are 5% of the positive MLP mass, the top 16,384 are
67%. u puts 41–48% of its norm² in Gemma's massive dims 104 / 2733;
zeroing them costs −0.055 [−0.114, −0.008] of the repair (+0.301 left).

## Sufficiency (59 rows, baseline 0.017, k = 4, gold address)
| write | delivered at | cos with u | gold rate | dP [95% CI] |
|---|---|---|---|---|
| u (WX, in-job) | L30 | 1 | 0.373 | +0.356 [+0.237, +0.475] |
| **top 8 heads** | L23–L29 | 0.77 | **0.496** | **+0.479 [+0.360, +0.597]** |
| 8 random heads ×2 | their layers | −0.4–0.5 | 0.04 | +0.03 / +0.02 |
| top 16 / 64 / 256 / 1024 neurons | their layers | 0.65–0.71 | 0.02–0.05 | +0.000 / +0.013 / +0.008 / +0.034 |
| top 64 neurons, summed | L30 | 0.66–0.71 | 0.021 | +0.004 [0.000, +0.013] |
| 64 random neurons ×2 | their layers | ≈ 0 | 0.000 | −0.017 |

HEAD_8 − U +0.123 [+0.021, +0.233] (descriptive). A neuron vector with
cosine 0.7 and the same projection on u is inert; the heads' vector at
cosine 0.77 repairs more than u. What repairs is the heads' write
itself, not "something near u".

## What reads it (stage 3)
With the U write on, clamping each MLP's output at the written positions
in L31–L36 to its no-write value leaves the repair intact (+0.369 vs
+0.356) — the written residual is read by later attention directly (as
WT's routing showed), with no span-MLP step. Clamping the top / random
64 span neurons: +0.373 / +0.360.

## Legibility (stage 4, the essay's test)
Top-20 contexts over 2,800 texts (WF documents + 2,000 fineweb docs);
gpt-5.4-mini explains, a separate call writes 10 positives / 10 near
misses / 10 distant texts from the explanation alone; win = positives
fire more often than near misses (fire = 20% of the neuron's mean top-20).
| neurons | win / tie / lose | win rate | fire rate pos / near / distant | fire scale (median) |
|---|---|---|---|---|
| lever source, top 32 | 3 / 23 / 6 | **0.09** | 0.21 / 0.21 / 0.18 | 1.4 |
| downstream, top 32 | 8 / 22 / 2 | 0.25 | 0.19 / 0.15 / 0.09 | 5.0 |
| random 32 | 20 / 9 / 3 | **0.63** | 0.49 / 0.27 / 0.00 | 6.2 |

18 of the source neurons' 23 ties are 0 / 0: the explanation's positives
never fire them. Random neurons behave as the essay says (specific,
silent on distant text); the neurons aligned with our lever are weak and
diffuse, and their explanations (mostly "proper names") do not predict
their firing.

## Reading
On this mechanism the clean unit is the attention head, not the neuron:
eight heads write the direction, attention reads it, and the MLPs push
against it on net. The essay's legibility claim reproduces on random
neurons (0.63 here vs its ≈ 0.90, with a stricter protocol: separate
explainer and tester calls), but the neurons that align with the lever
are the illegible ones — a selection effect worth stating against
neuron-first auditing of a known mechanism. Caveats: one model, one
task, 59 rows; the heads were selected on donors and tested on held-out
rows (cross-fit), but the rider was registered after stage-1 data showed
the concentration; stage 4 uses an LLM judge with 10 texts per kind.
