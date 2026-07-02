# Erasure Control Matching — Verdict (Jobs 457207-457210)

Reviewer-insurance follow-up to the multi-layer correctness-direction erasure
(`docs/causal_handle_directions.md`, "Erasure control matching"). Same erasure
family and layers (`L15/L30/L40/L45/L53`), same 16 balanced S1 h3/h4 property
rows, k=4 per condition per shard (576 generations total). New ingredients:
a deliberately between-run height control direction (`height_ge_4` probe,
cosine vs raw −0.05 to −0.01), dose-response for orthogonal/Gaussian controls
(scales 0.25/0.5/1), and within-forward-pass positional projection-variance
telemetry per direction.

## Arms (row-level cluster bootstrap vs in-job regenerated baseline)

| condition | P(strong) | P(weak) | parse fail | P(strong\|parsed) | dP (CI95) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.359 | 0.766 | 0.062 | 0.391 | — |
| erase_raw_s1 | 0.391 | 0.672 | 0.078 | 0.438 | +0.031 [−0.094, +0.172] |
| erase_height_s1 | 0.312 | 0.625 | 0.062 | 0.318 | −0.047 [−0.188, +0.125] |
| erase_orthogonal_s0p25 | 0.312 | 0.500 | 0.203 | 0.357 | −0.047 [−0.188, +0.094] |
| erase_orthogonal_s0p5 | 0.266 | 0.359 | 0.344 | 0.357 | −0.094 [−0.297, +0.109] |
| erase_orthogonal_s1 | 0.016 | 0.141 | 0.344 | 0.019 | −0.344 [−0.547, −0.156] |
| erase_gaussian_s0p25 | 0.375 | 0.531 | 0.047 | 0.375 | +0.016 [−0.141, +0.188] |
| erase_gaussian_s0p5 | 0.328 | 0.438 | 0.031 | 0.333 | −0.031 [−0.234, +0.172] |
| erase_gaussian_s1 | 0.125 | 0.125 | 0.500 | 0.295 | −0.234 [−0.438, −0.016] |

Strict row flips (row P(strong) crossing 0.5): raw 0/0 (false→true/true→false),
height 1/0, orthogonal_s1 0/4, gaussian_s1 0/3. Dose-response is monotone:
both controls are null-ish at scale 0.5 and destructive only at scale 1.
`P(strong|parsed)` reproduces the earlier precision/demolition split:
orthogonal destroys correctness even among parsed outputs (0.391 → 0.019)
while Gaussian damage is substantially format (parse fail 0.50, parsed
precision 0.295).

## Within-forward-pass positional projection variance (sd², call-weighted)

| layer | raw | height | orthogonal | gaussian |
| --- | ---: | ---: | ---: | ---: |
| L15 | 10.65 | 0.01 | 818.37 | 381.73 |
| L30 | 0.84 | 0.00 | 3.13 | 75.32 |
| L40 | 0.14 | 0.00 | 0.11 | 0.33 |
| L45 | 0.06 | 0.00 | 0.14 | 4.32 |
| L53 | 0.03 | 0.00 | 0.01 | 0.02 |

## Pre-registered decision-rule outcome

The rule (`docs/causal_handle_directions.md` item 1) said: if the raw
direction's within-run positional variance is far below the controls', the
constant-offset account is live and the between-run control governs wording.
That branch fired. At the high-variance layers the raw direction carries far
less within-run structure than the matched controls (L15: 10.65 vs 818/382;
L30: 0.84 vs 3.1/75.3), and the by-construction between-run height direction
(~zero within-run variance) erases as harmlessly as raw (−0.047, CI straddles
zero).

## Verdict

Non-necessity stands and gains a second harmless arm: erasing the correctness
readout axis costs nothing, and neither does erasing a semantically unrelated
between-run (height) axis. But the destructive-control contrast is confounded
by within-run variance: harmlessness tracks low within-forward-pass projection
variance, not the semantic identity of the erased direction, and the
norm-matched controls inject one to two orders of magnitude more positional
perturbation than the raw erasure does. Wording rule for the paper: cite the
Gemma control separation as evidence that the erasure procedure has behavioral
teeth at matched norm, NOT as evidence that the raw axis is specially inert
among same-norm directions. The safe claim remains: the readable correctness
axis is not load-bearing (claim 2's first sentence), with the constant-offset
account explicitly acknowledged.

Artifacts: `docs/erasure_control_matching_27b_property_k4_shard{0..3}of4.json`,
`results/stage2/erasure/erasure_control_matching_27b_property_k4_shard*.jsonl`
and `*_directions.npz`.
