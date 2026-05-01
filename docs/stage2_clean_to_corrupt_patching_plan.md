# Stage 2 Clean-To-Corrupt Patching Plan

Date: 2026-05-01

## Purpose

The forced-choice hard-foil result closed the probe-direction steering branch:
the answer-polarity direction is perfectly decodable but does not causally move
free-form or binary choices. Patching asks a different question: whether any
compact residual state from an easier correct prompt can repair a harder
incorrect prompt.

This also uses the hard-foil observation that baseline MCQ accuracy was 14/16
on rows where free-form generation was wrong. The model can recognize the
answer when shown candidates, so the likely failure is answer deployment rather
than ontology comprehension.

## Pair Construction

Use strict natural pairs from existing S1 test rows:

- clean row: height 1, parsed, strong-correct;
- corrupt row: height 4, parsed, strong-incorrect;
- full gold hypothesis match: same target concept, predicate, and polarity;
- corrupt foil: first parsed wrong hypothesis from the corrupt Stage 1 output.

The existing 27B `infer_property` data has 11 S1-test strict pair capacity.
The first pilot uses 8 pairs selected by seed `20260501`. Exact same-ontology
cross-height pairs do not exist in the shipped data, so this is the cleanest
natural pairing before synthetic h1 construction.

## Patch Sites

Patch residual-stream states at semantic landmarks rather than raw token index:

- `last_prompt`: final token before generation;
- `subject`: last tokenized occurrence of the gold subject before the question
  stem;
- `predicate`: last tokenized occurrence of the gold predicate before the
  question stem;
- `question_stem`: first token of "Please come up with hypothesis".

Layers for the pilot: `30,35,40,45,50`. This weights late answer-deployment
layers while still checking one earlier/mid layer.

## Intervention

For each pair, layer, and landmark:

- `clean`: replace the corrupt residual vector at that landmark with the clean
  h1 residual vector from the matching landmark.
- `noise`: add norm-matched random noise to the corrupt vector, matching the
  L2 norm of the clean-minus-corrupt delta.

The pilot is margin-first. It scores the corrupt original prompt under the gold
hypothesis versus the corrupt model's wrong foil. The recovery metric is:

```text
(patched_margin - corrupt_baseline_margin)
/ (clean_reference_margin - corrupt_baseline_margin)
```

Values near `1` indicate full recovery toward the clean prompt margin; values
near `0` indicate no effect; negative values indicate anti-recovery.

## Run

Script:

- `scripts/stage2_patch_clean_to_corrupt.py`

Wrapper:

- `scripts/stage2_patch_clean_to_corrupt_27b_property_margin_pilot.sbatch`

Outputs:

- `docs/clean_to_corrupt_patching_27b_property_margin_pilot.json`
- `results/stage2/patching/clean_to_corrupt_27b_property_margin_pilot.jsonl`

## Decision Rule

If clean patches recover margin at specific late-layer landmarks above
norm-matched noise, refine around those sites and rerun with generation enabled
to measure free-form false-to-true flips.

If clean patches do not beat noise anywhere, the causal story becomes stronger:
not only are probe-derived directions non-causal, but transplanting strict
matched h1 residual states at the tested landmarks does not repair h4 output
margins.

## Pilot Result

Slurm job `452478` completed on 2026-05-01. It evaluated 8 strict
h1-correct to h4-incorrect pairs, 5 layers, 4 landmarks, and 2 patch modes,
for 320 total rows. Runtime was 879 seconds. No landmarks were missing.

The only sites with meaningful aggregate movement were late `last_prompt`
patches:

| Site | Mode | Mean recovery | Mean margin delta | Improved pairs |
|---|---|---:|---:|---:|
| L35 `last_prompt` | clean | 0.108 | +2.836 | 6/8 |
| L40 `last_prompt` | clean | 0.079 | +1.526 | 5/8 |
| L45 `last_prompt` | clean | 0.046 | +0.161 | 3/8 |
| L50 `last_prompt` | clean | 0.071 | +1.088 | 5/8 |
| L45 `last_prompt` | noise | 0.085 | +3.497 | 7/8 |
| L50 `last_prompt` | noise | 0.115 | +4.345 | 5/8 |

Thus the apparent last-prompt effect is not cleanly state-specific:
norm-matched random perturbations are comparable to, and sometimes stronger
than, transplanting the h1 residual state.

The per-pair headroom check supports this interpretation rather than rescuing
a hidden localization claim. For the four high-headroom pairs
(`recovery_denominator >= 45`), clean last-prompt patches did not beat noise:

| Site | Clean mean recovery | Noise mean recovery |
|---|---:|---:|
| L35 `last_prompt` | -0.006 | +0.013 |
| L40 `last_prompt` | -0.048 | -0.011 |
| L45 `last_prompt` | -0.090 | +0.080 |
| L50 `last_prompt` | -0.068 | +0.120 |

For the four lower/mid-headroom pairs, clean patches did beat noise at
L35-L45, but this split is not enough to claim repair because the aggregate
effect is small, pair-dependent, and not stable under the matched noise
control.

## Interpretation

This pilot rules out a specific missing-state story at the tested sites. If
the h4 failure were simply caused by the absence of a compact residual state
present in the easier h1 prompt, clean h1-to-h4 patches should outperform
norm-matched noise. They do not.

The more specific reading is that the h4 `last_prompt` state is already
committed to the wrong free-form answer. Small perturbations at late
`last_prompt` positions can partially loosen that commitment, slightly
improving the gold-vs-foil margin, but they do not transplant the correct
answer computation. This connects to the forced-choice hard-foil result:
the model often recognizes the correct answer when shown the correct and
wrong hypotheses, but its free-form generation pathway remains committed to
the wrong hypothesis.

Caveat: these are strict natural h1/h4 pairs sharing the full gold hypothesis,
not exact same-ontology pairs. The h1 and h4 prompts therefore differ in more
than depth. Exact same-ontology cross-height pairs were not available in the
shipped data.

## Reverse-Patching Follow-Up

We ran the minimal reverse intervention as Slurm job `452492`: patch h4
incorrect `last_prompt` residual states into the matching h1 correct prompt,
using the same 8 pairs, layers L35/L40/L45/L50, and norm-matched noise
controls. The signed metric remains `gold - foil`; for this reverse run,
negative margin deltas indicate collapse of the h1 correct margin. The report
also records a positive `breakage_fraction` equal to `-delta / denominator`.

Outputs:

- `docs/corrupt_to_clean_patching_27b_property_margin_pilot.json`
- `results/stage2/patching/corrupt_to_clean_27b_property_margin_pilot.jsonl`

Aggregate breakage:

| Site | Mode | Mean breakage | Mean margin delta | Decreased pairs |
|---|---|---:|---:|---:|
| L35 `last_prompt` | corrupt | 0.100 | -2.508 | 6/8 |
| L40 `last_prompt` | corrupt | 0.136 | -2.842 | 6/8 |
| L45 `last_prompt` | corrupt | 0.120 | -2.214 | 5/8 |
| L50 `last_prompt` | corrupt | 0.177 | -4.704 | 6/8 |
| L35 `last_prompt` | noise | -0.015 | +1.200 | 3/8 |
| L40 `last_prompt` | noise | 0.018 | -0.711 | 4/8 |
| L45 `last_prompt` | noise | -0.065 | +2.097 | 3/8 |
| L50 `last_prompt` | noise | 0.023 | +0.227 | 4/8 |

This is an asymmetric result: h4 states patched into h1 prompts reduce the h1
gold-vs-foil margin more than matched noise, while the forward h1-to-h4
patches did not repair h4 beyond noise. The asymmetry is strongest at L50 and
is visible at all four tested layers by the aggregate mean.

The effect is not uniform across headroom. For the four high-headroom pairs
(`recovery_denominator >= 45`), corrupt patches do not robustly beat noise:

| Site | Corrupt mean breakage | Noise mean breakage |
|---|---:|---:|
| L35 `last_prompt` | 0.012 | -0.034 |
| L40 `last_prompt` | -0.012 | 0.019 |
| L45 `last_prompt` | -0.036 | -0.011 |
| L50 `last_prompt` | 0.017 | -0.026 |

For the four lower/mid-headroom pairs, corrupt patches break h1 much more than
noise:

| Site | Corrupt mean breakage | Noise mean breakage |
|---|---:|---:|
| L35 `last_prompt` | 0.187 | 0.004 |
| L40 `last_prompt` | 0.283 | 0.018 |
| L45 `last_prompt` | 0.277 | -0.120 |
| L50 `last_prompt` | 0.337 | 0.071 |

Interpretation: wrong h4 late `last_prompt` states can partially degrade weaker
h1 correct-answer margins, but they do not reliably collapse strong h1
commitments. Together with the forward null, this supports an asymmetric
commitment story rather than a missing-state repair story: incorrect
free-form commitments can be disruptive when transplanted into easier prompts,
but correct h1 states do not repair the hard h4 prompt.
