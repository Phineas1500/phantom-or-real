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
