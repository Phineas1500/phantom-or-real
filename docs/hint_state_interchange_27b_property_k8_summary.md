# Hint-State Interchange — Powered k=8 Results

Job 456990 (batched generation: 560 generations in 1h25). Pools
`results/stage2/erasure/hint_state_interchange_27b_property_manifest.jsonl`;
concept-resolved row-paired analysis in
`docs/hint_state_interchange_27b_property_concept_analysis.json`. Supersedes
the k=2 pilot (`docs/hint_state_interchange_27b_property_manifest_k2.json`).

## Design

14 recognition-gap manifest rows, 5 arms, k=8 samples per (row, arm) at
temperature 0.7, L30/L40/L45 context-block patching from behaviorally
validated hint-first donors. All statistics are row-paired with row-level
cluster bootstrap (n=14 rows is the effective n).

## Results

| arm | P(strong) | dP vs baseline (95% CI) | F->T rows | targets_wrong delta (CI) |
| --- | --- | --- | --- | --- |
| baseline | 0.196 | — | — | — |
| patch_hint_gold | 0.688 | +0.491 [+0.277, +0.705] | 7/14 | +0.098 [-0.080, +0.295] |
| patch_shuffled | 0.589 | +0.393 [+0.152, +0.634] | 5/14 | +0.062 [-0.098, +0.241] |
| patch_hint_wrong | 0.241 | +0.045 [-0.009, +0.107] | 0/14 | +0.018 [-0.214, +0.259] |
| noise_matched | 0.027 | -0.170 [-0.357, -0.009] | 0/14 | -0.295 [-0.527, -0.089] |

Gold-patch minus shuffled-donor, row-paired: +0.098, CI [+0.009, +0.214].

## Verdict

1. **The causal repair handle is confirmed at power.** Gold-hint-state
   patching repairs +0.491 with a CI far from zero, against a destructive
   matched-noise control. This is the program's first powered positive
   activation-level causal result.
2. **The carrier is predominantly content, not position structure.** The
   position-shuffled donor retains most of the repair (+0.393); the ordered
   patch adds a small but nonzero increment (+0.098, CI excludes zero).
   This raises the prior that a low-rank/mean-shift hint delta can
   reproduce the effect — the add-to-unhinted arm of the hint-delta program
   is the priority follow-up.
3. **Misdirection does not transfer through activations.** The wrong-donor
   arm is null on accuracy (+0.045) and on the concept-resolved metric
   built to detect it (targets_wrong +0.018, CI [-0.21, +0.26]) — while the
   same wrong hint through tokens misdirects completely (0.000 correct).
   The defensible framing is **asymmetric controllability**: token-reachable
   and intervention-reachable states diverge, consistent with the
   non-surjectivity result. Candidate mechanism (untested): the hint works
   partly through decode-time attention to the literal hint tokens, which a
   context-only patch cannot reproduce; the KV-transplant variant tests
   this.

## Caveats

- 14 rows, property task, Gemma only; recognition-gap rowset (free-form
  strong-wrong rows), so repair is measured where baseline is mostly wrong.
- The shuffled control shows position-shuffling within the block is mild;
  it does not rule out that block-level placement matters.
- Baseline outputs already target the gold concept in 67% of generations
  (wrong form/polarity); part of the repair is form correction, not only
  concept selection — the concept-resolved analysis separates these only
  partially.
