# Item G6 — Qwen Content-Transfer Ladder: PRIMARY NULL, but the Dissociation INVERTS — raw potent, projection dilutes

Job 458474 (Scholar, 2× A40, HF pathway, bf16), completed 2026-07-07
~00:55, 102 min. `scripts/stage2_qwen_g6_hf.py` per the G6 registration.
Raw rows: `results/stage2/erasure/qwen_g6_ladder.jsonl`; metadata + F(i)-
analog: `docs/qwen_g6_ladder.json`. 24 test rows (15 G3-core + 9 fresh),
96 protocol-matched sources (48/48).

## Integrity gates — all pass

Unhinted + rank16 verbatim vs G3′ on the shared 15 rows (240/240
token-identical; 0.192/0.308 exactly). Max parse-fail 1.0%. Free bonus:
rank16 on the 9 held-out fresh rows repairs at ~+0.17 — the channel
basis generalizes to rows that never touched it.

## The F(i)-analog rider (descriptive, the night's pivotal diagnostic)

Natural correct-vs-incorrect at L43 concept positions, 96 rows:
**full-dim CV AUC 0.504 — chance** (Gemma same-design ceiling: 0.807);
rank-16 slice 0.587 and rank-64 slice 0.521, both inside their 200-draw
random-slice nulls (p95 0.627/0.606). Qwen does NOT stage linearly-
decodable outcome information at the channel's address.

## Pooled results (24 rows, row-cluster bootstrap 10k, seed 20260704)

| arm | dP vs baseline (0.177) | 95% CI |
|---|---:|---|
| rank16_loo (gate) | +0.135 | [+0.052, +0.224] ✓ |
| protomatched_proj16 (PRIMARY) | +0.021 | [−0.042, +0.083] |
| protomatched_proj64 (secondary) | **+0.104** | **[+0.005, +0.198]** ✓ |
| protomatched_proj16_x2 (secondary) | +0.010 | [−0.062, +0.089] |
| **class_mean_raw96** (descriptive) | **+0.120** | **[+0.016, +0.224]** ✓ |
| shuffled96_proj16 | +0.036 | [−0.031, +0.115] |
| signflip96_proj16 | −0.115 | [−0.177, −0.047] — CI < 0 |
| rand_norm_perpos | +0.031 | [−0.047, +0.104] |

Key contrasts: raw − proj16 paired **+0.099 [+0.005, +0.193]** (the
inversion, CI-supported); raw − rand (amplitude-matched) +0.089
[−0.010, +0.193] (just misses); proj64 − proj16 +0.083 [+0.000, +0.172]
(boundary); PRIMARY-2 (proj16 − shuffled) −0.016 (dead).

## Verdict (registered branches)

- **PRIMARY: FAIL** (both criteria). With the F(i)-analog, this is now
  EXPLAINED rather than under-powered: rank-16 projection extracts
  nothing because nothing linearly decodable lives at this address to
  extract — the shuffled vector does exactly as well (+0.036).
- **Named secondary proj64: PASSES its CI** (+0.104 [+0.005, +0.198])
  — reported with the registered flag: control-unmatched at rank-64
  (shuffled/rand ran at rank-16 norms); no claim moves on it alone.
- **Sign-flip: registered prediction ≤ 0 CONFIRMED again** (−0.115,
  CI < 0; fourth confirmed registered prediction of the program). At
  rank 16 the forward direction is inert but its negation is
  destructive while shuffled is null — the slice carries genuine
  label-derived polarity, asymmetrically: breaking is easy, helping
  needs the full vector.
- **x2: null** — amplitude cannot substitute for the missing
  directions.
- **The all-null licensed sentence does NOT apply**: raw96 and proj64
  both exclude zero. The honest landed sentence: **in Qwen the
  dissociation INVERTS — the raw 96-row class-mean repairs (+0.120,
  CI > 0) and projection into the compact channel basis DILUTES it
  (raw − proj16 +0.099, CI > 0), the mirror image of Gemma's
  raw-null/proj-potent (+0.043 vs +0.341).** Consistent with
  everything else Qwen has shown: channel energy, channel rank, and now
  content are all DISTRIBUTED where Gemma's are CONCENTRATED.

## What is and is not claimable

Claimable now (descriptive, CIs attached, controls partial): the
inversion pattern and the sign-flip asymmetry. NOT yet claimable:
"answer-free repair transfers to Qwen" — raw96 was a descriptive rider,
its label-specificity control (shuffled-RAW) and a second
amplitude-matched draw do not exist, and raw − rand misses its CI.
**G6′ (registered separately) is the one-job nail-down**: raw96 as
PRIMARY, shuffled-raw ×2 draws, rand-perpos ×2 draws, signflip-raw,
proj64 + shuffled-proj64-at-64-norms (closing the registered flag).
If raw survives that battery, the cross-model sentence becomes:
"answer-free content transfer works in both models — through the
compact channel in Gemma, through the full state in Qwen."

## Cross-model table (updated)

| coordinate | Gemma 3 27B | Qwen3.5 27B |
|---|---|---|
| carrier depth (rel) | 0.48 | 0.67 |
| channel rank | 8 (78%) | ~16 (67%; 95% @ 64) |
| outcome info at channel address | decodable (0.807) | absent (0.504) |
| class-mean transfer | proj-potent, raw-null | raw-potent, proj-diluted |
| sign-flip | harmful | harmful (every design) |
