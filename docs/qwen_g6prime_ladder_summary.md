# Item G6′ — Raw-Transfer Control Battery: PRIMARY PASSES — Answer-Free, Label-Specific Content Transfer Works in Qwen (Through the Full State)

Job 458512 (Scholar, 2× A40, HF pathway, bf16), completed 2026-07-07
~03:30, 110 min. `scripts/stage2_qwen_g6prime_hf.py` per the G6′
registration. Raw rows: `results/stage2/erasure/qwen_g6prime_ladder.jsonl`.
Same 24 test rows, 96-source class-mean, per-position norms as G6.

## Integrity gates — all pass

All THREE replication arms reproduce G6 verbatim: unhinted (0.177),
rank16 (0.312 all-24), class_mean_raw96 (0.297) — 576/576 generations
token-identical across jobs. Max parse-fail 0.5%.

## Pooled results (24 rows, row-cluster bootstrap 10k, seed 20260704)

| arm | dP vs baseline (0.177) | 95% CI |
|---|---:|---|
| rank16_loo (gate) | +0.135 | [+0.052, +0.224] ✓ |
| **class_mean_raw96 (gate/PRIMARY)** | **+0.120** | **[+0.016, +0.224] ✓** |
| shuffled_raw96 d1 | −0.099 | [−0.167, −0.026] — CI < 0 |
| shuffled_raw96 d2 | −0.026 | [−0.099, +0.052] |
| signflip_raw96 | −0.094 | [−0.146, −0.042] — CI < 0 |
| rand_norm_perpos d2 / d3 | −0.016 / +0.026 | both straddle 0 |
| shuffled96_proj64 @ 64-norms | −0.115 | [−0.167, −0.057] — CI < 0 |

Registered contrasts — every one lands:

| contrast | dP | 95% CI | verdict |
|---|---:|---|---|
| PRIMARY-1: raw96 vs baseline | +0.120 | [+0.016, +0.224] | ✓ |
| **PRIMARY-2: raw96 − shuffled-raw family** | **+0.182** | **[+0.070, +0.299]** | **✓ PASS** |
| raw96 − rand family (3 draws w/ G6 d1) | +0.106 | [+0.009, +0.214] | ✓ (G6's near-miss now excludes zero) |
| signflip-raw vs baseline | −0.094 | CI < 0 | registered ≤ 0 prediction CONFIRMED (5th) |
| proj64 − shuffled-proj64 @ 64-norms | +0.219 | [+0.125, +0.318] | ✓ proj64's flag CLOSED |

## Verdict (registered branch: PASS)

**The registered PASS wording applies: answer-free, label-specific
content transfer works in Qwen through the FULL state — Gemma
concentrates, Qwen distributes; the recipe is cross-model, the
compression is not.**

- The labels carry the effect: real class-mean +0.120 while
  label-scrambled versions of the SAME geometry/norms/positions range
  from null to actively harmful (d1 −0.099 CI < 0) — paired real-vs-
  scrambled +0.182 [+0.070, +0.299]. Both G4′-era ambiguities are
  resolved: not amplitude (rand family null, paired CI > 0), not
  geometry (shuffled family ≤ 0).
- Sign-flip harmful again, unprojected (−0.094, CI < 0) — the fifth
  confirmed registered prediction of the program.
- proj64's control-unmatched flag is CLOSED in the strong direction:
  shuffled-into-64 at 64-norms is actively destructive (−0.115, CI < 0)
  while the real vector at identical treatment repairs (+0.104, G6) —
  paired +0.219. The rank-64 pass was labels, not basis-plus-energy.
- Descriptive texture: every scrambled/flipped variant is null-to-
  harmful; the ONLY helpful direction found in five Qwen content jobs
  is the true natural correct-minus-incorrect axis.

## The landed cross-model story (G-series complete: G0–G3′, G4′, G6, G6′)

| coordinate | Gemma 3 27B | Qwen3.5 27B |
|---|---|---|
| recognition-gap failure mode | ✓ | ✓ (hint lift +0.52) |
| carrier depth (rel) | 0.48 | 0.67 |
| channel rank | 8 (78% of carrier) | ~16 (67%; 95% @ 64) |
| outcome info at channel address | decodable (0.807) | NOT decodable (0.504) |
| answer-free class-mean transfer | ✓ +0.341, PROJECTED | ✓ +0.120, RAW (projection dilutes) |
| label battery (shuffled/signflip) | ✓ passes | ✓ passes (shuffled ≤ 0, flip < 0) |
| self-knowledge/channel co-location | co-located | separated |

Wording consequences applied: claim 13 upgrades to include the
label-specific transfer; ledger §3 cross-model bullet gains the PASS;
§1 claims remain Gemma-scoped (Qwen still lacks the erasure battery).
The current paper's cross-model section may cite this as the registered
one-line addendum (decision at drafting, per the G6 registration).
Open next-paper thread: the Q1 information map — where DOES Qwen stage
its outcome information, given it repairs from an address where nothing
is linearly decodable?
