# Item G4′ — Qwen Answer-Free Class-Mean at L43/rank-16: PRIMARY NULL (content transfer not established at this power)

Job 458472 (Scholar, 2× A40, HF pathway, bf16), completed 2026-07-06
22:13, 64 min. `scripts/stage2_qwen_g4_hf.py` per the G4′ registration
in `causal_handle_directions.md`. Raw rows:
`results/stage2/erasure/qwen_g4_classmean.jsonl`; run metadata:
`docs/qwen_g4_classmean.json`.

## Integrity gates (registered) — all pass

- Replication: unhinted_baseline and rank16_loo reproduce G3′ **verbatim**
  (240/240 token-identical; P 0.192 / 0.308 exactly).
- Source screening: 20 correct + 20 incorrect rows confirmed
  (majority-of-4 in-job votes), disjoint from test/G0 rows.
- Parse integrity: max parse-fail 0.8% (gate < 5%).
- Scale telemetry: class-mean norm 6.69 → pooled channel norm 23.81
  (~3.6× amplification; the natural difference is small, as in Gemma).

## Pooled results (15 test rows, row-cluster bootstrap 10k, seed 20260704)

| arm | dP vs baseline (0.192) | 95% CI |
|---|---:|---|
| rank16_loo (gate) | +0.117 | [+0.017, +0.217] ✓ |
| class_mean_raw | +0.058 | [−0.058, +0.175] |
| **class_mean_proj16** | **+0.050** | **[−0.058, +0.167]** |
| shuffled_label (d1 / d2) | −0.067 / +0.008 | both straddle 0 |
| signflip_proj16 | −0.083 | [−0.158, −0.017] — CI < 0 |
| rand_norm16 | −0.092 | [−0.175, +0.000] |

Registered contrasts: PRIMARY-1 (proj16 vs baseline) +0.050
[−0.058, +0.167] — straddles zero. PRIMARY-2 (proj16 − shuffled family)
+0.079 [−0.008, +0.167] — straddles zero. Rider (raw − proj16 paired)
+0.008 [−0.100, +0.108].

## Verdict (registered branches)

**PRIMARY NULL.** Both criteria miss. Registered wording applies:
**"channel established (G3′), content transfer not established at this
power"** — MDE ≈ 0.12, observed +0.050. No claim moves; ledger §1
claims 1–2 remain Gemma-scoped, exactly as they were.

**Sign-flip: registered prediction CONFIRMED** (predicted ≤ 0; landed
−0.083 with CI entirely below zero) — the program's third confirmed
registered prediction (item D branch, item H mechanism, this). The
natural correct-minus-incorrect axis has the right POLARITY in Qwen:
pushing against it reliably hurts, even though pushing along it at this
norm/rank/n does not reliably help.

Branches that did NOT fire: "shuffled matches real" cannot be claimed
(the paired contrast leans positive but straddles); "raw repairs too"
is moot (both raw and proj are null); no label-specificity claim in
either direction is licensed for Qwen.

## Reading (descriptive, not registered)

The asymmetry is the informative texture: signflip harmful (CI < 0) and
rand_norm16 borderline-harmful (upper bound exactly 0.000) while the
real direction is the ONLY non-gate arm whose point estimate is
positive. The natural-outcome axis is not inert in Qwen — it just
doesn't repair at the amplitude/rank/sample size this design bought.
Candidate explanations for the Gemma/Qwen gap, all untested here:
(i) power — Gemma's F(ii) effect (+0.341) is ~7× this point estimate,
so an effect 1/3 of Gemma's would be invisible at n=15; (ii) the
donor-frame gap — Gemma's class-mean was captured at GOLD-concept
positions of guard-row prompts, Qwen's at each source row's own concept
positions (the only available anchor for natural rows); (iii) Qwen's
wider channel (k*=16 of a 5,376-dim state) may dilute a
40-row-estimated mean more than Gemma's rank-8 core does. A follow-up
would need more rows and/or a donor-frame-matched capture — registered
separately if ever run.

## Wording consequences applied

- Claims table claim 13: caveat gains "G4′ answer-free content transfer
  NOT established (null at MDE 0.12; sign-flip polarity confirmed)".
- Ledger §3 cross-model bullet: motif sentence unchanged; add the
  content-transfer null with the sign-flip confirmation.
- The paper's cross-model section states both: the channel replicates
  (G3′), the answer-free content result is Gemma-only at current power.
