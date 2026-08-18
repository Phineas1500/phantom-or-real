# Item M (SJ/OC factorization) — COMPLETE: the gauge reads objective correctness (GAUGE~OC, T1), and the lever repairs regardless of self-judgement (LEVER-IGNORES-SJ, 8th registered prediction CONFIRMED)

Registered 2026-08-14 (commit 20b4b35); M0 job 462065 + battery, M2
jobs 462066/462067, all Aug 18. Frame: docs/sjoc_m0_frame.json
(128 rows, 32/cell, seed 20260816, soft-census SJ labels).

## M0 — gates and T1 (docs/sjoc_m0_battery.json)
- Sanity gate: frozen gauge reads OC at AUC 0.960 on the frame (PASS).
- SJ-readability gate: dir_SJ decodes held-out SJ at 0.811 (PASS) —
  self-judgement IS linearly present at the gauge's site.
- Factorization: erasing dir_SJ drops SJ 0.331 / moves OC 0.007;
  erasing dir_OC drops OC 0.177 / moves SJ 0.002 (clean).
- **T1 (no registered prediction): GAUGE~OC** — on 64 held-out
  conflict rows the frozen gauge agrees with OC 84.4% vs SJ 15.6%
  (diff −0.688 [−0.844, −0.500]). Against 2607.16799's probes-track-SJ
  claim: this gauge tracks objective correctness even where the
  model's verbal self-assessment disagrees, with the SJ signal
  readably present in the same states.

## M2 — the 8th registered prediction (docs/sjoc_m2_pooled.json)
Rank-8 in-job LOO repair (L′ protocol, arm base 120) on the pinned
sets (24 each; 23/23 prepared — one row per set skipped for missing
concept positions):
- CONFIDENT-WRONG: 0.054 → 0.674, dP **+0.620 [+0.462, +0.772]**;
  delivery +0.228 CI>0.
- ORDINARY-WRONG: 0.114 → 0.614, dP **+0.500 [+0.337, +0.663]**;
  delivery +0.179 CI>0.
- Across-set difference +0.120 [−0.114, +0.342] — contains 0
  (MDE 0.228, stated). Parse 8.7%/4.6%.
- **VERDICT: LEVER-IGNORES-SJ** — both sets repair with CI>0 and the
  difference straddles zero. The largest single-arm repairs in the
  program, on a frame drawn without regard to repairability.

## The factorization, complete
Gauge and lever both live on the OBJECTIVE-CORRECTNESS side: the
gauge reads OC (not SJ) on conflict rows; the lever repairs
identically whether the model verbally insists it was right or admits
it was wrong. The verbal self-report channel — elicitation-fragile in
the censuses — is causally bypassed by both objects. 29 registered
verdicts; 8 confirmed directional predictions.
