# Item L″ verdict — registered branch 3: PROTOCOL INSTABILITY ACROSS ARM STRUCTURE. The composition's own gold branch fails to reproduce L′'s repair (−0.009 vs +0.279); the verdict is confined to reporting this.

Pre-registered 2026-08-14 (docs/causal_handle_directions.md, item L″;
registered after L′'s TRANSFERS verdict, before any L″ data; no
registered prediction; exploratory — no §1 claim moves on any branch).
Jobs 461663–461666 (Scholar bf16 2×A40 constraint J, 4×16-row shards,
2h16–2h44 each), `--selfaddress-loo` mode. Pooled by
`scripts/stage2_selfaddress_loo_pool.py`; full numbers in
`docs/selfaddress_loo_27b_property_pooled.json`; row-level jsonls
checked in under `results/stage2/erasure/`.

## Gates

- **Baseline verbatim gate: PASS 456/456** vs item L1's recorded
  baselines — the third exact regeneration of these generations
  (row-keyed seeds; L1 → L′ → L″).
- Parse: baseline 8.3% fail, branch arm 7.8% — both FLAG (5–20%).
  P(strong|parsed): baseline 0.096, branch arm 0.110, gold branches
  0.087 — the flagged-regime read matches the headline (no
  parse-artifact rescue).

## The registered decision cascade (n = 57 rows, 8.7 candidates/row, k=4)

| quantity | dP vs in-job baseline | CI95 | sign |
|---|---:|---|---|
| **gold branch (the arm's own oracle)** | **−0.009** | [−0.035, +0.018] | straddle ≈ 0 |
| gauge-select [registered PRIMARY] | +0.009 | [−0.031, +0.053] | straddle |
| random-select (20 draws) | +0.008 | [−0.017, +0.038] | straddle |
| self-ratify | +0.018 | [−0.015, +0.057] | straddle |
| L1-recorded bestofN majority | −0.018 | [−0.042, +0.007] | straddle |
| L1-recorded bestofN any-correct (ceiling, calibration only) | +0.175 | [+0.088, +0.270] | pos |
| wrong-address collateral (non-gold pooled) | +0.010 | [−0.016, +0.040] | straddle |

PRIMARY fails, but the operative fact sits above it: **the arm's own
gold branch — the same LOO write, at the gold address, on the same
rows where L′ measured +0.279 — repairs nothing** (−0.009, MDE
half-width 0.026). Branch 2 (selector–write interference) does NOT
fire: there is no working write for the selector to interfere with.
This is pre-named **branch 3: protocol instability across arm
structure; the verdict is confined to reporting this; flagged for the
rescue-set analysis.**

## Texture

- The selector survives the write: gauge-select still picks the gold
  branch **75.4%** of the time (43/57) under LOO firing — degraded
  from item L's gauge==oracle on clean steered branches, but far
  above the 11.5% random rate. The addressing half keeps working;
  the write half is what collapsed.
- No wrong-address harm: misaddressed fires are indistinguishable
  from baseline (+0.010 straddle), consistent with item L's frozen
  arm — off-target writes are inert, not destructive, in both
  protocols.

## What differs between L′ (+0.279) and L″ (−0.009) — post-hoc, for the rescue-set analysis (unregistered)

1. **LOO basis pool size**: L′ fit each row's rank-8 basis from 31
   in-shard donors (2×32-row shards); L″ from 15 (4×16-row shards).
   If basis quality degrades sharply below ~30 donors, the write dies
   with the shard layout.
2. **Arm structure**: L′ fired during full-length generation arms;
   L″ fires inside item L's per-candidate branch machinery (k=4
   branches, gauge prefill capture interleaved). Any interaction
   between the branch-generation context and the write is
   uncontrolled across the two.
3. Both differences are confounded here; the rescue-set analysis
   (registered as the branch-3 follow-up path) would cross them:
   L″-structure with 31-donor bases vs L′-structure with 15-donor
   bases. Cheap version first: refit L″ rows' bases from the union
   donor pool offline and check recon-norm/overlap diagnostics
   against L′'s bases before spending GPU hours.

## What moves

- Nothing in §1; the paper's selector language was already demoted to
  descriptive-pending-composition (review round 3), and the
  composition now reads: **attempted once, branch 3 — the loop is not
  closed; the write's potency is a property of the full protocol
  (donor pool + arm structure), not of the vector.** This SHARPENS
  the L-series' central lesson (in-job fitting is part of the recipe;
  now: so is the surrounding protocol), and it is the honest
  counterweight to L′'s +0.279 wherever that number is quoted.
- L‴ (fresh-draw confirmation) is NOT triggered — it was obligated
  only on PASS.
- Item M launches next on Scholar (the L-series has drained the
  queue); the rescue-set diagnostic is offline/CPU and can run
  anytime.
