# Item L — Self-Addressing the Lever: ORACLE GATE FAILS — the frozen-basis donor-free protocol does not transfer to fresh failing rows at pinned norm; selectors descriptive only (and the gauge picked the gold address anyway)

Jobs 461624 (L0) + 461628/461629/461630/461632/461633 (L1: shards
0–2 of 4 plus 3of8+7of8, whose union equals the registered shard 3of4;
row-keyed seeds), Scholar bf16, 2× A40, completed 2026-08-13.
Registered in `causal_handle_directions.md` item L before any data.
Pooled battery: `docs/selfaddress_27b_property_pooled.json`
(`scripts/stage2_selfaddress_pool.py`); L0 gates:
`docs/selfaddress_l0_27b_property_shard0of1.json`.

## Gates

- **L0(a) natural sanity: PASS** — gauge AUC 0.936 on natural states
  (bar 0.75).
- **L0(c) selection signal: PASS** — gold-branch gauge score beats
  non-gold branches by +31.8 [+23.8, +40.1] on the 12 calibration rows.
- **Pinned-norm checks**: hard-fail gate passed in every job (frozen
  basis recomputation = 3708.2628 exactly).
- **Parse**: all three arms in the flag band pooled (baseline 8.3%,
  percand 8.5%, bestofN 10.9%; no arm voided) — higher than prior row
  sets; all scoring on dP(strong), unparsed = not-strong, as registered.
- **ORACLE GATE (registered to be evaluated first): FAIL.** Firing the
  frozen K-convention vector at the GOLD concept's positions on 57
  fresh failing rows (baseline 0.088) produces nothing: dP −0.004
  [−0.022, +0.015].

## Registered consequence (verbatim branch)

"The frozen-basis donor-free protocol does not transfer to fresh
failing rows at pinned norm." The K-convention protocol — full frozen
26-row basis, single pinned amplitude — had never been run on failing
rows; the registration named this a genuine risk, and it bit. Under
the registered rules: **selectors are reported descriptively, no
selector claims land, and the addressing question returns to the
registry unresolved.**

## Pooled numbers (57 rows, ~8.7 candidates/row)

| policy / arm | P(strong) | dP vs baseline (CI95) |
| --- | ---: | ---: |
| unhinted baseline | 0.088 | — |
| ORACLE (gold branch) | 0.083 | −0.004 [−0.022, +0.015] |
| GAUGE-SELECT | 0.083 | −0.004 [−0.022, +0.015] |
| RANDOM-SELECT (20 draws) | 0.099 | +0.012 [−0.009, +0.036] |
| SELF-RATIFY | 0.110 | +0.022 [−0.007, +0.055] |
| best-of-N majority vote | 0.070 | −0.018 [−0.042, +0.007] |
| best-of-N any-correct ceiling | 0.263 | +0.175 [+0.088, +0.270] |

Fire texture: gold branches 0.083 vs non-gold 0.103 — no gold
advantage under this protocol. Branch gauge-score vs outcome AUC 0.63
(weak). Paired gauge−random −0.016 [−0.031, −0.002].

## What the wreckage still shows (descriptive, no claims)

1. **The selector did its job; the lever did not fire.** GAUGE-SELECT
   equals ORACLE to three decimals — the gauge, scoring steered
   prefills it was never trained on, picked the gold-equivalent branch
   essentially always (consistent with L0's +31.8 separation). The
   composition's addressing half worked; there was nothing downstream
   to select FOR, because the write itself was inert here.
2. **Protocol transfer is the story.** The same construction helps
   naturally-correct rows (+0.160, item K collateral) and does nothing
   for fresh failing rows (−0.004) — while F(ii)-c's per-row LOO-basis
   variant repaired guard-row failures at +0.447. The repair-side
   difference between the frozen-pooled protocol and the in-job LOO
   protocol (basis provenance, per-row norms) is now the isolated
   variable. Hypothesis only; its test is a fresh registration
   (L′: F(ii)-c protocol on the L rows), not this doc.
3. **The sampling texture replicates on fresh rows**: majority vote
   −0.018 (correlated errors), any-correct ceiling only 0.263 from
   ~35 samples/row.

## Consequences

- Per the registration, the paper's §5.5 addressing caveat is REPLACED
  by this outcome sentence: the answer-free addressing loop was built
  and its selector works, but the frozen donor-free write does not
  repair fresh failures at pinned norm, so the addressing gap remains
  open — now with the failure localized to protocol transfer rather
  than selection.
- No registered prediction was at stake (none was made — the
  registration refused one on the primary, correctly, again). Count
  stays 7.
- Registry: addressing question returns unresolved; candidate next
  registration L′ = the F(ii)-c LOO protocol on these same 57 rows
  (one arm + gates), which cleanly separates basis-provenance from
  row-population explanations.
- Scoop note (2026-08-12 sweep): the composition scaffold is now in
  the literature (2608.08829). Our result is currently a NEGATIVE for
  the composition at this protocol — worth stating publicly precisely
  because the field is about to try it: the selector-works /
  write-doesn't dissociation is not a result anyone else has.
