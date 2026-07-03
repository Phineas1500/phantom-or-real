# Subtype L35 Targeted Replication — Pooled Verdict, Jobs 458387 + 458388

Per-seed outputs: `docs/subtype_discriminator_27b_l35rep2_seed{A,B}.json`;
row-level generations in
`results/stage2/erasure/subtype_discriminator_27b_l35rep2_seed{A,B}.jsonl`.
Same 16 recognition-gap manifest rows as job 457170, ladder 30/35/40/45 with
top-offtrio=1 (selects L35 in both seeds, row-mean delta norm 6012.6), fresh
generation/control seeds 20260702 and 20260703, k=8 per seed. First
submission (jobs 458376/458377) lost the ladder to sbatch `--export`
comma-splitting and ran old-trio-only; this rerun set env vars in the
submitting shell.

## Pooled new-seed arms (16 rows, k=16; paired row-cluster bootstrap)

| arm | P(strong) | dP vs baseline (CI95) |
| --- | ---: | ---: |
| baseline | 0.191 | — |
| L35_concept_replace | 0.309 | **+0.117 [+0.008, +0.254]** |
| L35_random_replace | 0.215 | +0.023 [−0.004, +0.062] (null) |
| L35_rank4_loo_add | 0.238 | +0.047 [+0.000, +0.109] (marginal) |
| old_trio_full_replace_L30_40_45 | 0.215 | +0.023 [−0.195, +0.242] (null) |

Meta-pool with 457170 (k=24): L35_concept_replace +0.112 [+0.003, +0.260];
L35_random_replace +0.013 [−0.008, +0.039].

## Pre-registered decision rule → outcome

Rule (recorded before unblinding): pooled-new CI excluding zero lands the
off-trio layer-mismatch repair and upgrades claim 12's wording (subtype
carrier reachable at L35); a null with half-width ≤ ~0.10 closes the hedge
as a bounded null. Either way `L35_random_replace` must stay null.

**Outcome: the positive branch fires.** Pooled-new CI [+0.008, +0.254]
excludes zero; the random control is null; the meta-pool concurs. The
subtype focus state is reachable through residual-state concept replacement
at L35 — the old trio was the wrong set of layers for subtype, and the
repeated old-trio nulls (three independent replicates now: 457005, 457170,
458376/458377, plus in-job here) were a layer-mismatch, not a task-level
absence of the variable.

## Caveats that go into the wording

- **Edge-of-significance magnitude.** +0.117 with a lower CI bound of
  +0.008 is a landed but modest effect — roughly a quarter of property's
  L30 repair (+0.491 interchange; +0.255 concept-replace on fresh rows).
- **Row-sparse.** The repair concentrates in 3 of 13 baseline-wrong rows
  (3080: 0→0.31, 3103: 0→0.62, 3137: 0.06→0.94); the other 10 never move
  at k=16. Three manifest rows are at baseline ceiling and uninformative.
- **No compact core landed on subtype.** `L35_rank4_loo_add` is marginal
  (+0.047, lower bound touching zero) — consistent with the property
  guard-v2 finding that rank-4 under-transfers; a subtype rank ladder was
  not run and stays future work.

## Wording for the paper (claim 12 / section 5.6)

Subtype localization partially replicates once the layer is chosen by the
capture ladder rather than inherited from property: residual-state concept
replacement at L35 repairs +0.117 [+0.008, +0.254] (k=16, matched random
control null), against repeatedly-null old-trio patches. State it as a
weaker, row-sparse analogue of the property repair — evidence that the
focus-state variable exists on subtype and sits at a different depth — not
as a matched replication of property's effect size, and keep claims 5–6's
strong quantitative story property-scoped.
