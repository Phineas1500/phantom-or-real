# Subtype Localization Patch — Verdict (Job 457005)

Pre-registered analysis of `docs/subtype_localization_patch_27b.json`
(512 generations, 78 min; 16 behaviorally validated recognition-gap rows from
`docs/subtype_recognition_gap_27b_manifest.json`). Job state shows FAILED due
to a benign sbatch line-continuation artifact AFTER all outputs were written
(stray `--layers` line; all unpassed flags equal their defaults — verified);
the run itself is complete.

## Arms (row-paired bootstrap, 16 rows, k=8)

| arm | P(strong) | dP vs baseline (CI95) |
| --- | --- | --- |
| baseline | 0.180 | — |
| full_patch | 0.219 | +0.039 [-0.156, +0.234] |
| subset_concept | 0.219 | +0.039 [-0.008, +0.117] |
| subset_random | 0.188 | +0.008 [-0.016, +0.031] |

## Pre-registered bin: both null — accessibility is task-dependent

On subtype, the full-block hint-state patch does NOT repair (+0.039 vs
property's +0.491), and the concept-position subset is marginal at best.
The control is clean (random null). Per the bins recorded before landing:
focus-state causal accessibility through L30/40/45 context encodings is
task-dependent; claims 5-6 scope to property, and the discussion gains a
task-dependence subsection.

## The unifying reading (pre-registered prediction for the KV program)

The behavioral donors are validated on subtype (hint-first repairs at 0.875
through TOKENS), yet the context-encoding patch transfers nothing. Combined
with the property necessity null (repair survives context-encoding reversion
because the literal hint tokens remain), one account covers both tasks:
**the hint always has two routes — in-place context-encoding writes and
decode-time access to the hint tokens — and their mix is task-dependent.**
Property: writes carry enough to repair via patching (+0.491) and the token
route makes them non-necessary. Subtype: the token route dominates; writes
at L30/40/45 carry too little to transfer.

Pre-registered predictions (historical; recorded while job 457009 ran, before
its results were seen): (a) if gold-KV repaired on property, the token route
would be demonstrated sufficient there too; (b) a subtype gold-KV arm was then
expected to repair despite the subtype context-patch null. Property gold-KV
later proved null despite attention flowing, so this old subtype gold-KV
prediction is superseded. The resolved follow-up is the capture-ladder
discriminator postscript below.

## Cross-lane note

In-job sampled baseline (0.180 at temp 0.7) sits well above the Modal
behavioral baseline (0.000) on these rows, unlike the hinted ceiling which
matched exactly at 1.000 on property. Baseline-level cross-lane comparisons
should use in-job numbers only; ceiling-level ones are lane-robust.

## Instrument verification (post-hoc, before prose; all offline)

- Resolved config verified from the run's own log (`using_hooks` resolved to
  L30/40/45; 16 rows; blocks 92-99 tokens) — not argparse reasoning.
  Program-wide fix adopted: future scripts dump resolved args into the report.
- Span audit: 16/16 rows clean — concept positions decode to the literal
  concept tokens, donor/receiver token ids match at every patched position,
  hint offset 11 tokens.
- Arm coincidence check: full_patch and subset_concept per-row patterns are
  entirely different (the equal +0.039 means is averaging coincidence).
- Delta-norm diagnostic: subtype per-position concept-delta norms EQUAL OR
  EXCEED property's at all three layers (L45: 8683 vs 8058). The hint writes
  to subtype context encodings at normal magnitude; the writes are present
  but not causally transferable at these layers.

## Rival account registered (same breath as pathway-mix)

**Layer mismatch**: subtype's causally potent hint-writes may live at layers
other than the property-derived 30/40/45. At registration time both accounts
predicted subtype gold-KV repair, so KV alone was not the intended
discriminator. Separation logic
(pre-registered): (1) capture-ladder triage — hinted-vs-unhinted delta norms
at a ladder of layers on the subtype rows, forward passes only (triage, not
verdict: norm is not causal content); (2) a targeted patch at any off-trio
peak is the discriminator — repairs -> layer-mismatch; null at every ladder
layer supports residual-route insufficiency. Job 457170 resolved this branch;
see the postscript below.

Terminology: the predicted pattern is a **cross-task route dissociation**
(property: both routes sufficient; subtype: token-route-only), NOT a double
dissociation. The true double-dissociation bin exists only if property
gold-KV unexpectedly fails (property writes-only + subtype tokens-only) —
named now since job 457009 reports on it within hours.


## Postscript: discriminator resolved (Job 457170)

The follow-up capture-ladder discriminator has now landed in
`docs/subtype_discriminator_27b_summary.md`. The ladder found large late
hinted-vs-unhinted writes and selected off-trio layers `L53,L50,L35`, but the
causal arms did not deliver a decisive layer-mismatch repair: old-trio replay
was +0.016 CI [-0.211,+0.227]; L53/L50 concept replacement were +0.031/+0.039;
L35 was the strongest at +0.102 CI [-0.008,+0.258], with matched
concept-minus-random +0.109 CI [+0.000,+0.266]; L53 rank4 LOO add was +0.000.

Resolved reading: the subtype writes are present and large, but simple
concept-position residual-state replacement at the largest off-trio layers is
not a robust repair route. Keep claims 5-6 scoped to property; treat L35 as a
possible replication target, not a landed subtype causal handle.
