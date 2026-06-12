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

Pre-registered predictions (recorded while job 457009 runs, before its
results are seen): (a) if gold-KV repairs on property, the token route is
demonstrated sufficient there too (both routes individually sufficient);
(b) a subtype gold-KV arm (future job, consuming the subtype manifest)
should repair DESPITE the subtype context-patch null — a cross-task double
dissociation that would convert the task-dependence scoping into a positive
pathway-mix finding. If subtype gold-KV is also null, the subtype hint
effect rides a carrier outside both tested routes (unpatched layers).

## Cross-lane note

In-job sampled baseline (0.180 at temp 0.7) sits well above the Modal
behavioral baseline (0.000) on these rows, unlike the hinted ceiling which
matched exactly at 1.000 on property. Baseline-level cross-lane comparisons
should use in-job numbers only; ceiling-level ones are lane-robust.
