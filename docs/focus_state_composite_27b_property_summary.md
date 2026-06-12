# Focus-State Composite — Verdict (Job 457002)

Pre-registered analysis of `docs/focus_state_composite_27b_property.json`
(1,008 generations, 2h19; all 9 arms completed). Offline geometry uses the
saved concept-position states and the L30/L40/L45 INLP stacks.

## Arms (row-paired bootstrap, 14 rows, k=8)

| arm | P(strong) | dP vs reference (CI95) | note |
| --- | --- | --- | --- |
| baseline (unhinted) | 0.163 | — | |
| hinted_baseline (in-job) | 1.000 | — | matches Modal exactly: zero harness tax |
| reverse_subset (necessity) | 1.000 | +0.000 [0, 0] vs hinted | **necessity null** |
| reverse_random (control) | 1.000 | +0.000 | |
| restricted_add x1 | 0.221 | +0.058 [-0.135, +0.279] | null-ish |
| restricted_add x4 | 0.000 | -0.163 [-0.346, -0.019] | destructive overshoot |
| spotlight_rank1 | 0.221 | +0.058; donor-tgt 0.096 | null |
| spotlight_perpos | 0.000 | -0.163 [-0.356, -0.019]; donor-tgt 0.087 | **disruptive, never misdirects** |
| complement | 0.240 | +0.077 [+0.000, +0.163] | marginal |

## Verdicts

1. **Commitment, not spotlight.** Foreign concept-position deltas at one's
   own concept positions disrupt and never misdirect (donor-targeting
   0.087-0.096). Per the pre-registered grid: the localized state is
   concept commitment written in place, not a movable attention operator.
2. **Necessity null -> conditional link 1 fires.** Reverting the hinted
   run's concept-position encodings (or random positions) leaves the 1.000
   repair untouched. The literal hint tokens remain in the prompt, so
   decode-time attention to them is the parallel pathway. **The KV/hint-span
   job jumps the queue**, now four arms: baseline, gold-KV (machinery
   control), wrong-KV, and hint-span ablation.
3. **Decomposition is superadditive.** subset_concept (+0.250, job 456999)
   plus complement (+0.077) falls well short of the full patch (+0.491):
   concept positions are the main carrier but full repair requires the
   block-level interaction.
4. **Rank-1 remains unsupported with a scale-window caveat.** x1 null-ish,
   x4 destructive: the pre-registered x2 follow-up is owed. Conditional
   link 2 (PCA gatekeeper): L30 is concentrated (top-1 = 49% of variance,
   top-3 = 61%) while L40/L45 are flat (top-3 = 31%/28%) — a layer-resolved
   causal rank-k arm at L30 is earned and joins the next job.
5. **The geometry headline survives restriction and is now bulletproof.**
   Concept-position delta vectors project onto the full 9-direction INLP
   readable subspace at 0.013-0.014 — below the 0.041 chance level, at all
   three layers. The causal focus content lies outside everything the
   correctness probes can read. Claims-table row 9 upgrades from
   provisional to landed.

## Updated one-line thesis

Concept commitment is written in place at the mention sites — sufficient to
transplant, redundant with the token pathway, invisible to the readable
subspace — while the correctness probes watch a gauge wired to nothing.
