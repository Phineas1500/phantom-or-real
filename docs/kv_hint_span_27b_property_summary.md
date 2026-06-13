# KV/Hint-Span Job — Verdict (Job 457009)

Pre-registered analysis of `docs/kv_hint_span_27b_property.json` (936
generations, 2h18; 13 rows × 9 arms × k=8; row 5292 self-skipped as
documented). All bins were written before the job ran.

## Arms (row-paired bootstrap vs each arm's reference)

| arm | P(strong) | dP (CI95) | attn-to-span | note |
| --- | --- | --- | --- | --- |
| unhinted_baseline | 0.183 | — | — | |
| hinted_baseline | 1.000 | — | 0.0050 | natural decode attention to hint: 0.5% |
| hint_span_masking | 1.000 | +0.000 [0, 0] | 0.0000 | mask verified; **token route does nothing** |
| masking_x_reversion | 0.981 | -0.019 [-0.058, 0] | 0.0000 | **survives-both bin: carrier = unpatched layers** |
| gold_kv_transplant | 0.192 | +0.010 [-0.058, +0.077] | 0.0179 | attention flows (3.6x natural), no repair: genuine insufficiency |
| wrong_kv_transplant | 0.192 | +0.010 [-0.058, +0.087] | 0.0174 | no misdirection (wrong-tgt 0.221 vs baseline 0.327) |
| perpos_add_own | 0.087 | -0.096 [-0.327, +0.125] | — | rank-full ADD fails where replacement worked |
| restricted_add_x2 | 0.000 | -0.183 [-0.385, -0.029] | — | scale curve x1/x2/x4 = +0.06/-0.18/-0.16: no window |
| **rank_k_L30** | **0.442** | **+0.260 [+0.115, +0.423]** | — | **reading rule fires: compactly structured** |

## Verdicts (pre-registered bins)

1. **The decode-time token route is dead on property.** The hinted run
   naturally puts only 0.5% of decode attention on the hint span; masking it
   costs exactly nothing; transplanting hint KV into unhinted runs repairs
   nothing despite receiving 3.6x the natural attention (telemetry validates
   the machinery — gold-KV null = genuine insufficiency, not splice bug);
   wrong-KV does not misdirect. The pathway-mix account's token route is
   ruled out for property.
2. **Exhaustive-necessity headline: survives-both.** Masking x reversion
   leaves repair at 0.981. With both tested routes removed simultaneously,
   the hint still works: the carrier is the unpatched layers. Necessity
   failure is now exhaustive over tested routes — the redundancy symmetry
   at its third granularity (directions, pathways, layers).
3. **The surprise: the focus state has a compact causal core at L30.**
   The rank-4 PCA reconstruction of the concept-position deltas, ADDED at
   L30 alone, repairs +0.260 — 104% of the three-layer subset-replacement
   effect (+0.250), where rank-1 was null at every scale and the rank-full
   ADD was null/harmful. Per the pre-registered reading rule (k<=4 recovering
   >=70%): **compactly structured.** The PCA gatekeeper pointed at L30
   (top-1 = 49% of variance) and the causal test confirmed it. Truncation
   HELPS the additive route: the discarded high-rank residual is not just
   inert but obstructive when added.
4. Scale insurance resolved: x1/x2/x4 is monotone harmful past x1 — the
   rank-1 direction has no working scale; the structure, not the magnitude,
   was missing.

## The program's closing picture (property task)

Concept commitment is written at mention sites redundantly across layers;
its causally sufficient core is a <=4-dimensional structured object at L30;
it is invisible to the readable correctness subspace; and the literal hint
tokens, though present and attended, carry none of it at decode time.
Subtype discriminator unchanged: capture ladder + targeted off-trio patch
(the property evidence now weights layer-mismatch over pathway-mix there,
since the token route did nothing even where it was most plausible).

## Next (final experimental items)

- Capture ladder + subtype targeted patch + property-positive-control rider
  (one job); erasure refinements (queued); Qwen erasure (critical path).
- Geometry rider, free: project the rank-4 L30 components onto the INLP
  stack and Gemma Scope decoders — is the compact causal core sparse in the
  learned basis?
