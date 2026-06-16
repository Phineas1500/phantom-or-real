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
   L30 alone, repairs +0.260 — matching the three-layer subset-replacement
   effect (+0.250) within CI (point ratio 104%), where rank-1 was null at
   every scale and the rank-full ADD was null/harmful. Per the pre-registered
   reading rule (k<=4 recovering >=70%): **compactly structured.** The PCA
   gatekeeper pointed at L30 (top-1 = 49% of variance) and the causal test
   confirmed it. Truncation HELPS the additive route: the discarded high-rank
   residual is not just inert but obstructive when added.
4. Scale insurance resolved: x1/x2/x4 is monotone harmful past x1 — the
   rank-1 direction has no working scale; the structure, not the magnitude,
   was missing.

## Rank-k guard (Job 457012)

The post-finale held-out-basis guard passes. Leave-one-row-out PCA at L30
repairs at rank 4: P(strong)=0.385, dP=+0.192 CI [+0.096,+0.288], or 77% of
the +0.250 subset-replacement effect. Rank 8 is stronger (P=0.423, dP=+0.231
CI [+0.087,+0.404], 92% of subset). The result is therefore not an in-sample
low-rank fit, but the ladder is graded rather than a hard rank-4 cliff.

## Rank-core geometry rider

The held-out-surviving L30 core is essentially absent from the INLP readable
subspace: rank4/rank8 subspace fractions are 0.0001/0.0002, below a random
subspace null mean of 0.0017. Gemma Scope gives the complementary result: the
core is dictionary-visible, with top decoder subspace sqrt-energy about 0.98 in
both 16k and 262k residual decoders. But it is not sparse-small: the top 20
decoder rows explain only 1.0% (16k) or 0.1% (262k) of decoder-overlap mass,
with thousands of rows showing nontrivial overlap.

## The program's closing picture (property task)

Concept commitment is written at mention sites redundantly across layers; a
compact low-dimensional causal core is present by rank 4 at L30, with useful
additional structure through rank 8; it is invisible to the readable
correctness subspace but visible in Gemma Scope as a highly redundant decoder
family rather than a compact sparse feature set; and the literal hint tokens,
though present and attended, carry none of it at decode time. The subtype
discriminator has now landed: the capture ladder found large late writes and
selected L53/L50/L35, but targeted off-trio residual-state patches did not
robustly repair (best L35 +0.102 CI [-0.008,+0.258]; L53 rank4 LOO add +0.000).
This scopes the compact-core positive claim to property and leaves subtype in a
residual-route-insufficiency bin rather than a clean layer-mismatch win.

## Next (final experimental items)

- Qwen erasure / cross-model necessity robustness (critical path); erasure
  refinements; optional L35 subtype replication only if we want to chase the
  suggestive but non-landed bump.
