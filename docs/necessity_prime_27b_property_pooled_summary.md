# Item K′ — Energy-Matched Necessity Controls: CONTENT-NECESSITY fires; but every content-destroying intervention floors — the necessity is the state's content wholesale, not a thin channel

Jobs 459847/459849/459850 (three 16-row shards on the item-K rows, 46
pooled), Scholar bf16, 2× A40, completed 2026-08-09 evening. Registered
in `causal_handle_directions.md` item K′ before any data (pre-rule
values computed from frozen artifacts at registration). Pooled battery:
`docs/necessity_prime_27b_property_pooled.json`
(`scripts/stage2_necessity_prime_pool.py`); per-shard reports
`docs/necessity_prime_27b_property_shard{0,1,2}of3.json`.

## Gates — all pass

- **Verbatim vs item K**: both gate arms (baseline + ablate_rank8)
  reproduce jobs 459836/459837/459838 token-for-token — 736/736
  generations, 0 mismatches, all three shards. (One operational note:
  the first shard-0/1 submissions, jobs 459845/459846, died on a
  telemetry-only assert before any hooked generation; the fix touched
  no generation path, and the verbatim gates here prove it.)
- **Baseline**: 0.764 ≥ 0.55 (identical rows and seeds to item K, as
  designed).
- **Parse**: meanablate pooled 11.7% — FLAG, below the 20% void bound,
  so K′-PRIMARY is scored; P(strong|parsed) = 0.000 preserves the
  verdict. statepca8 8.7%, rank1/2 6.0/7.3%, keeponly8 5.4%, rank8
  gate 7.1% — all flagged, none voided; baseline 2.7%, dose012 3.5%,
  rank4 4.1% pass clean.
- **Pre-rules (recorded at registration, confirmed in-job)**: statepca8
  interpretable (principal-angle mean cos 0.472 vs the delta basis);
  no MEAN-FAR flag (median ‖h − mean‖/‖h‖ = 0.175).

## Pooled arms (46 rows, k=8; row-cluster bootstrap 10k, seed 20260704)

| arm | P(strong) | dP vs baseline (CI95) |
| --- | ---: | ---: |
| correct_unhinted_baseline | 0.764 | — |
| ablate_rank8 (K gate, verbatim) | 0.098 | −0.666 [−0.769, −0.557] |
| **meanablate (K′-PRIMARY)** | 0.000 | **−0.764 [−0.851, −0.668]** |
| statepca8 (matched-energy) | 0.000 | −0.764 [−0.848, −0.671] |
| keeponly8 (delete complement) | 0.000 | −0.764 [−0.851, −0.671] |
| ablate_rank1 / rank2 / rank4 | 0.161 / 0.101 / 0.057 | −0.603 / −0.663 / −0.707 |
| dose012 (12% proportional shrink) | 0.661 | −0.103 [−0.158, −0.054] |

Paired: meanablate − rank8 **−0.098 [−0.190, −0.024]**; statepca8 −
rank8 −0.098 [−0.188, −0.024]. The three floor arms are all censored
at literal zero (no natural success survives any of them), so
"breaks at least as hard" is the strongest statement the floor allows.

## K′-PRIMARY → CONTENT-NECESSITY (registered branch 1)

Mean-ablation replaces gold-position states with a state that is
typical by construction (0.175 median relative distance; the giant-norm
infrastructure dims preserved at their dataset-typical values) — and it
is MORE destructive than zeroing the delta subspace. Row-specific
content at gold concept positions is load-bearing even at preserved
typical energy. **Item K's deferred §6 movement fires in the registered
hedged form** — with the rider synthesis below built into the sentence.

## What the riders add: the necessity is broad, not thin

- **keeponly8 → INSUFFICIENT-AT-SITE**: preserving the full 8-dim
  delta-span content while deleting the 5,368-dim complement (21% of
  norm) also floors performance. The rank-8 basis is not a sufficient
  carrier of the position's function.
- **statepca8**: the energy-optimal generic removal is exactly as
  lethal as the targeted one (interpretable per pre-rule).
- **Ladder**: rank-1 removal already costs −0.603; saturation by
  rank 4. Most of the break arrives with the first (giant-norm-aligned)
  direction.
- **dose012**: a proportional 12% shrink of the delta-span component
  (≈ 3,650-norm perturbation, matched to item K's add family) costs
  only −0.103 — vs −0.351 for ADDING random noise at the same
  magnitude (item K) and −0.696 for the sign-flipped class-mean.
  Direction dwarfs magnitude: shrinking existing content is ~3.4×
  cheaper than injecting foreign content at matched norm.

Synthesis: **every intervention that destroys the content of the state
at gold concept positions floors natural success — regardless of which
subspace it targets** (delta-span, energy-optimal, or the complement) —
while content-preserving perturbations are cheap (proportional shrink
−0.103; item K's matched-rank random removals −0.072/−0.038) and the
class-mean add is beneficial (+0.160). The necessary object is the
state's specific content at these positions, wholesale.

## Landed wording (resolves item K's deferred §6 movement)

"At the lever's site, natural success requires the specific content of
the residual state at gold concept-mention tokens: replacing it with a
typical state, deleting its dominant subspace, or keeping only the
repair basis's 8 dimensions each abolish natural success, while
content-preserving perturbations at matched magnitude are an order
cheaper. Necessity at this site is real and content-specific but NOT
compact: the rank-8 basis that suffices to WRITE repairs is neither
necessary alone nor sufficient alone to carry natural success — the
lever is write-compact, but what it writes into is carried broadly."

Scope: site/protocol-scoped as registered; never channel-identity; no
§1 claim moves. The thin-channel-necessity reading is closed (against;
by the item-K energy telemetry and the keeponly8/statepca8 riders
jointly). The write-compact/carry-broad asymmetry is a drafting
candidate for §6, one sentence.

## Consequences

- Item K's verdict stands as scoped ("state content at gold positions
  necessary"); its deferred movement is now resolved per branch 1 with
  the rider-tempered sentence above.
- No new registered predictions were at stake (as registered); the
  count stays 7.
- Follow-up ideas (unregistered, for the next-paper queue only): the
  meanablate parse-degradation gap (mean-ablation garbles format 11.7%
  vs zero-ablation 7.1% — replacing content disrupts more than
  removing it); a positional-generality check (are non-concept
  positions equally content-necessary, i.e., is this specific to the
  lever's site at all?) — that control is the natural K″ if the paper
  needs it.
