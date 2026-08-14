# Item L′ — Protocol-Transfer Adjudication: TRANSFERS — the F(ii)-c protocol repairs fresh failing rows (+0.279); item L's null was the frozen protocol, not fresh-draw fragility

Jobs 461648/461649 (two 32-row shards over the item-L 64-row selection),
Scholar bf16, 2× A40, completed 2026-08-13 night. Registered in
`causal_handle_directions.md` item L′ before any data. Pooled battery:
`docs/selfaddress_prime_27b_property_pooled.json`
(`scripts/stage2_selfaddress_prime_pool.py`).

## Gates

- **Baseline verbatim vs item L1: PASS** — all 456 baseline generations
  token-identical across a DIFFERENT shard layout (2×32 here vs
  16/16/16/8/8 in L1): the row-keyed seed convention proven
  layout-independent once more.
- **Parse**: fixednorm arm clean; baseline 8.3% (flag band, matching
  L1's own baseline on this population; scoring on dP(strong) as
  registered).

## L′-PRIMARY → branch 1: TRANSFERS

| arm | P(strong) | dP vs baseline (CI95) |
| --- | ---: | ---: |
| unhinted baseline | 0.088 | — |
| **fixednorm_proj (F(ii)-c protocol)** | 0.366 | **+0.279 [+0.182, +0.377]** |

The registered wording lands: **"the F(ii)-c protocol repairs fresh
failing rows — item L's null is attributed to the frozen-pooled
protocol (basis provenance / norm convention), not fresh-draw
fragility."**

- **Within-row protocol contrast** (secondary, claim-bearing under
  branch 1): L′ LOO-protocol fire minus item-L1's recorded
  gold-branch frozen-protocol fire, on the same 57 rows:
  **+0.283 [+0.189, +0.384]**, CI > 0. The two protocols differ
  decisively on identical rows — the cleanest form of the question.
- **Fresh-draw replication rider**: this is the donor-free repair's
  first replication on a draw disjoint from the guard set: +0.279 =
  62% of the +0.447 guard anchor (descriptive; expected attenuation
  from a lower-baseline, harder population).

## Registered consequences

- Paper §5.5: claims stand, plus one protocol-scope sentence —
  **in-job LOO basis fitting and per-row norm conventions are part of
  the recipe**; the frozen-pooled variant (K-convention) transfers on
  correct rows (+0.160) but not to failing-row repair.
- Item L's headline refines to: the gauge-selector works
  (GAUGE-SELECT == ORACLE) and the repair works on fresh rows (this
  item) — what item L composed was the working selector with the
  non-transferring write protocol. The addressing question is now
  BACK IN PLAY: the natural L″ is the same candidate sweep with the
  F(ii)-c write protocol. Registry note only; its own registration
  required.
- No registered prediction was at stake (none made). Count stays 7.
- Program tally: 23 registered verdicts.

## One-paragraph arc (for the drafting pass)

Across L/L′: the readable gauge — deletable, unsteerable — turns out
to rank steered futures well enough to pick the gold address
essentially always; and the answer-free repair replicates on a second
independent draw of failing rows under its original protocol. The two
halves of a closed answer-free repair loop each work in isolation;
their first composition failed only because it paired the selector
with a write variant that does not transfer. That is a sharper,
more optimistic addressing story than item H left us with — and it is
entirely pre-registered, gates-verbatim, and control-bracketed.
