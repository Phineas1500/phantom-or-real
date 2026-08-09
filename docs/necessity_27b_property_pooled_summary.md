# Item K — Necessity Ablation on Natural Successes: state content at gold positions IS necessary; subspace-specificity at matched energy NOT established (energy-confound flag); prediction (i) CONFIRMED (7th)

Jobs 459836/459837/459838 (correct-side shards, 46 rows pooled of 48
selected) + 459839 (anchor), Scholar bf16, 2× A40, completed 2026-08-09.
Registered in `causal_handle_directions.md` item K before any data.
Pooled battery: `docs/necessity_27b_property_pooled.json` (built by
`scripts/stage2_necessity_pool.py`); per-shard reports
`docs/necessity_27b_property_shard{0,1,2}of3.json`,
`docs/necessity_anchor_27b_property_shard0of1.json`.

## Gates — all pass

- **Anchor verbatim (gate i)**: all 416 generations of both anchor arms
  reproduce job 458431 token-for-token (0 text mismatches, 0 score
  mismatches); anchor repair +0.447 [+0.288, +0.611] on baseline 0.120.
  Fourth consecutive verbatim replication gate in the program.
- **Baseline (gate ii)**: pooled correct-side baseline 0.764 ≥ 0.55.
- **Parse (gate iii)**: four arms in the flag band, none near the 20%
  void bound — ablate_rank8 7.1%, signflip_100 6.0%, signflip_200 7.1%,
  rand_norm_d1 5.7%. P(strong|parsed) preserves every ordering (ablate
  0.105, signflip_100 0.072, signflip_200 0.000, rand_norm_d1 0.424):
  the breaks are wrong answers, not format demolition — ablated outputs
  are well-formed hypotheses about the wrong concepts.

## Pooled arms (46 rows, k=8; row-cluster bootstrap 10k, seed 20260704)

| arm | P(strong) | dP vs baseline (CI95) |
| --- | ---: | ---: |
| correct_unhinted_baseline | 0.764 | — |
| **ablate_rank8 (K-PRIMARY)** | 0.098 | **−0.666 [−0.769, −0.557]** |
| ablate_rand8 family (d1/d2 pooled) | 0.692 | −0.072 [−0.124, −0.024] |
| ablate_perm8 (flag layer) | 0.726 | −0.038 [−0.079, −0.003] |
| signflip_fixednorm_100 | 0.068 | **−0.696 [−0.791, −0.598]** |
| signflip_fixednorm_200 | 0.000 | −0.764 [−0.851, −0.668] |
| rand_norm family (d1/d2 pooled) | 0.413 | −0.351 [−0.444, −0.261] |
| correct_fixednorm_100 (collateral) | 0.924 | **+0.160 [+0.062, +0.261]** |

Paired contrasts: ablate − rand8 family **−0.594 [−0.702, −0.481]**;
ablate − perm8 −0.628 [−0.739, −0.514]; signflip_100 − rand_norm family
**−0.345 [−0.455, −0.243]**; signflip_200 − signflip_100 −0.068
[−0.139, −0.014] (dose-monotone). Per-shard K-PRIMARY replication:
−0.692 / −0.664 / −0.642 on three disjoint 15–16-row draws.

## K-PRIMARY → branch CHANNEL-IN-USE (b) fires mechanically — with an ENERGY-CONFOUND flag from the registered telemetry

The registered conditions select **CHANNEL-IN-USE, sub-case (b)**
(ablate CI < 0, paired vs rand8 family CI < 0, rand8 family CI < 0 at
−0.072), and the perm8 flag layer passes (ablate − perm8 CI < 0). The
mechanical wording would be "breaks beyond generic rank-8 removal
damage — necessity supported with a generic-fragility component."

**The registered projection-variance telemetry scopes this hard.** The
fitted rank-8 basis carries **97.7% of the state norm** at gold
positions (mean state norm ≈ 31,163; removed ≈ 30,452). The controls do
not operate at that energy: rand8 removes 3.3% (≈ 1,000), perm8 1.3%
(≈ 400), and the add arms perturb at the pinned 3,708. The ablation is
therefore a near-deletion of the residual state at gold concept
positions — a perturbation ~8× larger than any control in the design.
"Matched rank" did not mean "matched energy": the delta-PCA basis
aligns with the few huge-norm directions that dominate Gemma's
residual, so removing 8 directions removes essentially the state.

**Landed wording (telemetry-scoped, per the item-D precedent of letting
telemetry constrain wording):** the state content at gold
concept-mention positions is necessary for natural success — its
near-deletion destroys 87% of successes (0.764 → 0.098) while
matched-rank structure controls are near-free — but
**subspace-specific necessity at matched energy is NOT established**,
and the §6 endogeneity movement licensed by the branch is **DEFERRED**
pending an energy-matched control (ablate-to-row-mean, or removal of a
matched-energy state-PCA subspace outside the delta span — next
registration, not run here). No §1 claim moves (as registered on every
branch).

## K-SECONDARY → registered prediction (i) CONFIRMED — the program's 7th

`signflip_100` CI entirely < 0 (−0.696) AND paired (signflip_100 −
rand_norm family) CI entirely < 0 (−0.345): **label-specific corruption
of natural successes**, the fourth confirmed sign-flip prediction
(after F(ii)-b, G4′, G6′). This comparison is clean — the sign-flip and
its random control are norm-matched by design at the pinned 3,708.
Texture: matched-norm noise alone breaks correct rows substantially
(−0.351; contrast item C's failing rows, where the same-scale noise was
only −0.088) — corruption has a norm-generic floor on correct rows, and
the true label roughly doubles the damage on top of it. Dose-monotone
(2× worse than 1×, paired CI < 0).

## Riders

- **Asymmetry readout (descriptive)**: break consumes 91% of the
  down-room (0.696/0.764) vs repair's 51% of the up-room (0.447/0.880,
  by gate) — **attractor-compatible asymmetry** (breaking natural
  success is easier than inducing it), dose-monotone, cross-population
  caveat as registered.
- **Collateral fresh-draw replication**: `correct_fixednorm_100`
  +0.160 [+0.062, +0.261] — **F(ii)-c's collateral-beneficial finding
  survives its first fresh draw** (direction CI > 0 on 46 fresh rows;
  0.764 → 0.924). Headroom fraction 0.68 vs the original 0.97 —
  attenuated from F(ii)-c's near-ceiling 16-row draw, point estimate
  60% of the original; the J1→J2 lesson applied and the finding held.
- **Sign telemetry**: pre-intervention ⟨class-mean, state⟩ is positive
  at gold positions on 46/46 correct rows (mean cosine +0.21) — the
  class-mean direction is present in every natural success, consistent
  with the sign-flip's destructiveness.

## Consequences

- Ledger: registered-prediction count 6 → **7**. No §1 claim moves.
- §6 endogeneity paragraph: may cite item K in one sentence — the
  honest form is "near-deletion of the state at concept positions
  destroys natural success while matched-rank random removal is free
  and the label-specific sign-flip at matched norm halves-to-zeroes it;
  whether a thin subspace (rather than the state content wholesale) is
  the necessary carrier awaits an energy-matched control."
- The anatomical mirror with item D is now measured from both sides but
  must be stated carefully: erasing everything READABLE (rank-9 stack,
  five layers, every token) costs nothing while matched-rank random
  stacks are catastrophic; deleting the state at the LEVER's site
  breaks natural success while matched-rank random removal there costs
  little. The asymmetry of what matched-rank controls do at the two
  sites is itself informative (readable-stack site: controls
  catastrophic; lever site: controls free) — but the item-K side
  carries the energy caveat until the follow-up lands.
- Next registration (before any stronger channel language):
  energy-matched necessity control — ablate-to-row-mean at gold
  positions, matched-energy state-PCA removal outside the delta span,
  and a rank ladder of the removal (1/2/4/8) to find where the break
  turns on.
- Qwen analog stays deferred as registered.
