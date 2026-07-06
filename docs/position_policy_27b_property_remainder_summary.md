# Rank-k Guard v2 (fresh rows) - Job 458440 - shard 0 of 1

Output JSON: `docs/position_policy_27b_property_remainder.json`
Rows: 8 prepared from 8 fresh-selection rows.

## Causal arms (row-paired bootstrap vs in-job unhinted baseline)

| arm | P(strong) | dP vs reference (CI95) | reference |
| --- | ---: | ---: | --- |
| fixednorm_gold_L30 | 0.406 | +0.391 [+0.125, +0.703] | unhinted_baseline |
| percand_fire_L30 | 0.053 | +0.037 [+0.009, +0.075] | unhinted_baseline |
| unhinted_baseline | 0.016 | - | none |

Hint-validated rows (hinted P(strong) >= 0.5): 0.

Reading rule: Gate: gold-candidate fires repair (CI excl. zero; fallback fixednorm_gold k=8). POLICY-VIABLE if P1 self-ratification beats baseline (paired CI excl. zero) AND >=50% of the oracle fire. Mechanism readout (wrong-concept fires, targets_fired_concept) reported regardless. Rules: item H. Exploratory.
