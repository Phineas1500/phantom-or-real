# Rank-Core Geometry Rider

Analysis of L30 rank-core PCA components from `results/stage2/erasure/focus_state_composite_27b_property_states.npz`.

## INLP overlap

| rank | subspace fraction in INLP | null mean | null p95 | component fractions |
| --- | ---: | ---: | ---: | --- |
| 4 | 0.0001 | 0.0017 | 0.0024 | 0.0000, 0.0001, 0.0001, 0.0002 |
| 8 | 0.0002 | 0.0017 | 0.0021 | 0.0000, 0.0001, 0.0001, 0.0002, 0.0002, 0.0002, 0.0008, 0.0003 |

## Gemma Scope decoder overlap

### layer_30_width_16k_l0_small

| rank | top decoder sqrt-energy | top20 fraction of decoder energy | count >=0.10 |
| --- | ---: | ---: | ---: |
| 4 | 0.9786 | 0.010566 | 10460 |
| 8 | 0.9801 | 0.010304 | 11003 |

Top features by subspace sqrt-energy:

- rank 4: 12625:0.9786, 2018:0.9782, 12253:0.9781, 12182:0.9778, 9214:0.9773, 15026:0.9768, 14617:0.9762, 11562:0.9760, 11910:0.9758, 15886:0.9732
- rank 8: 2018:0.9801, 12625:0.9790, 12253:0.9786, 12182:0.9784, 9214:0.9779, 15026:0.9778, 14617:0.9777, 11910:0.9777, 11562:0.9771, 15886:0.9749

### layer_30_width_262k_l0_small

| rank | top decoder sqrt-energy | top20 fraction of decoder energy | count >=0.10 |
| --- | ---: | ---: | ---: |
| 4 | 0.9816 | 0.001132 | 106663 |
| 8 | 0.9823 | 0.001103 | 112506 |

Top features by subspace sqrt-energy:

- rank 4: 1056:0.9816, 108101:0.9808, 14222:0.9804, 14038:0.9802, 1279:0.9795, 159794:0.9787, 7151:0.9778, 57329:0.9769, 3978:0.9764, 85244:0.9763
- rank 8: 1056:0.9823, 108101:0.9817, 14038:0.9813, 14222:0.9809, 1279:0.9798, 159794:0.9790, 3978:0.9783, 7151:0.9782, 57329:0.9781, 20513:0.9781

## Verdict

The held-out-surviving L30 core is essentially outside the INLP readable subspace: rank-4 and rank-8 overlap are far below the random-subspace null. Gemma Scope gives a different answer: decoder rows align strongly with the causal-core subspace and the first few PCA components, so the dictionary exposes the object. But the exposure is highly redundant rather than sparse-small: thousands of decoder rows have nontrivial overlap, and the top decoder rows explain only a tiny fraction of total decoder-overlap mass. The loop therefore closes as gauge-orthogonal but dictionary-visible, not as a compact handful of Gemma Scope features.
