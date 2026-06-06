# Prefix-Conditioned Margin Trajectory

Generated: `2026-06-06T18:48:53.131922+00:00`

Purpose: score gold, hard foil, and selected hypotheses after generated-prefix checkpoints on the Gemma manifest recognition-gap rows.

## Summary

- Rows: `14`; trajectory rows: `112`.
- Selected hypothesis available on `13/14` rows.
- Selected-vs-gold nonnegative at checkpoint 0 on `13/14` rows.
- Gold-vs-foil nonnegative at checkpoint 0 on `1/14` rows.

## Interpretation Note

This is not a causal result. It shows that the regenerated selected hypothesis is already more prompt-likely than gold at prefix 0 on 13/13 parsed rows, while gold beats the hard foil at prefix 0 on only 1/14 rows. That is a strong predictive trajectory signature of early wrong-hypothesis preference, not proof that the preference is causally responsible for the final answer.

## Checkpoint Summary

| checkpoint | prefix tokens | n | selected avail. | selected-vs-gold mean | selected>=gold | gold-vs-foil mean | gold>=foil |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 14 | 13 | 17.318 | 13/13 | -11.650 | 1/14 |
| 1 | 1 | 14 | 13 | 7.832 | 9/13 | -1.874 | 5/14 |
| 4 | 4 | 14 | 13 | 8.168 | 11/13 | 3.364 | 8/14 |
| 8 | 8 | 14 | 13 | -11.764 | 3/13 | 5.820 | 8/14 |
| 16 | 12-16 | 14 | 13 | -7.999 | 7/13 | 9.647 | 8/14 |
| 32 | 12-32 | 14 | 13 | -0.528 | 7/13 | 7.023 | 10/14 |
| 64 | 12-64 | 14 | 13 | 4.056 | 11/13 | 4.030 | 7/14 |
| final | 12-96 | 14 | 13 | 2.693 | 9/13 | 3.288 | 8/14 |

## Row Commitment Checkpoints

| row | example | h | strong | parse fail | first selected>=gold | first gold>=foil |
| --- | --- | --- | --- | --- | --- | --- |
| 3073 | property_h3_00073 | 3 | False | False | 0 | 32 |
| 3290 | property_h3_00290 | 3 | True | False | 0 | 0 |
| 3415 | property_h3_00415 | 3 | False | False | 0 | 1 |
| 4322 | property_h3_01322 | 3 | True | False | 0 | 1 |
| 4675 | property_h3_01675 | 3 | False | False | 0 | 4 |
| 5292 | property_h3_02292 | 3 | False | True | None | 1 |
| 6188 | property_h4_00188 | 4 | False | False | 0 | None |
| 6327 | property_h4_00327 | 4 | False | False | 0 | 4 |
| 8035 | property_h4_02035 | 4 | False | False | 0 | 8 |
| 8298 | property_h4_02298 | 4 | False | False | 0 | 4 |
| 8874 | property_h4_02874 | 4 | False | False | 0 | 32 |
| 9549 | property_h4_03549 | 4 | False | False | 0 | 8 |
| 10079 | property_h4_04079 | 4 | False | False | 0 | 1 |
| 10714 | property_h4_04714 | 4 | True | False | 0 | 4 |

## Causal-Abstraction Claim

Predictive trajectory diagnostic only. It tests whether the selected hypothesis is already more likely than gold under generated-prefix contexts; it does not perform a causal intervention.

