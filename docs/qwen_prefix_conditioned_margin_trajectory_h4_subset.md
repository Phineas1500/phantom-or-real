# Qwen Prefix-Conditioned Margin Trajectory

Generated: `2026-06-06T19:43:13.427946+00:00`

Purpose: score gold and the Stage 1 hard-foil hypothesis after original Qwen output-prefix checkpoints on recognition-gap rows.

## Summary

- Rows: `14`; trajectory rows: `112`.
- Available recognition-gap rows before limit: `43`.
- Selected hypothesis source: `stage1_hard_foil`.
- Selected-vs-gold nonnegative at checkpoint 0 on `14/14` rows.
- Gold-vs-foil nonnegative at checkpoint 0 on `0/14` rows.

## Interpretation Note

This is not a causal result. It uses Qwen's original Stage 1 free-form output as the prefix trajectory and treats the hard foil as the emitted selected hypothesis. It is comparable to the Gemma prefix-conditioned diagnostic only as cross-model recognition-gap trajectory evidence; it is not a matched regenerated decode-trace replication.

## Checkpoint Summary

| checkpoint | prefix tokens | n | selected avail. | selected-vs-gold mean | selected>=gold | gold-vs-foil mean | gold>=foil |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 14 | 14 | 12.864 | 14/14 | -12.864 | 0/14 |
| 1 | 1 | 14 | 14 | 9.787 | 14/14 | -9.787 | 0/14 |
| 4 | 4 | 14 | 14 | 6.043 | 14/14 | -6.043 | 0/14 |
| 8 | 8 | 14 | 14 | 6.807 | 14/14 | -6.807 | 0/14 |
| 16 | 16 | 14 | 14 | 6.830 | 14/14 | -6.830 | 0/14 |
| 32 | 17-31 | 14 | 14 | 8.611 | 14/14 | -8.611 | 0/14 |
| 64 | 17-31 | 14 | 14 | 8.611 | 14/14 | -8.611 | 0/14 |
| final | 17-31 | 14 | 14 | 8.611 | 14/14 | -8.611 | 0/14 |

## Row Commitment Checkpoints

| row | example | h | original strong | MCQ correct | first selected>=gold | first gold>=foil |
| --- | --- | --- | --- | --- | --- | --- |
| 6085 | ontology_h4_00085 | 4 | False | True | 0 | None |
| 6100 | ontology_h4_00100 | 4 | False | True | 0 | None |
| 6184 | ontology_h4_00184 | 4 | False | True | 0 | None |
| 6293 | ontology_h4_00293 | 4 | False | True | 0 | None |
| 6322 | ontology_h4_00322 | 4 | False | True | 0 | None |
| 6610 | ontology_h4_00610 | 4 | False | True | 0 | None |
| 6676 | ontology_h4_00676 | 4 | False | True | 0 | None |
| 6757 | ontology_h4_00757 | 4 | False | True | 0 | None |
| 6925 | ontology_h4_00925 | 4 | False | True | 0 | None |
| 6932 | ontology_h4_00932 | 4 | False | True | 0 | None |
| 6971 | ontology_h4_00971 | 4 | False | True | 0 | None |
| 7145 | ontology_h4_01145 | 4 | False | True | 0 | None |
| 7306 | ontology_h4_01306 | 4 | False | True | 0 | None |
| 7452 | ontology_h4_01452 | 4 | False | True | 0 | None |

## Causal-Abstraction Claim

Predictive trajectory diagnostic only. It tests whether the Stage 1 hard-foil hypothesis is already more likely than gold under prompt and generated-prefix contexts; it does not perform a causal intervention.
