# Item G2 — Qwen Layer Sweep Verdict (job 458463)

15 fresh Qwen3.5-27B failure rows (G0 rows excluded), HF pathway, k=8;
row-cluster bootstrap 10k. Baseline P(strong)=0.192, hinted 0.817.

| layer (rel depth) | concept dP [CI95] | random dP | paired (c−r) [CI95] |
| --- | ---: | ---: | ---: |
| L26 (0.40) | −0.042 [−0.100,+0.017] | +0.008 | −0.050 |
| L32 (0.50) | +0.083 [−0.025,+0.200] | +0.067 | +0.017 |
| L38 (0.60) | +0.108 [−0.017,+0.233] | +0.033 | +0.075 |
| **L43 (0.67)** | **+0.175 [+0.100,+0.258]** | +0.017 | **+0.158 [+0.033,+0.292]** |
| L48 (0.75) | +0.075 [−0.017,+0.175] | +0.025 | +0.050 |

**Pre-registered winner rule: L43 qualifies (uniquely)** — concept CI
excludes zero AND paired concept-vs-random CI excludes zero. The
commitment carrier exists in Qwen and is reachable at relative depth
0.67 — deeper than Gemma's 0.48, consistent with depth being the
non-transferable coordinate (cf. subtype L30→L35). Effect ~69% of
Gemma's concept_replace (+0.175 vs +0.255). G3 (specificity ladder at
L43) is unlocked.
