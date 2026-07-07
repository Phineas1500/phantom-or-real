# Item J1 — J-aware F(i)-analog: PARTIAL-AT-THRESHOLD on the registered question; the descriptive headline is POSITION-SEPARATED STAGING

Job 458524 (Scholar, 2× A40, bf16), completed 2026-07-07, 13 min.
`scripts/stage2_qwen_j1_hf.py` per the J1 registration. No generation:
96 rows × 12 deterministic forwards + CPU battery. Full numbers:
`docs/qwen_j1_probe.json`.

## Gates (registered) — both pass

- raw_concept = **0.504** — reproduces G6's F(i)-analog exactly.
- raw_ctrl_final (L53, final token) = **0.808** ≥ 0.75 — the stage-1
  gauge region reproduces; pipeline validated.

## The battery (CV AUC, natural correct vs incorrect, 96 rows 48/48)

| feature | AUC |
|---|---:|
| raw L43 @ concept positions (the G6 cell) | 0.504 |
| **raw L43 @ final prompt token** | **0.806** |
| raw L53 @ final prompt token (control) | 0.808 |
| raw penultimate @ final token | 0.579 |
| response (ε-inject own content @ concept) read @ penultimate | 0.663 |
| response read @ L53 final token | 0.792 (no matched null — descriptive) |
| random-tangent null family (10 draws) @ penultimate | 0.388–0.591 |
| label-shuffle p95 for the response feature | 0.660 |

## Registered verdict: PARTIAL-AT-THRESHOLD (inconclusive)

PRIMARY required response@pen ≥ 0.70 AND above both nulls: **not met**
(0.663 < 0.70). The partial branch (above both nulls, < 0.70) fires by
the letter — 0.663 clears the random-tangent family cleanly (max 0.591)
but clears the label-shuffle p95 by 0.003 (0.663 vs 0.660), inside
Monte-Carlo noise. Per the F(i)-nuance precedent, the honest call:
**a consumption-readable signal at concept positions is suggested by
the random-tangent contrast but NOT established** — treat as
inconclusive, do not cite as a positive finding.

## The descriptive headline (NOT a registered decision cell — flagged)

**raw_L43_final = 0.806: the warehouse was never empty; we were
checking the wrong shelf.** Qwen's natural-outcome information is
linearly decodable at Gemma-strength at the SAME layer (L43) at the
FINAL prompt token — just not at the concept-mention positions where
the repair channel operates. The G6 "no decodable info at the channel
address" finding was position-specific, not layer-specific.

Sharpened cross-model picture:
- Gemma: information AND channel co-located at concept positions
  (0.807 decodable where the write works).
- Qwen: information at the final token (0.806), causal write-port at
  concept positions (chance decodability, 0.504), same layer —
  **separated by POSITION, not by layer.** The weak (inconclusive)
  consumption trace at concept positions and the strong class-dependent
  response at the L53 gauge region (0.792, descriptive) are consistent
  with attention routing content between the two sites.

Status of this cell: it was in the registered battery as a descriptive
Q1-map cell, not a decision cell, and this is its first unblinding. To
CLAIM position-separated staging it needs one confirmatory registration
(J2 candidate: fresh 96-row replication + a position sweep — final
token vs concept mentions vs random positions × layers {43, 53} — and
the attention-routing readout). Until then it is a strong hypothesis
with one clean data point.

## Consequences

- The J-space/consumption-staging hypothesis (qwen_jspace_connection.md)
  is NOT the primary explanation — position separation is simpler and
  now better supported. The J-reading remains live only as the account
  of the weak concept-position trace; demoted accordingly.
- The next-paper opening experiment updates from "Q1 information map +
  J column" to **J2: confirm position-separated staging, then test
  routing** (does blocking attention from concept positions to the
  final-token staging site kill the gauge? does the repair write at
  concept positions flow TO the staging site?).
- Current paper: §6's mystery sentence can now say — hypothesis-flagged,
  one line — "preliminary evidence suggests the information sits at a
  different POSITION (final token) at the same layer, i.e., staging and
  consumption sites are position-separated in Qwen where Gemma
  co-locates them."
- Explainer's "open hunt" paragraph: update to "we found the shelf."
