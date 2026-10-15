# Item NX — does the frozen write decompose into a few legible MLP neurons? (SCOPE DRAFT, 2026-10-15 — not registered)

Prompted by Cammarata, "Neural Networks Are Already Clean" (Goodfire,
2026-10-15): ~90% of Qwen3-32B neuron poles have one main job, probes
"belong at the end", superposition is rare. If that holds for Gemma-3-27B,
the frozen hint-delta direction u that repairs WikiHop failures (WX +0.360
at L30 × 2×; WV: flipped/random inert; WT: routes L31–33 attention onto the
written span ×8) should be (i) produced by, and/or (ii) read through, a
small set of MLP neurons with nameable jobs. This item tests both, on rows,
pins, donors and controls already on disk.

Prior from our own data (cuts against the essay): on InAbHyD the rank-core
direction was dictionary-visible but **massively redundant, not
sparse-small** in Gemma Scope residual SAEs (top-20 rows ≈ 1% of overlap
mass; docs/REPORT_NOTES.md). SAE latents are not neurons, so this is a
prior, not a test.

## Frame (all frozen)
- Gemma-3-27B bf16; the 59 WX rows, cross-fit A/B exactly as WX (neurons
  are SELECTED on the donor shard, TESTED on the held-out shard).
- Write: L30, rung 2×, donor norm target N, at whole-word candidate
  mentions; gold address + the three seeded non-gold addresses WX fired;
  k = 4 samples; WX baseline / WX write / WV flip+random reused as arms.
- Neuron = one coordinate of the MLP down_proj input (GeGLU product
  gate·up), signed. ~21.5k per layer × 62 layers.

## Stage 1 — source attribution (prefill only, donor shards; ~$2)
Hint-first vs std prefills (the forwards WX already uses to fit u). At the
gold-mention positions, per layer ℓ ≤ 30, per neuron i:
c_i = Δact_i · (W_down[:, i] · u), averaged over donor positions; and the
same for each attention head's output (o_proj slice) so the accounting
is complete, not neuron-only.
Reported: share of Σ projection onto u carried by MLP neurons vs heads vs
embeddings; concentration curve (top-k share for k = 16 … 4,096); the
massive-dim share of u (dims 104, 2733) and of each top neuron's write.

## Stage 2 — sufficiency: write the neurons instead of u (behaviour; ~$12)
Arm NEURON-k: at the candidate's mentions, add Σ_{i∈top-k} Δ̄act_i ·
W_down[:, i] at each neuron's own layer (Δ̄ = donor mean), rescaled so its
L30 projection onto u matches the WX write's. k ∈ {16, 64, 256, 1024}.
Arm RANDOM-k: k neurons drawn from the same layers, same total norm,
seed fixed at registration (two draws).
Arm u-minus-massive: u with dims 104/2733 zeroed, renormalized (checks
that the lever is not riding Gemma's massive coordinates).

**Prediction 43 (registered direction): NEURON-64 at the gold address
repairs at ≥ half of WX (dP ≥ +0.18, CI lower bound > 0) and beats
RANDOM-64 row-paired (CI > 0).** Failure reading: DISTRIBUTED-SOURCE —
the lever is a direction many neurons contribute to, not a handful.
Descriptive: the k-ladder, specificity at non-gold addresses, top-k
attention routing at L32 (WT reader, ≤ 1,600-token rows).

## Stage 3 — necessity downstream: is the write read by neurons or by attention keys directly? (~$5)
Gate arm FREEZE-MLP: WX write on, but the MLP output at the written
positions in L30–L35 clamped to its no-write value. If the repair
survives, the route is residual → QK directly and the neuron question is
answered "no neuron in between" (reported as such; P44 then not read).
Otherwise: rank written-position neurons in L30–L35 by write-induced
Δact × attribution to the L32 span-attention gain (donor shard); clamp
top-64 to no-write values under the write vs RANDOM-64.
**Prediction 44: clamping the top-64 downstream neurons removes ≥ half of
the WX repair and more than RANDOM-64 (row-paired CI > 0).** Failure
reading: QK-DIRECT or DISTRIBUTED-READ, by the gate.

## Stage 4 — legibility, the essay's own test (~$3 GPU + ~$5 judge)
For the top-32 neurons from each of stages 1 and 3, plus 32 random
neurons from the same layers: top activations over ~2M tokens (WikiHop
documents + a generic web sample, fixed at registration). An explainer
model (gpt-5.4-mini) writes a one-line explanation from top contexts; a
SEPARATE call that sees only the explanation writes 10 positives, 10
near misses, 10 distant texts; Gemma scores them. Win = positives fire
more than near misses (the essay's criterion; its benchmark 164/183 win).
Departure from the essay, by design: explainer and tester are separate
calls, and near misses are also drawn from the neuron's own top contexts
of OTHER neurons in the set (harder than self-written ones).
**Prediction 45: the stage-1 top-32 win at a rate ≥ 0.75 and above the
random-32 rate (two-proportion CI > 0).** Descriptive: what the
explanations say (pre-named bins: entity/copy, document reference,
question relation, answer-format, other).

## Readings
- P43 + P44 + P45 confirmed → the lever is a few legible neurons; the
  paper's mechanism sentence gets a neuron-level account and supports the
  essay on a reasoning task.
- P43 not, P45 confirmed → neurons are legible but the lever is a
  distributed direction across them (the InAbHyD SAE prior).
- FREEZE-MLP gate passes → write is read by attention keys directly;
  neurons are not the interface (a counterpoint to the neuron-first view
  for attention-mediated levers).

## Cost and logistics
≈ $27 expected (≈ 16 jobs), cap $45 with the usual full-disk retries
(no cache mount; ~$2/failure). Two-shard jobs on givemeanode H100 as in
WX/WV; stage 4 judge on OPENAI_API_KEY_GPT. Reuses wikihop_wt_job.py
hooks (write delivery + delivery audit), WX pins, WV control records.
New code: stage-1 attribution capture, neuron-write / clamp hooks (with
a targets-fired delivery gate per the steering-delivery lesson), a
fake-model dry run of every arm, and one reader. ≈ 1 day to build,
≈ 1 day of jobs.

## Open choices before registration
1. Write layer: keep L30 (all WX/WV/WT records pair row-by-row) or move to
   the ladder peak L25 (+0.449, but no paired controls on disk).
2. k = 64 as the registered size, and the ≥ half-of-WX bar.
3. Whether stage 4 runs regardless of stages 2–3 (cheap; recommended).
