# Qwen3.8-27B gauge hunt — the gauge exists in the DeltaNet hybrid at near-pure-attention strength (AUC 0.86), architecture-neutral, familiar staging

Registration-free reconnaissance, step two after the day-one smoke
(`docs/qwen38_smoke_summary.md`). givemeanode H100 batch jobs, mission
https://givemeanode.com/missions/qwen38-gauge, 2026-08-14, ≈$2.50 total:
job-wxzwp (labels 2,000 + hinted 500 + 65-layer capture, 18 min) and
job-33j8u (greedy-label rider, 12 min; job-7rrfm was a `python`-vs-`python3`
no-op failure, $0). 2,000 seeded stage-1 property rows (1,000 h3 + 1,000 h4,
seed 20260815), thinking disabled throughout (smoke-v2 regime). Probes trained
locally: per-layer logistic regression on the final-prompt-token residual
stream, StandardScaler, 3-fold stratified CV (seed 20260815). Entrypoints
`scripts/qwen38_gauge_job.py` (in-job) + `scripts/qwen38_gauge_analysis.py`
(local); row-level data and ladders in `results/qwen38_gauge/` (the 1.3 GB
fp16 capture `gauge_hidden_final.npy`, shape [65, 2000, 5120], stays
untracked on scratch; artifact sha256
e9af8ca757280fd5b560e2ed08e9c0746b61d6b4216a712aa45dc185a2120e24).

## Behavioral, at G0 scale (n = 2,000 / hinted n = 500)

| quantity | value |
|---|---:|
| unhinted P(strong), temp 0.7 k=1 | **0.309** (h3 0.361 / h4 0.257) |
| unhinted P(strong), greedy | 0.3035 |
| hinted (`hint_concept_first`) P(strong), k=4 | **0.8345** |
| hinted P(strong) on the 350 unhinted-failing rows | **0.8107** |
| temp-0.7 vs greedy label agreement | 0.7805 |

The smoke's numbers replicate at 20× scale (0.307 → 0.309) and the
recognition gap is, if anything, larger than Qwen3.5's: the concept hint
repairs 81% of failing rows.

## The probe ladder

Three ladders over all 65 depths (embedding + 64 layers):

| ladder | peak | where | attn mean | dnet mean |
|---|---:|---|---:|---:|
| temp-0.7 labels, C=1 | 0.668 | L54 | 0.601 | 0.600 |
| temp-0.7 labels, C=3e-4 (tuned) | 0.741 | L51–54 plateau | 0.667 | 0.667 |
| **greedy labels, C=3e-4** | **0.860** | **L53 (plateau L51–54 ≥ 0.858)** | 0.753 | 0.752 |

Shape (greedy ladder): monotone rise from 0.55 at L0 through 0.81 at L47 to
the 0.86 plateau at L51–54 (~82% relative depth), then genuine decline to
0.80 at the final layer. C-sweep at the peak plateaus at 1e-4–3e-4;
learning curve mostly flat by n=1,400.

## Findings

1. **The gauge exists in a linear-attention hybrid.** First measurement of
   upcoming-correctness readability in a DeltaNet-style architecture:
   AUC 0.86 at matched (greedy) label protocol — near Gemma's 0.90, below
   Qwen3.5's 0.94, with smaller probe-training data than either (residual
   gap is within plausible sample-size effects; the learning curve was
   still rising slightly at n=1,400 under weaker C).
2. **The label-noise lesson.** The first ladder read 0.741 and invited an
   "architecture-sensitive gauge" story. The confound was ours: temp-0.7
   single-sample labels agree with greedy labels on only 78% of rows, and
   state cannot predict sampling luck. Matching the stage-1 label protocol
   (greedy) recovered +0.12 AUC. Any cross-model gauge comparison must
   match label protocol first.
3. **Architecture-neutral readout.** At every depth band, probes read the
   residual stream after attention layers and after DeltaNet layers equally
   well (0.753 vs 0.752 mean; the interleave is 1:3). The correctness
   signal lives in the shared residual stream, not in a mixer-specific
   pocket.
4. **Familiar staging.** Peak at ~82% relative depth mirrors Gemma
   (L53/61 ≈ 87%) and the late-depth Qwen3.5 shelf — the gauge's
   depth-geography survives the change of sequence mixer.

## Consequences

- The Qwen3.8 bridge is fully de-risked at the behavioral and readability
  gates: recognition gap (+0.81 repair on failing rows) and gauge
  (0.86 @ L51–54) both present. A registered bridge would start at the
  carrier sweep / channel-rank stages, on the H100 lane, with hooks via
  plain transformers (`Qwen3_5ForCausalLM` text tower).
- The J1-analog (concept-position vs final-token staging under recurrent
  state) is the architecture-differentiating follow-up; the capture for it
  would need concept-position vectors, not just final-token.
- No registered claim moves; reconnaissance only, but seeded and documented
  for citation by any future registration.

## Caveats

- Labels are k=1 (greedy) — fine for probe training, but P(correct|state)
  calibration analyses would want k≥8 sampled.
- Single task (property), two heights, one capture position; thinking mode
  disabled — the thinking regime is unprobed.
- Probe ladder is 3-fold CV on 2,000 rows; the registered gauges were
  trained on larger stage-1 sets, so cross-model AUC comparisons remain
  approximate even at matched label protocol.
