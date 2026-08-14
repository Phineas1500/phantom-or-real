# Qwen3.8-27B day-one smoke — inference works, and the recognition gap exists in a linear-attention hybrid (hint lift +0.575)

Reconnaissance, registration-free. givemeanode H100 batch jobs
job-k4mn5 (v1) + job-ayipz (v2), mission
https://givemeanode.com/missions/qwen38-smoke, 2026-08-14 — the day of
the model's release; total ≈$2. 100 seeded stage-1 property rows
(50/50 h3/h4, seed 20260814), unhinted + `hint_concept_first` arms,
k=4 at temp 0.7, plain transformers bf16 on one H100 (the pinned
Scholar/vLLM stacks predate the release). Raw generations scored
locally with the house parser. Row-level:
`results/qwen38_smoke/qwen38_smoke_v2.jsonl`; job entrypoint
`scripts/qwen38_smoke_job.py`.

## Facts established

- **Model**: Qwen/Qwen3.8-27B — Apache 2.0, ungated, multimodal
  wrapper; hybrid Gated-DeltaNet + Gated-Attention per the card. The
  loaded text tower reports class `Qwen3_5ForCausalLM` (so the pinned
  phantom env's transformers 5.5.4 may load it — untested; relevant if
  a bridge ever wants Scholar hooks).
- **v1 lesson (96-token budget): thinking-mode default.** All outputs
  were task-engaged CoT that exhausted the budget before any final
  hypothesis (25–41% parse-fail, all-zero scores). Not a capability
  fact — a harness fact.
- **v2 (`enable_thinking=False`, 640-token budget)**: parse 1.2%/0.0%;
  outputs are clean bare hypothesis lines.

## The numbers (100 rows, k=4)

| arm | P(strong) | by height |
|---|---:|---|
| unhinted | **0.307** | h3 0.370 / h4 0.245 |
| hinted (concept-first) | **0.767** | — |
| hint lift on the 63 unhinted-failing rows | **+0.575** | — |

## Reading

1. **Normal inference: yes** — 800 generations in 120s on one H100
   (~400 gens/min at 640 tokens), no special support needed beyond
   current transformers.
2. **Familiar accuracy regime**: unhinted 0.307 sits near Qwen3.5's
   0.352 and shows the same height gradient.
3. **The recognition gap exists in the hybrid**: +0.575 hint lift on
   failing rows, matching Qwen3.5's registered G0 lift (+0.523) —
   the behavioral precondition of the entire causal program is present
   in a fundamentally different sequence-mixing architecture. A
   Qwen3.8 bridge (G-series analog: gauge probe → carrier sweep →
   channel rank → content battery) is a live candidate, with its G0
   behavioral gate effectively pre-screened and the H100 lane as its
   native venue.

## Caveats

- Reconnaissance scale (100 rows, no registration); thinking DISABLED
  — the thinking-mode regime is its own (interesting) variable, not
  probed here.
- The multimodal wrapper and hybrid layers mean hook-point naming and
  any activation work need a fresh look (64 layers, hidden 5,120; the
  DeltaNet blocks carry recurrent state — position-based staging
  analyses may behave qualitatively differently, which is precisely
  the scientific draw).
