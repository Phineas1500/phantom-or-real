# Phantom or Real? → Gauge, Not Lever

Mechanistic-interpretability study of whether a language model's internal
"I am about to get this right/wrong" signal is causally load-bearing — or
just a gauge on the dashboard. Models: **Gemma 3 27B-IT** (primary) and
**Qwen3.5-27B** (cross-model port). The project began as a stage-1
behavioral/probing pipeline on depth-controlled ontology-reasoning tasks
(InAbHyD; see `docs/BEHAVIORAL_DATA_PLAN.md`), grew into a pre-registered
causal program on those tasks (34 verdicts, closed 2026-08-19), and then
took the repair loop to natural text: WikiHop, HotpotQA, SQuAD,
knowledge-conflict frames, and long table documents (46 registered
directional predictions: 29 confirmed, 15 not, 1 intermediate, 1 not
read; ≈ $211 / 181 H100 jobs).

**Status (2026-10-30): all experiments are closed. The current milestone
is the paper**: `paper/gauge_not_lever.tex` covers the InAbHyD program
(through 2026-08-19); the natural-text, table, and neuron-decomposition
results are not yet folded in.

Plain-language explainer of every finding, for newcomers:
`docs/artifacts/phantom_or_real_findings.html` (published as the
"Phantom or Real?" artifact).

## The thesis in one paragraph

In a model that visibly predicts its own failures, the readable
correctness signal and the causal repair channel are **different
objects**: the entire readable subspace can be deleted without behavioral
cost, while repair flows through a low-rank channel at concept-mention
tokens that the readout does not occupy — gauge and lever, anatomically
separated. On natural text the lever survives as a single frozen
"attend to this" direction whose target is set by where it is written,
written by a handful of attention heads and read by attention; what does
not survive is blind deployment, because knowing which answers are wrong
(and where to write) is the bottleneck.

## Headline findings (all pre-registered, row-cluster bootstrap CIs)

### Part 1 — InAbHyD, Gemma 3 27B (the paper's main battery)

- **The gauge**: correctness is linearly readable pre-generation
  (AUC 0.90 at L53; Qwen 0.94), robust to name-scrambling, beats metadata.
- **The gauge is not necessary**: erasing everything linearly readable —
  the full rank-9 INLP stack, five layers, every token — costs nothing
  (+0.047 [−0.070, +0.203]), while matched-rank random erasures are
  catastrophic (−0.38). (Item D)
- **The lever is elsewhere**: a rank-8 channel at concept-mention tokens
  (L30) repairs held-out failures (+0.245) and passes the full
  specificity ladder — random subspaces null, matched noise destructive.
  (Item C)
- **The necessity dissociation**: a natural correct-minus-incorrect
  class-mean is null at full dimension (+0.043) and potent only when
  concentrated into the channel (+0.341) — and only with real labels
  (shuffled ≈ 0; sign-flipped actively harmful). (F(ii), F(ii)-b)
- **The inversion**: the channel that causally controls success is not
  itself decodably privileged (probe inside the random-subspace null).
  (F(i))
- **Answer-free repair**: a donor-free variant reaches **+0.399** on a
  0.12 baseline where majority-vote self-consistency scores 0.000.
  Position addressing remains the answer-adjacent ingredient (item H:
  the vector is a positional commitment command). (F(ii)-b/c, H)
- **The loop closes in the sandbox** (L-series): an answer-free
  write-then-select loop matches the oracle (+0.241 / +0.390); a frozen
  direction transfers (+0.498 / +0.513). (L″-r, L‴)

**Qwen3.5 27B (G/J/N series):** the motif transfers, the coordinates
don't — carrier at rel. depth 0.67 (vs 0.48), channel rank ~16 (vs 8).
The dissociation inverts: Qwen's *raw* class-mean repairs (+0.120
[+0.016, +0.224]) and projection dilutes it (Gemma concentrates, Qwen
distributes); label battery passes. Gauge and lever separate by
position in Qwen, by subspace in Gemma.

### Part 2 — natural text (W-series, Gemma unless noted)

- **A fixable quarter**: ~27% of document-dependent WikiHop failures flip
  under a one-line concept hint (24–26% on HotpotQA/SQuAD); the rest are
  firm beliefs. All repair numbers are on this slice. (WH, WR, WP)
- **One frozen hint-delta direction** at L30 on the candidate's mentions
  repairs +0.33 / +0.36 on 0/8 rows, replicated on a disjoint frame;
  identity is carried by the address. +0.65 when entities are renamed
  (memory removed). (WL, WF, WX, WA)
- **The loop closes on natural text**; the best selector is the model's
  own acceptance of the nudge (0.362 vs the probe's 0.218; oracle
  0.686). (WO)
- **Mechanism**: the write routes attention onto the addressed span (×8
  at L32); flipped and random vectors repair and route nothing; works in
  a mid-depth band (Gemma L20–30, Qwen L31–37). (WT, WV, WY)
- **Transfer**: Qwen write +0.372 (selector must be a probe there);
  WikiHop's vector beats each task's own on HotpotQA/SQuAD. (WQ, WY, WP)
- **Blind deployment is small**: always-answer breaks ~10% of correct
  rows; frame net ≈ 0 on real WikiHop. A groundedness-gated loop gains
  +0.050 / +0.033 blind on conflict frames (WK′, WS), but a one-sentence
  follow-the-documents instruction gains +5–8 and CAD +10–11. Answer-token
  probes are a swap artifact (0.87 on real text); P(True) at chance.
  (WD, WE, WK, WI, WZ, WC)

### Part 3 — long table documents (WM/WN/WW/WW′)

- The residual write corrupts table rows (−0.42 aimed at the gold row).
- Attention steering in 14 retrieval heads at the gold row repairs
  **+0.625**, the largest repair in the program; control inert.
- Aim is the wall: retrieval heads put the gold row top-3 on 52% of
  fixable failures (BM25 23%), but steering at their picks leaves answers
  unchanged, and the model's own pick names gold 15/90.

### Part 4 — neuron decomposition (NX, 2026-10-15)

- The hint delta at the address is written by **8 attention heads**
  (L23–29, same set in both shards); writing only their contribution
  repairs **+0.479** (> the frozen vector by +0.123); random heads +0.025.
- Top-64 aligned MLP neurons +0.013; freezing MLPs keeps 104% (read by
  attention, QK-direct). Aligned neurons are *less* legible than random
  ones on the explanation test (3/32 vs 20/32). Prompted by Goodfire's
  "Neural Networks Are Already Clean".

## Where to look

| what | where |
|---|---|
| Plain-language explainer (all parts) | `docs/artifacts/phantom_or_real_findings.html` |
| Visual status overview (living doc) | `docs/causal_status_overview.html` |
| Paper draft (InAbHyD program) | `paper/gauge_not_lever.tex` |
| All pre-registrations (decision rules before unblinding) | `docs/causal_handle_directions.md` |
| InAbHyD per-item verdicts | `docs/*_summary.md` (C, D, E, F-series, H, G-series, J-series); sandbox loop: `docs/selfaddress_*_summary.md`, `docs/loop_screen_summary.md` |
| Natural-text verdicts | `docs/wikihop_*_summary.md`, `docs/wikihop_levers_summary.md` |
| Table-chain verdicts | `docs/wm_summary.md`, `docs/wn_summary.md`, `docs/ww_summary.md`, `docs/wwp_summary.md` |
| Neuron decomposition | `docs/nx_summary.md`, `docs/nx_gates.json`, `docs/nx_explanations.json` |
| Claims × evidence × caveats | `docs/next_paper_claims_table.md` |
| Paper outline / contribution ledger | `docs/paper_outline_final.md`, `docs/contribution_ledger_2026-07-05.md` |
| Statistical hardening (BCa/LOO/MDE) | `docs/stat_hardening_2026-07.md` |
| Running lab notebook | `docs/REPORT_NOTES.md` |
| Bibliography (verification-graded) | `docs/bibliography.md` |
| Artifact page sources | `docs/artifacts/` |

## Methodology notes

Every causal item was **pre-registered with branch-complete decision
rules before unblinding**; see the tallies above. All
effect sizes are pooled P(strong-correct) deltas vs in-job baselines
with row-cluster bootstrap CIs (10k, fixed seed); claim-bearing
contrasts near zero additionally get BCa + leave-one-row-out. Job
determinism is seed-exact: replication gate arms reproduce prior jobs
token-for-token (verified 576/576 generations across three jobs).
Controls used throughout: matched-norm noise, random subspaces/stacks,
shuffled-label class-means, sign-flips, dose ladders, sampling
baselines (self-consistency, best-of-N), collateral slices.

## Stage 1 — behavioral/probing pipeline (the dataset everything runs on)

Generates InAbHyD reasoning examples, runs Gemma 3 4B/27B (+ Qwen)
inference, scores correctness, annotates structure, ships labeled JSONL
(44k rows/model). Full spec: `docs/BEHAVIORAL_DATA_PLAN.md`; operating
layout, credentials, and common commands: `CLAUDE.md`.

```bash
# One-time setup
conda env create -f environment.yml            # or environment.lock.yml for exact pins
conda activate phantom

# You need a clone of beyond-deduction somewhere — options:
#   - set BD_PATH=/path/to/beyond-deduction
#   - or ln -s /path/to/beyond-deduction third_party_beyond_deduction
#   - or clone it to ~/beyond-deduction

# Generate the full dataset (~44k examples)
python -m src.generate_examples --counts full

# InAbHyD causal harnesses: scripts/stage2_*.py (SLURM sbatch per item).
# Natural-text / table / NX jobs: scripts/wikihop_*_job.py, scripts/nx_*_job.py,
# run as givemeanode H100 jobs; *_context.sh builds each job's input tarball and
# *_gates.py reads the registered verdict from the returned rows.
```

## What's next

1. **The paper**: fold Parts 2–4 into `paper/gauge_not_lever.tex` and add
   related work published since August (arXiv 2605.05715, 2609.36138,
   2610.20067, PatchBench).
2. Optional runs: whether Qwen also writes its lever with a few heads
   (NX port); the Qwen3.8 bridge on H100 (entry gates pre-cleared, see
   `docs/qwen38_gauge_summary.md`).
