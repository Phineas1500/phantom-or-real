# Item WY — Qwen at Gemma's relative depth on a fresh frame (20th + 21st predictions CONFIRMED), and the depth ladder of the frozen hint-delta write on both models (2026-09-03)

Registered: docs/causal_handle_directions.md item WY (before any data).
Frame: a third real-text draw (800 rows, seed 20260858, zero overlap with
all prior frames). Stage 1 (Qwen grade_hint **job-nfejy**, $1.94): std
0.537, closed-book 0.440, hint-first 0.710; doc-dependent 140;
hint-repairable ∧ doc-dependent **36 (25.7%)**, 23 reading-driven;
cross-fit pins A/B 18/18. Tie-break gauge: the WQ L48 probe fit on the
WF frame, frozen. Stage 2: Qwen L31 registered pair **job-4ukdh /
job-bb9gw**; Qwen ladder L19/25/37/43/49 (10 jobs, 5 resubmitted after
full-disk failures); Gemma ladder L15/20/25/35/40/45 on the 59 WX rows
(12 jobs, 5 resubmitted). WY cost $54.30 over 35 jobs, of which $18.7
went to 10 jobs that died at model download on machines whose disk was
full (a platform fault: the shared cache guarantees 32 GiB and both 27B
checkpoints are 54 GB; without the cache mount the same machines' own
disks were also full). Readers: `scripts/wikihop_wl_gates.py --frozen`
(→ `docs/wikihop_wy_write_gates.json`, `docs/wikihop_wy_{gemma,qwen}_L*_gates.json`),
`scripts/wikihop_wo_gates.py --tie-key primary_L48` (→ `docs/wikihop_wy_gates.json`);
ladder table `docs/wikihop_wy_ladder.json`.

## Registered: Qwen3.5-27B at L31 on 36 fresh rows (delivery valid on 3,408 branches)
Text ceiling 0.795 (+0.771 [+0.698, +0.840]).
| gold-address frozen write | gold rate | dP vs baseline 0.024 | specificity |
|---|---|---|---|
| 1× | 0.194 | +0.170 [+0.094, +0.253] | +0.172 [+0.097, +0.256] |
| **2× (pinned)** | **0.396** | **+0.372 [+0.253, +0.497]** | **+0.368 [+0.250, +0.491]** |

**20th prediction CONFIRMED — QWEN-WRITE-TRANSFERS-AT-GEMMA-DEPTH.** On
rows the frozen direction never saw, the write at relative depth 0.48
repairs Qwen as strongly as it repairs Gemma (+0.360 on Gemma's real-
text rows), with the same specificity. WQ's descriptive +0.255 on 27
rows was, if anything, an under-estimate.

| selector on identical 2× branches (19.7 per row) | select | gold argmax | vs baseline 0.024 | vs random 0.061 |
|---|---|---|---|---|
| **L48 gauge-select — REGISTERED** | **0.188** | 13/36 | **[+0.062, +0.281]** | **[+0.030, +0.239]** |
| output-first (answers-fired, L48 tie-break) — descriptive | 0.111 | 4/36 | [+0.003, +0.188] | [−0.033, +0.151] |
| oracle | 0.396 | | | |

**21st prediction CONFIRMED — QWEN-PROBE-SELECTOR-CLOSES-THE-LOOP** (47%
of the oracle). Decomposition of the output-first failure: gold branch
out-fired by a wrong branch on 15/36 rows, never fired 13, tied-lost 4,
selected 4; the gold branch is fully accepted on 8 rows while some wrong
branch is fully accepted on 32 — the same "beaten" regime as WQ. On
Qwen the probe wins and the acceptance signal loses, twice on disjoint
frames; on Gemma the reverse. Which selector works is a property of
the model; the write is not.

## The depth ladder (pre-named descriptive): the frozen hint-delta write at 2×
| model | layer | relative depth | rows | gold rate | dP [95% CI] | specificity [95% CI] |
|---|---|---|---|---|---|---|
| Gemma-3-27B | L15 | 0.24 | 59 | 0.127 | +0.110 [+0.042, +0.186] | +0.110 |
| Gemma-3-27B | L20 | 0.32 | 59 | 0.424 | +0.407 [+0.284, +0.530] | +0.401 |
| Gemma-3-27B | **L25** | **0.40** | 59 | **0.466** | **+0.449 [+0.331, +0.568]** | +0.444 |
| Gemma-3-27B | L30 (WX) | 0.48 | 59 | 0.377 | +0.360 [+0.242, +0.483] | +0.346 |
| Gemma-3-27B | L35 | 0.56 | 59 | 0.034 | +0.017 [−0.034, +0.085] | +0.006 |
| Gemma-3-27B | L40 | 0.65 | 59 | 0.000 | −0.017 [−0.051, 0.000] | −0.035 |
| Gemma-3-27B | L45 | 0.73 | 59 | 0.000 | −0.017 [−0.051, 0.000] | −0.031 |
| Qwen3.5-27B | L19 | 0.30 | 36 | 0.132 | +0.104 [+0.035, +0.188] | +0.090 |
| Qwen3.5-27B | L25 | 0.39 | 36 | 0.194 | +0.167 [+0.087, +0.260] | +0.159 |
| Qwen3.5-27B | **L31** | **0.48** | 36 | **0.396** | **+0.372 [+0.253, +0.497]** | +0.368 |
| Qwen3.5-27B | L37 | 0.58 | 36 | 0.375 | +0.347 [+0.236, +0.458] | +0.351 |
| Qwen3.5-27B | L43 (WQ's layer) | 0.67 | 36 | 0.194 | +0.167 [+0.090, +0.250] | +0.161 |
| Qwen3.5-27B | L49 | 0.77 | 36 | 0.028 | 0.000 [−0.031, +0.028] | −0.016 |

Every delivery audit valid. Both models carry a **band**, about a
quarter of their depth wide, in which one frozen direction at the
candidate's address repairs at +0.35 to +0.45, with shoulders at about
+0.1 to +0.17 and nothing beyond. Gemma's band is L20–L30 (0.32–0.48),
peak L25 — the chain's L30 sat at its upper edge, and the class-mean
sweep that "failed everywhere" (item W) failed because the vector was
wrong, not the depth. Qwen's band is L31–L37 (0.48–0.58), peak L31;
WQ's registered L43 (0.67, Qwen's class-mean carrier) is on the shoulder
at +0.17, which is exactly what WQ measured (+0.162). The pre-named
question — does each peak sit near 0.5? — gets a qualified yes: the
bands overlap at 0.48, but Qwen's is shifted about a tenth of depth
deeper than Gemma's, and neither peak is where the class-mean
correctness carrier lives.

## Verdict
**QWEN-WRITE-TRANSFERS-AT-GEMMA-DEPTH / QWEN-PROBE-SELECTOR-CLOSES-THE-LOOP**,
with the ladder as the mechanism sentence: the addressable hint-delta
signal occupies a mid-depth band in both models, model-specific by
about a tenth of depth, distinct from the correctness carrier.

## Program tally after WY (and WT)
22 registered directional predictions: **19 confirmed**, 3 not (13th,
14th, 19th). WikiHop chain W → … → WQ → WT → WY complete; ≈ $105
across 88 H100 jobs ($18.7 of it lost to platform disk failures).
