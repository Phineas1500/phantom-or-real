# Next-Paper Claims Table (Pre-Registered)

Each claim mapped to landed or queued evidence. Written before job 457002
unblinded, per review: the composite job's outcome fills the last variable
rather than opening a new branch. Companion to the dashboard (evidence rows)
and `docs/causal_handle_directions.md` (designs and pre-registrations).

| # | Claim | Status | Evidence | Caveat / remaining |
| --- | --- | --- | --- | --- |
| 1 | Correctness is linearly readable from pre-generation activations, beyond metadata, across models and name-scrambling. | landed | ICLR-paper probe results; Qwen L53 raw AUC 0.940/0.920; scramble results | none new |
| 2 | The correctness readout direction is causally epiphenomenal: erasable at all readable layers with no behavioral cost, while matched control erasures are destructive. | landed | jobs 456912/456915-918 (property), 456963-966 (subtype): raw CI straddles zero both tasks; orthogonal CI excludes zero; P(correct\|parsed) separates precision from demolition | between-run control direction + within-run variance test queued (pre-registered prediction recorded); Qwen erasure queued |
| 3 | Correctness information is redundant (INLP barely decays) yet no readable axis is load-bearing: directionally epiphenomenal, informationally redundant. | landed | docs/probe_on_erased_activations_27b_property*.json | full-subspace LEACE erasure listed as future work, not claimed |
| 4 | The recognition-generation gap is a proposal failure localized to target-concept focus: recognition 14/14, candidates 0.839, concept-hint 0.955-1.000, self-critique +0.15, gold absent from 16-sample proposals on ~2/3 of rows. | landed | Modal behavioral suite artifacts (candidates, proposal distribution, self-critique, hint gradient v1/v2) | property/Gemma only; subtype behavioral replication queued (task 10) |
| 5 | The focus state is causally accessible: patching hint-conditioned context encodings repairs free-form generation (+0.491 CI [+0.28,+0.71] at k=8) against destructive matched noise. | landed | job 456990; concept-resolved row-paired analysis | misdirection does not transfer through residual patches (asymmetric controllability); KV job queued with gold-KV positive control |
| 6 | The focus state is positionally localized at concept-mention tokens (~6x per-token potency vs matched random positions); it is not captured by a uniform rank-1 block summary. | landed | job 456999; dilution caveat recorded — "not low-rank" NOT claimed | restricted-delta ladder + rank-k PCA gatekeeper in job 457002 + offline |
| 7 | Necessity of concept-position encodings: no detectable effect at ceiling (1.000 vs in-job hinted baseline; 112 generations cannot see sub-ceiling degradation). Conditional link 1 fired: a parallel pathway exists — candidates are the hint-span KV and the unpatched layers. | landed (ceiling-limited null, pre-registered routing) | job 457002 arms 2-4; summary doc | KV/hint-span job spec v2: decode-only masking + the combination arm (ablation x reversion) enumerates the carriers |
| 8 | Commitment, not spotlight: foreign concept-position deltas disrupt (-0.163) and never misdirect (donor-targeting <0.10); rank-1 unsupported at tested scales (x2 owed); decomposition superadditive (subset +0.250 + complement +0.077 < full +0.491). | landed | job 457002 arms 5-9; summary doc | x2 scale + layer-resolved rank-k at L30 (PCA top-1=49%) join the next job |
| 9 | Geometry: the causal focus content lies outside the readable correctness subspace — survives restriction, quantified. | landed | restricted deltas project at 0.013-0.014 vs empirical null 0.040+/-0.009 (z~-2.8 per delta; 86% of 360 per-position deltas below null-2SD) | one flagged mechanism-speculation sentence allowed |
| 10 | Unification: each prompt format stores selection state where its candidates live (MCQ at option tokens — context patch null; hints at concept mentions — patch repairs). | landed | jobs 456913 + 456990/456999 read jointly | wording only; no new experiment |
| 11 | Cross-model robustness of the necessity/epiphenomenality result (Qwen). | queued — CRITICAL PATH | needs HF-hooks erasure variant | reviewer asymmetry: readability is claimed cross-model (claim 1), so epiphenomenality must be too, or claim 1's cross-model sentence gets scoped |
| 12 | Cross-task robustness of the localization result (subtype). | behavioral half landed | docs/subtype_recognition_gap_27b_manifest.json: recognition gap 33/48; on 16 gap rows baseline 0.000, candidates 1.000, hint-first 0.875 | concept-position patch job queued; consumes gap_row_indices |

Negative-results table (for the appendix): CAA/raw steering, optimized
vectors, DAS L45/L50, AtP/exact patching, raw-z decode gates, prompt-margin
gated decode correction, recognition-context patch, block-mean delta add,
cross-row block-delta transplant, attractor-strength account — each with its
control set and the post-hoc explanation the focus-state account provides.

Scope rule: claims 7-9 are filled by jobs already queued or saved states;
anything not in this table goes to future work, not into the draft.
