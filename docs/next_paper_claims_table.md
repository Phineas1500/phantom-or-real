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
| 7 | Necessity at the same positions: removing concept-position hint states collapses the hint repair (vs random-position control); reversion-to-attractor scored. | queued | job 457002 arms 2-4 (in-job hinted baseline, reverse subset, reverse random) | conditional link pre-registered: a null here routes to the KV job (decode-time attention account) |
| 8 | Spotlight vs commitment: whether the localized state transfers across rows (movable attention operator) or is concept/context-bound. | queued | job 457002 arms 5-9 (restricted ladder, rank-1 + per-position spotlight, complement) | outcome grid + ladder logic pre-registered |
| 9 | Geometry: the causal focus content lies outside the readable correctness subspace. | provisional | block-mean deltas project below chance onto the 9-direction INLP stack | MUST survive recomputation on concept-position-restricted deltas (states saved by 457002) before becoming prose |
| 10 | Unification: each prompt format stores selection state where its candidates live (MCQ at option tokens — context patch null; hints at concept mentions — patch repairs). | landed | jobs 456913 + 456990/456999 read jointly | wording only; no new experiment |
| 11 | Cross-model robustness of the necessity/epiphenomenality result (Qwen). | queued | needs HF-hooks erasure variant | not started |
| 12 | Cross-task robustness of the localization result (subtype). | queued | task 10: Modal behavioral prerequisites, then one concept-position patch job | not started; first reviewer question for any localization claim |

Negative-results table (for the appendix): CAA/raw steering, optimized
vectors, DAS L45/L50, AtP/exact patching, raw-z decode gates, prompt-margin
gated decode correction, recognition-context patch, block-mean delta add,
cross-row block-delta transplant, attractor-strength account — each with its
control set and the post-hoc explanation the focus-state account provides.

Scope rule: claims 7-9 are filled by jobs already queued or saved states;
anything not in this table goes to future work, not into the draft.
