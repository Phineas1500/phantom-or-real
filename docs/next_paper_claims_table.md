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
| 7 | Exhaustive necessity over tested routes is null: masking the hint span costs nothing (natural decode attention to hint is 0.5%), the combination arm (masking x reversion) survives at 0.981, and KV transplants are insufficient with attention verifiably flowing. Carrier = unpatched layers; commitment is multiply realized across layers. | landed (survives-both bin) | jobs 457002 + 457009; docs/kv_hint_span_27b_property_summary.md | the token route is ruled out on property; remaining necessity question is layer-exhaustive ablation (future work) |
| 8 | Commitment, not spotlight — and the focus state has a COMPACT CAUSAL CORE: rank-4 PCA add at L30 alone repairs +0.260 CI [+0.115,+0.423] = 104% of the subset-replacement effect (pre-registered reading rule: compactly structured), where rank-1 was null at every scale (x1/x2/x4 monotone harmful) and rank-full ADD was null. Foreign deltas never misdirect. | landed (compactly-structured bin) | jobs 457002 + 457009; docs/kv_hint_span_27b_property_summary.md | geometry rider queued: project the rank-4 L30 components onto INLP stack + Gemma Scope decoders |
| 9 | Geometry: the causal focus content lies outside the readable correctness subspace — survives restriction, quantified. | landed | restricted deltas project at 0.013-0.014 vs empirical null 0.040+/-0.009 (z~-2.8 per delta; 86% of 360 per-position deltas below null-2SD) | one flagged mechanism-speculation sentence allowed |
| 10 | Unification: each prompt format stores selection state where its candidates live (MCQ at option tokens — context patch null; hints at concept mentions — patch repairs). | landed | jobs 456913 + 456990/456999 read jointly | wording only; no new experiment |
| 11 | Cross-model robustness of the necessity/epiphenomenality result (Qwen). | queued — CRITICAL PATH | needs HF-hooks erasure variant | reviewer asymmetry: readability is claimed cross-model (claim 1), so epiphenomenality must be too, or claim 1's cross-model sentence gets scoped |
| 12 | Cross-task: the behavioral story replicates (recognition gap 33/48; candidates 1.000; hint-first 0.875) but context-encoding patch accessibility does NOT (full patch +0.039 null vs property +0.491; clean random control). Pre-registered bin: claims 5-6 scope to property; pathway-mix account pre-registered with its rival (layer-mismatch) and separation logic; null instrument-verified (span audit 16/16, delta norms normal, config from run log). | landed (task-dependence bin) | docs/subtype_localization_patch_27b_summary.md; job 457005 | predicted pattern = cross-task ROUTE dissociation (not double); subtype gold-KV + capture ladder + property-positive-control rider queued |

Negative-results table (for the appendix): CAA/raw steering, optimized
vectors, DAS L45/L50, AtP/exact patching, raw-z decode gates, prompt-margin
gated decode correction, recognition-context patch, block-mean delta add,
cross-row block-delta transplant, attractor-strength account — each with its
control set and the post-hoc explanation the focus-state account provides.

Scope rule: claims 7-9 are filled by jobs already queued or saved states;
anything not in this table goes to future work, not into the draft.
