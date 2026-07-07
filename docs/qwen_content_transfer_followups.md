# Why G4′ Might Have Under-Delivered — Protocol Audit & Follow-Up Menu

Written 2026-07-06 late evening, after the G4′ null. Motivation: the
null is unlikely to mean "no content transfer." Three signals say the
effect is real but starved: (1) sign-flip CI < 0 (registered
prediction confirmed — the axis polarity is right); (2) per-row
structure — every G4′ proj responder is a rank-16 responder (5/5
subset, per-row corr +0.73 with the hint-delta arm; row 3543:
proj +0.625 ≥ rank16 +0.500), i.e., the class-mean moves the SAME rows
through the SAME channel, just less often; (3) the real direction is
the only non-gate arm with a positive point estimate.

## The audit: every protocol difference vs Gemma F(ii) (+0.341)

| dimension | Gemma F(ii) | Qwen G4′ | expected cost |
|---|---|---|---|
| class-mean source rows | **96** balanced (stage-1 labels, no screen) | **40** (majority-of-4 screened) | mean noise ×1.55; in 5k-dim, a 40-row mean is substantially less aligned with the true class direction, and projection cannot remove within-subspace noise |
| norm convention | **per-position, per-row** target = that row's LOO recon norms | **one pooled scalar** (23.8) for all rows/positions | energy misallocated across positions; Gemma's convention puts amplitude where the channel naturally operates |
| basis rank / channel coverage | rank 8 ≈ **78%** of carrier | rank 16 ≈ **67%** of carrier (G3′ curve: 95% needs rank 64) | ~⅓ of the channel invisible to the projection |
| amplitude regime | matched to recon norms (sufficient in Gemma) | same — but G3′ showed **Qwen rewards fuller amplitude** (fixednorm-8 at full-delta norm beat natural-norm rank-8) | possibly under-driven |
| test rows / power | 26 (2 shards), observed +0.341 | 15, MDE 0.12, observed +0.050 | an effect ⅓ of Gemma's is invisible |
| F(i) decodability diagnostic | run (null — informative) | **never run on Qwen** | unknown whether natural-outcome info even sits at L43 concept positions |
| capture frame | row's own unhinted prompt, gold-concept positions | same | — |
| layer | L30 (rel 0.48) | L43 (rel 0.67) | matched to each model's carrier — correct by design |

None of these differences was wrong to make — each was registered — but
they all point the same direction, and they compound. A ×0.5 from mean
noise, ×0.85 from channel coverage, and ×0.9 from norm allocation turns
a Gemma-sized +0.34 into ~+0.13 — near what we saw against an MDE the
design couldn't beat.

## Follow-up menu (ranked; G6 = one job, ~2 h on 2× A40)

1. **Protocol-matched class-mean (the big one).** 96 source rows
   (48 correct + 48 incorrect, stage-1 labels, NO majority screen —
   exactly Gemma's selection), per-position/per-row norm matching to
   the LOO recon (exactly Gemma's convention). Cheap: capture is
   forwards-only, no screening generations.
2. **Rank-64 projection arm.** G3′ says rank 64 carries 95% of Qwen's
   channel; project the class-mean there. If Qwen's content is as
   distributed as its channel, this is where it shows.
3. **Dose arm.** Protocol-matched vector at ×2 — the G3′ dilution
   lesson applied to content. The projection-headroom prediction says
   projected vectors tolerate amplitude that raw ones don't.
4. **Power.** Add 9 fresh test rows (24 total; the original 15 keep
   their exact seeds so both gates stay verbatim). MDE drops ~0.12 →
   ~0.09.
5. **Free rider (CPU, in-job):** the F(i)-analog on the 96 captured
   states — is the natural outcome decodable at L43 concept positions,
   in rank-16/64 slices vs random? Diagnoses whether the content is
   even present where we're reading it, independent of any repair arm.
6. **Controls:** shuffled-label d1 + sign-flip on the protocol-matched
   arm; unhinted + rank16 verbatim gates.

Suggested arm list (10): gates ×2 · protomatched_proj16 ·
protomatched_proj64 · protomatched_proj16_x2 · shuffled_d1 · signflip ·
class_mean_raw (96-row, for the dissociation rider) · rand_norm d1 ·
(one spare slot if the 24-row expansion is dropped).

## What this is and is not

- NOT a rescue mission for the current paper: the G4′ verdict stands
  as written; the paper's cross-model section does not change unless
  G6 runs and passes under its own registration.
- IS the cheapest decisive test of the user-facing question "should
  projection help in Qwen too?" — every suspect difference gets its
  own arm, so a pass localizes the cause and a full null (protocol-
  matched, rank-64, ×2 dose, 24 rows, MDE 0.09) would justify the
  stronger sentence "content transfer genuinely differs across models."
- Requires a fresh registration section (G6) before launch; decision
  rules should mirror G4′ (PRIMARY on the protocol-matched arm vs
  baseline AND vs shuffled family) with the rank-64 and dose arms as
  named secondaries.
