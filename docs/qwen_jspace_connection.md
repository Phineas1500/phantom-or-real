# The Qwen Mystery ↔ J-Space Connection (design note, 2026-07-07)

Prompted by the user reading the explainer's open question — "Qwen's
success/failure difference isn't linearly readable where the repair
channel lives (0.50 vs Gemma's 0.81), yet writing it there repairs" —
against Anthropic's global-workspace paper
(transformer-circuits.pub/2026/workspace) and Neel Nanda's review with
a Qwen 3.6 27B replication
(lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S).

## The hypothesis

The workspace paper's methodological core: linear probes on raw states
under-read staged content, because downstream nonlinearities transform
it; J-Lens (reading through the Jacobian — the model's own consumption
map) recovers what Tuned Lens misses. Applied to our G6/G6′ anomaly:

**Qwen's natural-outcome information at L43 concept positions may be
present but staged in consumption-readable (Jacobian-visible), not
linearly-probe-readable, form.** The raw class-mean write repairs
because addition speaks the downstream circuit's input language even
when a linear probe on states cannot see the class structure. Under
this reading, the Gemma/Qwen contrast reduces to one architectural
property: Gemma stages outcome content in linearly-readable,
concentrated form at the channel address (probe 0.807, rank-8 core);
Qwen in consumption-readable, distributed form (probe 0.504, content
past rank 16). "Concentrates vs distributes" and the workspace J-space
story may be two views of the same fact.

Why this is now testable in-family: Nanda replicated the J-space
machinery on **Qwen 3.6 27B** — Jacobians to the penultimate layer, 25
Pile prompts, 128 tokens, first-4-token skip — with most probing and
causal effects reproducing (weaker but positive). The structure exists
in this model family and the recipe is published and cheap.

## The experiment ladder (next-paper scope; register before running)

1. **J-aware F(i)-analog (the decisive first step).** Same 96 captured
   natural states at L43 concept positions (G6 capture, reproducible by
   seed). Compute per-position Jacobian-projected features (VJP/JVP
   products to the penultimate layer per Nanda's recipe — full J not
   needed; project onto the top-k J-image directions or use J·h
   features) and re-run the CV-AUC battery: raw states (known: 0.504)
   vs J-projected states, each against matched random-projection nulls.
   - Outcome (a): J-projected AUC ≫ 0.5 → warehouse never empty; it is
     written in a script only the model reads. Reframes G6′: the raw
     write works because it is in-language.
   - Outcome (b): still chance → content genuinely elsewhere; fall back
     to the Q1 information map with a J-axis added (layer × position ×
     {raw, J-projected}).
   - Outcome (c): partial → proceed to (2).
2. **Channel ↔ J-space overlap.** Compute Qwen's J-space at L43
   (their construction) and measure principal-angle overlap with our
   rank-16/64 repair basis, against random-basis nulls. This is the
   bridging experiment already reserved in the comparison matrix
   ("J-lens ↔ rank-8 overlap"), now Qwen-native and mechanistically
   motivated. Same test in Gemma for the two-regime contrast.
3. **Gemma control.** Run (1) on Gemma's F(i) capture (exists,
   458416): prediction — J-projection adds little where linear reading
   already works (0.807). A concentrated regime should be
   probe-and-J readable; a distributed regime J-only. That asymmetry,
   if found, IS the mechanism behind "concentrates vs distributes."

## Scope and caveats

- Their variable is verbalizable concepts, not correctness; the link is
  the READING METHOD and the staging geometry, not the variable. Keep
  wording mechanistic.
- Nanda's replication is Qwen **3.6**; ours is Qwen **3.5** — same
  family, adjacent release. If the J-machinery matters here, the G5
  (3.6) stretch gains a reason to exist beyond redundancy.
- Cost estimate: capture is already done; JVPs for ~150 positions × a
  few dozen directions on 2× A40 ≈ one short job. The Q1 map and this
  ladder should merge into one registration (the map's decodability
  grid gains the J-projected column).
- Current paper: NO change to any claim. At most, §6 discussion may
  cite Nanda's review alongside the workspace paper when naming the
  open question, and may phrase the mystery as "consistent with
  staging in consumption-readable form (cf. J-Lens vs Tuned Lens gap)"
  — flagged as hypothesis, not finding.
