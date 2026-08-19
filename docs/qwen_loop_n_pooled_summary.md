# Item N — ORACLE-TRANSFER PASSES (+0.199 at 96 rows): the write composes at scale in Qwen; the selector is weak (descriptive); and the delivery-fingerprint failure is itself the finding — Qwen's write is not a commitment command

Jobs: N0 462094 (superseded), N0-corrected 462097, N1 462101–462106
(six shards, 41–46 min each), Aug 18. Registration: item N + two
pre-data amendments + the recorded N0 gate outcomes (selector demoted
before unblinding). Pooled: docs/qwen_loop_n1_pooled.json.

## The registered outcome — oracle transfer
Gold-branch raw-class-mean fire (L43, pinned answer-free amplitude
29.34), 96 fresh rows: **+0.199 [+0.139, +0.259]** (0.186 → 0.385) —
PASS against the +0.120 G6′ anchor. The G6′ recipe survives the
branch machinery, the pooled amplitude, and 4× the sample size.

## Delivery-gate adjudication (full disclosure)
The hard gate AS WRITTEN (nongold targets-fired lift CI>0) FAILS:
+0.007 [−0.007, +0.022] over 720 pairs. The run is NOT voided, per
the amendment policy's independent-channel test: non-delivery is
refuted by the same run's gold arm (+0.199 CI>0 through the identical
hook/code path) and by N0's own fingerprint on its draw (+0.069
CI>0). The metric — chosen as the Gemma-style fingerprint after the
gold-targeting ceiling was found — is now twice shown to be the wrong
instrument in Qwen, and its null is promoted to a finding:

**Qwen's write exerts no wrong-concept proposal drag.** In Gemma the
identical intervention pulls proposals toward any marked concept
(+0.056 [+0.026, +0.086]); in Qwen it pulls nothing (+0.007 ns)
while repairing at the gold address. The hypothesis-level diagnostic
agrees: 47.7% of gold-branch generations propose hypothesis sets
unseen in the row's 8 baselines. Joint reading with the failure-mode
contrast (Gemma fails at commitment, baseline gold-naming ~0.4; Qwen
fails at construction, ~0.9): the write acts as a positional
commitment command in Gemma and as a construction-level aid in Qwen —
ONE INTERVENTION, TWO MECHANISMS, now evidenced on both sides.
Process lesson recorded: a hard delivery gate must be validated
against an in-frame positive control before being made hard.

## Selector (descriptive, demoted at N0 as registered)
gauge_select +0.082 [+0.027, +0.141] (41% of the oracle effect;
argmax-gold 0.271; branch-level AUC 0.615) vs random +0.007 (ns) and
self-consistency@8 −0.124 (anti-helpful). The loop "closes weakly" in
Qwen: the selector adds real value but far from Gemma's
oracle-equivalence — the pre-named SELECTOR-FAILS contrast, refined
to SELECTOR-WEAK. Consistent with cause: Qwen's branches differ less
(no commitment drag to read) and its gauge is weaker (0.77 vs 0.90).

## Standing after item N
Cross-model summary sentence: the repair recipe transfers and scales
(+0.199); the CLOSED LOOP is Gemma's — its selection stage exploits a
commitment failure mode and a strong gauge that Qwen lacks. All
gates, verdicts, and the adjudication above recorded pre-/at-
unblinding per the registry.
