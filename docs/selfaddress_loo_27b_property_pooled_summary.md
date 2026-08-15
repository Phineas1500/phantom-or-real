# Item L″ (composition) — EXECUTION-INVALID, and the invalidity reaches back into item L: branch generations were never steered. Corrected rerun (L″-r) + frozen-transfer rider launched same night.

Jobs 461663/461664/461665/461666 (shards C0–C3, Aug 14; 2:16–2:44 each,
all COMPLETED). Pooled by `scripts/stage2_selfaddress_loo_pool.py`;
report and row-level data preserved under `docs/invalid_20260814/` and
`results/stage2/erasure/invalid_20260814/`. Registry correction entry:
`docs/causal_handle_directions.md` (2026-08-14, late), commit 03b0eae.

## What the pooled analysis showed, and how the defect surfaced

- Baseline verbatim gate **456/456 PASS** (job-stitching integrity: the
  fresh in-job baselines reproduce L1's token-for-token).
- ORACLE GATE FAILS: gold-branch dP −0.009 [−0.035, +0.018] — on the
  same 57 rows where L′ repaired +0.279 with a byte-identical write
  construction the day before. Non-overlapping CIs on identical
  protocol constants → not sampling noise → code diff.
- The diff: in the selfaddress/selfaddress-loo branch loop, the
  intervention hooks wrapped only the gauge-scoring forward
  (`gauge_read`); the branch `generate_sample_batch` call was never
  inside `with model.hooks(...)`. (The L′ arms loop wraps generation
  correctly — L′ is unaffected.)
- Data-level confirmation (both L1 and L″): branch generations carry
  NO item-H steering signature — gold-branch targets-fired 0.412/0.399
  ≈ the 0.395 natural baseline rate; item H's delivered writes pull
  ~0.50. The "steered branches" were unhinted samples.

## Withdrawn

1. **L1's "ORACLE GATE FAILS / frozen protocol does not transfer"** —
   the write never touched the generations; frozen transfer is
   UNTESTED, not refuted.
2. **All behavioral selector equivalences** from L1/L″ ("gauge-select
   == oracle to three decimals" was trivially guaranteed when every
   branch is a baseline draw).
3. **L′'s within-row protocol contrast as a frozen-vs-fresh claim** —
   its frozen comparator was the invalid L1 gold branch; the contrast
   collapses to fresh-fit vs baseline, which is L′'s primary and
   stands.
4. **L″'s primary outcome** — void, not a null.

## Stands

- L0 gates (natural gauge AUC 0.936; selection signal +31.8
  [+23.8, +40.1]) and the L″ per-shard selection-signal replications
  (C0: +21.8 [+15.7, +28.3]) — steered-STATE scores, real.
- **State-level selector evidence**: gauge argmax over steered-state
  branch scores picks the gold address **54/57 (0.947)** under the
  frozen write and **43/57 (0.754)** under the stronger LOO write.
- L′ in full: +0.279 [+0.182, +0.377].
- All verbatim/determinism gates.

## Corrective actions (all same-night, pre-registered before any new data)

1. Fix: branch generation wrapped in the branch hooks
   (stage2_rank_k_guard_v2.py; commit 03b0eae).
2. **L″-r** relaunched: jobs 461886–461889 (C0–C3), identical rows,
   seeds, write construction, decision rules; baselines verbatim-gate
   against L1 again.
3. **C4 rider** (job 461890, `--gold-only`): the original item-L
   frozen-write oracle-gate test executed correctly — gold-address
   frozen fires, k=8, same 64 rows; FROZEN-TRANSFERS vs
   FROZEN-DOES-NOT-TRANSFER branches against L′'s +0.279 anchor.
4. **New hard gate** (registered): fired arms must show targets-fired
   lift over baseline (CI>0) or the run is execution-invalid by rule.
   The defect was catchable in advance by exactly this
   steering-signature positive control; it is now mandatory.
5. Venue note: the cluster's new Lua submit filter rejects 2-GPU jobs
   above 3h wall (bisected: 3:00 passes, 3:30 crashes the filter);
   rerun shards run at 3:00:00 against observed 2:16–2:45 runtimes,
   with the interleave-split precedent (3of8 ∪ 7of8 ≡ 3of4) as the
   overrun remedy.

## Paper impact

§5.5/§6/intro-(vi)/appendix corrected in commit 12ef30d: the
execution defect is disclosed in-text, the frozen-write null restated
as void, the selector claims restated at state level (54/57, 43/57),
and the composition marked as re-executing at submission time. No
other section touches the invalidated arms.
