# SJ census (2k rows) — verbalized self-judgement is near-uniformly "no": the pre-named confidence-blindness case, in extreme form; the SJ/OC registration takes the probe-SJ fallback

Registration-free reconnaissance per the Tier-1 SJ/OC design
(future-directions 2026-08-09, step 1). givemeanode batch job
`job-rt39f` (H100, offline vLLM, ~26 min, ≈$1.30; mission
https://givemeanode.com/missions/sj-census), completed 2026-08-14.
Protocol: seeded 2,000-row balanced draw (500 per height-3/4 ×
strong-correct/incorrect cell, seed 20260812), the model's own stage-1
answer replayed as an assistant turn, then the frozen question
"Was your final proposed hypothesis exactly correct? Reply with a
single word: yes or no." — k=4 at temp 0.7, majority vote. Row-level
data: `results/sj_census/sj_census_2k.jsonl` (sha256-verified
artifact); client `scripts/sj_census.py` (selection) +
`scripts/sj_census_job.py` (entrypoint). Venue note: black-box
generation on the H100 lane — venue-agnostic by design (reconnaissance
only; the registered SJ/OC arms elicit fresh judgements in-job on
Scholar).

## The census (rows; OC = stage-1 strong correctness, SJ = majority verbalized self-judgement)

| | SJ = yes | SJ = no | tie |
|---|---:|---:|---:|
| **OC = correct** (1,000) | 145 | **850** | 5 |
| **OC = wrong** (1,000) | 18 | 981 | 1 |

- Overall SJ-yes rate: **8.2%**. Unanimity across k=4: **97.5%**.
  Parse: 100.0%.
- **The model says "no, not exactly correct" on 85% of the answers it
  actually got right.** Verbalized self-judgement is close to a
  constant on this task.
- The conflict cell the factorization design most needs
  (confident-wrong: SJ=yes ∧ OC=wrong) has **18 rows in 2,000**
  (0.9%) — far too thin to estimate an SJ direction from verbalized
  labels.
- Signal is not literally zero: the rare yeses are informative
  (yes-rate 14.5% on correct vs 1.8% on wrong — an 8× lift), and
  height-3 draws more yeses than height-4 (97 vs 66) — but variance,
  not validity, is the binding constraint.

## Reading

1. **The pre-named fallback fires.** The census design (per the
   Tier-1 registration sketch and 2605.09502's confidence-blindness
   warning) named exactly this outcome: if verbalized SJ lacks
   variance, the SJ/OC factorization estimates the SJ direction from
   a PROBE on the elicitation states rather than from verbalized
   labels. That is now the registered-design path.
2. **A knowing-saying gap on our own task, for free.** The same
   model's internal gauge reads upcoming correctness at AUC 0.90–0.94
   — while its verbalized self-assessment is ~degenerate at "no."
   Whatever the stage-1 probe reads, it is not the model's verbal
   self-report channel: a clean local instance of the
   knowing-without-saying dissociation (cf. 2607.08456, 2603.17839),
   and a sharpening question for 2607.16799's probes-track-SJ claim —
   here verbal SJ barely exists, yet the probe tracks OC at 0.94.
   Worth one sentence in §6 at the drafting pass.
3. **Elicitation sensitivity caveat** (2605.27752): the census used
   one frozen question, and "exactly correct" plausibly invites
   hedging. A softer-elicitation variant census (drop "exactly";
   or ask "is your answer more likely right or wrong?") costs ≈$1.30
   and ~30 min as another batch job — worth running BEFORE the SJ/OC
   registration is written, since an elicitation that unlocks
   variance would restore the cheaper verbalized-label design. Both
   censuses would be reported (no cherry-picking the elicitation:
   the registration must pin whichever protocol it uses and cite
   both counts).

## Consequences

- SJ/OC registration (when written): probe-SJ fallback as the default
  design; optionally preceded by one softer-elicitation census
  (report both). Elicitation protocol, answer slot, and scoring to be
  pinned in the registration per 2605.27752.
- No claims move anywhere (reconnaissance; registration-free by
  design).
