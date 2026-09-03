# Item WT — what the write does to attention: WRITE-ROUTES-ATTENTION-AT-L38 (22nd prediction CONFIRMED) (2026-09-03)

Registered: docs/causal_handle_directions.md item WT (before any data).
Job **job-8wnju** ($0.51; eager attention over all 62 layers of
Gemma-3-27B; the first submission job-6ydrz was cancelled unbilled before
start to add per-candidate no-write masses). Rows: the WX cross-fit test
rows with prompts ≤ 1,600 tokens — 30 of 59 (A 11, B 19; 29 skipped as
too long); frozen direction from each job's donors exactly as in WX
(norm targets 3,289 / 3,217). Per row: one prefill with no write and one
per candidate (gold + the three non-gold candidates WX fired) with the
frozen write at L30 × 2× at that candidate's mentions — 120 write
records, delivery audit valid (zero-prefill 0, zero-positions 0).
Reader `scripts/wikihop_wt_gates.py` → `docs/wikihop_wt_gates.json`;
row-level `results/loop_screen/wikihop_wt.jsonl`.

Reading: the final prompt token's attention mass (mean over heads) onto
the written span, write minus no-write, row×candidate-paired.

## Primary (pinned layer L38)
| | paired difference | 95% CI |
|---|---|---|
| all 120 writes, written span | **+0.0030** (baseline mass 0.0048) | **[+0.0024, +0.0036]** |
| gold writes only (30) | +0.0035 | [+0.0020, +0.0048] |
| non-gold writes, own span (90) | +0.0028 | [+0.0022, +0.0035] |
| non-gold writes, gold span | −0.00034 | [−0.00047, −0.00023] |
| max over heads, all writes | +0.0072 | [−0.0086, +0.0223] |

**22nd prediction CONFIRMED — WRITE-ROUTES-ATTENTION-AT-L38.** The
write raises the answer position's attention onto the addressed span
(by 62% of its baseline mass at L38), for gold and non-gold addresses
alike, and slightly lowers attention onto the gold span when the
address is elsewhere.

## The per-layer curve (descriptive)
| layer | baseline written-span mass | write − none [CI] | gold only | non-gold, own span | non-gold, gold span |
|---|---|---|---|---|---|
| L31 | 0.0068 | +0.0359 [+0.0317, +0.0407] | +0.0389 | +0.0350 | −0.0004 |
| L32 | 0.0059 | **+0.0483 [+0.0406, +0.0572]** | +0.0546 | +0.0463 | −0.0008 |
| L33 | 0.0086 | +0.0326 [+0.0279, +0.0378] | +0.0385 | +0.0306 | −0.0013 |
| L35 | 0.0253 | +0.0183 [+0.0127, +0.0245] | +0.0231 | +0.0167 | −0.0018 |
| L36 | 0.0085 | +0.0269 [+0.0236, +0.0303] | +0.0336 | +0.0246 | −0.0008 |
| L38 (pinned) | 0.0048 | +0.0030 [+0.0024, +0.0036] | +0.0035 | +0.0028 | −0.0003 |
| L42 | 0.0065 | +0.0154 [+0.0128, +0.0184] | +0.0209 | +0.0136 | −0.0005 |
| L53 | 0.0223 | +0.0173 [+0.0132, +0.0215] | +0.0309 | +0.0128 | −0.0010 |
| L58 | 0.0068 | +0.0073 [+0.0060, +0.0086] | +0.0108 | +0.0061 | −0.0004 |

The routing is strongest in the two or three layers right after the
write (L32: eight times the baseline mass) and persists, smaller, to the
last layer; the pinned L38 sits at a local trough of the curve. Every
layer from L31 to L61 has a CI above zero. Gemma's local-attention
layers see only the last 1,024 tokens, which bounds the mass any single
layer can move.

## Attention gain and acceptance (descriptive)
Joined to WX's answers-fired at 2× for the same (row, candidate): the
written-span mass at L38 correlates with acceptance at r = 0.42 over
120 branches; branches the model accepts at least half the time carry
twice the mass of those it does not (0.0131 vs 0.0066). The write is
a routing intervention, and how much it routes predicts whether the
model adopts the candidate.

## Verdict
**WRITE-ROUTES-ATTENTION-AT-L38.** "Attend to this one" is now a
measurement: one frozen vector at a span makes the answer position look
at that span, most in the layers just above the write, with a
matching small withdrawal from the gold span when the address is wrong.

## Program tally after WT
22 registered directional predictions: 17 confirmed, 3 not (13th, 14th,
19th), 2 in flight (20th, 21st — item WY).
