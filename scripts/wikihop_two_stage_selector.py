#!/usr/bin/env python3
"""Descriptive (post-hoc, unregistered): a two-stage blind rule on the WD/WO
rows. Stage 1, the unsteered gauge score on the baseline state decides
whether the row is a failure (score < 0). Stage 2, on judged failures the
output-first loop runs with the model's own baseline answer removed from a
tie at the top (its nudge trivially confirms it); judged-correct rows keep
the baseline. Variants: answer the gauge tie-break among the remaining tied
branches, or abstain unless exactly one remains. An oracle detector gives
the upper bound."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_w1_gates import boot_ci  # noqa: E402
from wikihop_loop_descriptives import RS, load, rows_from, select  # noqa: E402


def stage2(row, variant):
    c = row["cands"]; top = max(v["af"] for v in c.values())
    tied = [x for x in c if c[x]["af"] == top]
    if top == 0:
        return row["base"], True
    non_base = [x for x in tied if x != row["base_modal"]]
    if not non_base:
        return row["base"], True
    if len(non_base) == 1:
        return c[non_base[0]]["corr"], False
    if variant == "answer":
        return c[max(non_base, key=lambda x: c[x]["gauge"])]["corr"], False
    return row["base"], True


def apply(rows, detector, variant, is_failure_stratum):
    vals, abst, flagged = [], [], 0
    for i in sorted(rows):
        row = rows[i]
        if detector == "oracle":
            fail = is_failure_stratum
        else:
            fail = row["base_gauge"].get(detector, 0.0) < 0
        flagged += int(fail)
        if fail:
            a, ab = stage2(row, variant)
        else:
            a, ab = row["base"], True
        vals.append(a - row["base"]); abst.append(ab)
    return np.array(vals), float(np.mean(abst)), flagged / len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pins", type=Path, default=Path("docs/wikihop_wd_pinned.json"))
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_two_stage_selector.json"))
    args = p.parse_args()
    pins = json.load(open(args.pins))
    w_rep, w_unrep = pins["pool"]["weight_repairable"], pins["pool"]["weight_unrepairable"]
    n_dd, n_cor, n_tot = pins["pool"]["n_doc_dependent"], pins["pool"]["n_correct_majority"], 507
    R = rows_from(load([RS / "wikihop_wo_a.jsonl", RS / "wikihop_wo_b.jsonl"]))
    U = rows_from(load([RS / "wikihop_wd_y1.jsonl", RS / "wikihop_wd_y2.jsonl"]))
    C = rows_from(load([RS / "wikihop_wd_c.jsonl"]))
    rng = np.random.default_rng(20260848)
    out = []
    for detector in ("primary_L48", "second_L38", "primary_L38", "oracle"):
        for variant in ("answer", "abstain"):
            d_rep, ab_rep, fl_rep = apply(R, detector, variant, True)
            d_unrep, ab_unrep, fl_unrep = apply(U, detector, variant, True)
            d_col, ab_col, fl_col = apply(C, detector, variant, False)
            draws = [w_rep * d_rep[rng.integers(0, len(d_rep), len(d_rep))].mean() + w_unrep * d_unrep[rng.integers(0, len(d_unrep), len(d_unrep))].mean() for _ in range(4000)]
            y = float(w_rep * d_rep.mean() + w_unrep * d_unrep.mean())
            rec = {"detector": detector, "variant": variant,
                   "detector_flags_failures": {"repairable": fl_rep, "unrepairable": fl_unrep, "correct_rows_flagged": fl_col},
                   "repairable_dP": float(d_rep.mean()), "repairable_ci": boot_ci(list(d_rep)), "repairable_abstained": ab_rep,
                   "unrepairable_dP": float(d_unrep.mean()), "unrepairable_ci": boot_ci(list(d_unrep)), "unrepairable_abstained": ab_unrep,
                   "yield_weighted": y, "yield_ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                   "collateral_dP": float(d_col.mean()), "collateral_ci": boot_ci(list(d_col)), "collateral_abstained": ab_col,
                   "frame_net": float(n_dd / n_tot * y + n_cor / n_tot * d_col.mean())}
            out.append(rec)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print("| detector | variant | flags fail: rep / unrep / correct | repairable dP [CI] | unrepairable dP [CI] | yield weighted [CI] | collateral [CI] | frame net |")
    print("|---|---|---|---|---|---|---|---|")
    for r in out:
        f = r["detector_flags_failures"]
        print(f"| {r['detector']} | {r['variant']} | {f['repairable']:.0%} / {f['unrepairable']:.0%} / {f['correct_rows_flagged']:.0%} | "
              f"{r['repairable_dP']:+.3f} [{r['repairable_ci'][0]:+.3f}, {r['repairable_ci'][1]:+.3f}] | "
              f"{r['unrepairable_dP']:+.3f} [{r['unrepairable_ci'][0]:+.3f}, {r['unrepairable_ci'][1]:+.3f}] | "
              f"{r['yield_weighted']:+.3f} [{r['yield_ci'][0]:+.3f}, {r['yield_ci'][1]:+.3f}] | "
              f"{r['collateral_dP']:+.3f} [{r['collateral_ci'][0]:+.3f}, {r['collateral_ci'][1]:+.3f}] | {r['frame_net']:+.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
