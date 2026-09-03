#!/usr/bin/env python3
"""Item WT reader. Primary (22nd prediction): at the pinned layer (L38)
the final token's attention mass onto the written span is higher under
the frozen write than without it — row×candidate-paired mean difference
with bootstrap CI > 0. Descriptive: the per-layer curve; gold-span mass
under non-gold writes vs none; the relation between a branch's attention
gain and WX's acceptance of that branch (answers-fired at 2x)."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_w1_gates import boot_ci  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, default=Path("results/loop_screen/wikihop_wt.jsonl"))
    p.add_argument("--wx", type=Path, nargs="*", default=[Path("results/loop_screen/wikihop_wx_a.jsonl"), Path("results/loop_screen/wikihop_wx_b.jsonl")])
    p.add_argument("--layer", type=int, default=38)
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_wt_gates.json"))
    args = p.parse_args()
    recs = [json.loads(l) for l in open(args.jsonl)]
    base = {r["id"]: r for r in recs if r["condition"] == "none"}
    writes = [r for r in recs if r["condition"] == "write"]
    layers = sorted(int(k) for k in next(iter(base.values()))["per_layer"])
    audit = {"n_write_records": len(writes), "n_rows": len(base),
             "n_zero_prefill_hook_calls": sum(1 for r in writes if r.get("hook_prefill_calls", 0) == 0),
             "n_zero_positions_written": sum(1 for r in writes if r.get("hook_positions_written", 0) == 0)}
    audit["valid"] = audit["n_zero_prefill_hook_calls"] == 0 and audit["n_zero_positions_written"] == 0 and len(writes) > 0

    def none_mass(r, L, stat):
        return base[r["id"]]["candidate_masses"][r["fired_candidate"]][str(L)][f"written_{stat}"]

    gold_writes = [r for r in writes if r["fired_is_gold"]]
    non_writes = [r for r in writes if not r["fired_is_gold"]]
    curve = {}
    for L in layers:
        d_all = [r["per_layer"][str(L)]["written_mean"] - none_mass(r, L, "mean") for r in writes]
        d_all_max = [r["per_layer"][str(L)]["written_max"] - none_mass(r, L, "max") for r in writes]
        d_gold = [r["per_layer"][str(L)]["written_mean"] - none_mass(r, L, "mean") for r in gold_writes]
        d_non_own = [r["per_layer"][str(L)]["written_mean"] - none_mass(r, L, "mean") for r in non_writes]
        d_non_gold_span = [r["per_layer"][str(L)]["gold_mean"] - base[r["id"]]["per_layer"][str(L)]["gold_mean"] for r in non_writes]
        curve[L] = {"written_span_write_minus_none_mean": float(np.mean(d_all)), "ci": boot_ci(d_all),
                    "written_span_write_minus_none_maxhead": float(np.mean(d_all_max)), "ci_maxhead": boot_ci(d_all_max),
                    "gold_writes_only": float(np.mean(d_gold)) if d_gold else None, "ci_gold_only": boot_ci(d_gold) if d_gold else None,
                    "nongold_writes_own_span": float(np.mean(d_non_own)) if d_non_own else None, "ci_nongold_own": boot_ci(d_non_own) if d_non_own else None,
                    "nongold_write_gold_span_minus_none": float(np.mean(d_non_gold_span)) if d_non_gold_span else None, "ci_nongold_gold_span": boot_ci(d_non_gold_span) if d_non_gold_span else None,
                    "baseline_written_span_mass_mean": float(np.mean([none_mass(r, L, "mean") for r in writes]))}
    L = args.layer
    primary = curve[L]
    # branch acceptance from WX (answers-fired at 2x) joined on id + candidate
    af = defaultdict(list)
    for f in args.wx:
        for line in open(f):
            r = json.loads(line)
            if r["condition"] == "delta_write" and r["rung"] == 2.0:
                af[(r["id"], r["fired_candidate"])].append(r["answers_fired"])
    pairs = []
    for r in gold_writes + non_writes:
        key = (r["id"], r["fired_candidate"])
        if key in af:
            gain = r["per_layer"][str(L)]["written_mean"]
            pairs.append((gain, float(np.mean(af[key])), r["fired_is_gold"]))
    corr = None
    if len(pairs) > 3:
        g = np.array([x[0] for x in pairs]); a = np.array([x[1] for x in pairs])
        corr = float(np.corrcoef(g, a)[0, 1]) if g.std() > 0 and a.std() > 0 else None
    acc_hi = [x[0] for x in pairs if x[1] >= 0.5]; acc_lo = [x[0] for x in pairs if x[1] < 0.5]
    out = {"layer": L, "delivery_audit": audit, "n_gold_writes": len(gold_writes), "n_nongold_writes": len(non_writes),
           "primary": {"paired_diff_written_span_mean_heads": primary["written_span_write_minus_none_mean"], "ci": primary["ci"], "pass": primary["ci"][0] > 0,
                       "baseline_written_span_mass": primary["baseline_written_span_mass_mean"]},
           "maxhead": {"paired_diff": primary["written_span_write_minus_none_maxhead"], "ci": primary["ci_maxhead"]},
           "gold_writes_only": {"paired_diff": primary["gold_writes_only"], "ci": primary["ci_gold_only"]},
           "nongold_writes_own_span": {"paired_diff": primary["nongold_writes_own_span"], "ci": primary["ci_nongold_own"]},
           "nongold_write_effect_on_gold_span": {"paired_diff": primary["nongold_write_gold_span_minus_none"], "ci": primary["ci_nongold_gold_span"]},
           "acceptance": {"n_pairs": len(pairs), "corr_written_mass_vs_answers_fired": corr,
                          "written_mass_accepted_ge_half": float(np.mean(acc_hi)) if acc_hi else None, "written_mass_accepted_lt_half": float(np.mean(acc_lo)) if acc_lo else None},
           "curve": {str(k): v for k, v in curve.items()}}
    out["verdict"] = "WRITE-ROUTES-ATTENTION-AT-L%d" % L if out["primary"]["pass"] else "WRITE-DOES-NOT-ROUTE-ATTENTION-AT-L%d" % L
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=1))
    print("| layer | baseline written-span mass | write − none, all writes (mean heads) [CI] | gold writes only | non-gold writes, own span | non-gold write, gold span |")
    print("|---|---|---|---|---|---|")
    for k in layers:
        c = curve[k]
        f = lambda x: "—" if x is None else f"{x:+.4f}"
        print(f"| L{k} | {c['baseline_written_span_mass_mean']:.4f} | {c['written_span_write_minus_none_mean']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] | {f(c['gold_writes_only'])} | {f(c['nongold_writes_own_span'])} | {f(c['nongold_write_gold_span_minus_none'])} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
