#!/usr/bin/env python3
"""Item WV reader — the content control. Gold-address repair at 2x on
the 59 WX rows under the frozen hint-delta (WX records), the sign-flipped
direction, and matched-norm random directions; row-paired differences
(23rd: hint − random > 0; 24th: hint − flip > 0); each arm's own dP vs
baseline and address-specificity; and the attention read (WT records for
hint, WV attention records for flip/random): final-token mass onto the
gold span at L32 and L38 under each vector, write minus none."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_w1_gates import boot_ci  # noqa: E402


def gold_rates(files, rung=2.0):
    base, gold, nongold = defaultdict(list), defaultdict(list), defaultdict(list)
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r["condition"] == "baseline":
                base[r["id"]].append(r["correct"])
            elif r["condition"] == "delta_write" and r["rung"] == rung:
                (gold if r["fired_is_gold"] else nongold)[r["id"]].append(r["correct"])
    return {i: float(np.mean(v)) for i, v in base.items()}, {i: float(np.mean(v)) for i, v in gold.items()}, {i: float(np.mean(v)) for i, v in nongold.items()}


def attention_gold_span(files, layers):
    base, write = {}, defaultdict(dict)
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r["condition"] == "none":
                base[r["id"]] = r["per_layer"]
            elif r["condition"] == "write" and r["fired_is_gold"]:
                write[r["id"]] = r["per_layer"]
    out = {}
    for L in layers:
        d = [write[i][str(L)]["gold_mean"] - base[i][str(L)]["gold_mean"] for i in write if i in base]
        out[str(L)] = {"n": len(d), "diff": float(np.mean(d)) if d else None, "ci": boot_ci(d) if d else None}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hint", type=Path, nargs="+", default=[Path("results/loop_screen/wikihop_wx_a.jsonl"), Path("results/loop_screen/wikihop_wx_b.jsonl")])
    p.add_argument("--flip", type=Path, nargs="+", required=True)
    p.add_argument("--random", type=Path, nargs="+", required=True, help="records of every random draw (pooled per row)")
    p.add_argument("--attn-hint", type=Path, nargs="*", default=[Path("results/loop_screen/wikihop_wt.jsonl")])
    p.add_argument("--attn-flip", type=Path, nargs="*", default=[])
    p.add_argument("--attn-random", type=Path, nargs="*", default=[])
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_wv_gates.json"))
    args = p.parse_args()
    arms = {"hint": gold_rates(args.hint), "flip": gold_rates(args.flip), "random": gold_rates(args.random)}
    rows = sorted(set(arms["hint"][1]) & set(arms["flip"][1]) & set(arms["random"][1]))
    base = arms["hint"][0]
    per_arm = {}
    for name, (b, g, ng) in arms.items():
        dP = [g[i] - b[i] for i in rows]
        spec = [g[i] - ng[i] for i in rows if i in ng]
        per_arm[name] = {"gold_rate": float(np.mean([g[i] for i in rows])), "baseline": float(np.mean([b[i] for i in rows])),
                         "dP": float(np.mean(dP)), "dP_ci": boot_ci(dP),
                         "nongold_address_gold_rate": float(np.mean([ng[i] for i in rows if i in ng])) if spec else None,
                         "specificity": float(np.mean(spec)) if spec else None, "specificity_ci": boot_ci(spec) if spec else None}
    d_rand = [arms["hint"][1][i] - arms["random"][1][i] for i in rows]
    d_flip = [arms["hint"][1][i] - arms["flip"][1][i] for i in rows]
    out = {"n_rows": len(rows), "arms": per_arm,
           "prediction_23_hint_minus_random": {"mean": float(np.mean(d_rand)), "ci": boot_ci(d_rand), "pass": boot_ci(d_rand)[0] > 0},
           "prediction_24_hint_minus_flip": {"mean": float(np.mean(d_flip)), "ci": boot_ci(d_flip), "pass": boot_ci(d_flip)[0] > 0},
           "attention": {}}
    for name, files in (("hint", args.attn_hint), ("flip", args.attn_flip), ("random", args.attn_random)):
        if files:
            out["attention"][name] = attention_gold_span(files, [31, 32, 33, 38, 42, 53])
    v = []
    v.append("CONTENT-MATTERS" if out["prediction_23_hint_minus_random"]["pass"] else "ANY-PERTURBATION-REPAIRS")
    v.append("SIGN-MATTERS" if out["prediction_24_hint_minus_flip"]["pass"] else "SIGN-DOES-NOT-MATTER")
    out["verdict"] = " / ".join(v)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "attention"}, indent=1))
    for name, a in out["attention"].items():
        print(name, "gold-span attention, write − none:", {L: (round(x["diff"], 4) if x["diff"] is not None else None, x["n"]) for L, x in a.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
