#!/usr/bin/env python3
"""Item WJ reader: depth × dose on NQ-Swap conflict rows. From the three
ladder records (write layers 20 / 25 / 30, rungs 1× / 2× / 3×, WikiHop XA
donors, no loop): per (layer, rung) the gold-address gold rate and dP over
baseline on the repairable rows, the non-gold-address gold rate and
specificity, the reach on the unrepairable rows; and the paired comparison
of every setting against L30 × 2× over the same rows."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_w1_gates import boot_ci  # noqa: E402


def load(path):
    per = defaultdict(lambda: defaultdict(list)); base = defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        if r["condition"] == "baseline":
            base[r["id"]].append(r["correct"])
        elif r["condition"] == "delta_write":
            per[r["id"]][("gold" if r["fired_is_gold"] else "nongold", r["rung"])].append(r["correct"])
    return {i: {"base": float(np.mean(base[i])), **{k: float(np.mean(v)) for k, v in per[i].items()}} for i in base}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", nargs="+", type=Path, required=True, help="layer=path pairs, e.g. 20=results/loop_screen/wikihop_wj_L20.jsonl")
    p.add_argument("--pins", type=Path, default=Path("docs/wikihop_wj_pinned.json"))
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_wj_gates.json"))
    args = p.parse_args()
    pins = json.load(open(args.pins)); rep, unrep = pins["test_rows"]["repairable"], pins["test_rows"]["unrepairable"]
    data = {int(s.split("=")[0]): load(Path(s.split("=")[1])) for s in args.records}
    rungs = sorted({k[1] for R in data.values() for r in R.values() for k in r if isinstance(k, tuple)})
    out = {"settings": {}, "paired_vs_L30x2": {}}
    ref = data.get(30)
    print("| layer × rung | repairable: gold rate | dP vs baseline [CI] | non-gold-address gold rate | specificity [CI] | unrepairable: gold rate (reach) | paired vs L30×2× on repairable [CI] |\n|---|---|---|---|---|---|---|")
    for L, R in sorted(data.items()):
        for g in rungs:
            gr = [R[i].get(("gold", g), np.nan) for i in rep]; b = [R[i]["base"] for i in rep]; ng = [R[i].get(("nongold", g), np.nan) for i in rep]
            ur = [R[i].get(("gold", g), np.nan) for i in unrep]
            d = [x - y for x, y in zip(gr, b)]; spec = [x - y for x, y in zip(gr, ng)]
            rec = {"gold_rate": float(np.nanmean(gr)), "dP": float(np.nanmean(d)), "dP_ci": boot_ci(d), "nongold_gold_rate": float(np.nanmean(ng)), "specificity": float(np.nanmean(spec)), "spec_ci": boot_ci(spec),
                   "unrepairable_reach": float(np.nanmean(ur)), "unrepairable_ci": boot_ci(ur)}
            pair = None
            if ref is not None and not (L == 30 and g == 2.0):
                pv = [R[i].get(("gold", g), np.nan) - ref[i].get(("gold", 2.0), np.nan) for i in rep]; pair = {"delta": float(np.nanmean(pv)), "ci": boot_ci(pv)}
                out["paired_vs_L30x2"][f"L{L}x{g}"] = pair
            out["settings"][f"L{L}x{g}"] = rec
            ps = f"{pair['delta']:+.3f} [{pair['ci'][0]:+.3f}, {pair['ci'][1]:+.3f}]" if pair else "(reference)"
            print(f"| L{L} × {g}× | {rec['gold_rate']:.3f} | {rec['dP']:+.3f} [{rec['dP_ci'][0]:+.3f}, {rec['dP_ci'][1]:+.3f}] | {rec['nongold_gold_rate']:.3f} | {rec['specificity']:+.3f} [{rec['spec_ci'][0]:+.3f}, {rec['spec_ci'][1]:+.3f}] | {rec['unrepairable_reach']:.3f} [{rec['unrepairable_ci'][0]:.3f}, {rec['unrepairable_ci'][1]:.3f}] | {ps} |")
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
