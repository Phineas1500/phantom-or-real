#!/usr/bin/env python3
"""Item WG comparison: loop readings under each gauge key recorded in the
records' 'gauge_scores' (plus the primary 'gauge_score'), on identical
branches; the 13th prediction = paired (primary-gauge loop − real-text
gauge loop) CI > 0. Also checks generation identity against the rider files."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_w1_gates import boot_ci  # noqa: E402


def loop_by_key(recs, key, rung=2.0):
    rows = sorted({r["id"] for r in recs})
    base = {i: float(np.mean([r["correct"] for r in recs if r["id"] == i and r["condition"] == "baseline"])) for i in rows}
    out = {}
    for i in rows:
        br = defaultdict(list)
        for r in recs:
            if r["id"] == i and r["condition"] == "delta_write" and r["rung"] == rung:
                br[r["fired_candidate"]].append(r)
        def score(c):
            r0 = br[c][0]
            return r0["gauge_score"] if key == "primary" else r0["gauge_scores"][key]
        best = max(br, key=score)
        gold = [c for c in br if br[c][0]["fired_is_gold"]][0]
        out[i] = {"loop": float(np.mean([x["correct"] for x in br[best]])), "oracle": float(np.mean([x["correct"] for x in br[gold]])),
                  "random": float(np.mean([np.mean([x["correct"] for x in v]) for v in br.values()])), "base": base[i],
                  "picked_gold": best == gold, "n": len(br)}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, nargs="+", required=True)
    p.add_argument("--rider-jsonl", type=Path, nargs="*", default=[])
    p.add_argument("--real-key", default="second_L38")
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_wg_compare.json"))
    args = p.parse_args()
    recs = [json.loads(l) for f in args.jsonl for l in open(f)]
    keys = ["primary"] + sorted({k for r in recs if r["condition"] == "delta_write" for k in (r.get("gauge_scores") or {})})
    per = {k: loop_by_key(recs, k) for k in keys}
    rows = sorted(per["primary"])
    summary = {}
    for k in keys:
        d = per[k]
        summary[k] = {"gauge_select": float(np.mean([d[i]["loop"] for i in rows])), "baseline": float(np.mean([d[i]["base"] for i in rows])),
                      "random": float(np.mean([d[i]["random"] for i in rows])), "oracle": float(np.mean([d[i]["oracle"] for i in rows])),
                      "dP_vs_baseline_ci": boot_ci([d[i]["loop"] - d[i]["base"] for i in rows]),
                      "dP_vs_random_ci": boot_ci([d[i]["loop"] - d[i]["random"] for i in rows]),
                      "argmax_gold_rate": float(np.mean([d[i]["picked_gold"] for i in rows])),
                      "oracle_recovered": float(np.mean([d[i]["loop"] for i in rows]) / max(np.mean([d[i]["oracle"] for i in rows]), 1e-9))}
    paired = [per["primary"][i]["loop"] - per[args.real_key][i]["loop"] for i in rows] if args.real_key in per else []
    pred = {"paired_primary_minus_real": float(np.mean(paired)) if paired else None, "ci": boot_ci(paired) if paired else None,
            "pass": bool(paired and boot_ci(paired)[0] > 0)}
    identity = None
    if args.rider_jsonl:
        rider = [json.loads(l) for f in args.rider_jsonl for l in open(f)]
        key = lambda r: (r["id"], r["condition"], r["rung"], r["fired_candidate"], r["sample_index"])
        a = {key(r): r["normalized_output"] for r in rider}
        b = {key(r): r["normalized_output"] for r in recs}
        common = set(a) & set(b)
        identity = {"n_common": len(common), "n_identical_outputs": sum(a[k] == b[k] for k in common), "fraction": (sum(a[k] == b[k] for k in common) / len(common)) if common else None}
    out = {"n_rows": len(rows), "per_gauge": summary, "prediction_13": pred, "generation_identity_vs_rider": identity,
           "verdict": ("GAUGE-REFIT-HELPS" if pred["pass"] else "SELECTOR-LIMIT-IS-NOT-DISTRIBUTION") if paired else "no real-gauge key"}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
