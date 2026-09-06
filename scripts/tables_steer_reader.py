#!/usr/bin/env python3
"""Item WW reader: per (row, arm) gold rate over k samples and the majority
answer; fixable-pool paired readings (38th, 39th), the blind two-stage
rule (40th), pre-named descriptive readings, and the delivery gate."""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

GATE = 0.1957; GATE_WIDE = 0.3201


def boot(diff, n=4000, seed=0):
    d = np.asarray(diff, float); rng = np.random.default_rng(seed)
    if not len(d):
        return [None, None]
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]; return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--records", nargs="+", required=True); ap.add_argument("--pins", default="docs/ww_pinned.json"); ap.add_argument("--out", default="docs/ww_gates.json"); a = ap.parse_args()
    pins = json.load(open(a.pins)); stratum = {rid: j["stratum"] for j in pins["jobs"].values() for rid in j["rows"]}
    samples = defaultdict(list); meta = {}
    for p in a.records:
        for line in open(p):
            r = json.loads(line); samples[(r["id"], r["arm"])].append(r); meta.setdefault(r["id"], r)
    rate, maj, deliv = {}, {}, {}
    for (rid, arm), rs in samples.items():
        rate[(rid, arm)] = float(np.mean([x["correct"] for x in rs]))
        top = Counter(x["normalized_output"] for x in rs).most_common(1)[0][0]; maj[(rid, arm)] = int(next(x["correct"] for x in rs if x["normalized_output"] == top))
        if rs[0].get("bias_fired") is not None:
            deliv[(rid, arm)] = (rs[0]["bias_fired"], rs[0]["highlight_share_before"], rs[0]["highlight_share_after"])
    ids = sorted({rid for rid, _ in samples}); arms = sorted({arm for _, arm in samples}, key=lambda s: (s != "baseline", s))
    out = {"n_rows": len(ids), "arms": arms, "delivery": {}, "fixable": {}, "blind": {}}
    fired = [v[0] >= 1 for v in deliv.values()]; rose = [v[2] > v[1] for v in deliv.values() if v[1] is not None and v[2] is not None]
    out["delivery"] = {"steered_row_arms": len(deliv), "bias_fired_all": bool(all(fired)) if fired else None, "share_rose_frac": round(float(np.mean(rose)), 3) if rose else None,
                       "mean_share_before": round(float(np.mean([v[1] for v in deliv.values() if v[1] is not None])), 3) if any(v[1] is not None for v in deliv.values()) else None,
                       "mean_share_after_by_arm": {arm: round(float(np.mean([v[2] for (rid, am), v in deliv.items() if am == arm and v[2] is not None])), 3) for arm in arms if any(am == arm and v[2] is not None for (_, am), v in deliv.items())}}
    print("delivery:", json.dumps(out["delivery"]))
    out["delivery"]["gate_pass"] = bool(out["delivery"]["bias_fired_all"] and (out["delivery"]["share_rose_frac"] or 0) >= 0.95)
    for pool in ("fixable", "blind"):
        rows = [rid for rid in ids if stratum.get(rid) == pool and (rid, "baseline") in rate]
        if not rows:
            continue
        res = {"n": len(rows), "baseline_rate": round(float(np.mean([rate[(rid, "baseline")] for rid in rows])), 3), "baseline_majority": round(float(np.mean([maj[(rid, "baseline")] for rid in rows])), 3),
               "aim_hits_gold": round(float(np.mean([meta[rid]["aim_hits_gold"] for rid in rows])), 3), "gated_frac": round(float(np.mean([meta[rid]["peak_share"] < GATE for rid in rows])), 3), "arms": {}}
        for arm in arms:
            if arm == "baseline" or any((rid, arm) not in rate for rid in rows):
                continue
            d = [rate[(rid, arm)] - rate[(rid, "baseline")] for rid in rows]; dm = [maj[(rid, arm)] - maj[(rid, "baseline")] for rid in rows]
            hit = [rid for rid in rows if meta[rid]["aim_hits_gold"]]; miss = [rid for rid in rows if not meta[rid]["aim_hits_gold"]]
            res["arms"][arm] = {"rate": round(float(np.mean([rate[(rid, arm)] for rid in rows])), 3), "dP": round(float(np.mean(d)), 3), "ci": boot(d), "up": int(sum(x > 0 for x in d)), "down": int(sum(x < 0 for x in d)),
                                "majority": round(float(np.mean([maj[(rid, arm)] for rid in rows])), 3), "d_majority": round(float(np.mean(dm)), 3), "ci_majority": boot(dm),
                                "dP_aim_hits_gold": round(float(np.mean([rate[(rid, arm)] - rate[(rid, "baseline")] for rid in hit])), 3) if hit else None, "dP_aim_misses": round(float(np.mean([rate[(rid, arm)] - rate[(rid, "baseline")] for rid in miss])), 3) if miss else None}
            print(f"{pool:8s} {arm:18s} rate {res['arms'][arm]['rate']:.3f} dP {res['arms'][arm]['dP']:+.3f} {res['arms'][arm]['ci']} up/down {res['arms'][arm]['up']}/{res['arms'][arm]['down']} | hit {res['arms'][arm]['dP_aim_hits_gold']} miss {res['arms'][arm]['dP_aim_misses']}")
        def paired(a1, a2):
            if any((rid, a1) not in rate or (rid, a2) not in rate for rid in rows):
                return None
            d = [rate[(rid, a1)] - rate[(rid, a2)] for rid in rows]; return {"diff": round(float(np.mean(d)), 3), "ci": boot(d)}
        res["steer_heads3_vs_control"] = paired("steer_heads3:0.01", "steer_control:0.01"); res["steer_heads3_vs_quote"] = paired("steer_heads3:0.01", "quote_heads3"); res["steer_gold_vs_heads3"] = paired("steer_gold:0.01", "steer_heads3:0.01")
        if pool == "blind":
            for name, thr in (("rule", GATE), ("rule_wide", GATE_WIDE)):
                for arm in ("steer_heads3:0.01", "steer_heads3:0.1"):
                    if any((rid, arm) not in maj for rid in rows):
                        continue
                    dm = [(maj[(rid, arm)] if meta[rid]["peak_share"] < thr else maj[(rid, "baseline")]) - maj[(rid, "baseline")] for rid in rows]
                    dr = [(rate[(rid, arm)] if meta[rid]["peak_share"] < thr else rate[(rid, "baseline")]) - rate[(rid, "baseline")] for rid in rows]
                    res[f"{name}_{arm}"] = {"frame_net_majority": round(float(np.mean(dm)), 3), "ci": boot(dm), "up": int(sum(x > 0 for x in dm)), "down": int(sum(x < 0 for x in dm)), "frame_net_rate": round(float(np.mean(dr)), 3), "ci_rate": boot(dr), "gated": int(sum(meta[rid]["peak_share"] < thr for rid in rows))}
                    print(f"blind {name} {arm}: net (majority) {res[f'{name}_{arm}']['frame_net_majority']:+.3f} {res[f'{name}_{arm}']['ci']} up/down {res[f'{name}_{arm}']['up']}/{res[f'{name}_{arm}']['down']} gated {res[f'{name}_{arm}']['gated']}; net (rate) {res[f'{name}_{arm}']['frame_net_rate']:+.3f} {res[f'{name}_{arm}']['ci_rate']}")
            corr = [rid for rid in rows if maj[(rid, "baseline")] == 1]; fail = [rid for rid in rows if maj[(rid, "baseline")] == 0]
            for arm in arms:
                if arm != "baseline" and all((rid, arm) in rate for rid in rows):
                    res["arms"][arm]["collateral_on_correct_rate"] = round(float(np.mean([rate[(rid, arm)] - rate[(rid, "baseline")] for rid in corr])), 3) if corr else None
                    res["arms"][arm]["gain_on_failures_rate"] = round(float(np.mean([rate[(rid, arm)] - rate[(rid, "baseline")] for rid in fail])), 3) if fail else None
            res["n_correct_majority"] = len(corr); res["n_failures_majority"] = len(fail)
        out[pool] = res
    fx, bl = out.get("fixable", {}).get("arms", {}), out.get("blind", {})
    g = fx.get("steer_gold:0.01"); h = fx.get("steer_heads3:0.01"); hc = out.get("fixable", {}).get("steer_heads3_vs_control")
    out["verdicts"] = {"38th": None if not g else {"dP": g["dP"], "ci": g["ci"], "confirmed": bool(g["dP"] >= 0.20 and g["ci"][0] > 0)},
                       "39th": None if not h or not hc else {"dP": h["dP"], "ci": h["ci"], "vs_control": hc, "confirmed": bool(h["dP"] >= 0.10 and h["ci"][0] > 0 and hc["ci"][0] > 0)},
                       "40th": None if "rule_steer_heads3:0.01" not in bl else {"net": bl["rule_steer_heads3:0.01"]["frame_net_majority"], "ci": bl["rule_steer_heads3:0.01"]["ci"], "confirmed": bool(bl["rule_steer_heads3:0.01"]["ci"][0] > 0)}}
    print("verdicts:", json.dumps(out["verdicts"]))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
