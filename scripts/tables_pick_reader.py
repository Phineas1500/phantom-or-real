#!/usr/bin/env python3
"""Item WW′ reader: pick-then-steer arms paired row-by-row against WW's
baseline / quote_heads3 / steer_gold / steer_heads3 records on the same
rows; pick accuracy; the disagreement-gated blind rule (41st, 42nd)."""
import argparse
import gzip
import json
import re
from collections import Counter, defaultdict

import numpy as np

GATE = 0.1957


def boot(d, n=4000, seed=0):
    d = np.asarray(d, float); rng = np.random.default_rng(seed)
    if not len(d):
        return [None, None]
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]; return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def norm_text(s):
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--records", nargs="+", required=True); ap.add_argument("--ww", nargs="+", required=True); ap.add_argument("--pins", default="docs/wwp_pinned.json")
    ap.add_argument("--frame", default="results/loop_screen/wtq_haystack_input.jsonl.gz"); ap.add_argument("--out", default="docs/wwp_gates.json"); a = ap.parse_args()
    pins = json.load(open(a.pins)); stratum = {rid: j["stratum"] for j in pins["jobs"].values() for rid in j["rows"]}
    frame = {}
    for line in gzip.open(a.frame, "rt"):
        r = json.loads(line); frame[r["id"]] = r
    samples = defaultdict(list); meta = {}
    for p in list(a.ww) + list(a.records):
        for line in open(p):
            r = json.loads(line); samples[(r["id"], r["arm"])].append(r)
            if r["arm"].startswith("pick"):
                meta.setdefault((r["id"], r["arm"]), r)
    rate, maj, majtxt = {}, {}, {}
    for key, rs in samples.items():
        rate[key] = float(np.mean([x["correct"] for x in rs])); top = Counter(x["normalized_output"] for x in rs).most_common(1)[0][0]
        maj[key] = int(next(x["correct"] for x in rs if x["normalized_output"] == top)); majtxt[key] = top
    ids = sorted({rid for rid, _ in samples if rid in stratum}); out = {"delivery": {}, "fixable": {}, "blind": {}}
    dl = [(x["bias_fired"], x["highlight_share_before"], x["highlight_share_after"]) for (rid, arm), rs in samples.items() if arm.startswith("pick") for x in rs[:1]]
    dl_ok = [(f, b, c) for f, b, c in dl if b is not None and c is not None]
    out["delivery"] = {"steered_row_arms": len(dl), "bias_fired_all": bool(all(f >= 1 for f, _, _ in dl)) if dl else None, "share_rose_frac": round(float(np.mean([c > b for _, b, c in dl_ok])), 3) if dl_ok else None}
    print("delivery:", out["delivery"])

    def base_row_of(rid):
        """Row holding the baseline majority answer (label-free): first span whose text contains it."""
        ans = majtxt.get((rid, "baseline"), ""); fr = frame[rid]
        if not ans:
            return None
        for i, sp in enumerate(fr["spans"]):
            if ans in norm_text(fr["docs"][sp["start"]:sp["end"]]):
                return i
        return None
    for pool in ("fixable", "blind"):
        rows = [rid for rid in ids if stratum[rid] == pool and (rid, "baseline") in rate]; res = {"n": len(rows), "baseline_rate": round(float(np.mean([rate[(rid, "baseline")] for rid in rows])), 3), "arms": {}}
        for arm in ("pick3_steer:0.01", "pick5_steer:0.01"):
            rs = [rid for rid in rows if (rid, arm) in rate]
            if not rs:
                continue
            d = [rate[(rid, arm)] - rate[(rid, "baseline")] for rid in rs]; dm = [maj[(rid, arm)] - maj[(rid, "baseline")] for rid in rs]
            pick_gold = [meta[(rid, arm)]["pick_is_gold"] for rid in rs]; parsed = [meta[(rid, arm)]["pick_parsed"] for rid in rs]; top1 = [meta[(rid, arm)]["pick_is_heads_top1"] for rid in rs]
            e = {"n": len(rs), "rate": round(float(np.mean([rate[(rid, arm)] for rid in rs])), 3), "dP": round(float(np.mean(d)), 3), "ci": boot(d), "up": int(sum(x > 0 for x in d)), "down": int(sum(x < 0 for x in d)),
                 "d_majority": round(float(np.mean(dm)), 3), "ci_majority": boot(dm), "pick_is_gold": round(float(np.mean(pick_gold)), 3), "pick_parsed": round(float(np.mean(parsed)), 3), "pick_is_heads_top1": round(float(np.mean(top1)), 3),
                 "dP_pick_gold": round(float(np.mean([rate[(rid, arm)] - rate[(rid, "baseline")] for rid in rs if meta[(rid, arm)]["pick_is_gold"]])), 3) if any(pick_gold) else None,
                 "dP_pick_wrong": round(float(np.mean([rate[(rid, arm)] - rate[(rid, "baseline")] for rid in rs if not meta[(rid, arm)]["pick_is_gold"]])), 3) if not all(pick_gold) else None}
            for other in ("quote_heads3", "steer_heads3:0.01", "steer_gold:0.01"):
                if all((rid, other) in rate for rid in rs):
                    dd = [rate[(rid, arm)] - rate[(rid, other)] for rid in rs]; e[f"vs_{other}"] = {"diff": round(float(np.mean(dd)), 3), "ci": boot(dd)}
            if pool == "blind":
                corr = [rid for rid in rs if maj[(rid, "baseline")] == 1]; fail = [rid for rid in rs if maj[(rid, "baseline")] == 0]
                e["collateral_on_correct"] = round(float(np.mean([rate[(rid, arm)] - rate[(rid, "baseline")] for rid in corr])), 3) if corr else None; e["gain_on_failures"] = round(float(np.mean([rate[(rid, arm)] - rate[(rid, "baseline")] for rid in fail])), 3) if fail else None
                gates = {"disagree": lambda rid: base_row_of(rid) is None or base_row_of(rid) != meta[(rid, arm)]["pick_row"], "peak": lambda rid: meta[(rid, arm)]["peak_share"] < GATE, "always": lambda rid: True}
                for gname, g in gates.items():
                    gated = [rid for rid in rs if g(rid)]; dmg = [(maj[(rid, arm)] if g(rid) else maj[(rid, "baseline")]) - maj[(rid, "baseline")] for rid in rs]; drg = [(rate[(rid, arm)] if g(rid) else rate[(rid, "baseline")]) - rate[(rid, "baseline")] for rid in rs]
                    e[f"rule_{gname}"] = {"gated": len(gated), "net_majority": round(float(np.mean(dmg)), 3), "ci": boot(dmg), "up": int(sum(x > 0 for x in dmg)), "down": int(sum(x < 0 for x in dmg)), "net_rate": round(float(np.mean(drg)), 3), "ci_rate": boot(drg)}
            res["arms"][arm] = e
            print(f"{pool:8s} {arm}: rate {e['rate']:.3f} dP {e['dP']:+.3f} {e['ci']} up/down {e['up']}/{e['down']} | pick gold {e['pick_is_gold']:.2f} parsed {e['pick_parsed']:.2f} top1 {e['pick_is_heads_top1']:.2f} | dP pick-gold {e['dP_pick_gold']} pick-wrong {e['dP_pick_wrong']} | " + " ".join(f"vs {k[3:]} {v['diff']:+.3f} {v['ci']}" for k, v in e.items() if k.startswith("vs_")))
            if pool == "blind":
                for gname in ("disagree", "peak", "always"):
                    v = e[f"rule_{gname}"]; print(f"   rule {gname}: gated {v['gated']} net(majority) {v['net_majority']:+.3f} {v['ci']} up/down {v['up']}/{v['down']} net(rate) {v['net_rate']:+.3f} {v['ci_rate']}  collateral {e['collateral_on_correct']} failures {e['gain_on_failures']}")
        out[pool] = res
    p3 = out["fixable"]["arms"].get("pick3_steer:0.01"); b3 = out["blind"]["arms"].get("pick3_steer:0.01")
    out["verdicts"] = {"41st": None if not p3 else {"dP": p3["dP"], "ci": p3["ci"], "confirmed": bool(p3["dP"] >= 0.20 and p3["ci"][0] > 0)},
                       "42nd": None if not b3 else {"net": b3["rule_disagree"]["net_majority"], "ci": b3["rule_disagree"]["ci"], "gated": b3["rule_disagree"]["gated"], "confirmed": bool(b3["rule_disagree"]["ci"][0] > 0)}}
    print("verdicts:", json.dumps(out["verdicts"])); json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
