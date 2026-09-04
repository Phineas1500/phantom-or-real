#!/usr/bin/env python3
"""Item WK reader: the blind frame test. Over the uniformly drawn test rows
(records of jobs A+B, own-frame donors; XA+XB, WikiHop donors), each rule
answers every row blind and the frame net is the mean of (rule correct −
baseline correct) with a row-bootstrap CI. Rules: ABSTENTION (registered,
28th prediction; the WD/WE rule: answer with the output-first branch only
when the top answers-fired rate is unique and ≥ 0.5, else keep the
baseline); ALWAYS (output-first, gauge tie-break); GROUNDED two-stage
(flag a row when its baseline answer is not a whole-word span of the
document; flagged rows run the loop with the baseline removed from ties,
others keep the baseline); ORACLE two-stage (flag = std failure) for the
ceiling. Per-stratum breakdown from the stage-1 rows file; the write
reading (gold-address dP and specificity at 2×) on the repairable conflict
rows in the draw; the delivery audit."""
from __future__ import annotations
import argparse, collections, gzip, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_w1_gates import boot_ci  # noqa: E402
from wikihop_common import normalize_answer  # noqa: E402


def whole_word(cand, text):
    return bool(cand) and re.search(r"(?<!\w)" + re.escape(cand) + r"(?!\w)", text, re.IGNORECASE) is not None


def per_row(recs, rung=2.0):
    out = {}
    for i in sorted({r["id"] for r in recs}):
        rr = [r for r in recs if r["id"] == i]
        brec = [r for r in rr if r["condition"] == "baseline"]
        base = float(np.mean([r["correct"] for r in brec])); base_modal = Counter(r["normalized_output"] for r in brec).most_common(1)[0][0]
        br = defaultdict(list); gold_by_rung = defaultdict(list); nongold_by_rung = defaultdict(list); text_gold = []
        audit_bad = 0
        for r in rr:
            if r["condition"] == "delta_write":
                if not r.get("hook_positions_written"):
                    audit_bad += 1
                if r["rung"] == rung:
                    br[r["fired_candidate"]].append(r)
                (gold_by_rung if r["fired_is_gold"] else nongold_by_rung)[r["rung"]].append(r["correct"])
            elif r["condition"] == "text_hint" and r["fired_is_gold"]:
                text_gold.append(r["correct"])
        cands = {c: {"af": float(np.mean([x["answers_fired"] for x in xs])), "corr": float(np.mean([x["correct"] for x in xs])),
                     "gauge": float(xs[0]["gauge_score"]), "gold": bool(xs[0]["fired_is_gold"])} for c, xs in br.items()}
        out[i] = {"base": base, "base_modal": base_modal, "cands": cands, "audit_bad": audit_bad,
                  "gold_write": {str(k): float(np.mean(v)) for k, v in gold_by_rung.items()}, "nongold_write": {str(k): float(np.mean(v)) for k, v in nongold_by_rung.items()},
                  "text_gold": float(np.mean(text_gold)) if text_gold else None}
    return out


def best(cands):
    return max(cands, key=lambda c: (cands[c]["af"], cands[c]["gauge"]))


def rule_value(row, rule, docs=None, is_failure=None):
    c = row["cands"]
    if not c:
        return row["base"], True
    top = max(v["af"] for v in c.values()); tied = [x for x in c if c[x]["af"] == top]
    if rule == "always":
        return c[best(c)]["corr"], False
    if rule == "abstention":
        fire = len(tied) == 1 and top >= 0.5
        return (c[tied[0]]["corr"] if fire else row["base"]), not fire
    if rule in ("grounded", "oracle"):
        flag = (not whole_word(row["base_modal"], docs)) if rule == "grounded" else is_failure
        if not flag or top == 0:
            return row["base"], True
        non_base = [x for x in tied if normalize_answer(x) != row["base_modal"]]
        if not non_base:
            return row["base"], True
        pick = non_base[0] if len(non_base) == 1 else max(non_base, key=lambda x: c[x]["gauge"])
        return c[pick]["corr"], False
    raise ValueError(rule)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", nargs="*", type=Path, default=[])
    p.add_argument("--xrecords", nargs="*", type=Path, default=[])
    p.add_argument("--rows", type=Path, default=Path("results/loop_screen/nqswap_rows.jsonl"))
    p.add_argument("--frame", type=Path, default=Path("results/loop_screen/wk_stage2_input.jsonl.gz"))
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_wk_gates.json"))
    p.add_argument("--prediction", default="28", choices=["28", "grounded-wikihop", "none"],
                   help="which registered statement the verdict line scores: 28 = abstention rule, own donors (WK); grounded-wikihop = grounded two-stage rule, WikiHop vector (WK′ 29th, WS 30th)")
    args = p.parse_args()
    frame = {json.loads(l)["id"]: json.loads(l) for l in gzip.open(args.frame, "rt")}
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(args.rows)}
    def stratum(i):
        r = rows.get(i, {})
        if r.get("conflict_failure"): return "conflict_failure_repairable" if r.get("hint_repairable") else "conflict_failure_unrepairable"
        if r.get("std_n_correct") == 0: return "other_failure"
        if r.get("correct_majority") or (r.get("std_n_correct", 0) >= 5): return "correct_majority"
        return "mixed"
    out = {"arms": {}}
    rng = np.random.default_rng(20260893)
    for arm, files in (("own_donors", args.records), ("wikihop_donors", args.xrecords)):
        if not files:
            continue
        R = per_row([json.loads(l) for f in files for l in open(f)])
        ids = sorted(R); n = len(ids)
        docs = {i: frame[i]["docs"].lower() for i in ids}
        res = {"n_rows": n, "baseline_accuracy": float(np.mean([R[i]["base"] for i in ids])), "delivery_bad_records": int(sum(R[i]["audit_bad"] for i in ids)),
               "n_branches": int(sum(len(R[i]["cands"]) for i in ids)), "rules": {}, "strata": {}}
        for rule in ("abstention", "always", "grounded", "oracle"):
            vals, abst = [], []
            for i in ids:
                v, a = rule_value(R[i], rule, docs=docs[i], is_failure=rows.get(i, {}).get("std_n_correct") == 0)
                vals.append(v - R[i]["base"]); abst.append(a)
            vals = np.array(vals)
            res["rules"][rule] = {"frame_net": float(vals.mean()), "ci95": boot_ci(list(vals)), "abstained_frac": float(np.mean(abst)),
                                  "rule_accuracy": float(np.mean([R[i]["base"] for i in ids]) + vals.mean()),
                                  "n_rows_improved": int((vals > 0).sum()), "n_rows_harmed": int((vals < 0).sum())}
        groups = defaultdict(list)
        for i in ids: groups[stratum(i)].append(i)
        for s, g in sorted(groups.items()):
            d = {"n": len(g), "baseline": float(np.mean([R[i]["base"] for i in g]))}
            for rule in ("abstention", "always", "grounded"):
                vals = [rule_value(R[i], rule, docs=docs[i], is_failure=rows.get(i, {}).get("std_n_correct") == 0)[0] - R[i]["base"] for i in g]
                d[rule + "_dP"] = float(np.mean(vals)); d[rule + "_ci95"] = boot_ci(vals)
            res["strata"][s] = d
        rep = [i for i in ids if stratum(i) == "conflict_failure_repairable"]
        if rep:
            g2 = [R[i]["gold_write"].get("2.0", np.nan) - R[i]["base"] for i in rep]; g1 = [R[i]["gold_write"].get("1.0", np.nan) - R[i]["base"] for i in rep]
            ng2 = [R[i]["nongold_write"].get("2.0", np.nan) for i in rep]
            spec = [a - b for a, b in zip([R[i]["gold_write"].get("2.0", np.nan) for i in rep], ng2) if not (np.isnan(a) or np.isnan(b))]
            res["write_on_repairable_conflict_rows"] = {"n": len(rep), "gold_1x_dP": float(np.nanmean(g1)), "gold_2x_dP": float(np.nanmean(g2)), "gold_2x_ci95": boot_ci([x for x in g2 if not np.isnan(x)]),
                                                        "nongold_address_gold_rate_2x": float(np.nanmean(ng2)), "specificity_2x": float(np.mean(spec)) if spec else None, "specificity_ci95": boot_ci(spec) if spec else None,
                                                        "text_hint_gold": float(np.nanmean([R[i]["text_gold"] if R[i]["text_gold"] is not None else np.nan for i in rep]))}
        out["arms"][arm] = res
    if args.prediction == "28":
        own = out["arms"].get("own_donors", {}).get("rules", {}).get("abstention")
        if own:
            out["prediction_28_BLIND_LOOP_HELPS_AT_FRAME_LEVEL"] = "CONFIRMED" if own["ci95"][0] > 0 else "NOT CONFIRMED"
    elif args.prediction == "grounded-wikihop":
        g = out["arms"].get("wikihop_donors", {}).get("rules", {}).get("grounded")
        if g:
            out["prediction_GROUNDED_TWO_STAGE_HELPS_BLIND"] = "CONFIRMED" if g["ci95"][0] > 0 else "NOT CONFIRMED"
    args.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    for arm, res in out["arms"].items():
        print(f"\n== {arm}: {res['n_rows']} rows, baseline {res['baseline_accuracy']:.3f}, branches {res['n_branches']}, delivery-bad {res['delivery_bad_records']}")
        print("| rule | frame net [CI] | rule accuracy | abstained | rows improved / harmed |\n|---|---|---|---|---|")
        for rule, d in res["rules"].items():
            print(f"| {rule} | **{d['frame_net']:+.3f}** [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | {d['rule_accuracy']:.3f} | {d['abstained_frac']:.0%} | {d['n_rows_improved']} / {d['n_rows_harmed']} |")
        print("| stratum | n | baseline | abstention dP | always dP | grounded dP |\n|---|---|---|---|---|---|")
        for s, d in res["strata"].items():
            print(f"| {s} | {d['n']} | {d['baseline']:.3f} | {d['abstention_dP']:+.3f} [{d['abstention_ci95'][0]:+.3f}, {d['abstention_ci95'][1]:+.3f}] | {d['always_dP']:+.3f} [{d['always_ci95'][0]:+.3f}, {d['always_ci95'][1]:+.3f}] | {d['grounded_dP']:+.3f} [{d['grounded_ci95'][0]:+.3f}, {d['grounded_ci95'][1]:+.3f}] |")
        if "write_on_repairable_conflict_rows" in res:
            print("write on repairable conflict rows:", json.dumps(res["write_on_repairable_conflict_rows"]))
    for k in ("prediction_28_BLIND_LOOP_HELPS_AT_FRAME_LEVEL", "prediction_GROUNDED_TWO_STAGE_HELPS_BLIND"):
        if k in out:
            print(f"\n{k}: {out[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
