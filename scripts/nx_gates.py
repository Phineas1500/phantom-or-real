#!/usr/bin/env python3
"""Item NX reader (registered: docs/causal_handle_directions.md item NX).
Stage 2: per-arm gold-address dP vs the in-job baseline; 43rd = NEURON_64
dP >= +0.18 with CI > 0 AND NEURON_64 - RANDOM_64 (draws pooled per row)
row-paired CI > 0. Stage 3: FREEZE_MLP gate (retains >= half of U's repair
-> QK-DIRECT); 44th = U - CLAMP_TOP >= 0.5 x (U - baseline) AND
CLAMP_RAND - CLAMP_TOP row-paired CI > 0. Delivery audit; decomposition
descriptives from the job summaries."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_w1_gates import boot_ci  # noqa: E402


def load(files):
    gold, nongold, base, audit = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), defaultdict(list), [0, 0]
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r["condition"] == "baseline":
                base[r["id"]].append(r["correct"])
                continue
            audit[0] += 1
            audit[1] += any(c < 1 for c in r["hook_prefill_calls"]) or any(c < 1 for c in r["hook_positions_touched"]) or not r["hook_prefill_calls"]
            (gold if r["fired_is_gold"] else nongold)[r["condition"]][r["id"]].append(r["correct"])
    m = lambda d: {a: {i: float(np.mean(v)) for i, v in rows.items()} for a, rows in d.items()}
    return m(gold), m(nongold), {i: float(np.mean(v)) for i, v in base.items()}, {"n_fired_records": audit[0], "n_undelivered": audit[1], "valid": audit[1] == 0}


def arm_table(gold, nongold, base):
    out = {}
    for arm, g in sorted(gold.items()):
        rows = sorted(set(g) & set(base))
        d = [g[i] - base[i] for i in rows]
        rec = {"n": len(rows), "gold_rate": float(np.mean([g[i] for i in rows])), "baseline": float(np.mean([base[i] for i in rows])),
               "dP": float(np.mean(d)), "dP_ci": boot_ci(d)}
        if arm in nongold:
            sp = [g[i] - nongold[arm][i] for i in rows if i in nongold[arm]]
            rec.update({"nongold_address_gold_rate": float(np.mean([nongold[arm][i] for i in rows if i in nongold[arm]])),
                        "specificity": float(np.mean(sp)), "specificity_ci": boot_ci(sp)})
        out[arm] = rec
    return out


def paired(a, b):
    rows = sorted(set(a) & set(b))
    d = [a[i] - b[i] for i in rows]
    return {"n": len(rows), "mean": float(np.mean(d)), "ci": boot_ci(d)}


def emit_neurons(source_npz, summaries, path, n=32, seed=20261020, n_inter=21504, rand_layers=37):
    """Stage-4 neuron list: stage-1 top n by c pooled (mean) over the two cross-fit
    jobs' full vectors, pole = sign of the donor-mean activation change; stage-3
    top n by score summed over the jobs' top-256 lists, pole likewise; n seeded
    random neurons from layers 0..rand_layers-1, positive pole."""
    flats = [np.load(f)["flat_c"] for f in source_npz]
    c = np.mean(flats, axis=0)
    src_layers = np.load(source_npz[0])["src_layers"]
    dact, read_score, read_dact = {}, defaultdict(float), {}
    for s in summaries:
        j = json.loads(s.read_text())
        for t in j.get("top_neurons", []):
            dact.setdefault((t["layer"], t["neuron"]), t["dact_mean"])
        for t in j.get("read_ranking", {}).get("top", []):
            read_score[(t["layer"], t["neuron"])] += t["score"]
            read_dact.setdefault((t["layer"], t["neuron"]), t["dact_mean"])
    out = []
    for i in np.argsort(-c)[:n]:
        L, j = int(src_layers[i // n_inter]), int(i % n_inter)
        out.append({"set": "source", "layer": L, "neuron": j, "pole": 1 if dact.get((L, j), 1.0) >= 0 else -1, "c_pooled": float(c[i])})
    for (L, j), sc in sorted(read_score.items(), key=lambda kv: -kv[1])[:n]:
        out.append({"set": "read", "layer": L, "neuron": j, "pole": 1 if read_dact[(L, j)] >= 0 else -1, "score_pooled": sc})
    for i in np.random.default_rng(seed).choice(rand_layers * n_inter, size=n, replace=False):
        out.append({"set": "random", "layer": int(i // n_inter), "neuron": int(i % n_inter), "pole": 1})
    path.write_text(json.dumps({"neurons": out}, indent=1) + "\n")
    return out


def legibility(explanations, scored, frac=0.2):
    """45th: per neuron, fire = max pole-signed activation >= frac x its corpus fire
    scale; win if positives fire more often than near misses."""
    ex = {f"{n['set']}:{n['layer']}:{n['neuron']}": n for n in json.loads(explanations.read_text())["neurons"]}
    per_set = defaultdict(list)
    rows = []
    for rec in json.loads(scored.read_text())["scored"]:
        n = ex[rec["key"]]
        thr = frac * n["fire_scale"]
        rate = {k: float(np.mean([v >= thr for v in vals])) if vals else float("nan") for k, vals in rec["maxact"].items()}
        outcome = "win" if rate["positive"] > rate["near_miss"] else ("tie" if rate["positive"] == rate["near_miss"] else "lose")
        per_set[n["set"]].append(outcome == "win")
        rows.append({"key": rec["key"], "set": n["set"], "explanation": n["explanation"], "bin": n["bin"], "fire_rates": rate, "outcome": outcome})
    summary = {}
    for k, v in per_set.items():
        summary[k] = {"n": len(v), "win_rate": float(np.mean(v)),
                      "outcomes": {o: sum(r["outcome"] == o for r in rows if r["set"] == k) for o in ("win", "tie", "lose")},
                      "bins": {b: sum(r["bin"] == b for r in rows if r["set"] == k) for b in sorted({r["bin"] for r in rows})}}
    src, rnd = np.array(per_set.get("source", []), float), np.array(per_set.get("random", []), float)
    rng = np.random.default_rng(20260822)
    diffs = [rng.choice(src, len(src)).mean() - rng.choice(rnd, len(rnd)).mean() for _ in range(4000)] if len(src) and len(rnd) else []
    ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))] if diffs else None
    p45 = bool(len(src) and src.mean() >= 0.75 and ci and ci[0] > 0)
    return {"sets": summary, "source_minus_random": {"diff": float(src.mean() - rnd.mean()) if diffs else None, "ci": ci},
            "prediction_45": {"pass": p45}, "neurons": rows}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, nargs="*", default=[])
    p.add_argument("--read", type=Path, nargs="*", default=[])
    p.add_argument("--summaries", type=Path, nargs="*", default=[])
    p.add_argument("--rider", type=Path, nargs="*", default=[], help="rider NX-h records (HEAD_8, RANDOM_HEAD_8_*, NEURON_64_L30)")
    p.add_argument("--out", type=Path, default=Path("docs/nx_gates.json"))
    p.add_argument("--emit-neurons", type=Path, help="write the stage-4 neuron list (needs --source-npz and --summaries) and exit")
    p.add_argument("--source-npz", type=Path, nargs="*", default=[])
    p.add_argument("--explanations", type=Path)
    p.add_argument("--scored", type=Path)
    args = p.parse_args()
    if args.emit_neurons:
        ns = emit_neurons(args.source_npz, args.summaries, args.emit_neurons)
        print(f"{len(ns)} neurons -> {args.emit_neurons}")
        return 0
    out = {"summaries": {}}
    for s in args.summaries:
        j = json.loads(s.read_text())
        out["summaries"][f"{j['stage']}_{j['wx_job']}"] = {k: j.get(k) for k in ("frozen", "decomposition", "arms", "audit", "read_ranking")
                                                          if k in j and k not in ("read_ranking",)} | (
            {"read_top_share_of_score": j["read_ranking"]["top_share_of_score"]} if "read_ranking" in j else {})
    verdict = []
    if args.source:
        gold, nongold, base, audit = load(args.source)
        arms = arm_table(gold, nongold, base)
        rand = defaultdict(list)
        for arm, g in gold.items():
            if arm.startswith("RANDOM_64"):
                for i, v in g.items():
                    rand[i].append(v)
        rand = {i: float(np.mean(v)) for i, v in rand.items()}
        n64 = arms.get("NEURON_64")
        vs_rand = paired(gold["NEURON_64"], rand) if n64 and rand else None
        p43 = bool(n64 and n64["dP"] >= 0.18 and n64["dP_ci"][0] > 0 and vs_rand and vs_rand["ci"][0] > 0)
        out["stage2"] = {"audit": audit, "arms": arms, "neuron64_minus_random64": vs_rand,
                         "u_nomassive_minus_u": paired(gold["U_NOMASSIVE"], gold["U"]) if "U_NOMASSIVE" in gold and "U" in gold else None,
                         "prediction_43": {"pass": p43, "bar_dP": 0.18}}
        verdict.append("FEW-NEURONS-SUFFICE" if p43 else "DISTRIBUTED-SOURCE")
        if not audit["valid"]:
            verdict[-1] += " (EXECUTION-INVALID)"
    if args.read:
        gold, nongold, base, audit = load(args.read)
        arms = arm_table(gold, nongold, base)
        rows = sorted(set(gold["U"]) & set(base))
        u_rep = float(np.mean([gold["U"][i] - base[i] for i in rows]))
        freeze_rep = arms["FREEZE_MLP"]["dP"]
        gate_qk = freeze_rep >= 0.5 * u_rep
        removed_top = u_rep - arms["CLAMP_TOP"]["dP"]
        rand_minus_top = paired(gold["CLAMP_RAND"], gold["CLAMP_TOP"])
        p44 = (not gate_qk) and removed_top >= 0.5 * u_rep and rand_minus_top["ci"][0] > 0
        out["stage3"] = {"audit": audit, "arms": arms, "u_in_job_repair": u_rep,
                         "freeze_gate": {"freeze_repair": freeze_rep, "retained_fraction": freeze_rep / u_rep if u_rep else None, "qk_direct": gate_qk},
                         "u_minus_freeze": paired(gold["U"], gold["FREEZE_MLP"]),
                         "clamp_top_removed": removed_top, "clamp_rand_minus_clamp_top": rand_minus_top,
                         "prediction_44": {"read": not gate_qk, "pass": p44}}
        u_ci = arms["U"]["dP_ci"]
        out["stage3"]["u_reproduced"] = u_ci[0] > 0
        if u_ci[0] <= 0:
            out["stage3"]["prediction_44"] = {"read": False, "pass": False}
            verdict.append("U-NOT-REPRODUCED (stage 3 void)")
        else:
            verdict.append("QK-DIRECT" if gate_qk else ("FEW-NEURONS-READ-IT" if p44 else "DISTRIBUTED-READ"))
        if not audit["valid"]:
            verdict[-1] += " (EXECUTION-INVALID)"
    if args.rider:
        gold, nongold, base, audit = load(args.rider)
        arms = arm_table(gold, nongold, base)
        rand = defaultdict(list)
        for arm, g in gold.items():
            if arm.startswith("RANDOM_HEAD_8"):
                for i, v in g.items():
                    rand[i].append(v)
        rand = {i: float(np.mean(v)) for i, v in rand.items()}
        h8 = arms["HEAD_8"]
        vs_rand = paired(gold["HEAD_8"], rand)
        p46 = h8["dP"] >= 0.18 and h8["dP_ci"][0] > 0 and vs_rand["ci"][0] > 0
        out["rider_h"] = {"audit": audit, "arms": arms, "head8_minus_random_head8": vs_rand, "prediction_46": {"pass": bool(p46), "bar_dP": 0.18}}
        verdict.append("FEW-HEADS-SUFFICE" if p46 else "HEADS-NOT-SUFFICIENT")
        if not audit["valid"]:
            verdict[-1] += " (EXECUTION-INVALID)"
    if args.explanations and args.scored:
        out["stage4"] = legibility(args.explanations, args.scored)
        verdict.append("SOURCE-NEURONS-LEGIBLE" if out["stage4"]["prediction_45"]["pass"] else "NOT-LEGIBLE-AT-THE-BAR")
    out["verdict"] = " / ".join(verdict)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "summaries"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
