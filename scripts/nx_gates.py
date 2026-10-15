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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, nargs="*", default=[])
    p.add_argument("--read", type=Path, nargs="*", default=[])
    p.add_argument("--summaries", type=Path, nargs="*", default=[])
    p.add_argument("--out", type=Path, default=Path("docs/nx_gates.json"))
    args = p.parse_args()
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
    out["verdict"] = " / ".join(verdict)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "summaries"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
