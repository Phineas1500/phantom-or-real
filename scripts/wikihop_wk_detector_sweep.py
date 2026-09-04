#!/usr/bin/env python3
"""Descriptive (post-hoc, unregistered): widening the groundedness detector on
the 360 blind rows already run (WK, WK′: NQ-Swap; WS: counterfactual SQuAD).
Rule G∨A(τ): flag a row when its baseline answer is not a whole-word span of
the passage OR the best non-baseline branch fires at ≥ τ; flagged rows run the
loop with the baseline removed from ties (gauge tie-break), others keep the
baseline. A(τ) alone is reported for reference. Frame net, rows improved /
harmed, flag rates by stratum, per frame and pooled."""
from __future__ import annotations
import argparse, gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_w1_gates import boot_ci  # noqa: E402
from wikihop_common import normalize_answer  # noqa: E402
from wikihop_wk_gates import per_row, whole_word  # noqa: E402

RS = Path("results/loop_screen")
FRAMES = {
    "WK (NQ-Swap draw 1)": dict(x=["wikihop_wk_xa.jsonl", "wikihop_wk_xb.jsonl"], rows="nqswap_rows.jsonl", frame="wk_stage2_input.jsonl.gz"),
    "WK′ (NQ-Swap draw 2)": dict(x=["wikihop_wkp_xa.jsonl", "wikihop_wkp_xb.jsonl"], rows="nqswap_rows.jsonl", frame="wk_stage2_input.jsonl.gz"),
    "WS (counterfactual SQuAD)": dict(x=["wikihop_ws_xa.jsonl", "wikihop_ws_xb.jsonl"], rows="squad_cf_rows.jsonl", frame="ws_stage2_input.jsonl.gz"),
}


def stratum(r):
    if r.get("conflict_failure"): return "repairable" if r.get("hint_repairable") else "unrepairable"
    if r.get("std_n_correct") == 0: return "other_failure"
    return "correct" if r.get("correct_majority") else "mixed"


def alt_max(row):
    c = row["cands"]
    return max((v["af"] for x, v in c.items() if normalize_answer(x) != row["base_modal"]), default=0.0)


def stage2(row):
    c = row["cands"]
    if not c:
        return row["base"]
    top = max(v["af"] for v in c.values()); tied = [x for x in c if c[x]["af"] == top]
    if top == 0:
        return row["base"]
    non_base = [x for x in tied if normalize_answer(x) != row["base_modal"]]
    if not non_base:
        return row["base"]
    pick = non_base[0] if len(non_base) == 1 else max(non_base, key=lambda x: c[x]["gauge"])
    return c[pick]["corr"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--taus", default="0.25,0.5,0.75,1.0")
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_wk_detector_sweep.json"))
    args = p.parse_args()
    taus = [float(t) for t in args.taus.split(",")]
    data = {}
    for name, cfg in FRAMES.items():
        R = per_row([json.loads(l) for f in cfg["x"] for l in open(RS / f)])
        rows = {json.loads(l)["id"]: json.loads(l) for l in open(RS / cfg["rows"])}
        frame = {json.loads(l)["id"]: json.loads(l) for l in gzip.open(RS / cfg["frame"], "rt")}
        for i, r in R.items():
            r["stratum"] = stratum(rows.get(i, {})); r["ungrounded"] = not whole_word(r["base_modal"], frame[i]["docs"].lower()); r["alt"] = alt_max(r)
        data[name] = R
    pooled = {f"{n}:{i}": r for n, R in data.items() for i, r in R.items()}
    out = {}
    print("| frame | rule | flags: rep / unrep / other / correct | frame net [CI] | up / down |\n|---|---|---|---|---|")
    for name, R in list(data.items()) + [("POOLED (360)", pooled)]:
        out[name] = {}
        rules = [("G (grounded only)", lambda r, t=None: r["ungrounded"])]
        for t in taus:
            rules.append((f"G ∨ A(≥{t})", lambda r, t=t: r["ungrounded"] or r["alt"] >= t))
        for t in taus:
            rules.append((f"A(≥{t}) alone", lambda r, t=t: r["alt"] >= t))
        rules.append(("oracle (all failures)", lambda r: r["stratum"] in ("repairable", "unrepairable", "other_failure")))
        for label, flag in rules:
            vals, fl = [], defaultdict(list)
            for i, r in R.items():
                f = flag(r); fl[r["stratum"]].append(f)
                vals.append((stage2(r) if f else r["base"]) - r["base"])
            vals = np.array(vals); ci = boot_ci(list(vals))
            rates = {s: float(np.mean(v)) for s, v in fl.items()}
            out[name][label] = {"frame_net": float(vals.mean()), "ci95": ci, "up": int((vals > 0).sum()), "down": int((vals < 0).sum()), "flag_rates": rates}
            g = lambda s: f"{rates.get(s, float('nan')):.0%}"
            print(f"| {name} | {label} | {g('repairable')} / {g('unrepairable')} / {g('other_failure')} / {g('correct')} | **{vals.mean():+.3f}** [{ci[0]:+.3f}, {ci[1]:+.3f}] | {int((vals > 0).sum())} / {int((vals < 0).sum())} |")
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
