#!/usr/bin/env python3
"""Item WM stage-2 reader. Per row and arm: mean correct over k samples.
Frame net of every arm vs the baseline (row-bootstrap CI); the 35th
(write @ retrieved rows > baseline, CI > 0); the 36th (write @ retrieved
rows > retrieval-text baseline, paired CI > 0); the pre-named descriptive
readings (substantial = ≥ +0.10; oracle row; control; retrieved table;
gain conditional on the address hitting the gold row; per-stratum from the
stage-1 rows file); the delivery audit."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_w1_gates import boot_ci  # noqa: E402

ARMS = ["baseline", "retrieved_rows", "retrieved_table", "gold_row", "control_rows", "hint", "retrieval_text"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", nargs="+", type=Path, required=True)
    p.add_argument("--rows", type=Path, default=None, help="stage-1 rows file for strata")
    p.add_argument("--out", type=Path, default=Path("docs/wm_gates.json"))
    args = p.parse_args()
    recs = [json.loads(l) for f in args.records for l in open(f)]
    per = defaultdict(lambda: defaultdict(list)); hits = {}; bad = 0; skipped = defaultdict(int)
    for r in recs:
        if r["condition"] == "skipped":
            skipped[r["arm"]] += 1; continue
        per[r["id"]][r["arm"]].append(r["correct"])
        if r["condition"] == "delta_write" and not r.get("hook_positions_written"): bad += 1
        if "address_hits_gold" in r: hits[(r["id"], r["arm"])] = r["address_hits_gold"]
    ids = sorted(per); strata = {}
    if args.rows and args.rows.exists():
        rows = {json.loads(l)["id"]: json.loads(l) for l in open(args.rows)}
        strata = {i: ("fixable_failure" if rows[i]["failing"] and rows[i]["hint_repairable"] else "unfixable_failure" if rows[i]["failing"] else "correct" if rows[i]["correct_majority"] else "mixed") for i in ids if i in rows}
    M = {a: np.array([np.mean(per[i][a]) if per[i].get(a) else np.nan for i in ids]) for a in ARMS}
    base = M["baseline"]; out = {"n_rows": len(ids), "delivery_bad_records": bad, "skipped_arms": dict(skipped), "baseline_accuracy": float(np.nanmean(base)), "arms": {}}
    print(f"rows {len(ids)} | baseline {np.nanmean(base):.3f} | delivery-bad {bad} | skipped {dict(skipped)}")
    print("| arm | accuracy | frame net vs baseline [CI] | rows up / down |\n|---|---|---|---|")
    for a in ARMS[1:]:
        d = M[a] - base; ok = ~np.isnan(d); dd = d[ok]
        out["arms"][a] = {"accuracy": float(np.nanmean(M[a])), "frame_net": float(dd.mean()), "ci95": boot_ci(list(dd)), "up": int((dd > 0).sum()), "down": int((dd < 0).sum()), "n": int(ok.sum())}
        o = out["arms"][a]; print(f"| {a} | {o['accuracy']:.3f} | **{o['frame_net']:+.3f}** [{o['ci95'][0]:+.3f}, {o['ci95'][1]:+.3f}] | {o['up']} / {o['down']} |")
    pr = M["retrieved_rows"] - M["retrieval_text"]; pr = pr[~np.isnan(pr)]
    out["paired_write_vs_retrieval_text"] = {"delta": float(pr.mean()), "ci95": boot_ci(list(pr))}
    a35 = out["arms"]["retrieved_rows"]; out["prediction_35"] = "CONFIRMED" if a35["ci95"][0] > 0 else "NOT CONFIRMED"
    out["prediction_36"] = "CONFIRMED" if out["paired_write_vs_retrieval_text"]["ci95"][0] > 0 else "NOT CONFIRMED"
    out["substantial_ge_0.10"] = a35["frame_net"] >= 0.10
    hit = np.array([hits.get((i, "retrieved_rows"), False) for i in ids]); d = M["retrieved_rows"] - base
    out["conditional_on_address"] = {"hit_gold": {"n": int(hit.sum()), "dP": float(np.nanmean(d[hit])) if hit.any() else None}, "miss_gold": {"n": int((~hit).sum()), "dP": float(np.nanmean(d[~hit])) if (~hit).any() else None}}
    print(f"\nwrite@retrieved rows vs retrieval-text (paired): {pr.mean():+.3f} {boot_ci(list(pr))}")
    print(f"conditional on the retriever hitting the gold row: hit n={int(hit.sum())} dP={out['conditional_on_address']['hit_gold']['dP']} | miss n={int((~hit).sum())} dP={out['conditional_on_address']['miss_gold']['dP']}")
    if strata:
        out["strata"] = {}
        print("\n| stratum | n | baseline | write@retrieved | retrieval-text | gold row | hint |\n|---|---|---|---|---|---|---|")
        for s in ("fixable_failure", "unfixable_failure", "correct", "mixed"):
            g = [k for k, i in enumerate(ids) if strata.get(i) == s]
            if not g: continue
            row = {a: float(np.nanmean(M[a][g])) for a in ARMS}; out["strata"][s] = {"n": len(g), **row}
            print(f"| {s} | {len(g)} | {row['baseline']:.3f} | {row['retrieved_rows']:.3f} | {row['retrieval_text']:.3f} | {row['gold_row']:.3f} | {row['hint']:.3f} |")
    print(f"\n35th: {out['prediction_35']} | 36th: {out['prediction_36']} | substantial (≥ +0.10): {out['substantial_ge_0.10']}")
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
