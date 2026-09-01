#!/usr/bin/env python3
"""Item W1 fallback (pre-named layer sweep) — per-layer write pins.

From a multi-layer candidate capture (`cand_L{L}`, `L{L}_sq_mean_cand`) and
the W0 pins: the class-mean write direction at each layer from the SAME
balanced donor sets W0 pinned (docs/wikihop_w0_pinned.json), and the
massive-dim-excluded per-position amplitude base per layer (same rule as W0:
dims carrying > 5% of pooled mean square are excluded). Also carries every
W0 gauge layer (L38/43/48/53) so the sweep job can read a gauge downstream of
any write layer. Output: wikihop_sweep_pinned.npz (+ JSON).
"""
from __future__ import annotations
import argparse, gzip, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_w0_fit import CAP_LAYERS, SEED_FRAME, fit_full  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frame", type=Path, default=Path("results/loop_screen/wikihop_port_input.jsonl.gz"))
    p.add_argument("--w0-capture", type=Path, default=Path("results/loop_screen/wikihop_w0_capture.npz"))
    p.add_argument("--w0-rows", type=Path, default=Path("results/loop_screen/wikihop_w0_rows.jsonl"))
    p.add_argument("--w0-pinned", type=Path, default=Path("docs/wikihop_w0_pinned.json"))
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-npz", type=Path, default=Path("results/loop_screen/wikihop_sweep_pinned.npz"))
    p.add_argument("--out-json", type=Path, default=Path("docs/wikihop_sweep_pinned.json"))
    args = p.parse_args()
    frame = {}
    for line in gzip.open(args.frame, "rt"):
        r = json.loads(line)
        frame[r["id"]] = r
    rows = {r["id"]: r for r in (json.loads(l) for l in open(args.w0_rows))}
    w0 = json.load(open(args.w0_pinned))
    donors_pos, donors_neg = w0["write"]["donors_correct"], w0["write"]["donors_incorrect"]
    cap = np.load(args.capture)
    man = json.load(open(args.manifest))
    layers = man["write_layers"]
    order = [m["id"] for m in man["rows"]]
    y = np.array([rows[i]["std_majority"] if rows[i]["std_majority"] is not None else -1 for i in order])
    arrays, report = {}, {"donors_correct": donors_pos, "donors_incorrect": donors_neg, "layers": {}}
    w0cap = np.load(args.w0_capture)
    keep_rows = np.array([rows[i]["std_majority"] is not None for i in order])
    for L in CAP_LAYERS:
        X = w0cap[f"L{L}_final"][keep_rows].astype(np.float64)
        w, b, mean = fit_full(X, y[keep_rows])
        arrays[f"gauge_w_L{L}"], arrays[f"gauge_b_L{L}"], arrays[f"gauge_mean_L{L}"] = w, np.array([b]), mean
    arrays["gauge_w"], arrays["gauge_b"], arrays["gauge_mean"] = arrays[f"gauge_w_L{w0['gauge']['primary_layer']}"], arrays[f"gauge_b_L{w0['gauge']['primary_layer']}"], arrays[f"gauge_mean_L{w0['gauge']['primary_layer']}"]
    arrays["gauge_layer"] = np.array([w0["gauge"]["primary_layer"]])
    main_npz = np.load(str(args.w0_pinned).replace("docs/", "results/loop_screen/").replace(".json", ".npz"))
    arrays["class_vector"], arrays["base_norm"], arrays["write_layer"] = main_npz["class_vector"], main_npz["base_norm"], main_npz["write_layer"]
    for L in layers:
        vecs = cap[f"cand_L{L}"].astype(np.float64)
        gold_vec = {}
        for m in man["rows"]:
            for c in m["candidates"]:
                if c["candidate"] == frame[m["id"]]["answer"]:
                    gold_vec[m["id"]] = vecs[c["vec_index"]]
        pos = [gold_vec[i] for i in donors_pos if i in gold_vec]
        neg = [gold_vec[i] for i in donors_neg if i in gold_vec]
        cv = np.stack(pos).mean(0) - np.stack(neg).mean(0)
        sq = cap[f"L{L}_sq_mean_cand"].astype(np.float64)
        pooled = sq.mean(axis=0)
        massive = [int(d) for d in np.argsort(-pooled)[:16] if pooled[d] / pooled.sum() > 0.05]
        keep = np.ones(sq.shape[1], dtype=bool)
        keep[massive] = False
        base = float(np.sqrt(sq[:, keep].sum(axis=1)).mean())
        literal = float(np.sqrt(sq.sum(axis=1)).mean())
        arrays[f"class_vector_L{L}"], arrays[f"base_norm_L{L}"] = cv, np.array([base])
        report["layers"][str(L)] = {"n_donors": [len(pos), len(neg)], "class_vector_norm": float(np.linalg.norm(cv)),
                                    "massive_dims_excluded": massive, "massive_share_of_norm_sq": float(1 - pooled[keep].sum() / pooled.sum()),
                                    "base_norm_pinned": base, "literal_rms_norm": literal, "middle_rung_0.5x": 0.5 * base,
                                    "class_vector_norm_over_base": float(np.linalg.norm(cv) / base)}
        print(f"L{L}: |cv|={np.linalg.norm(cv):.1f} base={base:.1f} (literal {literal:.0f}, massive {massive}) donors {len(pos)}/{len(neg)}", flush=True)
    np.savez(args.out_npz, **arrays)
    args.out_json.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.out_npz} {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
