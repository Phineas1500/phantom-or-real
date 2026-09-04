#!/usr/bin/env python3
"""Item WK pins (the knowledge-conflict regime, NQ-Swap). From stage-1
grades (std / closed-book / hint-first, k=8) on the NQ-Swap frame: the
memory rate (closed-book modal == the original answer), conflict rows,
conflict failures (memory ∧ 0/8 std against the document's answer), the
memory-answer share among them, hint-repairable (≥ 4/8 hint-first), the
27th-prediction reading with a row-bootstrap CI, the instrument gate, and
the stage-2 pins: a uniform seeded draw of --n-draw rows from the frame
split into cross-fit halves (job A tests the rows drawn from half 1 with
donors = repairable conflict failures of half 0 outside the draw, B the
reverse; XA/XB the same test rows with the WikiHop WX donors of item WP).
Also writes the stage-2 input frame (NQ-Swap rows + the WikiHop donor rows)."""
from __future__ import annotations
import argparse, collections, gzip, json, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_common import exact_match, normalize_answer  # noqa: E402


def boot_share(flags, n=4000, seed=20260889):
    a = np.array(flags, dtype=float); rng = np.random.default_rng(seed)
    if len(a) == 0:
        return [float("nan"), float("nan")]
    d = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grades", type=Path, default=Path("results/loop_screen/nqswap_grades.jsonl"))
    p.add_argument("--frame", type=Path, default=Path("results/loop_screen/nqswap_input.jsonl.gz"))
    p.add_argument("--wikihop-frame", type=Path, default=Path("results/loop_screen/wikihop_fresh_input.jsonl.gz"))
    p.add_argument("--wp-pins", type=Path, default=Path("docs/wikihop_wp_pinned.json"))
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_wk_pinned.json"))
    p.add_argument("--out-rows", type=Path, default=Path("results/loop_screen/nqswap_rows.jsonl"))
    p.add_argument("--out-stage2-input", type=Path, default=Path("results/loop_screen/wk_stage2_input.jsonl.gz"))
    p.add_argument("--n-draw", type=int, default=120)
    p.add_argument("--max-donors", type=int, default=30)
    p.add_argument("--half-seed", type=int, default=20260890)
    p.add_argument("--draw-seed", type=int, default=20260891)
    p.add_argument("--donor-seed", type=int, default=20260892)
    p.add_argument("--exclude-pins", nargs="*", type=Path, default=[], help="earlier pins whose drawn test rows are excluded from this draw and from the donor pools")
    p.add_argument("--label", default="item WK")
    args = p.parse_args()
    frame = {json.loads(l)["id"]: json.loads(l) for l in gzip.open(args.frame, "rt")}
    per = collections.defaultdict(lambda: collections.defaultdict(list)); outs = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(args.grades):
        g = json.loads(line)
        per[g["id"]][g["arm"]].append(bool(exact_match(g["model_output"], frame[g["id"]]["answer"])))
        outs[g["id"]][g["arm"]].append(g["model_output"])
    ids = list(frame)
    assert all(len(per[i][a]) == 8 for i in ids for a in ("std", "closed", "hint_first")), "grades incomplete"
    def modal(o): return collections.Counter(normalize_answer(x) for x in o).most_common(1)[0][0]
    rows = []
    for i in ids:
        r = frame[i]; s, c, h = (sum(per[i][a]) for a in ("std", "closed", "hint_first"))
        orig = normalize_answer(r["answer_original"])
        closed_mem = sum(1 for x in outs[i]["closed"] if normalize_answer(x) == orig); std_mem = sum(1 for x in outs[i]["std"] if normalize_answer(x) == orig)
        has_memory = modal(outs[i]["closed"]) == orig
        rows.append({"id": i, "std_n_correct": s, "closed_n_correct": c, "hint_first_n_correct": h, "closed_n_memory": closed_mem, "std_n_memory": std_mem,
                     "has_memory": has_memory, "conflict_failure": has_memory and s == 0, "memory_answer": modal(outs[i]["std"]) == orig,
                     "doc_dependent_failing": s == 0 and c == 0, "hint_repairable": h >= 4, "correct_majority": s >= 5,
                     "std_outputs": outs[i]["std"], "closed_outputs": outs[i]["closed"], "hint_outputs": outs[i]["hint_first"]})
    with open(args.out_rows, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = {r["id"]: r for r in rows}
    conflict = [r for r in rows if r["has_memory"]]; cf = [r for r in rows if r["conflict_failure"]]
    rep_cf = [r for r in cf if r["hint_repairable"]]
    failing0 = [r for r in rows if r["std_n_correct"] == 0]; rep_f0 = [r for r in failing0 if r["hint_repairable"]]
    std_acc = sum(r["std_n_correct"] for r in rows) / (8 * len(rows))
    share = len(rep_cf) / max(len(cf), 1); share_ci = boot_share([r["hint_repairable"] for r in cf])
    verdict = "CONFIRMED" if share >= 0.5 and share_ci[0] > 0.25 else "INTERMEDIATE" if share >= 0.25 else "NOT CONFIRMED"
    gate = 0.10 <= std_acc <= 0.90
    excluded = set()
    for f in args.exclude_pins:
        for j in json.load(open(f))["jobs"].values():
            excluded |= set(j["test_rows"])
    halves = list(ids); random.Random(args.half_seed).shuffle(halves); H0, H1 = set(halves[0::2]), set(halves[1::2])
    draw = set(random.Random(args.draw_seed).sample(sorted(set(ids) - excluded), args.n_draw))
    test1, test0 = sorted(draw & H1), sorted(draw & H0)
    def donors(H):
        pool = sorted(r["id"] for r in rep_cf if r["id"] in H and r["id"] not in draw and r["id"] not in excluded)
        return sorted(random.Random(args.donor_seed).sample(pool, args.max_donors)) if len(pool) > args.max_donors else pool
    dA, dB = donors(H0), donors(H1)
    wp = json.load(open(args.wp_pins))["jobs"]; xa, xb = wp["XA"]["donor_rows"], wp["XB"]["donor_rows"]
    wh = {json.loads(l)["id"]: json.loads(l) for l in gzip.open(args.wikihop_frame, "rt")}
    with gzip.open(args.out_stage2_input, "wt") as f:
        for i in ids:
            f.write(json.dumps(frame[i], ensure_ascii=False) + "\n")
        for i in sorted(set(xa) | set(xb)):
            f.write(json.dumps(wh[i], ensure_ascii=False) + "\n")
    strata = lambda t: collections.Counter("conflict_failure_repairable" if by[i]["conflict_failure"] and by[i]["hint_repairable"] else "conflict_failure_unrepairable" if by[i]["conflict_failure"]
                                           else "other_failure" if by[i]["std_n_correct"] == 0 else "correct_majority" if by[i]["correct_majority"] else "mixed" for i in t)
    out = {"registered": f"docs/causal_handle_directions.md {args.label}", "excluded_rows": sorted(excluded), "frame": str(args.frame), "grades": str(args.grades), "n_rows": len(rows),
           "std_accuracy_vs_document": std_acc, "closed_accuracy_vs_document": sum(r["closed_n_correct"] for r in rows) / (8 * len(rows)),
           "hint_first_accuracy": sum(r["hint_first_n_correct"] for r in rows) / (8 * len(rows)),
           "memory_rate_closed_modal_is_original": len(conflict) / len(rows), "closed_memory_sample_rate": sum(r["closed_n_memory"] for r in rows) / (8 * len(rows)),
           "std_memory_sample_rate": sum(r["std_n_memory"] for r in rows) / (8 * len(rows)),
           "n_failing_zero_of_8": len(failing0), "n_conflict_rows": len(conflict), "n_conflict_failures": len(cf),
           "memory_answer_share_of_conflict_failures": sum(r["memory_answer"] for r in cf) / max(len(cf), 1),
           "hint_first_hist_on_conflict_failures": {str(k): v for k, v in sorted(collections.Counter(r["hint_first_n_correct"] for r in cf).items())},
           "n_hint_repairable_conflict_failures": len(rep_cf), "hint_repairable_share_of_conflict_failures": share, "hint_repairable_share_ci95": share_ci,
           "hint_repairable_share_of_all_zero_of_8": len(rep_f0) / max(len(failing0), 1), "n_doc_dependent_failing_wikihop_definition": sum(r["doc_dependent_failing"] for r in rows),
           "n_correct_majority": sum(r["correct_majority"] for r in rows),
           "prediction_27_FIXABLE_MAJORITY": verdict, "instrument_gate_std_in_[0.10,0.90]": gate,
           "seeds": {"half": args.half_seed, "draw": args.draw_seed, "donor": args.donor_seed}, "n_draw": args.n_draw,
           "draw_strata": dict(strata(sorted(draw))),
           "rule": "uniform seeded draw of n_draw rows from the frame, blind to stage 1; halves H0/H1 by seeded shuffle; job A tests the drawn rows of H1 with donors = hint-repairable conflict failures of H0 outside the draw (cap max_donors, seeded); B the reverse; XA/XB = same test rows with the WikiHop WX donors of item WP (cross-task vector)",
           "jobs": {"A": {"test_rows": test1, "donor_rows": dA}, "B": {"test_rows": test0, "donor_rows": dB}, "XA": {"test_rows": test1, "donor_rows": xa}, "XB": {"test_rows": test0, "donor_rows": xb}},
           "pools": {"conflict_failures": [r["id"] for r in cf], "repairable_conflict_failures": [r["id"] for r in rep_cf], "correct_majority": [r["id"] for r in rows if r["correct_majority"]]},
           "stage2_input": str(args.out_stage2_input)}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("n_rows", "std_accuracy_vs_document", "closed_accuracy_vs_document", "hint_first_accuracy", "memory_rate_closed_modal_is_original",
                                          "std_memory_sample_rate", "n_failing_zero_of_8", "n_conflict_failures", "memory_answer_share_of_conflict_failures",
                                          "n_hint_repairable_conflict_failures", "hint_repairable_share_of_conflict_failures", "hint_repairable_share_ci95",
                                          "hint_repairable_share_of_all_zero_of_8", "n_correct_majority", "prediction_27_FIXABLE_MAJORITY", "instrument_gate_std_in_[0.10,0.90]", "draw_strata")}, indent=1))
    print("jobs:", {k: (len(v["test_rows"]), len(v["donor_rows"])) for k, v in out["jobs"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
