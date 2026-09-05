#!/usr/bin/env python3
"""Item WC reader: the chain-of-thought baseline. From cot grades (k=4, the
answer parsed from the last 'Answer:' line): accuracy against the document's
answer vs std; repair share of conflict failures (or doc-dependent failures
on WikiHop) and of hint-repairable rows, split by whether the std wrong
answer was grounded; collateral on correct-majority rows."""
from __future__ import annotations
import argparse, collections, gzip, json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_common import exact_match, normalize_answer, parse_cot_answer  # noqa: E402
from wikihop_w1_gates import boot_ci  # noqa: E402


def ww(a, t): return bool(a) and re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", t, re.I) is not None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grades", type=Path, required=True); p.add_argument("--rows", type=Path, required=True); p.add_argument("--frame", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--failure-key", default="conflict_failure")
    args = p.parse_args()
    frame = {json.loads(l)["id"]: json.loads(l) for l in gzip.open(args.frame, "rt")}
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(args.rows)}
    cot = collections.defaultdict(list); raw = collections.defaultdict(list)
    for l in open(args.grades):
        g = json.loads(l); a = parse_cot_answer(g["model_output"]); cot[g["id"]].append(bool(exact_match(a, frame[g["id"]]["answer"]))); raw[g["id"]].append(a)
    ids = [i for i in frame if i in cot]; k = len(cot[ids[0]])
    modal = lambda o: collections.Counter(normalize_answer(x) for x in o).most_common(1)[0][0]
    std_acc = float(np.mean([rows[i]["std_n_correct"] for i in ids]) / 8); cot_acc = float(np.mean([np.mean(cot[i]) for i in ids]))
    fail = [i for i in ids if rows[i].get(args.failure_key)]; rep = [i for i in fail if rows[i]["hint_repairable"]]
    fixed = lambda i: np.mean(cot[i]) >= 0.5
    grounded = {i: ww(modal(rows[i]["std_outputs"]), frame[i]["docs"].lower()) for i in fail}
    correct = [i for i in ids if rows[i]["std_n_correct"] >= 5]; broken = [i for i in correct if np.mean(cot[i]) <= 0.25]
    out = {"n_rows": len(ids), "k": k, "std_accuracy": std_acc, "cot_accuracy": cot_acc, "delta": cot_acc - std_acc,
           "n_failures": len(fail), "cot_repair_share_of_failures": float(np.mean([fixed(i) for i in fail])) if fail else None, "cot_repair_ci": boot_ci([float(fixed(i)) for i in fail]) if fail else None,
           "hint_repairable": len(rep), "cot_repair_share_of_hint_repairable": float(np.mean([fixed(i) for i in rep])) if rep else None,
           "cot_repair_by_groundedness": {"grounded_wrong_answer": float(np.mean([fixed(i) for i in fail if grounded[i]])) if any(grounded.values()) else None,
                                          "ungrounded_wrong_answer": float(np.mean([fixed(i) for i in fail if not grounded[i]])) if not all(grounded.values()) else None,
                                          "n_grounded": int(sum(grounded.values())), "n_ungrounded": int(len(fail) - sum(grounded.values()))},
           "collateral": {"n_correct_majority": len(correct), "n_broken": len(broken), "share": len(broken) / max(len(correct), 1), "cot_accuracy_on_correct_rows": float(np.mean([np.mean(cot[i]) for i in correct]))},
           "parse_empty_share": float(np.mean([a == "" for i in ids for a in raw[i]]))}
    args.out.write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
