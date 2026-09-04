#!/usr/bin/env python3
"""Item WI reader: the context-faithful instruction baseline on a conflict
frame. From grade_hint grades with the faithful arms (std / closed /
hint_first / faithful / faithful_hint, k=8): accuracy against the document's
answer per arm; the instruction's repair share of conflict failures (memory ∧
std 0/8 → faithful ≥ 5/8) against the hint's; overlap between the two repair
sets; collateral on correct-majority rows (faithful ≤ 3/8); the instruction +
hint ceiling."""
from __future__ import annotations
import argparse, collections, gzip, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_common import exact_match, normalize_answer  # noqa: E402
from wikihop_w1_gates import boot_ci  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grades", type=Path, required=True)
    p.add_argument("--frame", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    frame = {json.loads(l)["id"]: json.loads(l) for l in gzip.open(args.frame, "rt")}
    per = collections.defaultdict(lambda: collections.defaultdict(list)); outs = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(args.grades):
        g = json.loads(line); per[g["id"]][g["arm"]].append(bool(exact_match(g["model_output"], frame[g["id"]]["answer"]))); outs[g["id"]][g["arm"]].append(g["model_output"])
    arms = sorted({a for i in per for a in per[i]})
    ids = list(frame); assert all(len(per[i][a]) == 8 for i in ids for a in arms), "grades incomplete"
    modal = lambda o: collections.Counter(normalize_answer(x) for x in o).most_common(1)[0][0]
    acc = {a: float(np.mean([sum(per[i][a]) for i in ids]) / 8) for a in arms}
    mem = {i: modal(outs[i]["closed"]) == normalize_answer(frame[i]["answer_original"]) for i in ids}
    cf = [i for i in ids if mem[i] and sum(per[i]["std"]) == 0]
    hint_fix = {i for i in cf if sum(per[i]["hint_first"]) >= 4}; faith_fix = {i for i in cf if sum(per[i]["faithful"]) >= 5}; both_fix = {i for i in cf if sum(per[i]["faithful_hint"]) >= 5}
    faith_any = {i for i in cf if sum(per[i]["faithful"]) >= 1}
    correct = [i for i in ids if sum(per[i]["std"]) >= 5]; collateral = [i for i in correct if sum(per[i]["faithful"]) <= 3]
    mem_share_faith = float(np.mean([sum(1 for x in outs[i]["faithful"] if normalize_answer(x) == normalize_answer(frame[i]["answer_original"])) for i in cf]) / 8) if cf else float("nan")
    mem_share_std = float(np.mean([sum(1 for x in outs[i]["std"] if normalize_answer(x) == normalize_answer(frame[i]["answer_original"])) for i in cf]) / 8) if cf else float("nan")
    out = {"n_rows": len(ids), "accuracy_vs_document": acc, "memory_rate": float(np.mean(list(mem.values()))), "n_conflict_failures": len(cf),
           "hint_repairable_share": len(hint_fix) / max(len(cf), 1), "hint_repairable_ci": boot_ci([i in hint_fix for i in cf]),
           "instruction_repair_share": len(faith_fix) / max(len(cf), 1), "instruction_repair_ci": boot_ci([i in faith_fix for i in cf]),
           "instruction_any_sample_share": len(faith_any) / max(len(cf), 1),
           "instruction_plus_hint_repair_share": len(both_fix) / max(len(cf), 1), "instruction_plus_hint_ci": boot_ci([i in both_fix for i in cf]),
           "overlap": {"hint_fixed_also_instruction_fixed": len(hint_fix & faith_fix) / max(len(hint_fix), 1), "instruction_fixed_also_hint_fixed": len(hint_fix & faith_fix) / max(len(faith_fix), 1),
                       "n_hint_only": len(hint_fix - faith_fix), "n_instruction_only": len(faith_fix - hint_fix), "n_both": len(hint_fix & faith_fix), "n_neither": len(cf) - len(hint_fix | faith_fix)},
           "memory_answer_sample_share_on_conflict_failures": {"std": mem_share_std, "faithful": mem_share_faith},
           "collateral": {"n_correct_majority": len(correct), "n_broken_by_instruction": len(collateral), "share": len(collateral) / max(len(correct), 1),
                          "faithful_accuracy_on_correct_rows": float(np.mean([sum(per[i]["faithful"]) for i in correct]) / 8) if correct else float("nan")},
           "faithful_hist_on_conflict_failures": {str(k): v for k, v in sorted(collections.Counter(sum(per[i]["faithful"]) for i in cf).items())}}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
