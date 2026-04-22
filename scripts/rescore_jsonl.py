"""Re-run the Gemma 3 parser + scorer over an existing JSONL in place.

Needed whenever the parser changes and we want the shipped dataset's
is_correct_strong / is_correct_weak / parse_failed / failure_mode fields
to reflect the new parser without rerunning inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bd_path import ensure_on_path  # noqa: E402
from src.gemma3_parse import parse_hypotheses  # noqa: E402
from src.inference import classify_failure  # noqa: E402

ensure_on_path()

from evaluate import (  # noqa: E402
    compute_quality,
    compute_strong_accuracy,
    compute_weak_accuracy,
    parse_ground_truth,
)


def rescore(in_path: Path, out_path: Path) -> dict:
    changed_correct = 0
    changed_parse = 0
    n = 0

    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            n += 1
            r = json.loads(line)

            reply = r["model_output"]
            pred_hyps = parse_hypotheses(reply)
            gt_hyps = parse_ground_truth(r["ground_truth"])
            raw = r["ontology_raw"]

            strong_acc = compute_strong_accuracy(pred_hyps, gt_hyps)
            weak_acc = compute_weak_accuracy(pred_hyps, gt_hyps, raw["observations"], raw["theories"])
            quality = compute_quality(pred_hyps, gt_hyps, raw["observations"], raw["theories"])
            if strong_acc == 1:
                weak_acc = 1
                quality = 1.0

            new_fm = classify_failure(reply, pred_hyps)
            new_parse_failed = new_fm is not None
            new_strong = bool(strong_acc)

            if new_strong != r["is_correct_strong"]:
                changed_correct += 1
            if new_parse_failed != r["parse_failed"]:
                changed_parse += 1

            r["is_correct_strong"] = new_strong
            r["is_correct_weak"] = bool(weak_acc)
            r["quality_score"] = float(quality)
            r["parse_failed"] = new_parse_failed
            r["failure_mode"] = new_fm
            # Preserve error_type; Teammate B may regenerate after this.
            if r["is_correct_strong"]:
                r["error_type"] = None

            fout.write(json.dumps(r, ensure_ascii=False))
            fout.write("\n")

    return {"path": str(in_path), "n": n, "changed_correct": changed_correct, "changed_parse": changed_parse}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--pattern", default="gemma3_*.jsonl")
    p.add_argument("--in-place", action="store_true", help="Rewrite input files instead of *.rescored.jsonl")
    args = p.parse_args()

    files = sorted(args.in_dir.glob(args.pattern))
    summaries = []
    for f in files:
        out = f if args.in_place else f.with_suffix(".rescored.jsonl")
        if args.in_place:
            tmp = f.with_suffix(f.suffix + ".tmp")
            s = rescore(f, tmp)
            tmp.replace(f)
        else:
            s = rescore(f, out)
        summaries.append(s)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
