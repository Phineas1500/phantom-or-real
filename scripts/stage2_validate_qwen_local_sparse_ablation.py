#!/usr/bin/env python3
"""Leave-one-block-out validation for the strongest Qwen local sparse concat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_probe_sparse_concat import run_sparse_concat_probe_grid  # noqa: E402
from src.stage2_probes import DEFAULT_C_VALUES, read_json, write_json  # noqa: E402


TASKS = ["infer_property", "infer_subtype"]
SPLIT_SUFFIX = {
    "s1": "s1",
    "s3": "s3",
}
FULL_REPORT = {
    "s1": Path("docs/qwen_scope_sparse_concat_probe_27b_l53_residual_local_mlpout_localtc_s1.json"),
    "s3": Path("docs/qwen_scope_sparse_concat_probe_27b_l53_residual_local_mlpout_localtc_s3.json"),
}
BLOCK_TEMPLATES = {
    "l53_resid_l0_50": "results/stage2/sae_features/qwen35_27b_{task}_L53_qwenscope_qwen35_27b_w80k_l0_50_top50",
    "l53_resid_l0_100": "results/stage2/sae_features/qwen35_27b_{task}_L53_qwenscope_qwen35_27b_w80k_l0_100_top100",
    "l53_local_mlpout": "results/stage2/sae_features/qwen35_27b_{task}_L53_mlp_out_local_topk_mlp_out_l53_w4096_k64_{split_family}_top64",
    "l53_localtc": "results/stage2/sae_features/qwen35_27b_{task}_L53_mlp_in_weighted_local_topk_tc_mlp_in_weighted_to_mlp_out_l53_w4096_k64_{split_family}_top64",
}


def patterns_for_split(split_family: str) -> dict[str, str]:
    return {
        name: template.format(task="{task}", split_family=split_family)
        for name, template in BLOCK_TEMPLATES.items()
    }


def aucs_by_task(report: dict[str, Any]) -> dict[str, float]:
    return {
        task: float(best["test_auc"])
        for task, best in report["best_by_task"].items()
        if best is not None and best.get("test_auc") is not None
    }


def best_c_values(report: dict[str, Any]) -> tuple[float, ...]:
    values = {
        float(best["best_c"])
        for best in report["best_by_task"].values()
        if best is not None and best.get("best_c") is not None
    }
    return tuple(sorted(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs"))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--c-mode",
        choices=("full-best", "full-grid"),
        default="full-best",
        help="Use the full concat's selected C value(s), or rerun the full regularization grid.",
    )
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", default="liblinear")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--rerun-existing", action="store_true")
    args = parser.parse_args()

    summary: dict[str, Any] = {
        "schema_version": 1,
        "validation": "leave_one_sparse_block_out",
        "full_combo": "qwen_l53_residual_local_mlpout_localtc",
        "block_templates": BLOCK_TEMPLATES,
        "splits": {},
    }

    for split_family, suffix in SPLIT_SUFFIX.items():
        full = read_json(FULL_REPORT[split_family])
        full_auc = aucs_by_task(full)
        c_values = best_c_values(full) if args.c_mode == "full-best" else DEFAULT_C_VALUES
        if not c_values:
            c_values = DEFAULT_C_VALUES

        blocks = patterns_for_split(split_family)
        split_summary: dict[str, Any] = {
            "full_report": str(FULL_REPORT[split_family]),
            "full_test_auc": full_auc,
            "c_values": list(c_values),
            "blocks": blocks,
            "leave_one_out": {},
        }
        for removed_block in blocks:
            patterns = [pattern for name, pattern in blocks.items() if name != removed_block]
            combo_name = f"qwen_l53_residual_local_mlpout_localtc_minus_{removed_block}"
            output = args.out_dir / f"qwen_scope_sparse_concat_ablation_27b_l53_residual_local_mlpout_localtc_minus_{removed_block}_{suffix}.json"
            if output.exists() and not args.rerun_existing:
                report = read_json(output)
                print(f"reusing {output}")
            else:
                report = run_sparse_concat_probe_grid(
                    combo_name=combo_name,
                    feature_patterns=patterns,
                    tasks=TASKS,
                    splits_path=args.splits,
                    split_family=split_family,
                    seed=args.seed,
                    drop_parse_failed=True,
                    c_values=c_values,
                    max_iter=args.max_iter,
                    solver=args.solver,
                    bootstrap_samples=args.bootstrap_samples,
                    dense_active=False,
                )
                write_json(output, report)
            ablated_auc = aucs_by_task(report)
            split_summary["leave_one_out"][removed_block] = {
                "report": str(output),
                "test_auc": ablated_auc,
                "delta_vs_full": {
                    task: ablated_auc[task] - full_auc[task]
                    for task in TASKS
                    if task in full_auc and task in ablated_auc
                },
            }
            print(f"{split_family} minus {removed_block}: {ablated_auc}")
        summary["splits"][split_family] = split_summary

    summary_path = args.out_dir / "qwen_scope_sparse_concat_ablation_27b_l53_residual_local_mlpout_localtc_summary.json"
    write_json(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
