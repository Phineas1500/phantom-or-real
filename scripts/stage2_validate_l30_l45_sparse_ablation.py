#!/usr/bin/env python3
"""Leave-one-block-out validation for the L30+L45 sparse concat probe."""

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
    "s3": "s3_target_symbol",
}
FULL_REPORT = {
    "s1": Path("docs/sparse_concat_probe_27b_l30_l45_all_sparse_broadc_s1.json"),
    "s3": Path("docs/sparse_concat_probe_27b_l30_l45_all_sparse_broadc_s3_target_symbol.json"),
}
BLOCKS = {
    "l30_resid16k": "results/stage2/sae_features/gemma3_27b_{task}_L30_layer_30_width_16k_l0_small_top128",
    "l30_resid262k": "results/stage2/sae_features/gemma3_27b_{task}_L30_layer_30_width_262k_l0_small_top128",
    "l45_resid16k": "results/stage2/sae_features/gemma3_27b_{task}_L45_layer_45_width_16k_l0_small_top128",
    "l45_resid262k": "results/stage2/sae_features/gemma3_27b_{task}_L45_layer_45_width_262k_l0_small_top128",
    "l45_tc16k": "results/stage2/sae_features/gemma3_27b_{task}_L45_mlp_in_weighted_layer_45_width_16k_l0_small_affine_top128",
    "l45_tc262k": "results/stage2/sae_features/gemma3_27b_{task}_L45_mlp_in_weighted_layer_45_width_262k_l0_small_affine_top128",
    "l45_mlpout16k": "results/stage2/sae_features/gemma3_27b_{task}_L45_mlp_out_hook_layer_45_width_16k_l0_small_top128",
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
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--rerun-existing", action="store_true")
    args = parser.parse_args()

    summary: dict[str, Any] = {
        "schema_version": 1,
        "validation": "leave_one_sparse_block_out",
        "full_combo": "l30_l45_all_sparse_broadc",
        "blocks": BLOCKS,
        "splits": {},
    }

    for split_family, suffix in SPLIT_SUFFIX.items():
        full = read_json(FULL_REPORT[split_family])
        full_auc = aucs_by_task(full)
        c_values = best_c_values(full) if args.c_mode == "full-best" else DEFAULT_C_VALUES
        if not c_values:
            c_values = DEFAULT_C_VALUES
        split_summary: dict[str, Any] = {
            "full_report": str(FULL_REPORT[split_family]),
            "full_test_auc": full_auc,
            "c_values": list(c_values),
            "leave_one_out": {},
        }
        for removed_block in BLOCKS:
            patterns = [pattern for name, pattern in BLOCKS.items() if name != removed_block]
            combo_name = f"l30_l45_all_sparse_minus_{removed_block}"
            output = args.out_dir / f"sparse_concat_ablation_27b_{combo_name}_{suffix}.json"
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

    summary_path = args.out_dir / "sparse_concat_ablation_27b_l30_l45_all_sparse_summary.json"
    write_json(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
