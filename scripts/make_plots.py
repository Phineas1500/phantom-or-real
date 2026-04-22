"""Phase 4.1 + 4.2 + 5.5 plots: accuracy-vs-depth, structural slicing, error-type bars."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis import build_summary  # noqa: E402

HEIGHTS = [1, 2, 3, 4]


def accuracy_vs_depth(per_task_summaries: dict[str, dict], out_path: Path) -> None:
    """Reproduce Figure 3 layout — one subplot per task, one line per model."""
    tasks = sorted({s["task"] for s in per_task_summaries.values()})
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        for key, s in per_task_summaries.items():
            if s["task"] != task:
                continue
            xs = []
            ys = []
            yerr_lo = []
            yerr_hi = []
            for h in HEIGHTS:
                hkey = f"h{h}"
                if hkey not in s["by_height"]:
                    continue
                acc = s["by_height"][hkey]["strong_accuracy"]
                ci = s["by_height"][hkey]["strong_ci95"]
                if acc is None:
                    continue
                xs.append(h)
                ys.append(acc)
                yerr_lo.append(acc - ci[0])
                yerr_hi.append(ci[1] - acc)
            ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], marker="o", capsize=3, label=s["model"])
        ax.set_xlabel("Tree height")
        ax.set_title(task)
        ax.set_xticks(HEIGHTS)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Strong accuracy")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def structural_bars(per_task_summaries: dict[str, dict], out_path: Path) -> None:
    fig, axes = plt.subplots(
        len(per_task_summaries), 1, figsize=(8, 3 * len(per_task_summaries)), sharex=True
    )
    if len(per_task_summaries) == 1:
        axes = [axes]

    for ax, (key, s) in zip(axes, per_task_summaries.items()):
        labels = HEIGHTS
        direct = []
        nodirect = []
        for h in HEIGHTS:
            hkey = f"h{h}"
            if hkey not in s["by_structure"]:
                direct.append(0)
                nodirect.append(0)
                continue
            d = s["by_structure"][hkey]["has_direct_member"]["strong_accuracy"] or 0
            nd = s["by_structure"][hkey]["no_direct_member"]["strong_accuracy"] or 0
            direct.append(d)
            nodirect.append(nd)
        x = [h - 0.2 for h in labels]
        x2 = [h + 0.2 for h in labels]
        ax.bar(x, direct, width=0.4, label="has_direct_member")
        ax.bar(x2, nodirect, width=0.4, label="no_direct_member")
        ax.set_title(key)
        ax.set_xticks(labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Strong accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Tree height")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def error_type_stack(per_task_summaries: dict[str, dict], out_path: Path) -> None:
    categories = ["wrong_direction", "trivial", "unnecessary", "hallucinated", "unclassified"]
    fig, axes = plt.subplots(
        len(per_task_summaries), 1, figsize=(8, 3 * len(per_task_summaries)), sharex=True
    )
    if len(per_task_summaries) == 1:
        axes = [axes]

    for ax, (key, s) in zip(axes, per_task_summaries.items()):
        bottom = [0.0, 0.0, 0.0, 0.0]
        for cat in categories:
            vals = []
            for h in HEIGHTS:
                hkey = f"h{h}"
                dist = s["error_types"].get(hkey, {})
                total = sum(dist.values()) or 1
                vals.append(dist.get(cat, 0) / total)
            ax.bar(HEIGHTS, vals, bottom=bottom, label=cat)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_title(key)
        ax.set_xticks(HEIGHTS)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of errors")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    axes[-1].set_xlabel("Tree height")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--jsonl", nargs="+", type=Path, required=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for path in args.jsonl:
        s = build_summary(path)
        if s.get("n") == 0:
            continue
        summaries[f"{s['model']}_{s['task']}"] = s

    if not summaries:
        raise SystemExit("No non-empty JSONL inputs")

    accuracy_vs_depth(summaries, args.output_dir / "accuracy_vs_depth.png")
    structural_bars(summaries, args.output_dir / "structural_slicing.png")
    error_type_stack(summaries, args.output_dir / "error_type_stack.png")

    with (args.output_dir / "summaries.json").open("w") as f:
        json.dump(summaries, f, indent=2, default=str)


if __name__ == "__main__":
    main()
