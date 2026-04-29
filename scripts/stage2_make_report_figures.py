#!/usr/bin/env python3
"""Generate report-ready Stage 2 figures from committed JSON outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TASKS = ("infer_property", "infer_subtype")
TASK_LABEL = {
    "infer_property": "Property",
    "infer_subtype": "Subtype",
}
SPLITS = ("s1", "s3")
SPLIT_LABEL = {
    "s1": "S1 random",
    "s3": "S3 target heldout",
}
SPLIT_SUFFIX = {
    "s1": "s1",
    "s3": "s3_target_symbol",
}
COLORS = {
    "metadata": "#6f6f6f",
    "raw": "#1f77b4",
    "sae": "#2ca02c",
    "sparse": "#9467bd",
    "error": "#d62728",
    "transcoder": "#ff7f0e",
    "mlp": "#17becf",
}


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def best_auc(path: Path, task: str) -> float:
    data = read_json(path)
    return float(data["best_by_task"][task]["test_auc"])


def raw_l45_auc(docs: Path, split: str, task: str) -> float:
    path = docs / f"raw_probe_27b_{SPLIT_SUFFIX[split]}.json"
    return float(read_json(path)["results"][task]["L45"]["test_auc"])


def baseline_auc(docs: Path, split: str, task: str) -> float:
    path = docs / f"stage2_b0_summary_27b_{split}.json"
    key = f"gemma3-27b__{task}"
    return float(read_json(path)["best_pre_output_baseline"][key][split]["test_auc"])


def sae_reconstruction_auc(docs: Path, split: str, sae_id: str, task: str, probe: str) -> float:
    data = read_json(docs / f"sae_reconstruction_probe_27b_l45_{SPLIT_SUFFIX[split]}.json")
    return float(data["results"][sae_id][task]["probes"][probe]["test_auc"])


def sae_energy(docs: Path, split: str, sae_id: str, task: str) -> float:
    data = read_json(docs / f"sae_reconstruction_probe_27b_l45_{SPLIT_SUFFIX[split]}.json")
    return float(data["results"][sae_id][task]["reconstruction_stats"]["energy_explained"])


def component_auc(docs: Path, split: str, width: str, task: str, component: str) -> float:
    path = docs / f"transcoder_component_probe_27b_l45_{width}_affine_exact_{SPLIT_SUFFIX[split]}.json"
    return float(read_json(path)["results"][task]["probes"][component]["test_auc"])


def add_row(rows: list[dict[str, Any]], figure: str, split: str, task: str, method: str, auc: float) -> None:
    rows.append(
        {
            "figure": figure,
            "split": split,
            "task": task,
            "method": method,
            "auc": f"{auc:.6f}",
        }
    )


def save_rows(rows: list[dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["figure", "split", "task", "method", "auc"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.set_ylim(0.45, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def plot_behavior(docs: Path, out_dir: Path) -> None:
    data = read_json(Path("results/full/summary_accuracy.json"))
    heights = [1, 2, 3, 4]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for task, color in (("infer_property", "#1f77b4"), ("infer_subtype", "#d62728")):
        summary = data[f"gemma3-27b__{task}"]
        ys = [summary["by_height"][f"h{height}"]["strong_accuracy"] for height in heights]
        lo = [ys[idx] - summary["by_height"][f"h{height}"]["strong_ci95"][0] for idx, height in enumerate(heights)]
        hi = [summary["by_height"][f"h{height}"]["strong_ci95"][1] - ys[idx] for idx, height in enumerate(heights)]
        ax.errorbar(
            heights,
            ys,
            yerr=[lo, hi],
            marker="o",
            linewidth=2,
            capsize=3,
            color=color,
            label=TASK_LABEL[task],
        )
    ax.set_title("Gemma 3 27B accuracy collapses with ontology depth")
    ax.set_xlabel("Ontology tree height")
    ax.set_ylabel("Strong accuracy")
    ax.set_xticks(heights)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "stage2_behavior_accuracy_27b.png", dpi=220)
    plt.close(fig)


def plot_probe_overview(docs: Path, out_dir: Path, rows: list[dict[str, Any]]) -> None:
    methods = [
        ("B0 metadata", "metadata", lambda split, task: baseline_auc(docs, split, task)),
        (
            "Residual SAE 16K",
            "sae",
            lambda split, task: best_auc(docs / f"sae_probe_27b_l45_16k_{SPLIT_SUFFIX[split]}.json", task),
        ),
        (
            "Residual SAE 262K",
            "sae",
            lambda split, task: best_auc(docs / f"sae_probe_27b_l45_262k_{SPLIT_SUFFIX[split]}.json", task),
        ),
        (
            "Best sparse concat",
            "sparse",
            lambda split, task: best_auc(
                docs / f"sparse_concat_probe_27b_l30_l45_all_sparse_broadc_{SPLIT_SUFFIX[split]}.json",
                task,
            ),
        ),
        ("Raw L45", "raw", lambda split, task: raw_l45_auc(docs, split, task)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), sharey=True)
    for row_idx, split in enumerate(SPLITS):
        for col_idx, task in enumerate(TASKS):
            ax = axes[row_idx][col_idx]
            labels = []
            values = []
            colors = []
            for label, color_key, getter in methods:
                auc = getter(split, task)
                labels.append(label)
                values.append(auc)
                colors.append(COLORS[color_key])
                add_row(rows, "probe_overview", split, task, label, auc)
            ax.bar(labels, values, color=colors)
            ax.set_title(f"{SPLIT_LABEL[split]}: {TASK_LABEL[task]}")
            style_axis(ax)
            if col_idx == 0:
                ax.set_ylabel("Test AUC")
    fig.suptitle("Sparse features improve over metadata, but raw activations remain strongest", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "stage2_probe_overview_auc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_error(docs: Path, out_dir: Path, rows: list[dict[str, Any]]) -> None:
    sae_ids = {
        "16K": "layer_45_width_16k_l0_small",
        "262K": "layer_45_width_262k_l0_small",
    }
    methods = [
        ("Raw L45", "raw"),
        ("16K recon", "sae"),
        ("16K error", "error"),
        ("262K recon", "sae"),
        ("262K error", "error"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), sharey=True)
    for row_idx, split in enumerate(SPLITS):
        for col_idx, task in enumerate(TASKS):
            ax = axes[row_idx][col_idx]
            values = [
                raw_l45_auc(docs, split, task),
                sae_reconstruction_auc(docs, split, sae_ids["16K"], task, "reconstruction"),
                sae_reconstruction_auc(docs, split, sae_ids["16K"], task, "error"),
                sae_reconstruction_auc(docs, split, sae_ids["262K"], task, "reconstruction"),
                sae_reconstruction_auc(docs, split, sae_ids["262K"], task, "error"),
            ]
            for (label, _), auc in zip(methods, values, strict=True):
                add_row(rows, "reconstruction_error", split, task, label, auc)
            ax.bar([label for label, _ in methods], values, color=[COLORS[key] for _, key in methods])
            e16 = sae_energy(docs, split, sae_ids["16K"], task)
            e262 = sae_energy(docs, split, sae_ids["262K"], task)
            ax.set_title(f"{SPLIT_LABEL[split]}: {TASK_LABEL[task]}\nenergy explained {e16:.3f}/{e262:.3f}")
            style_axis(ax)
            if col_idx == 0:
                ax.set_ylabel("Test AUC")
    fig.suptitle("SAE reconstruction error retains near-raw correctness signal", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "stage2_reconstruction_error_auc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sparse_progression(docs: Path, out_dir: Path, rows: list[dict[str, Any]]) -> None:
    methods = [
        (
            "L45 SAE 262K",
            "sae",
            lambda split, task: best_auc(docs / f"sae_probe_27b_l45_262k_{SPLIT_SUFFIX[split]}.json", task),
        ),
        (
            "Exact TC 262K",
            "transcoder",
            lambda split, task: best_auc(
                docs / f"transcoder_probe_27b_l45_262k_affine_exact_{SPLIT_SUFFIX[split]}.json",
                task,
            ),
        ),
        (
            "L45 four-block",
            "sparse",
            lambda split, task: best_auc(
                docs / f"sparse_concat_probe_27b_l45_resid16k_resid262k_exact_tc262k_mlpout16k_lowc_{SPLIT_SUFFIX[split]}.json",
                task,
            ),
        ),
        (
            "L45 five-block",
            "sparse",
            lambda split, task: best_auc(
                docs / f"sparse_concat_probe_27b_l45_resid16k_resid262k_exacttc16k_exacttc262k_mlpout16k_broadc_{SPLIT_SUFFIX[split]}.json",
                task,
            ),
        ),
        (
            "L30+L45 all",
            "sparse",
            lambda split, task: best_auc(
                docs / f"sparse_concat_probe_27b_l30_l45_all_sparse_broadc_{SPLIT_SUFFIX[split]}.json",
                task,
            ),
        ),
        ("Raw L45", "raw", lambda split, task: raw_l45_auc(docs, split, task)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharey=True)
    for row_idx, split in enumerate(SPLITS):
        for col_idx, task in enumerate(TASKS):
            ax = axes[row_idx][col_idx]
            labels = []
            values = []
            colors = []
            for label, color_key, getter in methods:
                auc = getter(split, task)
                labels.append(label)
                values.append(auc)
                colors.append(COLORS[color_key])
                add_row(rows, "sparse_progression", split, task, label, auc)
            ax.plot(labels, values, color="#333333", linewidth=1.5, marker="o", zorder=3)
            ax.bar(labels, values, color=colors, alpha=0.75, zorder=2)
            ax.set_title(f"{SPLIT_LABEL[split]}: {TASK_LABEL[task]}")
            style_axis(ax)
            if col_idx == 0:
                ax.set_ylabel("Test AUC")
    fig.suptitle("Sparse feature additions narrow, but do not close, the raw gap", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "stage2_sparse_progression_auc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_site_transcoder(docs: Path, out_dir: Path, rows: list[dict[str, Any]]) -> None:
    methods = [
        (
            "Raw mlp_in",
            "raw",
            lambda split, task: best_auc(docs / f"raw_probe_27b_l45_mlp_in_weighted_{SPLIT_SUFFIX[split]}.json", task),
        ),
        (
            "TC 16K",
            "transcoder",
            lambda split, task: best_auc(
                docs / f"transcoder_probe_27b_l45_16k_affine_exact_lowc_{SPLIT_SUFFIX[split]}.json",
                task,
            ),
        ),
        (
            "TC 262K",
            "transcoder",
            lambda split, task: best_auc(
                docs / f"transcoder_probe_27b_l45_262k_affine_exact_{SPLIT_SUFFIX[split]}.json",
                task,
            ),
        ),
        (
            "TC full 262K",
            "transcoder",
            lambda split, task: component_auc(docs, split, "262k", task, "full"),
        ),
        (
            "MLP-out SAE",
            "mlp",
            lambda split, task: best_auc(docs / f"sae_probe_27b_l45_mlp_out_hook_16k_{SPLIT_SUFFIX[split]}.json", task),
        ),
        (
            "Raw mlp_out",
            "raw",
            lambda split, task: best_auc(docs / f"raw_probe_27b_l45_mlp_out_hook_{SPLIT_SUFFIX[split]}.json", task),
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharey=True)
    for row_idx, split in enumerate(SPLITS):
        for col_idx, task in enumerate(TASKS):
            ax = axes[row_idx][col_idx]
            labels = []
            values = []
            colors = []
            for label, color_key, getter in methods:
                auc = getter(split, task)
                labels.append(label)
                values.append(auc)
                colors.append(COLORS[color_key])
                add_row(rows, "site_transcoder", split, task, label, auc)
            ax.bar(labels, values, color=colors)
            ax.set_title(f"{SPLIT_LABEL[split]}: {TASK_LABEL[task]}")
            style_axis(ax)
            if col_idx == 0:
                ax.set_ylabel("Test AUC")
    fig.suptitle("Exact-hook sparse MLP artifacts improve, but same-site raw activations still lead", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "stage2_site_transcoder_auc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(out_dir: Path) -> None:
    text = """# Stage 2 Report Figures

Generated by `scripts/stage2_make_report_figures.py` from committed JSON
outputs in `docs/` and `results/full/summary_accuracy.json`.

- `stage2_behavior_accuracy_27b.png`: 27B strong accuracy vs ontology height.
- `stage2_probe_overview_auc.png`: metadata, residual SAE, best sparse concat,
  and raw L45 probes.
- `stage2_reconstruction_error_auc.png`: SAE reconstruction/error diagnostic.
- `stage2_sparse_progression_auc.png`: sparse-family improvements vs raw.
- `stage2_site_transcoder_auc.png`: exact-hook MLP/transcoder sparse artifacts
  compared with same-site raw activations.
- `stage2_figure_data.csv`: plotted AUC values.
"""
    (out_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures/stage2"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    plot_behavior(args.docs_dir, args.output_dir)
    plot_probe_overview(args.docs_dir, args.output_dir, rows)
    plot_reconstruction_error(args.docs_dir, args.output_dir, rows)
    plot_sparse_progression(args.docs_dir, args.output_dir, rows)
    plot_site_transcoder(args.docs_dir, args.output_dir, rows)
    save_rows(rows, args.output_dir / "stage2_figure_data.csv")
    write_readme(args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
