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
    "control": "#8c8c8c",
    "patch": "#bcbd22",
    "recognition": "#009e73",
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


def best_sparse_concat_auc(docs: Path, split: str, task: str) -> float:
    candidates = [
        docs / f"sparse_concat_probe_27b_l30_l45_all_sparse_broadc_{SPLIT_SUFFIX[split]}.json",
        docs / f"sparse_concat_probe_27b_l30_l40_l45_all_sparse_broadc_{SPLIT_SUFFIX[split]}.json",
    ]
    return max(best_auc(path, task) for path in candidates)


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


def save_causal_rows(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
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
            lambda split, task: best_sparse_concat_auc(docs, split, task),
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
            for (label, _), auc in zip(methods, values):
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
        (
            "L30+L40+L45 all",
            "sparse",
            lambda split, task: best_auc(
                docs / f"sparse_concat_probe_27b_l30_l40_l45_all_sparse_broadc_{SPLIT_SUFFIX[split]}.json",
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


def raw_4b_auc(docs: Path, split: str, task: str) -> float:
    path = docs / f"raw_probe_4b_{SPLIT_SUFFIX[split]}.json"
    data = read_json(path)
    return float(data["best_by_task"][task]["test_auc"])


def plot_cross_model_property(docs: Path, out_dir: Path, rows: list[dict[str, Any]]) -> None:
    methods = [
        (
            "Raw",
            "raw",
            lambda model, split: raw_l45_auc(docs, split, "infer_property")
            if model == "27B"
            else raw_4b_auc(docs, split, "infer_property"),
        ),
        (
            "Residual SAE 262K",
            "sae",
            lambda model, split: best_auc(
                docs
                / (
                    f"sae_probe_27b_l45_262k_{SPLIT_SUFFIX[split]}.json"
                    if model == "27B"
                    else f"sae_probe_4b_l22_262k_{SPLIT_SUFFIX[split]}.json"
                ),
                "infer_property",
            ),
        ),
        (
            "Big-L0 TC",
            "transcoder",
            lambda model, split: best_auc(
                docs
                / (
                    f"transcoder_probe_27b_l45_262k_big_affine_exact_top512_{SPLIT_SUFFIX[split]}.json"
                    if model == "27B"
                    else f"transcoder_probe_4b_l22_262k_big_affine_exact_top512_{SPLIT_SUFFIX[split]}.json"
                ),
                "infer_property",
            ),
        ),
        (
            "Best sparse stack",
            "sparse",
            lambda model, split: best_sparse_concat_auc(docs, split, "infer_property")
            if model == "27B"
            else best_auc(docs / f"sparse_concat_probe_4b_l20_l22_all_sparse_{SPLIT_SUFFIX[split]}.json", "infer_property"),
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    bar_width = 0.18
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]
    for ax, model in zip(axes, ("27B", "4B")):
        xs = list(range(len(SPLITS)))
        for (label, color_key, getter), offset in zip(methods, offsets):
            values = [getter(model, split) for split in SPLITS]
            for split, auc in zip(SPLITS, values):
                add_row(rows, "cross_model_property", split, "infer_property", f"{model} {label}", auc)
            ax.bar([x + offset for x in xs], values, width=bar_width, label=label, color=COLORS[color_key])
        ax.set_title(f"Gemma 3 {model} property")
        ax.set_xticks(xs)
        ax.set_xticklabels([SPLIT_LABEL[split] for split in SPLITS])
        ax.set_ylim(0.68, 0.93)
        ax.grid(axis="y", alpha=0.25)
        if ax is axes[0]:
            ax.set_ylabel("Test AUC")
        else:
            ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("4B mirrors the 27B property pattern: sparse improves, raw remains strongest", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "stage2_cross_model_property_auc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def probe_auc(data: dict[str, Any]) -> float:
    for key in ("probe_direction", "bundle_fit", "sparse_probe"):
        if key in data and "test_auc" in data[key]:
            return float(data[key]["test_auc"])
    raise KeyError("No probe test AUC found")


def max_successful_repairs(data: dict[str, Any], prefixes: tuple[str, ...], answer_content: bool = False) -> tuple[int, int]:
    flips = data["summary"].get("flips_vs_baseline") or data["summary"].get("answer_flips_vs_baseline")
    best = 0
    paired_n = 0
    for label, metrics in flips.items():
        if not label.startswith(prefixes):
            continue
        paired_n = max(paired_n, int(metrics.get("paired_n", 0)))
        if answer_content:
            count = max(
                int(metrics.get("polarity_flips_toward_gold", 0)),
                int(metrics.get("predicate_flips_toward_gold", 0)),
                int(metrics.get("strong_false_to_true", 0)),
            )
        else:
            count = int(metrics.get("false_to_true", 0))
        best = max(best, count)
    return best, paired_n


def plot_steering_predictive_vs_causal(docs: Path, out_dir: Path, causal_rows: list[dict[str, Any]]) -> None:
    experiments = [
        ("27B raw\ncorrectness", docs / "raw_steering_27b_l45_property_decode_sweep.json", ("raw_",), False),
        ("27B sparse\nbundle", docs / "sparse_probe_bundle_steering_27b_l45_property.json", ("bundle_",), False),
        ("4B raw\ncorrectness", docs / "raw_steering_4b_l22_property_decode_sweep.json", ("raw_",), False),
        ("4B SAE-error\ncorrectness", docs / "error_steering_4b_l22_16k_property_decode_sweep.json", ("raw_",), False),
        ("4B sparse\nbundle", docs / "bundle_steering_4b_l22_big_affine_property_decode_sweep.json", ("bundle_",), False),
        ("4B answer\npolarity", docs / "answer_property_steering_4b_l22_polarity_decode_sweep.json", ("toward_gold_",), True),
        ("27B answer\npolarity", docs / "answer_property_steering_27b_l45_polarity_smoke.json", ("toward_gold_",), True),
    ]
    labels: list[str] = []
    aucs: list[float] = []
    repairs: list[int] = []
    paired_ns: list[int] = []
    for label, path, prefixes, answer_content in experiments:
        data = read_json(path)
        auc = probe_auc(data)
        repair_count, paired_n = max_successful_repairs(data, prefixes, answer_content=answer_content)
        labels.append(label)
        aucs.append(auc)
        repairs.append(repair_count)
        paired_ns.append(paired_n)
        causal_rows.append(
            {
                "figure": "steering_predictive_vs_causal",
                "experiment": label.replace("\n", " "),
                "probe_auc": f"{auc:.6f}",
                "successful_repairs": repair_count,
                "paired_n": paired_n,
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].bar(labels, aucs, color=COLORS["raw"])
    axes[0].set_ylim(0.80, 1.02)
    axes[0].set_ylabel("Probe test AUC")
    axes[0].set_title("Readout is strong")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, repairs, color=COLORS["error"])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Best directed repair count")
    axes[1].set_title("Directed repairs stay at zero")
    axes[1].grid(axis="y", alpha=0.25)
    for idx, (count, n) in enumerate(zip(repairs, paired_ns)):
        axes[1].text(idx, count + 0.035, f"{count}/{n}", ha="center", va="bottom", fontsize=8)

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
    fig.suptitle("High probe AUC did not translate into reliable answer repair", y=1.04)
    fig.tight_layout()
    fig.savefig(out_dir / "stage2_steering_predictive_vs_causal.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_forced_choice_hardfoil(docs: Path, out_dir: Path, causal_rows: list[dict[str, Any]]) -> None:
    data = read_json(docs / "answer_property_margins_27b_l45_polarity_hardfoil.json")
    baseline = data["summary"]["by_condition"]["baseline"]
    n = int(baseline["n"])
    mcq_correct = int(round(float(baseline["mcq_choice_accuracy"]) * n))
    conditions = [
        ("Toward gold", "toward_gold_pos2sd", COLORS["raw"]),
        ("Orthogonal", "orthogonal_pos2sd", COLORS["control"]),
        ("Away gold", "away_gold_pos2sd", COLORS["error"]),
    ]
    original_deltas = []
    mcq_deltas = []
    for label, key, _ in conditions:
        delta = data["summary"]["margin_deltas_vs_baseline"][key]
        original_deltas.append(float(delta["mean_original_margin_delta"]))
        mcq_deltas.append(float(delta["mean_mcq_margin_delta"]))
        causal_rows.append(
            {
                "figure": "forced_choice_hardfoil",
                "condition": label,
                "mean_original_margin_delta": f"{original_deltas[-1]:.6f}",
                "mean_mcq_margin_delta": f"{mcq_deltas[-1]:.6f}",
                "mcq_choice_changed": int(delta["mcq_choice_changed"]),
                "paired_n": int(delta["paired_n"]),
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    axes[0].bar(["Free-form\nselected wrong", "MCQ\ngold vs own foil"], [0.0, mcq_correct / n], color=[COLORS["error"], COLORS["recognition"]])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Recognition survives free-form failure")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].text(0, 0.04, f"0/{n}", ha="center", va="bottom", fontsize=9)
    axes[0].text(1, mcq_correct / n + 0.04, f"{mcq_correct}/{n}", ha="center", va="bottom", fontsize=9)

    xs = list(range(len(conditions)))
    width = 0.34
    axes[1].bar([x - width / 2 for x in xs], original_deltas, width=width, label="Original prompt margin", color=COLORS["raw"])
    axes[1].bar([x + width / 2 for x in xs], mcq_deltas, width=width, label="MCQ margin", color=COLORS["recognition"])
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([label for label, _, _ in conditions])
    axes[1].set_ylabel("Mean delta vs baseline")
    axes[1].set_title("No choice flips; margin shifts are small/non-antisymmetric")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].text(1, max(original_deltas + mcq_deltas) * 0.85, "0 MCQ choice flips", ha="center", fontsize=9)

    fig.suptitle("Hard-foil forced choice: model recognizes the answer, steering still does not flip it", y=1.04)
    fig.tight_layout()
    fig.savefig(out_dir / "stage2_forced_choice_hardfoil.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_patching_asymmetry(docs: Path, out_dir: Path, causal_rows: list[dict[str, Any]]) -> None:
    forward = read_json(docs / "clean_to_corrupt_patching_27b_property_margin_pilot.json")
    reverse = read_json(docs / "corrupt_to_clean_patching_27b_property_margin_pilot.json")
    layers = [35, 40, 45, 50]
    series = [
        ("h1->h4 clean repair", "forward_clean", COLORS["patch"]),
        ("h1->h4 noise repair", "forward_noise", COLORS["control"]),
        ("h4->h1 corrupt breakage", "reverse_corrupt", COLORS["error"]),
        ("h4->h1 noise breakage", "reverse_noise", "#bdbdbd"),
    ]
    values: dict[str, list[float]] = {key: [] for _, key, _ in series}
    for layer in layers:
        fc = float(forward["summary"][f"clean_L{layer}_last_prompt"]["mean_recovery_fraction"])
        fn = float(forward["summary"][f"noise_L{layer}_last_prompt"]["mean_recovery_fraction"])
        rc = float(reverse["summary"][f"corrupt_L{layer}_last_prompt"]["mean_breakage_fraction"])
        rn = float(reverse["summary"][f"noise_L{layer}_last_prompt"]["mean_breakage_fraction"])
        values["forward_clean"].append(fc)
        values["forward_noise"].append(fn)
        values["reverse_corrupt"].append(rc)
        values["reverse_noise"].append(rn)
        causal_rows.extend(
            [
                {"figure": "patching_asymmetry", "layer": layer, "series": "h1->h4 clean repair", "value": f"{fc:.6f}"},
                {"figure": "patching_asymmetry", "layer": layer, "series": "h1->h4 noise repair", "value": f"{fn:.6f}"},
                {"figure": "patching_asymmetry", "layer": layer, "series": "h4->h1 corrupt breakage", "value": f"{rc:.6f}"},
                {"figure": "patching_asymmetry", "layer": layer, "series": "h4->h1 noise breakage", "value": f"{rn:.6f}"},
            ]
        )

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    xs = list(range(len(layers)))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for (label, key, color), offset in zip(series, offsets):
        ax.bar([x + offset for x in xs], values[key], width=width, label=label, color=color)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{layer}" for layer in layers])
    ax.set_ylabel("Mean fraction\n(repair or breakage)")
    ax.set_title("Forward repair is noise-like; reverse corrupt-state disruption is larger than noise")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "stage2_patching_asymmetry.png", dpi=220, bbox_inches="tight")
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
- `stage2_cross_model_property_auc.png`: 27B/4B property AUC comparison.
- `stage2_steering_predictive_vs_causal.png`: probe AUCs versus directed
  repair counts.
- `stage2_forced_choice_hardfoil.png`: MCQ recognition despite free-form
  failures and hard-foil margin deltas.
- `stage2_patching_asymmetry.png`: forward repair and reverse breakage
  patching effects against matched noise controls.
- `stage2_figure_data.csv`: plotted AUC values.
- `stage2_causal_figure_data.csv`: plotted causal-test values.
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
    plot_cross_model_property(args.docs_dir, args.output_dir, rows)
    causal_rows: list[dict[str, Any]] = []
    plot_steering_predictive_vs_causal(args.docs_dir, args.output_dir, causal_rows)
    plot_forced_choice_hardfoil(args.docs_dir, args.output_dir, causal_rows)
    plot_patching_asymmetry(args.docs_dir, args.output_dir, causal_rows)
    save_rows(rows, args.output_dir / "stage2_figure_data.csv")
    save_causal_rows(causal_rows, args.output_dir / "stage2_causal_figure_data.csv")
    write_readme(args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
