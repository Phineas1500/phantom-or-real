#!/usr/bin/env python3
"""Subtype capture-ladder and targeted patch discriminator.

Consumes docs/subtype_recognition_gap_27b_manifest.json. First measures the
hinted-vs-unhinted concept-position delta norm across a ladder of residual
layers, then dynamically selects the strongest off-L30/40/45 layers for
causal tests. The targeted arms distinguish a simple layer-mismatch account
from a broader "not carried by this residual-state family" null.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass(frozen=True)
class Arm:
    label: str
    kind: str
    reference: str
    layers: tuple[int, ...] = ()
    rank_k: int | None = None
    basis_mode: str | None = None


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"expected a comma-separated integer list, got {text!r}")
    return values


def unique_preserve(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def load_gap_rows(jsonl: Path, manifest: Path) -> list[dict[str, Any]]:
    indices = set(json.load(manifest.open())["gap_row_indices"])
    rows: list[dict[str, Any]] = []
    with jsonl.open() as f:
        for row_index, line in enumerate(f):
            if row_index in indices:
                row = json.loads(line)
                row["row_index"] = row_index
                rows.append(row)
    return rows


def summarize_layer_deltas(delta_by_row: dict[int, np.ndarray]) -> dict[str, Any]:
    row_means: list[float] = []
    row_maxes: list[float] = []
    all_norms: list[np.ndarray] = []
    for row_id in sorted(delta_by_row):
        arr = np.asarray(delta_by_row[row_id], dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        norms = np.linalg.norm(arr, axis=1)
        all_norms.append(norms)
        row_means.append(float(norms.mean()))
        row_maxes.append(float(norms.max()))
    if not row_means:
        return {
            "n_rows": 0,
            "n_positions": 0,
            "row_mean_delta_norm_mean": 0.0,
            "row_mean_delta_norm_median": 0.0,
            "row_max_delta_norm_mean": 0.0,
            "position_delta_norm_mean": 0.0,
            "position_delta_norm_median": 0.0,
            "position_delta_norm_rms": 0.0,
            "position_delta_norm_max": 0.0,
        }
    pooled = np.concatenate(all_norms)
    return {
        "n_rows": len(row_means),
        "n_positions": int(pooled.shape[0]),
        "row_mean_delta_norm_mean": float(np.mean(row_means)),
        "row_mean_delta_norm_median": float(np.median(row_means)),
        "row_max_delta_norm_mean": float(np.mean(row_maxes)),
        "position_delta_norm_mean": float(np.mean(pooled)),
        "position_delta_norm_median": float(np.median(pooled)),
        "position_delta_norm_rms": float(np.sqrt(np.mean(pooled**2))),
        "position_delta_norm_max": float(np.max(pooled)),
    }


def select_offtrio_layers(
    capture_ladder: list[dict[str, Any]],
    old_layers: list[int],
    top_n: int,
    *,
    metric: str = "row_mean_delta_norm_mean",
) -> list[int]:
    old = set(old_layers)
    candidates = [entry for entry in capture_ladder if int(entry["layer"]) not in old]
    ranked = sorted(candidates, key=lambda entry: (-float(entry.get(metric, 0.0)), int(entry["layer"])))
    return [int(entry["layer"]) for entry in ranked[: max(0, top_n)]]


def build_arms(old_layers: list[int], selected_layers: list[int], rank_k: int) -> list[Arm]:
    arms = [
        Arm("baseline", "none", "none"),
        Arm("old_trio_full_replace_L" + "_".join(str(layer) for layer in old_layers), "full_replace", "baseline", tuple(old_layers)),
    ]
    for layer in selected_layers:
        arms.append(Arm(f"L{layer}_concept_replace", "concept_replace", "baseline", (layer,)))
    for layer in selected_layers:
        arms.append(Arm(f"L{layer}_random_replace", "random_replace", "baseline", (layer,)))
    if selected_layers and rank_k > 0:
        arms.append(Arm(f"L{selected_layers[0]}_rank{rank_k}_loo_add", "rank_k_add", "baseline", (selected_layers[0],), rank_k, "leave_one_row_out"))
    return arms


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def fit_pca_basis(delta_by_row: dict[int, np.ndarray], rank_k: int, exclude_rows: set[int] | None = None) -> dict[str, Any]:
    exclude_rows = exclude_rows or set()
    source = {row: arr for row, arr in delta_by_row.items() if row not in exclude_rows}
    if not source:
        raise ValueError("cannot fit PCA basis with no source rows")
    pooled = np.concatenate([source[row] for row in sorted(source)], axis=0).astype(np.float64)
    mean = pooled.mean(axis=0)
    centered = pooled - mean[None, :]
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    keep = min(rank_k, vt.shape[0])
    denom = float((singular**2).sum())
    explained = 0.0 if denom == 0.0 else float((singular[:keep] ** 2).sum() / denom)
    return {
        "mean": mean,
        "components": vt[:keep],
        "requested_rank_k": rank_k,
        "effective_rank_k": keep,
        "source_rows": sorted(source),
        "exclude_rows": sorted(exclude_rows),
        "n_pooled_positions": int(pooled.shape[0]),
        "explained_variance_ratio": explained,
    }


def rank_k_reconstruction(delta: Any, basis: dict[str, Any]) -> Any:
    import torch

    arr = delta.detach().cpu().numpy().astype(np.float64)
    centered = arr - basis["mean"][None, :]
    coords = centered @ basis["components"].T
    recon = basis["mean"][None, :] + coords @ basis["components"]
    return torch.from_numpy(recon.astype(np.float32))


def make_replace_hook(states: Any, positions: list[int]):
    def hook_fn(act: Any, hook: Any) -> Any:  # noqa: ARG001
        if not positions or act.shape[1] <= max(positions):
            return act
        donor = states.to(device=act.device, dtype=act.dtype)
        for row_index, pos in enumerate(positions):
            act[:, pos, :] = donor[row_index]
        return act

    return hook_fn


def make_position_add_hook(matrix: Any, positions: list[int], scale: float):
    def hook_fn(act: Any, hook: Any) -> Any:  # noqa: ARG001
        if not positions or act.shape[1] <= max(positions):
            return act
        add = (scale * matrix).to(device=act.device, dtype=act.dtype)
        for row_index, pos in enumerate(positions):
            act[:, pos, :] += add[row_index]
        return act

    return hook_fn


def row_bootstrap_ci(deltas: np.ndarray, rng: np.random.Generator, n_boot: int = 10000) -> tuple[float, float]:
    if deltas.size == 0:
        return float("nan"), float("nan")
    draws = [float(np.mean(deltas[rng.integers(0, len(deltas), len(deltas))])) for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_generation_rows(rows_out: list[dict[str, Any]], arms: list[Arm], seed: int) -> dict[str, Any]:
    by = defaultdict(lambda: defaultdict(list))
    for row in rows_out:
        by[row["condition"]][row["source_row_index"]].append(row)
    refs = {arm.label: arm.reference for arm in arms}
    rng = np.random.default_rng(seed)
    summary: dict[str, Any] = {}
    for cond in sorted(by):
        per_row = {i: float(np.mean([sample["is_correct_strong"] for sample in samples])) for i, samples in by[cond].items()}
        flat = [sample for samples in by[cond].values() for sample in samples]
        entry: dict[str, Any] = {
            "n_rows": len(per_row),
            "n_samples": len(flat),
            "strong_accuracy": float(np.mean([sample["is_correct_strong"] for sample in flat])),
            "reference": refs.get(cond, "none"),
        }
        ref = refs.get(cond, "none")
        if ref != "none" and ref in by:
            ref_rows = {i: float(np.mean([sample["is_correct_strong"] for sample in samples])) for i, samples in by[ref].items()}
            deltas = np.array([per_row[i] - ref_rows[i] for i in sorted(ref_rows) if i in per_row], dtype=float)
            lo, hi = row_bootstrap_ci(deltas, rng)
            entry["paired_delta_vs_reference"] = float(np.mean(deltas)) if deltas.size else float("nan")
            entry["paired_ci95"] = [lo, hi]
        summary[cond] = entry
    return summary


def write_markdown_summary(path: Path, report: dict[str, Any]) -> None:
    job = report.get("slurm_job_id") or "local"
    lines = [
        f"# Subtype Capture-Ladder Discriminator - Job {job}",
        "",
        f"Output JSON: `{report['output']}`",
        f"Rows: {report['prepared_rows']} prepared from {report['selected_rows']} manifest rows.",
        "",
        "## Capture ladder",
        "",
        "| layer | old trio? | row-mean delta norm | position mean | position rms |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for entry in report["capture_ladder"]:
        lines.append(
            "| {layer} | {old} | {row:.3f} | {mean:.3f} | {rms:.3f} |".format(
                layer=entry["layer"],
                old="yes" if entry["is_old_trio"] else "no",
                row=entry["row_mean_delta_norm_mean"],
                mean=entry["position_delta_norm_mean"],
                rms=entry["position_delta_norm_rms"],
            )
        )
    lines.extend(
        [
            "",
            "Selected off-trio layers: `" + ",".join(str(layer) for layer in report["selected_layers"]) + "`.",
            "",
            "## Causal arms",
            "",
            "| arm | P(strong) | dP vs reference (CI95) | reference |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for cond, entry in sorted(report["summary"].items()):
        if "paired_delta_vs_reference" in entry:
            ci = entry["paired_ci95"]
            delta = f"{entry['paired_delta_vs_reference']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        else:
            delta = "-"
        lines.append(f"| {cond} | {entry['strong_accuracy']:.3f} | {delta} | {entry['reference']} |")
    lines.extend(
        [
            "",
            "Reading rule: off-trio concept replacement repair supports layer mismatch; old-trio null plus off-trio null, especially with large capture norms and null random controls, supports insufficiency of this residual-state route rather than a splice bug.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_subtype.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/subtype_recognition_gap_27b_manifest.json"))
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--task", default="infer_subtype")
    parser.add_argument("--ladder-layers", default="15,20,25,30,35,40,45,50,53")
    parser.add_argument("--old-trio-layers", default="30,40,45")
    parser.add_argument("--top-offtrio-layers", type=int, default=3)
    parser.add_argument("--rank-k", type=int, default=4)
    parser.add_argument("--min-block-tokens", type=int, default=32)
    parser.add_argument("--samples-per-row", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=20260620)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/subtype_discriminator_27b.jsonl"))
    parser.add_argument("--states-output", type=Path, default=Path("results/stage2/erasure/subtype_discriminator_27b_states.npz"))
    parser.add_argument("--output", type=Path, default=Path("docs/subtype_discriminator_27b.json"))
    parser.add_argument("--summary-md", type=Path, default=Path("docs/subtype_discriminator_27b_summary.md"))
    args = parser.parse_args()
    started = time.time()

    ladder_layers = unique_preserve(parse_int_list(args.ladder_layers))
    old_layers = unique_preserve(parse_int_list(args.old_trio_layers))
    ladder_layers = unique_preserve(ladder_layers + [layer for layer in old_layers if layer not in ladder_layers])
    selected_rows = load_gap_rows(args.jsonl, args.manifest)
    planned_arm_count = 2 + 2 * args.top_offtrio_layers + (1 if args.rank_k > 0 and args.top_offtrio_layers > 0 else 0)
    total = len(selected_rows) * planned_arm_count * args.samples_per_row
    print(
        f"selected_rows={len(selected_rows)} ladder_layers={ladder_layers} "
        f"old_trio_layers={old_layers} planned_arms={planned_arm_count} total_generations_pre_skip={total}",
        flush=True,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "rows": [int(row["row_index"]) for row in selected_rows],
                    "ladder_layers": ladder_layers,
                    "old_trio_layers": old_layers,
                    "top_offtrio_layers": args.top_offtrio_layers,
                    "rank_k": args.rank_k,
                    "planned_arm_count_after_ladder": planned_arm_count,
                    "total_generations_pre_skip": total,
                    "dynamic_selection_metric": "row_mean_delta_norm_mean",
                },
                indent=2,
                default=json_default,
            ),
            flush=True,
        )
        return 0

    import torch
    from dotenv import load_dotenv

    from scripts.stage2_decode_time_correction import torch_dtype  # noqa: E402
    from scripts.stage2_hint_delta import concept_positions, prompt_cache  # noqa: E402
    from scripts.stage2_hint_state_interchange import generate_sample_batch  # noqa: E402
    from scripts.stage2_interchange_concept_analysis import canon, subjects_of  # noqa: E402
    from scripts.stage2_proposal_hints import make_user_prompt  # noqa: E402
    from scripts.stage2_recognition_state_patch import longest_common_token_block  # noqa: E402
    from src.activations import load_tl_model, render_chat_text, validate_hooks  # noqa: E402
    from src.bd_path import ensure_on_path  # noqa: E402
    from src.stage2_steering import score_reply  # noqa: E402

    load_dotenv()
    torch.set_grad_enabled(False)
    ensure_on_path()
    model = load_tl_model(args.model, n_devices=args.n_devices, n_ctx=args.n_ctx, dtype=torch_dtype(args.dtype), load_mode=args.load_mode)
    dtype = torch_dtype(args.dtype)
    hook_names = validate_hooks(model, ladder_layers)
    hook_by_layer = dict(zip(ladder_layers, hook_names))
    tokenizer = model.tokenizer
    print(f"using_hooks={hook_by_layer}", flush=True)

    prepared = []
    delta_by_layer: dict[int, dict[int, np.ndarray]] = {layer: {} for layer in ladder_layers}
    for stage1_row in selected_rows:
        source_row_index = int(stage1_row["row_index"])
        gold_concept = stage1_row["ontology_fol_structured"]["hypothesis"]["subject"]
        receiver_text = render_chat_text(tokenizer, system=stage1_row["system_prompt"], user=stage1_row["prompt_text"], model_name=args.model, add_generation_prompt=True)
        receiver_ids = tokenizer(receiver_text, add_special_tokens=False)["input_ids"]
        hinted_text = render_chat_text(tokenizer, system=stage1_row["system_prompt"], user=make_user_prompt(stage1_row, "hint_concept_first"), model_name=args.model, add_generation_prompt=True)
        hinted_ids = tokenizer(hinted_text, add_special_tokens=False)["input_ids"]
        h_start, r_start, block_len = longest_common_token_block(hinted_ids, receiver_ids)
        if block_len < args.min_block_tokens:
            print(f"skip row {source_row_index}: block={block_len}", flush=True)
            continue
        positions_r = concept_positions(tokenizer, receiver_text, gold_concept, r_start, block_len)
        if not positions_r:
            print(f"skip row {source_row_index}: no concept positions", flush=True)
            continue
        rel = [pos - r_start for pos in positions_r]
        rng = random.Random(args.sample_seed + source_row_index)
        random_rel = sorted(rng.sample(range(block_len), len(rel)))
        hinted_cache = prompt_cache(model, hinted_ids, hook_names)
        unhinted_cache = prompt_cache(model, receiver_ids, hook_names)
        per_layer: dict[int, dict[str, Any]] = {}
        for layer in ladder_layers:
            hook_name = hook_by_layer[layer]
            h_block = hinted_cache[hook_name][h_start : h_start + block_len].detach().cpu()
            u_block = unhinted_cache[hook_name][r_start : r_start + block_len].detach().cpu()
            concept_delta = h_block[rel] - u_block[rel]
            per_layer[layer] = {"h_block": h_block, "concept_delta": concept_delta}
            delta_by_layer[layer][source_row_index] = concept_delta.numpy().astype(np.float32)
        prepared.append(
            {
                "row": stage1_row,
                "source_row_index": source_row_index,
                "gold_concept": gold_concept,
                "receiver_ids": receiver_ids,
                "r_start": r_start,
                "block_len": block_len,
                "rel": rel,
                "random_rel": random_rel,
                "per_layer": per_layer,
            }
        )
        print(f"prepared row {source_row_index}: block={block_len} concept_tokens={len(rel)}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    capture_ladder = []
    for layer in ladder_layers:
        entry = {
            "layer": layer,
            "is_old_trio": layer in set(old_layers),
            **summarize_layer_deltas(delta_by_layer[layer]),
        }
        capture_ladder.append(entry)
    selected_layers = select_offtrio_layers(capture_ladder, old_layers, args.top_offtrio_layers)
    arms = build_arms(old_layers, selected_layers, args.rank_k)
    total_after_skip = len(prepared) * len(arms) * args.samples_per_row
    print(f"selected_offtrio_layers={selected_layers} arms={[arm.label for arm in arms]} total_generations={total_after_skip}", flush=True)

    basis_cache: dict[tuple[int, int, int], dict[str, Any]] = {}

    def basis_for(layer: int, row_id: int, rank_k: int) -> dict[str, Any]:
        key = (layer, row_id, rank_k)
        if key not in basis_cache:
            basis_cache[key] = fit_pca_basis(delta_by_layer[layer], rank_k, exclude_rows={row_id})
        return basis_cache[key]

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    basis_records: list[dict[str, Any]] = []
    with args.out_jsonl.open("w", encoding="utf-8") as fout:
        for arm_index, arm in enumerate(arms):
            arm_started = time.time()
            for prep_index, prep in enumerate(prepared):
                torch.manual_seed(args.sample_seed + prep["source_row_index"] * 10007 + arm_index * 101)
                start = prep["r_start"]
                fwd_hooks = []
                basis = None
                if arm.kind == "full_replace":
                    for layer in arm.layers:
                        positions = [start + rel_pos for rel_pos in range(prep["block_len"])]
                        states = prep["per_layer"][layer]["h_block"]
                        fwd_hooks.append((hook_by_layer[layer], make_replace_hook(states, positions)))
                elif arm.kind in {"concept_replace", "random_replace"}:
                    layer = arm.layers[0]
                    rel_positions = prep["rel"] if arm.kind == "concept_replace" else prep["random_rel"]
                    positions = [start + rel_pos for rel_pos in rel_positions]
                    states = prep["per_layer"][layer]["h_block"][rel_positions]
                    fwd_hooks.append((hook_by_layer[layer], make_replace_hook(states, positions)))
                elif arm.kind == "rank_k_add":
                    layer = arm.layers[0]
                    assert arm.rank_k is not None
                    basis = basis_for(layer, prep["source_row_index"], arm.rank_k)
                    recon = rank_k_reconstruction(prep["per_layer"][layer]["concept_delta"], basis)
                    positions = [start + rel_pos for rel_pos in prep["rel"]]
                    fwd_hooks.append((hook_by_layer[layer], make_position_add_hook(recon, positions, 1.0)))
                    basis_records.append(
                        {
                            "condition": arm.label,
                            "source_row_index": prep["source_row_index"],
                            "layer": layer,
                            "rank_k": arm.rank_k,
                            "basis_mode": arm.basis_mode,
                            "n_source_rows": len(basis["source_rows"]),
                            "excluded": basis["exclude_rows"],
                            "explained_variance_ratio": basis["explained_variance_ratio"],
                        }
                    )
                with model.hooks(fwd_hooks=fwd_hooks):
                    batch = generate_sample_batch(
                        model=model,
                        token_ids=prep["receiver_ids"],
                        n_samples=args.samples_per_row,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        stop_at_eos=True,
                        cache_dtype=dtype,
                    )
                for sample_index, (new_ids, reply) in enumerate(batch):
                    score = score_reply(prep["row"], reply)
                    out = {
                        "schema_version": 1,
                        "source_row_index": prep["source_row_index"],
                        "example_id": prep["row"].get("example_id"),
                        "height": prep["row"].get("height"),
                        "model": args.model,
                        "task": args.task,
                        "condition": arm.label,
                        "arm_kind": arm.kind,
                        "reference": arm.reference,
                        "patch_layers": list(arm.layers) if arm.layers else None,
                        "rank_k": arm.rank_k,
                        "basis_mode": arm.basis_mode,
                        "basis_exclude_rows": basis["exclude_rows"] if basis else None,
                        "basis_source_rows": basis["source_rows"] if basis else None,
                        "basis_explained_variance_ratio": basis["explained_variance_ratio"] if basis else None,
                        "sample_index": sample_index,
                        "method": "subtype_capture_ladder_discriminator",
                        "target_variable": "target_concept",
                        "representation_type": "patched_residual_state",
                        "gold_concept": prep["gold_concept"],
                        "n_concept_positions": len(prep["rel"]),
                        "targets_gold_concept": canon(prep["gold_concept"]) in subjects_of(reply),
                        "generated_token_count": len(new_ids),
                        "model_output": reply,
                        **score,
                    }
                    rows_out.append(out)
                    fout.write(json.dumps(out, ensure_ascii=False, default=json_default) + "\n")
                    fout.flush()
                strong_rate = float(np.mean([row["is_correct_strong"] for row in rows_out[-args.samples_per_row :]]))
                print(f"arm {arm_index + 1}/{len(arms)} {arm.label} row {prep_index + 1}/{len(prepared)}: P(strong)={strong_rate:.2f}", flush=True)
            print(f"ARM DONE {arm.label}: {time.time() - arm_started:.0f}s elapsed_total={time.time() - started:.0f}s", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    state_arrays = {}
    for layer in ladder_layers:
        for row_id, arr in delta_by_layer[layer].items():
            state_arrays[f"L{layer}_row{row_id}_concept_delta"] = arr.astype(np.float32)
    args.states_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.states_output, **state_arrays)

    summary = summarize_generation_rows(rows_out, arms, args.sample_seed)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_subtype_discriminator.py",
        "model": args.model,
        "task": args.task,
        "target_variable": "target_concept",
        "method": "subtype_capture_ladder_discriminator",
        "representation_type": "patched_residual_state",
        "manifest": str(args.manifest),
        "selected_rows": len(selected_rows),
        "prepared_rows": len(prepared),
        "ladder_layers": ladder_layers,
        "old_trio_layers": old_layers,
        "selected_layers": selected_layers,
        "selection_metric": "row_mean_delta_norm_mean",
        "capture_ladder": capture_ladder,
        "arms": [asdict(arm) for arm in arms],
        "generation": {"samples_per_row": args.samples_per_row, "temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
        "summary": summary,
        "basis_records": basis_records,
        "states_output": str(args.states_output),
        "out_jsonl": str(args.out_jsonl),
        "output": str(args.output),
        "summary_md": str(args.summary_md),
        "n": len(rows_out),
        "resolved_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "controls": [
            "same-run baseline",
            "old L30/40/45 full-block patch replay",
            "dynamic off-trio layer selection from pre-generation capture ladder",
            "matched random-position replacements at selected layers",
            "leave-one-row-out rank-k add at top selected layer",
        ],
        "reading_rule": (
            "Repair from selected off-trio concept replacement supports layer mismatch. "
            "Null selected-layer repairs with clean random controls and visible capture norms supports "
            "insufficiency of this residual-state route rather than a splice bug."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")
    write_markdown_summary(args.summary_md, report)
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.summary_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
