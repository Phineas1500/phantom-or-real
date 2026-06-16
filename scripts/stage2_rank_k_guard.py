#!/usr/bin/env python3
"""Rank-k guard for the L30 compact-core claim.

Runs the lead-claim guard after the KV/hint-span finale: a rank ladder over
PCA reconstructions of concept-position deltas, with in-sample and
leave-one-row-out bases. The LOO basis is the key reviewer guard: the held-out
row receives a reconstruction from a subspace it did not define.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
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
    rank_k: int | None = None
    basis_mode: str | None = None


def parse_rank_list(text: str) -> list[int]:
    ranks = [int(part) for part in text.split(",") if part.strip()]
    if not ranks or any(rank <= 0 for rank in ranks):
        raise ValueError(f"rank list must contain positive integers: {text!r}")
    return ranks


def parse_basis_modes(text: str) -> list[str]:
    modes = [part.strip() for part in text.split(",") if part.strip()]
    allowed = {"in_sample", "leave_one_row_out"}
    bad = sorted(set(modes) - allowed)
    if bad:
        raise ValueError(f"unsupported basis modes {bad}; allowed={sorted(allowed)}")
    if not modes:
        raise ValueError("at least one basis mode is required")
    return modes


def build_arms(ranks: list[int], basis_modes: list[str]) -> list[Arm]:
    arms = [Arm("unhinted_baseline", "none", "none")]
    for mode in basis_modes:
        suffix = "loo" if mode == "leave_one_row_out" else "in_sample"
        for rank in ranks:
            arms.append(Arm(f"rank{rank}_{suffix}_L30", "rank_k_add", "unhinted_baseline", rank, mode))
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


def make_position_add_hook(matrix: Any, positions: list[int], scale: float):
    def hook_fn(act: Any, hook: Any) -> Any:  # noqa: ARG001
        if not positions or act.shape[1] <= max(positions):
            return act
        add = (scale * matrix).to(device=act.device, dtype=act.dtype)
        for row_index, pos in enumerate(positions):
            act[:, pos, :] += add[row_index]
        return act

    return hook_fn


def load_delta_by_row(states_npz: Path, layer: int) -> dict[int, np.ndarray]:
    data = np.load(states_npz)
    pattern = re.compile(rf"^L{layer}_row(\d+)_concept_delta$")
    out: dict[int, np.ndarray] = {}
    for key in data.files:
        match = pattern.match(key)
        if match:
            out[int(match.group(1))] = data[key].astype(np.float64)
    if not out:
        raise ValueError(f"no L{layer} concept-delta arrays in {states_npz}")
    return dict(sorted(out.items()))


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


def summarize_basis(basis_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in basis_records:
        by_condition[record["condition"]].append(record)
    summary = {}
    for condition, records in sorted(by_condition.items()):
        ev = np.array([r["explained_variance_ratio"] for r in records], dtype=float)
        summary[condition] = {
            "rank_k": records[0]["rank_k"],
            "basis_mode": records[0]["basis_mode"],
            "n_bases": len(records),
            "source_rows_min": int(min(r["n_source_rows"] for r in records)),
            "source_rows_max": int(max(r["n_source_rows"] for r in records)),
            "explained_variance_ratio_mean": float(ev.mean()),
            "explained_variance_ratio_min": float(ev.min()),
            "explained_variance_ratio_max": float(ev.max()),
        }
    return summary


def main() -> int:
    import torch
    from dotenv import load_dotenv

    from scripts.stage2_decode_time_correction import torch_dtype  # noqa: E402
    from scripts.stage2_hint_delta import concept_positions, prompt_cache  # noqa: E402
    from scripts.stage2_hint_state_interchange import generate_sample_batch  # noqa: E402
    from scripts.stage2_prompt_margin_gated_decode_correction import select_prefix_rows  # noqa: E402
    from scripts.stage2_proposal_hints import make_user_prompt  # noqa: E402
    from scripts.stage2_recognition_state_patch import longest_common_token_block  # noqa: E402
    from src.activations import load_tl_model, render_chat_text, validate_hooks  # noqa: E402
    from src.bd_path import ensure_on_path  # noqa: E402
    from src.stage2_steering import parse_int_list, score_reply  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--prefix-trajectory-jsonl", type=Path, default=Path("results/stage2/decode_time/prefix_conditioned_margin_trajectory_gemma_manifest.jsonl"))
    parser.add_argument("--prefix-checkpoint", default="0")
    parser.add_argument("--selection-limit", type=int, default=None)
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layers", default="30,40,45")
    parser.add_argument("--rank-list", default="1,2,3,4,6,8")
    parser.add_argument("--basis-modes", default="leave_one_row_out,in_sample")
    parser.add_argument("--min-block-tokens", type=int, default=32)
    parser.add_argument("--samples-per-row", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=20260617)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--states-npz", type=Path, default=Path("results/stage2/erasure/focus_state_composite_27b_property_states.npz"))
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/rank_k_guard_27b_property.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/rank_k_guard_27b_property.json"))
    args = parser.parse_args()
    load_dotenv()
    torch.set_grad_enabled(False)
    started = time.time()

    layers = parse_int_list(args.layers)
    rank_layer = layers[0]
    ranks = parse_rank_list(args.rank_list)
    basis_modes = parse_basis_modes(args.basis_modes)
    arms = build_arms(ranks, basis_modes)
    selected_rows, selection_summary = select_prefix_rows(
        prefix_jsonl=args.prefix_trajectory_jsonl,
        source_jsonl=args.jsonl,
        checkpoint=args.prefix_checkpoint,
        limit=args.selection_limit,
        prompt_gold_vs_foil_threshold=0.0,
    )
    delta_by_row = load_delta_by_row(args.states_npz, rank_layer)
    total = len(selected_rows) * len(arms) * args.samples_per_row
    print(f"selected_rows={len(selected_rows)} arms={[arm.label for arm in arms]} total_generations_pre_skip={total}", flush=True)
    if args.dry_run:
        print(json.dumps({
            "selection": selection_summary,
            "rank_layer": rank_layer,
            "rank_list": ranks,
            "basis_modes": basis_modes,
            "available_state_rows": sorted(delta_by_row),
            "arms": [arm.__dict__ for arm in arms],
            "total_generations_pre_skip": total,
        }, indent=2, default=json_default), flush=True)
        return 0

    ensure_on_path()
    model = load_tl_model(args.model, n_devices=args.n_devices, n_ctx=args.n_ctx, dtype=torch_dtype(args.dtype), load_mode=args.load_mode)
    dtype = torch_dtype(args.dtype)
    hook_names = validate_hooks(model, [rank_layer])
    hook_by_layer = {rank_layer: hook_names[0]}
    tokenizer = model.tokenizer
    print(f"using_rank_hook={hook_by_layer[rank_layer]}", flush=True)

    prepared = []
    for stage1_row in selected_rows:
        source_row_index = int(stage1_row["row_index"])
        if source_row_index not in delta_by_row:
            print(f"skip row {source_row_index}: no saved L{rank_layer} state", flush=True)
            continue
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
        rel = [p - r_start for p in positions_r]
        unhinted_cache = prompt_cache(model, receiver_ids, hook_names)
        hinted_cache = prompt_cache(model, hinted_ids, hook_names)
        u_block = unhinted_cache[hook_by_layer[rank_layer]][r_start : r_start + block_len]
        h_block = hinted_cache[hook_by_layer[rank_layer]][h_start : h_start + block_len]
        concept_delta = h_block[rel] - u_block[rel]
        prepared.append({
            "row": stage1_row,
            "source_row_index": source_row_index,
            "gold_concept": gold_concept,
            "receiver_ids": receiver_ids,
            "positions_r": positions_r,
            "concept_delta": concept_delta,
        })
        print(f"prepared row {source_row_index}: block={block_len} concept_tokens={len(rel)}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    basis_cache: dict[tuple[str, int, int | None], dict[str, Any]] = {}

    def basis_for(arm: Arm, row_id: int) -> dict[str, Any]:
        assert arm.rank_k is not None and arm.basis_mode is not None
        if arm.basis_mode == "in_sample":
            key = (arm.basis_mode, arm.rank_k, None)
            exclude: set[int] = set()
        else:
            key = (arm.basis_mode, arm.rank_k, row_id)
            exclude = {row_id}
        if key not in basis_cache:
            basis_cache[key] = fit_pca_basis(delta_by_row, arm.rank_k, exclude)
        return basis_cache[key]

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    basis_records: list[dict[str, Any]] = []
    with args.out_jsonl.open("w") as fout:
        for arm_index, arm in enumerate(arms):
            arm_started = time.time()
            for prep_index, prep in enumerate(prepared):
                torch.manual_seed(args.sample_seed + prep["source_row_index"] * 10007 + arm_index * 101)
                fwd_hooks = []
                basis = None
                if arm.kind == "rank_k_add":
                    basis = basis_for(arm, prep["source_row_index"])
                    recon = rank_k_reconstruction(prep["concept_delta"], basis)
                    fwd_hooks.append((hook_by_layer[rank_layer], make_position_add_hook(recon, prep["positions_r"], 1.0)))
                    basis_records.append({
                        "condition": arm.label,
                        "source_row_index": prep["source_row_index"],
                        "rank_k": arm.rank_k,
                        "basis_mode": arm.basis_mode,
                        "n_source_rows": len(basis["source_rows"]),
                        "excluded": basis["exclude_rows"],
                        "explained_variance_ratio": basis["explained_variance_ratio"],
                    })
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
                        "condition": arm.label,
                        "arm_kind": arm.kind,
                        "reference": arm.reference,
                        "rank_k": arm.rank_k,
                        "basis_mode": arm.basis_mode,
                        "basis_exclude_rows": basis["exclude_rows"] if basis else None,
                        "basis_source_rows": basis["source_rows"] if basis else None,
                        "basis_explained_variance_ratio": basis["explained_variance_ratio"] if basis else None,
                        "sample_index": sample_index,
                        "method": "rank_k_guard",
                        "target_variable": "target_concept",
                        "representation_type": "patched_residual_state",
                        "patch_layers": [rank_layer] if arm.kind == "rank_k_add" else None,
                        "gold_concept": prep["gold_concept"],
                        "n_concept_positions": len(prep["positions_r"]),
                        "generated_token_count": len(new_ids),
                        "model_output": reply,
                        **score,
                    }
                    rows_out.append(out)
                    fout.write(json.dumps(out, ensure_ascii=False, default=json_default) + "\n")
                    fout.flush()
                strong_rate = float(np.mean([r["is_correct_strong"] for r in rows_out[-args.samples_per_row:]]))
                print(f"arm {arm_index + 1}/{len(arms)} {arm.label} row {prep_index + 1}/{len(prepared)}: P(strong)={strong_rate:.2f}", flush=True)
            print(f"ARM DONE {arm.label}: {time.time() - arm_started:.0f}s elapsed_total={time.time() - started:.0f}s", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    by = defaultdict(lambda: defaultdict(list))
    for row in rows_out:
        by[row["condition"]][row["source_row_index"]].append(row)
    rng = np.random.default_rng(args.sample_seed)
    refs = {arm.label: arm.reference for arm in arms}
    summary = {}
    for cond in sorted(by):
        per_row = {i: float(np.mean([s["is_correct_strong"] for s in v])) for i, v in by[cond].items()}
        flat = [s for v in by[cond].values() for s in v]
        entry: dict[str, Any] = {
            "n_rows": len(per_row),
            "strong_accuracy": float(np.mean([s["is_correct_strong"] for s in flat])),
            "reference": refs.get(cond, "none"),
        }
        ref = refs.get(cond, "none")
        if ref != "none" and ref in by:
            ref_rows = {i: float(np.mean([s["is_correct_strong"] for s in v])) for i, v in by[ref].items()}
            deltas = np.array([per_row[i] - ref_rows[i] for i in sorted(ref_rows) if i in per_row])
            boots = [float(np.mean(deltas[rng.integers(0, len(deltas), len(deltas))])) for _ in range(10000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            entry["paired_delta_vs_reference"] = float(np.mean(deltas))
            entry["paired_ci95"] = [float(lo), float(hi)]
        summary[cond] = entry

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_rank_k_guard.py",
        "model": args.model,
        "task": args.task,
        "target_variable": "target_concept",
        "method": "rank_k_guard",
        "representation_type": "patched_residual_state",
        "rank_layer": rank_layer,
        "rank_list": ranks,
        "basis_modes": basis_modes,
        "states_npz": str(args.states_npz),
        "selection": selection_summary,
        "arms": [arm.__dict__ for arm in arms],
        "generation": {"samples_per_row": args.samples_per_row, "temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
        "summary": summary,
        "basis_summary": summarize_basis(basis_records),
        "n": len(rows_out),
        "controls": [
            "same-run_unhinted_baseline",
            "leave_one_row_out_basis_excludes_receiver_row",
            "rank_ladder_1_2_3_4_6_8",
        ],
        "interpretation_note": (
            "Lead-claim guard for the compact-core result: if leave-one-row-out rank-4 repair "
            "survives, the L30 compact core is not merely an in-sample PCA fit. The rank ladder "
            "turns the point result into an intrinsic-dimensionality curve."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
