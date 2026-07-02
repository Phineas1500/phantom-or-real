#!/usr/bin/env python3
"""Rank-k guard v2: fresh-row expansion of the L30 compact-core claim.

The 457012 guard passed leave-one-row-out on the 13 rows that defined the
compact core. This job re-tests sufficiency on rows that contributed to
neither the PCA bases nor the original row selection: it draws fresh
baseline-wrong property rows, captures hinted/unhinted L30 states in-job,
and runs the rank-4/rank-8 LOO adds against an in-job concept-replacement
denominator. Pre-registered in docs/causal_handle_directions.md
("Pre-Registered Manuscript-Hardening Jobs", item A).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_subtype_discriminator import (  # noqa: E402
    Arm,
    fit_pca_basis,
    json_default,
    make_position_add_hook,
    make_replace_hook,
    parse_int_list,
    rank_k_reconstruction,
    row_bootstrap_ci,
    summarize_generation_rows,
)

COMPOSITE_MANIFEST_ROWS = [3073, 3290, 3415, 4322, 4675, 6188, 6327, 8035, 8298, 8874, 9549, 10079, 10714]


def select_fresh_rows(
    jsonl: Path,
    *,
    exclude: set[int],
    heights: list[int],
    per_height: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Seeded balanced draw of parse-ok, strong-incorrect rows outside `exclude`."""
    pool: dict[int, list[dict[str, Any]]] = {height: [] for height in heights}
    with jsonl.open() as f:
        for row_index, line in enumerate(f):
            if row_index in exclude or not line.strip():
                continue
            row = json.loads(line)
            height = row.get("height")
            if height not in pool or row.get("parse_failed") or row.get("is_correct_strong"):
                continue
            row["row_index"] = row_index
            pool[height].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for height in heights:
        candidates = pool[height]
        if len(candidates) < per_height:
            raise ValueError(f"height {height}: only {len(candidates)} eligible rows, need {per_height}")
        selected.extend(rng.sample(candidates, per_height))
    return sorted(selected, key=lambda row: row["row_index"])


def shard_rows(rows: list[dict[str, Any]], shard_index: int, shard_count: int) -> list[dict[str, Any]]:
    """Interleaved slice so each shard keeps the height balance."""
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index {shard_index} out of range for shard_count {shard_count}")
    by_height: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        by_height.setdefault(row.get("height"), []).append(row)
    picked: list[dict[str, Any]] = []
    for height in sorted(by_height):
        picked.extend(by_height[height][shard_index::shard_count])
    return sorted(picked, key=lambda row: row["row_index"])


def build_arms(ranks: list[int], layer: int) -> list[Arm]:
    arms = [
        Arm("unhinted_baseline", "none", "none"),
        Arm("hinted_baseline", "hinted_prompt", "unhinted_baseline"),
        Arm(f"L{layer}_concept_replace", "concept_replace", "unhinted_baseline", (layer,)),
        Arm(f"L{layer}_random_replace", "random_replace", "unhinted_baseline", (layer,)),
    ]
    for rank in ranks:
        arms.append(Arm(f"rank{rank}_loo_add_L{layer}", "rank_k_add", "unhinted_baseline", (layer,), rank, "leave_one_row_out"))
    return arms


def hint_validated_summary(
    rows_out: list[dict[str, Any]],
    arms: list[Arm],
    seed: int,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Secondary pre-registered slice: rows where the hinted prompt repairs."""
    hinted: dict[int, list[float]] = {}
    for row in rows_out:
        if row["condition"] == "hinted_baseline":
            hinted.setdefault(row["source_row_index"], []).append(float(row["is_correct_strong"]))
    validated = {row_id for row_id, vals in hinted.items() if float(np.mean(vals)) >= threshold}
    subset = [row for row in rows_out if row["source_row_index"] in validated]
    return {
        "threshold": threshold,
        "validated_rows": sorted(validated),
        "n_validated_rows": len(validated),
        "summary": summarize_generation_rows(subset, arms, seed) if validated else {},
    }


def write_markdown_summary(path: Path, report: dict[str, Any]) -> None:
    job = report.get("slurm_job_id") or "local"
    shard = report["shard"]
    lines = [
        f"# Rank-k Guard v2 (fresh rows) - Job {job} - shard {shard['index']} of {shard['count']}",
        "",
        f"Output JSON: `{report['output']}`",
        f"Rows: {report['prepared_rows']} prepared from {report['selected_rows']} fresh-selection rows.",
        "",
        "## Causal arms (row-paired bootstrap vs in-job unhinted baseline)",
        "",
        "| arm | P(strong) | dP vs reference (CI95) | reference |",
        "| --- | ---: | ---: | --- |",
    ]
    for cond, entry in sorted(report["summary"].items()):
        if "paired_delta_vs_reference" in entry:
            ci = entry["paired_ci95"]
            delta = f"{entry['paired_delta_vs_reference']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        else:
            delta = "-"
        lines.append(f"| {cond} | {entry['strong_accuracy']:.3f} | {delta} | {entry['reference']} |")
    hv = report["hint_validated"]
    lines.extend(
        [
            "",
            f"Hint-validated rows (hinted P(strong) >= {hv['threshold']}): {hv['n_validated_rows']}.",
            "",
            "Reading rule: pooled-shard rank4/rank8 LOO CI excluding zero at >=70% of the",
            "pooled in-job concept-replace effect confirms claim 8 on fresh rows; a null",
            "concept-replace on fresh rows scopes the compact-core claim to",
            "recognition-gap-style rows rather than failing the guard.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--rank-list", default="4,8")
    parser.add_argument("--heights", default="3,4")
    parser.add_argument("--per-height", type=int, default=16)
    parser.add_argument("--selection-seed", type=int, default=20260702)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--exclude-rows", default=",".join(str(r) for r in COMPOSITE_MANIFEST_ROWS))
    parser.add_argument("--min-block-tokens", type=int, default=32)
    parser.add_argument("--samples-per-row", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=20260702)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-jsonl", type=Path, default=None)
    parser.add_argument("--states-output", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    args = parser.parse_args()
    started = time.time()

    stem = f"rank_k_guard_v2_27b_property_shard{args.shard_index}of{args.shard_count}"
    out_jsonl = args.out_jsonl or Path(f"results/stage2/erasure/{stem}.jsonl")
    states_output = args.states_output or Path(f"results/stage2/erasure/{stem}_states.npz")
    output = args.output or Path(f"docs/{stem}.json")
    summary_md = args.summary_md or Path(f"docs/{stem}_summary.md")

    ranks = parse_int_list(args.rank_list)
    heights = parse_int_list(args.heights)
    exclude = {int(part) for part in args.exclude_rows.split(",") if part.strip()}
    all_rows = select_fresh_rows(
        args.jsonl, exclude=exclude, heights=heights, per_height=args.per_height, seed=args.selection_seed
    )
    selected_rows = shard_rows(all_rows, args.shard_index, args.shard_count)
    arms = build_arms(ranks, args.layer)
    total = len(selected_rows) * len(arms) * args.samples_per_row
    print(
        f"selected_rows={len(selected_rows)} of {len(all_rows)} "
        f"(shard {args.shard_index}/{args.shard_count}) arms={[arm.label for arm in arms]} "
        f"total_generations={total}",
        flush=True,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "rows": [int(row["row_index"]) for row in selected_rows],
                    "heights": {f"h{h}": sum(1 for row in selected_rows if row.get("height") == h) for h in heights},
                    "arms": [asdict(arm) for arm in arms],
                    "total_generations": total,
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
    hook_name = validate_hooks(model, [args.layer])[0]
    tokenizer = model.tokenizer
    print(f"using_hook={hook_name}", flush=True)

    prepared = []
    delta_by_row: dict[int, np.ndarray] = {}
    for stage1_row in selected_rows:
        source_row_index = int(stage1_row["row_index"])
        gold_concept = stage1_row["ontology_fol_structured"]["hypothesis"]["subject"]
        receiver_text = render_chat_text(tokenizer, system=stage1_row["system_prompt"], user=stage1_row["prompt_text"], model_name=args.model, add_generation_prompt=True)
        receiver_ids = tokenizer(receiver_text, add_special_tokens=False)["input_ids"]
        hinted_user = make_user_prompt(stage1_row, "hint_concept_first")
        hinted_text = render_chat_text(tokenizer, system=stage1_row["system_prompt"], user=hinted_user, model_name=args.model, add_generation_prompt=True)
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
        hinted_cache = prompt_cache(model, hinted_ids, [hook_name])
        unhinted_cache = prompt_cache(model, receiver_ids, [hook_name])
        h_block = hinted_cache[hook_name][h_start : h_start + block_len].detach().cpu()
        u_block = unhinted_cache[hook_name][r_start : r_start + block_len].detach().cpu()
        concept_delta = h_block[rel] - u_block[rel]
        delta_by_row[source_row_index] = concept_delta.numpy().astype(np.float32)
        prepared.append(
            {
                "row": stage1_row,
                "source_row_index": source_row_index,
                "gold_concept": gold_concept,
                "receiver_ids": receiver_ids,
                "hinted_ids": hinted_ids,
                "r_start": r_start,
                "block_len": block_len,
                "rel": rel,
                "random_rel": random_rel,
                "h_block": h_block,
                "concept_delta": concept_delta,
            }
        )
        print(f"prepared row {source_row_index}: block={block_len} concept_tokens={len(rel)}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_after_skip = len(prepared) * len(arms) * args.samples_per_row
    print(f"prepared_rows={len(prepared)} total_generations={total_after_skip}", flush=True)

    basis_cache: dict[tuple[int, int], dict[str, Any]] = {}

    def basis_for(row_id: int, rank_k: int) -> dict[str, Any]:
        key = (row_id, rank_k)
        if key not in basis_cache:
            basis_cache[key] = fit_pca_basis(delta_by_row, rank_k, exclude_rows={row_id})
        return basis_cache[key]

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    basis_records: list[dict[str, Any]] = []
    with out_jsonl.open("w", encoding="utf-8") as fout:
        for arm_index, arm in enumerate(arms):
            arm_started = time.time()
            for prep_index, prep in enumerate(prepared):
                torch.manual_seed(args.sample_seed + prep["source_row_index"] * 10007 + arm_index * 101)
                start = prep["r_start"]
                token_ids = prep["receiver_ids"]
                fwd_hooks = []
                basis = None
                if arm.kind == "hinted_prompt":
                    token_ids = prep["hinted_ids"]
                elif arm.kind in {"concept_replace", "random_replace"}:
                    rel_positions = prep["rel"] if arm.kind == "concept_replace" else prep["random_rel"]
                    positions = [start + rel_pos for rel_pos in rel_positions]
                    states = prep["h_block"][rel_positions]
                    fwd_hooks.append((hook_name, make_replace_hook(states, positions)))
                elif arm.kind == "rank_k_add":
                    assert arm.rank_k is not None
                    basis = basis_for(prep["source_row_index"], arm.rank_k)
                    recon = rank_k_reconstruction(prep["concept_delta"], basis)
                    positions = [start + rel_pos for rel_pos in prep["rel"]]
                    fwd_hooks.append((hook_name, make_position_add_hook(recon, positions, 1.0)))
                    basis_records.append(
                        {
                            "condition": arm.label,
                            "source_row_index": prep["source_row_index"],
                            "layer": args.layer,
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
                        token_ids=token_ids,
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
                        "method": "rank_k_guard_v2_fresh_rows",
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

    state_arrays = {f"L{args.layer}_row{row_id}_concept_delta": arr for row_id, arr in delta_by_row.items()}
    states_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(states_output, **state_arrays)

    summary = summarize_generation_rows(rows_out, arms, args.sample_seed)
    hint_validated = hint_validated_summary(rows_out, arms, args.sample_seed)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_rank_k_guard_v2.py",
        "model": args.model,
        "task": args.task,
        "target_variable": "target_concept",
        "method": "rank_k_guard_v2_fresh_rows",
        "representation_type": "patched_residual_state",
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "selection": {
            "jsonl": str(args.jsonl),
            "rule": "parse_ok, strong_incorrect, heights balanced, excluding composite-manifest rows",
            "heights": heights,
            "per_height": args.per_height,
            "selection_seed": args.selection_seed,
            "excluded_rows": sorted(exclude),
            "all_selected_rows": [int(row["row_index"]) for row in all_rows],
            "shard_rows": [int(row["row_index"]) for row in selected_rows],
        },
        "selected_rows": len(selected_rows),
        "prepared_rows": len(prepared),
        "layer": args.layer,
        "rank_list": ranks,
        "arms": [asdict(arm) for arm in arms],
        "generation": {"samples_per_row": args.samples_per_row, "temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
        "summary": summary,
        "hint_validated": hint_validated,
        "basis_records": basis_records,
        "states_output": str(states_output),
        "out_jsonl": str(out_jsonl),
        "output": str(output),
        "summary_md": str(summary_md),
        "n": len(rows_out),
        "resolved_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "controls": [
            "in-job unhinted baseline",
            "hinted-prompt per-row validation arm",
            "in-job concept-position replacement denominator",
            "matched random-position replacement",
            "leave-one-row-out PCA bases fit within shard",
        ],
        "pre_registered_decision_rule": (
            "Claim 8 survives if pooled rank4_loo or rank8_loo CI excludes zero and reaches >=70% of "
            "the pooled in-job L30_concept_replace effect. A null concept_replace on fresh rows scopes "
            "the compact-core claim to recognition-gap-style rows rather than failing the guard."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")
    write_markdown_summary(summary_md, report)
    print(f"wrote {output}", flush=True)
    print(f"wrote {summary_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
