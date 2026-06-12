#!/usr/bin/env python3
"""Subtype replication of the focus-state localization result.

Consumes docs/subtype_recognition_gap_27b_manifest.json (behaviorally
validated gap rows): baseline, full-block hint-state patch, concept-position
subset patch, and matched random-subset control. Replicates claims-table
rows 5-6 on infer_subtype and saves concept-position states for the
restricted geometry replication. Claims-table row 12.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_decode_time_correction import json_default, torch_dtype  # noqa: E402
from scripts.stage2_focus_state_composite import make_replace_hook  # noqa: E402
from scripts.stage2_hint_delta import concept_positions, prompt_cache  # noqa: E402
from scripts.stage2_hint_state_interchange import generate_sample_batch  # noqa: E402
from scripts.stage2_interchange_concept_analysis import canon, subjects_of  # noqa: E402
from scripts.stage2_proposal_hints import make_user_prompt  # noqa: E402
from scripts.stage2_recognition_state_patch import longest_common_token_block  # noqa: E402
from src.activations import load_tl_model, render_chat_text, validate_hooks  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.stage2_steering import parse_int_list, score_reply  # noqa: E402

ARMS = ("baseline", "full_patch", "subset_concept", "subset_random")


def load_gap_rows(jsonl: Path, manifest: Path) -> list[dict[str, Any]]:
    indices = set(json.load(manifest.open())["gap_row_indices"])
    rows = []
    with jsonl.open() as f:
        for row_index, line in enumerate(f):
            if row_index in indices:
                row = json.loads(line)
                row["row_index"] = row_index
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_subtype.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/subtype_recognition_gap_27b_manifest.json"))
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--task", default="infer_subtype")
    parser.add_argument("--layers", default="30,40,45")
    parser.add_argument("--min-block-tokens", type=int, default=32)
    parser.add_argument("--samples-per-row", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=20260615)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/subtype_localization_patch_27b.jsonl"))
    parser.add_argument("--states-output", type=Path, default=Path("results/stage2/erasure/subtype_localization_patch_27b_states.npz"))
    parser.add_argument("--output", type=Path, default=Path("docs/subtype_localization_patch_27b.json"))
    args = parser.parse_args()
    load_dotenv()
    torch.set_grad_enabled(False)
    started = time.time()

    layers = parse_int_list(args.layers)
    selected_rows = load_gap_rows(args.jsonl, args.manifest)
    total = len(selected_rows) * len(ARMS) * args.samples_per_row
    print(f"selected_rows={len(selected_rows)} arms={list(ARMS)} total_generations={total}", flush=True)
    if args.dry_run:
        print(json.dumps({"rows": [r['row_index'] for r in selected_rows], "total_generations": total}, indent=2), flush=True)
        return 0

    ensure_on_path()
    model = load_tl_model(args.model, n_devices=args.n_devices, n_ctx=args.n_ctx, dtype=torch_dtype(args.dtype), load_mode=args.load_mode)
    dtype = torch_dtype(args.dtype)
    hook_names = validate_hooks(model, layers)
    hook_by_layer = dict(zip(layers, hook_names))
    tokenizer = model.tokenizer
    print(f"using_hooks={hook_by_layer}", flush=True)

    prepared = []
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
        rel = [p - r_start for p in positions_r]
        rng = random.Random(args.sample_seed + source_row_index)
        random_rel = sorted(rng.sample(range(block_len), len(rel)))
        hinted_cache = prompt_cache(model, hinted_ids, hook_names)
        unhinted_cache = prompt_cache(model, receiver_ids, hook_names)
        per_layer = {}
        for layer in layers:
            h_block = hinted_cache[hook_by_layer[layer]][h_start : h_start + block_len]
            u_block = unhinted_cache[hook_by_layer[layer]][r_start : r_start + block_len]
            per_layer[layer] = {"h_block": h_block, "u_block": u_block}
        prepared.append({
            "row": stage1_row, "source_row_index": source_row_index, "gold_concept": gold_concept,
            "receiver_ids": receiver_ids, "r_start": r_start, "block_len": block_len,
            "rel": rel, "random_rel": random_rel, "per_layer": per_layer,
        })
        print(f"prepared row {source_row_index}: block={block_len} concept_tokens={len(rel)}", flush=True)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    with args.out_jsonl.open("w") as fout:
        for arm_index, arm in enumerate(ARMS):
            arm_started = time.time()
            for prep_index, prep in enumerate(prepared):
                torch.manual_seed(args.sample_seed + prep["source_row_index"] * 10007 + arm_index * 101)
                start = prep["r_start"]
                fwd_hooks = []
                if arm != "baseline":
                    for layer in layers:
                        h_block = prep["per_layer"][layer]["h_block"]
                        if arm == "full_patch":
                            positions = [start + p for p in range(prep["block_len"])]
                            states = h_block
                        elif arm == "subset_concept":
                            positions = [start + p for p in prep["rel"]]
                            states = h_block[prep["rel"]]
                        else:
                            positions = [start + p for p in prep["random_rel"]]
                            states = h_block[prep["random_rel"]]
                        fwd_hooks.append((hook_by_layer[layer], make_replace_hook(states, positions)))
                with model.hooks(fwd_hooks=fwd_hooks):
                    batch = generate_sample_batch(model=model, token_ids=prep["receiver_ids"], n_samples=args.samples_per_row, max_new_tokens=args.max_new_tokens, temperature=args.temperature, stop_at_eos=True, cache_dtype=dtype)
                for sample_index, (new_ids, reply) in enumerate(batch):
                    score = score_reply(prep["row"], reply)
                    out = {
                        "schema_version": 1,
                        "source_row_index": prep["source_row_index"],
                        "example_id": prep["row"].get("example_id"),
                        "height": prep["row"].get("height"),
                        "model": args.model,
                        "task": args.task,
                        "condition": arm,
                        "sample_index": sample_index,
                        "method": "subtype_localization_patch",
                        "target_variable": "target_concept",
                        "representation_type": "patched_residual_state",
                        "patch_layers": layers,
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
                strong_rate = float(np.mean([r["is_correct_strong"] for r in rows_out[-args.samples_per_row:]]))
                print(f"arm {arm_index + 1}/{len(ARMS)} {arm} row {prep_index + 1}/{len(prepared)}: P(strong)={strong_rate:.2f}", flush=True)
            print(f"ARM DONE {arm}: {time.time() - arm_started:.0f}s elapsed_total={time.time() - started:.0f}s", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    state_arrays = {}
    for prep in prepared:
        idx = prep["source_row_index"]
        for layer in layers:
            pl = prep["per_layer"][layer]
            delta = pl["h_block"][prep["rel"]] - pl["u_block"][prep["rel"]]
            state_arrays[f"L{layer}_row{idx}_concept_delta"] = delta.numpy().astype(np.float32)
    args.states_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.states_output, **state_arrays)

    from collections import defaultdict
    by = defaultdict(lambda: defaultdict(list))
    for r in rows_out:
        by[r["condition"]][r["source_row_index"]].append(bool(r["is_correct_strong"]))
    base = {i: float(np.mean(v)) for i, v in by["baseline"].items()}
    rng = np.random.default_rng(args.sample_seed)
    summary = {}
    for cond in sorted(by):
        per_row = {i: float(np.mean(v)) for i, v in by[cond].items()}
        flat = [s for v in by[cond].values() for s in v]
        entry: dict[str, Any] = {"n_rows": len(per_row), "strong_accuracy": float(np.mean(flat))}
        if cond != "baseline":
            deltas = np.array([per_row[i] - base[i] for i in sorted(base) if i in per_row])
            boots = [float(np.mean(deltas[rng.integers(0, len(deltas), len(deltas))])) for _ in range(10000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            entry["paired_delta_p_strong"] = float(np.mean(deltas))
            entry["paired_ci95"] = [float(lo), float(hi)]
        summary[cond] = entry

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_subtype_localization_patch.py",
        "model": args.model,
        "task": args.task,
        "target_variable": "target_concept",
        "method": "subtype_localization_patch",
        "representation_type": "patched_residual_state",
        "layers": layers,
        "manifest": str(args.manifest),
        "arms": list(ARMS),
        "generation": {"samples_per_row": args.samples_per_row, "temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
        "summary": summary,
        "states_output": str(args.states_output),
        "n": len(rows_out),
        "controls": ["regenerated_baseline", "matched_random_position_subset", "behaviorally_validated_donors"],
        "causal_abstraction_claim": (
            "Cross-task replication of the localization result: full-block hint-state patch and "
            "concept-position subset patch vs matched random subsets on the subtype "
            "recognition-gap rowset. Claims-table row 12."
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
