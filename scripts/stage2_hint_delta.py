#!/usr/bin/env python3
"""Combined hint-delta job: the low-rank-vs-distributed decider.

Per row, computes the mean (hinted - unhinted) residual delta over the matched
context block, one vector per layer. Arms: baseline; delta-add at scale sweep
(low-rank sufficiency); subset patching restricted to concept-mention
positions vs matched random subsets (which positions carry the content);
cross-row delta transplant (concept specificity — row B under row A's delta
should target A's concept). Saves per-row and mean delta vectors and reports
their cosine to the correctness probe directions. See
docs/causal_handle_directions.md review follow-up 5.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_decode_time_correction import json_default, torch_dtype  # noqa: E402
from scripts.stage2_hint_state_interchange import generate_sample_batch  # noqa: E402
from scripts.stage2_interchange_concept_analysis import canon, subjects_of  # noqa: E402
from scripts.stage2_prompt_margin_gated_decode_correction import select_prefix_rows  # noqa: E402
from scripts.stage2_proposal_hints import make_user_prompt  # noqa: E402
from scripts.stage2_recognition_state_patch import longest_common_token_block  # noqa: E402
from src.activations import input_device_for_model, load_tl_model, render_chat_text, validate_hooks  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.stage2_steering import parse_float_list, parse_int_list, score_reply  # noqa: E402


@dataclass(frozen=True)
class DeltaArm:
    label: str
    kind: str
    scale: float = 1.0


def prompt_cache(model, token_ids: list[int], hook_names: list[str]) -> dict[str, torch.Tensor]:
    device = input_device_for_model(model)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda name: name in set(hook_names), return_type=None
        )
    return {name: cache[name][0].detach().float().cpu() for name in hook_names}


def concept_positions(
    tokenizer, prompt_text: str, concept: str, block_start: int, block_len: int
) -> list[int]:
    encoding = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    stem = concept.lower().rstrip("s")
    lowered = prompt_text.lower()
    spans = []
    start = 0
    while True:
        hit = lowered.find(stem, start)
        if hit < 0:
            break
        spans.append((hit, hit + len(stem)))
        start = hit + 1
    positions = []
    for index, (a, b) in enumerate(offsets):
        if not (block_start <= index < block_start + block_len):
            continue
        if any(a < span_end and b > span_start for span_start, span_end in spans):
            positions.append(index)
    return positions


def make_add_hook(vector: torch.Tensor, start: int, length: int, scale: float):
    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        if act.shape[1] < start + length:
            return act
        add = (scale * vector).to(device=act.device, dtype=torch.float32)
        act[:, start : start + length, :] += add.to(act.dtype)
        return act

    return hook_fn


def make_subset_patch_hook(donor_block: torch.Tensor, receiver_start: int, block_len: int, positions: list[int]):
    relative = [p - receiver_start for p in positions]

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        if act.shape[1] < receiver_start + block_len or not relative:
            return act
        donor = donor_block.to(device=act.device, dtype=torch.float32)
        for rel in relative:
            act[:, receiver_start + rel, :] = donor[rel].to(act.dtype)
        return act

    return hook_fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--prefix-trajectory-jsonl", type=Path, default=Path("results/stage2/decode_time/prefix_conditioned_margin_trajectory_gemma_manifest.jsonl"))
    parser.add_argument("--prefix-checkpoint", default="0")
    parser.add_argument("--selection-limit", type=int, default=None)
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layers", default="30,40,45")
    parser.add_argument("--delta-scales", default="1,2")
    parser.add_argument("--min-block-tokens", type=int, default=32)
    parser.add_argument("--with-random-add", action="store_true", help="Add matched-norm random-direction add arms (conditional control if delta_add is positive).")
    parser.add_argument("--samples-per-row", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=20260613)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--probe-direction-npz", type=Path, default=Path("results/stage2/erasure/subspace_erasure_27b_property_smoke_directions.npz"))
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/hint_delta_27b_property_manifest.jsonl"))
    parser.add_argument("--delta-output", type=Path, default=Path("results/stage2/erasure/hint_delta_27b_property_manifest_deltas.npz"))
    parser.add_argument("--output", type=Path, default=Path("docs/hint_delta_27b_property_manifest.json"))
    args = parser.parse_args()
    load_dotenv()
    torch.set_grad_enabled(False)
    started = time.time()

    layers = parse_int_list(args.layers)
    scales = parse_float_list(args.delta_scales)
    arms: list[DeltaArm] = [DeltaArm("baseline", "baseline")]
    arms += [DeltaArm(f"delta_add_s{scale:g}".replace(".", "p"), "delta_add", float(scale)) for scale in scales]
    arms += [
        DeltaArm("subset_concept", "subset_concept"),
        DeltaArm("subset_random", "subset_random"),
        DeltaArm("cross_row_delta", "cross_row_delta"),
    ]
    if args.with_random_add:
        arms += [DeltaArm(f"random_add_s{scale:g}".replace(".", "p"), "random_add", float(scale)) for scale in scales]

    selected_rows, selection_summary = select_prefix_rows(
        prefix_jsonl=args.prefix_trajectory_jsonl,
        source_jsonl=args.jsonl,
        checkpoint=args.prefix_checkpoint,
        limit=args.selection_limit,
        prompt_gold_vs_foil_threshold=0.0,
    )
    total = len(selected_rows) * len(arms) * args.samples_per_row
    print(f"selected_rows={len(selected_rows)} arms={[arm.label for arm in arms]} total_generations={total}", flush=True)
    if args.dry_run:
        print(json.dumps({"selection": selection_summary, "layers": layers, "arms": [arm.__dict__ for arm in arms], "total_generations": total}, indent=2, default=json_default), flush=True)
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
        donor_text = render_chat_text(tokenizer, system=stage1_row["system_prompt"], user=make_user_prompt(stage1_row, "hint_concept_first"), model_name=args.model, add_generation_prompt=True)
        donor_ids = tokenizer(donor_text, add_special_tokens=False)["input_ids"]
        d_start, r_start, block_len = longest_common_token_block(donor_ids, receiver_ids)
        if block_len < args.min_block_tokens:
            print(f"skip row {source_row_index}: block={block_len}", flush=True)
            continue
        donor_cache = prompt_cache(model, donor_ids, hook_names)
        receiver_cache = prompt_cache(model, receiver_ids, hook_names)
        deltas = {}
        donor_blocks = {}
        for layer in layers:
            donor_block = donor_cache[hook_by_layer[layer]][d_start : d_start + block_len]
            receiver_block = receiver_cache[hook_by_layer[layer]][r_start : r_start + block_len]
            donor_blocks[layer] = donor_block
            deltas[layer] = (donor_block - receiver_block).mean(dim=0)
        positions = concept_positions(tokenizer, receiver_text, gold_concept, r_start, block_len)
        rng = random.Random(args.sample_seed + source_row_index)
        block_positions = list(range(r_start, r_start + block_len))
        random_positions = sorted(rng.sample(block_positions, min(len(positions), block_len))) if positions else []
        prepared.append({
            "row": stage1_row,
            "source_row_index": source_row_index,
            "gold_concept": gold_concept,
            "receiver_ids": receiver_ids,
            "r_start": r_start,
            "block_len": block_len,
            "deltas": deltas,
            "donor_blocks": donor_blocks,
            "concept_positions": positions,
            "random_positions": random_positions,
        })
        print(f"prepared row {source_row_index}: block={block_len} concept_positions={len(positions)} delta_norms=" + ",".join(f"L{layer}={float(deltas[layer].norm()):.1f}" for layer in layers), flush=True)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    n_rows = len(prepared)
    with args.out_jsonl.open("w") as fout:
        for prep_index, prep in enumerate(prepared):
            stage1_row = prep["row"]
            cross = prepared[(prep_index + 1) % n_rows]
            for arm_index, arm in enumerate(arms):
                torch.manual_seed(args.sample_seed + prep["source_row_index"] * 10007 + arm_index * 101)
                fwd_hooks = []
                donor_concept = None
                if arm.kind == "delta_add":
                    for layer in layers:
                        fwd_hooks.append((hook_by_layer[layer], make_add_hook(prep["deltas"][layer], prep["r_start"], prep["block_len"], arm.scale)))
                elif arm.kind == "random_add":
                    gen = torch.Generator().manual_seed(args.sample_seed + prep["source_row_index"] * 31 + arm_index)
                    for layer in layers:
                        delta = prep["deltas"][layer]
                        rand = torch.randn(delta.shape, generator=gen, dtype=torch.float32)
                        rand = rand * (float(delta.norm()) / float(rand.norm()))
                        fwd_hooks.append((hook_by_layer[layer], make_add_hook(rand, prep["r_start"], prep["block_len"], arm.scale)))
                elif arm.kind == "cross_row_delta":
                    donor_concept = cross["gold_concept"]
                    for layer in layers:
                        fwd_hooks.append((hook_by_layer[layer], make_add_hook(cross["deltas"][layer], prep["r_start"], prep["block_len"], 1.0)))
                elif arm.kind == "subset_concept":
                    for layer in layers:
                        fwd_hooks.append((hook_by_layer[layer], make_subset_patch_hook(prep["donor_blocks"][layer], prep["r_start"], prep["block_len"], prep["concept_positions"])))
                elif arm.kind == "subset_random":
                    for layer in layers:
                        fwd_hooks.append((hook_by_layer[layer], make_subset_patch_hook(prep["donor_blocks"][layer], prep["r_start"], prep["block_len"], prep["random_positions"])))
                with model.hooks(fwd_hooks=fwd_hooks):
                    batch = generate_sample_batch(model=model, token_ids=prep["receiver_ids"], n_samples=args.samples_per_row, max_new_tokens=args.max_new_tokens, temperature=args.temperature, stop_at_eos=True, cache_dtype=dtype)
                for sample_index, (new_ids, reply) in enumerate(batch):
                    score = score_reply(stage1_row, reply)
                    subjects = subjects_of(reply)
                    out = {
                        "schema_version": 1,
                        "source_row_index": prep["source_row_index"],
                        "example_id": stage1_row.get("example_id"),
                        "height": stage1_row.get("height"),
                        "model": args.model,
                        "condition": arm.label,
                        "arm_kind": arm.kind,
                        "delta_scale": arm.scale,
                        "sample_index": sample_index,
                        "method": "hint_delta_program",
                        "target_variable": "target_concept",
                        "representation_type": "patched_residual_state",
                        "patch_layers": layers,
                        "gold_concept": prep["gold_concept"],
                        "cross_donor_row_index": cross["source_row_index"] if arm.kind == "cross_row_delta" else None,
                        "cross_donor_concept": donor_concept,
                        "n_concept_positions": len(prep["concept_positions"]),
                        "targets_gold_concept": canon(prep["gold_concept"]) in subjects,
                        "targets_donor_concept": (canon(donor_concept) in subjects) if donor_concept else None,
                        "generated_token_count": len(new_ids),
                        "model_output": reply,
                        **score,
                    }
                    rows_out.append(out)
                    fout.write(json.dumps(out, ensure_ascii=False, default=json_default) + "\n")
                    fout.flush()
                strong_rate = float(np.mean([r["is_correct_strong"] for r in rows_out[-args.samples_per_row:]]))
                print(f"row {prep_index + 1}/{n_rows} {arm.label}: P(strong)={strong_rate:.2f}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    delta_arrays = {}
    mean_deltas = {}
    for layer in layers:
        stack = torch.stack([prep["deltas"][layer] for prep in prepared])
        mean_deltas[layer] = stack.mean(dim=0)
        delta_arrays[f"L{layer}_mean_delta"] = mean_deltas[layer].numpy().astype(np.float32)
        for prep in prepared:
            delta_arrays[f"L{layer}_row{prep['source_row_index']}_delta"] = prep["deltas"][layer].numpy().astype(np.float32)
    args.delta_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.delta_output, **delta_arrays)

    geometry = {}
    if args.probe_direction_npz.exists():
        probe = np.load(args.probe_direction_npz)
        for layer in layers:
            key = f"L{layer}_raw_unit"
            if key in probe:
                unit = probe[key].astype(np.float64)
                delta = mean_deltas[layer].numpy().astype(np.float64)
                row_cos = []
                for prep in prepared:
                    vec = prep["deltas"][layer].numpy().astype(np.float64)
                    row_cos.append(float(vec @ unit / (np.linalg.norm(vec) * np.linalg.norm(unit))))
                geometry[f"L{layer}"] = {
                    "mean_delta_norm": float(np.linalg.norm(delta)),
                    "cos_mean_delta_vs_correctness_probe": float(delta @ unit / (np.linalg.norm(delta) * np.linalg.norm(unit))),
                    "row_delta_cos_vs_probe_mean": float(np.mean(row_cos)),
                    "row_delta_cos_vs_probe_minmax": [float(min(row_cos)), float(max(row_cos))],
                }
    print("geometry:", json.dumps(geometry, indent=2), flush=True)

    from collections import defaultdict
    by = defaultdict(lambda: defaultdict(list))
    for r in rows_out:
        by[r["condition"]][r["source_row_index"]].append(r)
    base = {i: np.mean([s["is_correct_strong"] for s in v]) for i, v in by["baseline"].items()}
    summary = {}
    rng = np.random.default_rng(args.sample_seed)
    for cond in sorted(by):
        per_row = {i: float(np.mean([s["is_correct_strong"] for s in v])) for i, v in by[cond].items()}
        entry: dict[str, Any] = {
            "n_rows": len(per_row),
            "strong_accuracy": float(np.mean([s["is_correct_strong"] for v in by[cond].values() for s in v])),
            "targets_gold_concept_rate": float(np.mean([s["targets_gold_concept"] for v in by[cond].values() for s in v])),
        }
        if cond == "cross_row_delta":
            entry["targets_donor_concept_rate"] = float(np.mean([s["targets_donor_concept"] for v in by[cond].values() for s in v]))
        if cond != "baseline":
            deltas_arr = np.array([per_row[i] - base[i] for i in sorted(base) if i in per_row])
            boots = [float(np.mean(deltas_arr[rng.integers(0, len(deltas_arr), len(deltas_arr))])) for _ in range(10000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            entry["paired_delta_p_strong"] = float(np.mean(deltas_arr))
            entry["paired_ci95"] = [float(lo), float(hi)]
        summary[cond] = entry

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_hint_delta.py",
        "model": args.model,
        "task": args.task,
        "target_variable": "target_concept",
        "method": "hint_delta_program",
        "representation_type": "patched_residual_state",
        "layers": layers,
        "delta_scales": list(scales),
        "selection": selection_summary,
        "arms": [arm.__dict__ for arm in arms],
        "generation": {"samples_per_row": args.samples_per_row, "temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
        "summary": summary,
        "geometry": geometry,
        "delta_output": str(args.delta_output),
        "n": len(rows_out),
        "controls": ["regenerated_baseline", "matched_random_position_subset", "cross_row_concept_specificity"],
        "causal_abstraction_claim": (
            "Low-rank-vs-distributed decider: if the per-layer mean hint-delta added to the "
            "unhinted run reproduces a large fraction of the full-patch repair, the focus state "
            "is a low-rank causal variable; subset arms localize which positions carry it; the "
            "cross-row arm tests concept specificity (row B under row A's delta should target "
            "A's concept)."
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
