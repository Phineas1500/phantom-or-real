#!/usr/bin/env python3
"""Composite focus-state job: necessity + expressivity ladder + spotlight.

Arms run in priority order (arms-outer loop) so a walltime kill costs the
cheapest arms: unhinted baseline; hinted in-job baseline; reverse subset
patch (necessity: unhinted states into the hinted run at concept positions)
with its random-position control; the restricted-delta ladder (own
concept-position mean delta at scales); rank-1 and per-position spotlight
(foreign deltas at own concept positions); complement patch. Saves
concept-position states and delta submatrices for offline restricted
geometry, rank-k PCA, and SAE feature-diff. KV transplant deferred to a
dedicated job. Spec: docs/causal_handle_directions.md NEXT CLUSTER JOB.
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
from scripts.stage2_hint_delta import concept_positions, prompt_cache  # noqa: E402
from scripts.stage2_hint_state_interchange import generate_sample_batch  # noqa: E402
from scripts.stage2_interchange_concept_analysis import canon, subjects_of  # noqa: E402
from scripts.stage2_prompt_margin_gated_decode_correction import select_prefix_rows  # noqa: E402
from scripts.stage2_proposal_hints import make_user_prompt  # noqa: E402
from scripts.stage2_recognition_state_patch import longest_common_token_block  # noqa: E402
from src.activations import load_tl_model, render_chat_text, validate_hooks  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.stage2_steering import parse_int_list, score_reply  # noqa: E402


@dataclass(frozen=True)
class Arm:
    label: str
    kind: str
    receiver: str  # "unhinted" | "hinted"
    reference: str  # which baseline arm pairs with it
    scale: float = 1.0


def build_arms(scales: list[float]) -> list[Arm]:
    arms = [
        Arm("baseline", "none", "unhinted", "none"),
        Arm("hinted_baseline", "none", "hinted", "none"),
        Arm("reverse_subset", "replace_concept_with_unhinted", "hinted", "hinted_baseline"),
        Arm("reverse_random", "replace_random_with_unhinted", "hinted", "hinted_baseline"),
    ]
    arms.append(Arm(f"restricted_add_s{scales[0]:g}".replace(".", "p"), "restricted_add", "unhinted", "baseline", scales[0]))
    arms.append(Arm("spotlight_rank1", "spotlight_rank1", "unhinted", "baseline"))
    for scale in scales[1:]:
        arms.append(Arm(f"restricted_add_s{scale:g}".replace(".", "p"), "restricted_add", "unhinted", "baseline", scale))
    arms.append(Arm("complement", "replace_complement_with_hinted", "unhinted", "baseline"))
    arms.append(Arm("spotlight_perpos", "spotlight_perpos", "unhinted", "baseline"))
    return arms


def make_replace_hook(states: torch.Tensor, positions: list[int]):
    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        if not positions or act.shape[1] <= max(positions):
            return act
        donor = states.to(device=act.device, dtype=torch.float32)
        for row_index, pos in enumerate(positions):
            act[:, pos, :] = donor[row_index].to(act.dtype)
        return act

    return hook_fn


def make_position_add_hook(matrix: torch.Tensor, positions: list[int], scale: float):
    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        if not positions or act.shape[1] <= max(positions):
            return act
        add = (scale * matrix).to(device=act.device, dtype=torch.float32)
        for row_index, pos in enumerate(positions):
            act[:, pos, :] += add[row_index].to(act.dtype)
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
    parser.add_argument("--restricted-scales", default="1,4")
    parser.add_argument("--min-block-tokens", type=int, default=32)
    parser.add_argument("--samples-per-row", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=20260614)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/focus_state_composite_27b_property.jsonl"))
    parser.add_argument("--states-output", type=Path, default=Path("results/stage2/erasure/focus_state_composite_27b_property_states.npz"))
    parser.add_argument("--output", type=Path, default=Path("docs/focus_state_composite_27b_property.json"))
    args = parser.parse_args()
    load_dotenv()
    torch.set_grad_enabled(False)
    started = time.time()

    layers = parse_int_list(args.layers)
    scales = [float(part) for part in args.restricted_scales.split(",")]
    arms = build_arms(scales)
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
        print(json.dumps({"selection": selection_summary, "arms": [arm.__dict__ for arm in arms], "total_generations": total}, indent=2, default=json_default), flush=True)
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
        positions_h = [p + h_start for p in rel]
        rng = random.Random(args.sample_seed + source_row_index)
        block_rel = list(range(block_len))
        random_rel = sorted(rng.sample(block_rel, len(rel)))
        complement_rel = [p for p in block_rel if p not in set(rel)]

        unhinted_cache = prompt_cache(model, receiver_ids, hook_names)
        hinted_cache = prompt_cache(model, hinted_ids, hook_names)
        per_layer = {}
        for layer in layers:
            u_block = unhinted_cache[hook_by_layer[layer]][r_start : r_start + block_len]
            h_block = hinted_cache[hook_by_layer[layer]][h_start : h_start + block_len]
            concept_delta = h_block[rel] - u_block[rel]
            per_layer[layer] = {
                "u_block": u_block,
                "h_block": h_block,
                "concept_delta": concept_delta,
                "restricted_mean": concept_delta.mean(dim=0),
            }
        prepared.append({
            "row": stage1_row,
            "source_row_index": source_row_index,
            "gold_concept": gold_concept,
            "receiver_ids": receiver_ids,
            "hinted_ids": hinted_ids,
            "r_start": r_start,
            "h_start": h_start,
            "block_len": block_len,
            "rel": rel,
            "random_rel": random_rel,
            "complement_rel": complement_rel,
            "per_layer": per_layer,
        })
        print(f"prepared row {source_row_index}: block={block_len} concept_tokens={len(rel)}", flush=True)

    n_rows = len(prepared)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    with args.out_jsonl.open("w") as fout:
        for arm_index, arm in enumerate(arms):
            arm_started = time.time()
            for prep_index, prep in enumerate(prepared):
                neighbor = prepared[(prep_index + 1) % n_rows]
                torch.manual_seed(args.sample_seed + prep["source_row_index"] * 10007 + arm_index * 101)
                token_ids = prep["hinted_ids"] if arm.receiver == "hinted" else prep["receiver_ids"]
                start = prep["h_start"] if arm.receiver == "hinted" else prep["r_start"]
                fwd_hooks = []
                donor_concept = None
                for layer in layers:
                    pl = prep["per_layer"][layer]
                    if arm.kind == "none":
                        break
                    if arm.kind == "replace_concept_with_unhinted":
                        fwd_hooks.append((hook_by_layer[layer], make_replace_hook(pl["u_block"][prep["rel"]], [start + p for p in prep["rel"]])))
                    elif arm.kind == "replace_random_with_unhinted":
                        fwd_hooks.append((hook_by_layer[layer], make_replace_hook(pl["u_block"][prep["random_rel"]], [start + p for p in prep["random_rel"]])))
                    elif arm.kind == "replace_complement_with_hinted":
                        fwd_hooks.append((hook_by_layer[layer], make_replace_hook(pl["h_block"][prep["complement_rel"]], [start + p for p in prep["complement_rel"]])))
                    elif arm.kind == "restricted_add":
                        matrix = pl["restricted_mean"].unsqueeze(0).expand(len(prep["rel"]), -1)
                        fwd_hooks.append((hook_by_layer[layer], make_position_add_hook(matrix, [start + p for p in prep["rel"]], arm.scale)))
                    elif arm.kind == "spotlight_rank1":
                        donor_concept = neighbor["gold_concept"]
                        matrix = neighbor["per_layer"][layer]["restricted_mean"].unsqueeze(0).expand(len(prep["rel"]), -1)
                        fwd_hooks.append((hook_by_layer[layer], make_position_add_hook(matrix, [start + p for p in prep["rel"]], 1.0)))
                    elif arm.kind == "spotlight_perpos":
                        donor_concept = neighbor["gold_concept"]
                        source = neighbor["per_layer"][layer]["concept_delta"]
                        cycled = torch.stack([source[i % source.shape[0]] for i in range(len(prep["rel"]))])
                        fwd_hooks.append((hook_by_layer[layer], make_position_add_hook(cycled, [start + p for p in prep["rel"]], 1.0)))
                    else:
                        raise ValueError(f"unknown arm kind {arm.kind!r}")
                with model.hooks(fwd_hooks=fwd_hooks):
                    batch = generate_sample_batch(model=model, token_ids=token_ids, n_samples=args.samples_per_row, max_new_tokens=args.max_new_tokens, temperature=args.temperature, stop_at_eos=True, cache_dtype=dtype)
                for sample_index, (new_ids, reply) in enumerate(batch):
                    score = score_reply(prep["row"], reply)
                    subjects = subjects_of(reply)
                    out = {
                        "schema_version": 1,
                        "source_row_index": prep["source_row_index"],
                        "example_id": prep["row"].get("example_id"),
                        "height": prep["row"].get("height"),
                        "model": args.model,
                        "condition": arm.label,
                        "arm_kind": arm.kind,
                        "receiver": arm.receiver,
                        "reference": arm.reference,
                        "scale": arm.scale,
                        "sample_index": sample_index,
                        "method": "focus_state_composite",
                        "target_variable": "target_concept",
                        "representation_type": "patched_residual_state",
                        "patch_layers": layers,
                        "gold_concept": prep["gold_concept"],
                        "n_concept_positions": len(prep["rel"]),
                        "spotlight_donor_row_index": neighbor["source_row_index"] if donor_concept else None,
                        "spotlight_donor_concept": donor_concept,
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
                print(f"arm {arm_index + 1}/{len(arms)} {arm.label} row {prep_index + 1}/{n_rows}: P(strong)={strong_rate:.2f}", flush=True)
            print(f"ARM DONE {arm.label}: {time.time() - arm_started:.0f}s elapsed_total={time.time() - started:.0f}s", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    state_arrays = {}
    for prep in prepared:
        idx = prep["source_row_index"]
        for layer in layers:
            pl = prep["per_layer"][layer]
            state_arrays[f"L{layer}_row{idx}_concept_delta"] = pl["concept_delta"].numpy().astype(np.float32)
            state_arrays[f"L{layer}_row{idx}_unhinted_concept_states"] = pl["u_block"][prep["rel"]].numpy().astype(np.float32)
            state_arrays[f"L{layer}_row{idx}_hinted_concept_states"] = pl["h_block"][prep["rel"]].numpy().astype(np.float32)
    args.states_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.states_output, **state_arrays)

    from collections import defaultdict
    by = defaultdict(lambda: defaultdict(list))
    for r in rows_out:
        by[r["condition"]][r["source_row_index"]].append(r)
    rng = np.random.default_rng(args.sample_seed)
    summary = {}
    refs = {arm.label: arm.reference for arm in arms}
    for cond in sorted(by):
        per_row = {i: float(np.mean([s["is_correct_strong"] for s in v])) for i, v in by[cond].items()}
        entry: dict[str, Any] = {
            "n_rows": len(per_row),
            "strong_accuracy": float(np.mean([s["is_correct_strong"] for v in by[cond].values() for s in v])),
            "targets_gold_concept_rate": float(np.mean([s["targets_gold_concept"] for v in by[cond].values() for s in v])),
            "reference": refs.get(cond, "none"),
        }
        donor_vals = [s["targets_donor_concept"] for v in by[cond].values() for s in v if s["targets_donor_concept"] is not None]
        if donor_vals:
            entry["targets_donor_concept_rate"] = float(np.mean(donor_vals))
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
        "script": "scripts/stage2_focus_state_composite.py",
        "model": args.model,
        "task": args.task,
        "target_variable": "target_concept",
        "method": "focus_state_composite",
        "representation_type": "patched_residual_state",
        "layers": layers,
        "selection": selection_summary,
        "arms": [arm.__dict__ for arm in arms],
        "generation": {"samples_per_row": args.samples_per_row, "temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
        "summary": summary,
        "states_output": str(args.states_output),
        "n": len(rows_out),
        "controls": [
            "in_job_hinted_baseline",
            "reverse_patch_random_positions",
            "matched_random_subsets_from_456999",
            "spotlight_uses_gold_validated_donors",
        ],
        "causal_abstraction_claim": (
            "Necessity (reverse subset vs hinted baseline, against random-position control), "
            "expressivity ladder (replacement -> per-position add -> rank-1 -> foreign rank-1), "
            "spotlight-vs-commitment, and the complement decomposition — at the concept-mention "
            "positions established by job 456999. KV transplant deferred to a dedicated job."
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
