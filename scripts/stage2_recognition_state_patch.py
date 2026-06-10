#!/usr/bin/env python3
"""Recognition-state cross-prompt patching on the commitment rowset.

Donor run: the hard-foil forced-choice prompt for a recognition-gap row, where
the model selects gold. Receiver run: the free-form generation prompt for the
same row, where free-form output was strong-wrong. The shared ontology-context
token block is located by longest-common-substring over token ids, and the
donor's residual activations over that block are patched into the receiver's
prompt forward at the chosen layers before generation. A repair claim requires
false-to-true repairs above position-shuffled donor and magnitude-matched
Gaussian controls. See docs/causal_handle_directions.md experiment 2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_decode_time_correction import (  # noqa: E402
    generate_one,
    json_default,
    package_version,
    torch_dtype,
)
from scripts.stage2_prompt_margin_gated_decode_correction import select_prefix_rows  # noqa: E402
from scripts.stage2_steer_answer_property_margins import build_margin_prompt  # noqa: E402
from scripts.stage2_steer_forced_choice_direction import parse_choice  # noqa: E402
from src.activations import load_tl_model, render_chat_text, validate_hooks  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.stage2_steering import parse_int_list, score_reply  # noqa: E402


@dataclass(frozen=True)
class PatchCondition:
    label: str
    patch_kind: str | None


def parse_condition_kinds(value: str) -> list[str]:
    allowed = {"baseline", "patch_recognition", "patch_shuffled", "noise_matched"}
    parsed = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("expected at least one condition")
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        raise ValueError(f"unknown condition kind(s): {unknown}")
    return parsed


def make_condition_plan(condition_kinds: list[str]) -> list[PatchCondition]:
    plan: list[PatchCondition] = []
    if "baseline" in condition_kinds:
        plan.append(PatchCondition("baseline", None))
    for kind in ("patch_recognition", "patch_shuffled", "noise_matched"):
        if kind in condition_kinds:
            plan.append(PatchCondition(kind, kind))
    if not plan or plan[0].label != "baseline":
        raise ValueError("baseline condition is required for paired interpretation")
    return plan


def longest_common_token_block(
    donor_ids: list[int], receiver_ids: list[int]
) -> tuple[int, int, int]:
    matcher = SequenceMatcher(None, donor_ids, receiver_ids, autojunk=False)
    match = matcher.find_longest_match(0, len(donor_ids), 0, len(receiver_ids))
    return match.a, match.b, match.size


def make_patch_hook(
    *,
    donor_block: torch.Tensor,
    receiver_start: int,
    patch_kind: str,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    block_len = int(donor_block.shape[0])
    state: dict[str, Any] = {"applied": 0, "delta_l2": None, "applied_l2": None}

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        if act.shape[1] < receiver_start + block_len:
            return act
        donor = donor_block.to(device=act.device, dtype=torch.float32)
        span = act[:, receiver_start : receiver_start + block_len, :].float()
        delta = donor.unsqueeze(0) - span
        delta_l2 = float(delta.norm().item())
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if patch_kind == "patch_recognition":
            replacement = donor.unsqueeze(0)
        elif patch_kind == "patch_shuffled":
            perm = torch.randperm(block_len, generator=generator)
            replacement = donor[perm].unsqueeze(0)
        elif patch_kind == "noise_matched":
            noise = torch.randn(span.shape, generator=generator, dtype=torch.float32).to(span.device)
            norm = float(noise.norm().item())
            if norm == 0.0:
                return act
            replacement = span + noise * (delta_l2 / norm)
        else:
            raise ValueError(f"unknown patch kind {patch_kind!r}")
        act[:, receiver_start : receiver_start + block_len, :] = replacement.to(act.dtype)
        state["applied"] += 1
        state["delta_l2"] = delta_l2
        state["applied_l2"] = float((replacement - span).norm().item())
        return act

    return hook_fn, state


def run_donor(
    *,
    model,
    token_ids: list[int],
    hook_names: list[str],
    max_choice_tokens: int,
    cache_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], str | None]:
    from src.activations import input_device_for_model

    input_device = input_device_for_model(model)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=input_device)
    with torch.inference_mode():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name in set(hook_names),
            return_type=None,
        )
    donor_cache = {name: cache[name][0].detach().float().cpu() for name in hook_names}
    _, choice_reply = generate_one(
        model=model,
        token_ids=token_ids,
        max_new_tokens=max_choice_tokens,
        do_sample=False,
        temperature=0.0,
        stop_at_eos=True,
        cache_dtype=cache_dtype,
    )
    return donor_cache, parse_choice(choice_reply)


def summarize_patch_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import defaultdict

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    def rate(subset: list[dict[str, Any]], key: str) -> float | None:
        return sum(bool(row[key]) for row in subset) / len(subset) if subset else None

    summary: dict[str, Any] = {}
    row_p: dict[str, dict[int, float]] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in condition_rows:
            by_row[int(row["source_row_index"])].append(row)
        row_p[condition] = {
            row_index: rate(samples, "is_correct_strong") for row_index, samples in by_row.items()
        }
        summary[condition] = {
            "n_generations": len(condition_rows),
            "n_rows": len(by_row),
            "strong_accuracy": rate(condition_rows, "is_correct_strong"),
            "weak_accuracy": rate(condition_rows, "is_correct_weak"),
            "parse_fail_rate": rate(condition_rows, "parse_failed"),
            "mean_quality": (
                sum(float(row["quality_score"]) for row in condition_rows) / len(condition_rows)
                if condition_rows
                else None
            ),
        }

    baseline_p = row_p.get("baseline", {})
    paired: dict[str, Any] = {}
    for condition, p_by_row in sorted(row_p.items()):
        if condition == "baseline":
            continue
        deltas = [
            p_by_row[row_index] - baseline_p[row_index]
            for row_index in p_by_row
            if row_index in baseline_p
        ]
        if not deltas:
            continue
        paired[condition] = {
            "paired_n": len(deltas),
            "mean_delta_p_strong": float(np.mean(deltas)),
            "rows_improved": int(sum(delta > 0 for delta in deltas)),
            "rows_degraded": int(sum(delta < 0 for delta in deltas)),
            "false_to_true": int(
                sum(
                    baseline_p[row_index] == 0.0 and p_by_row[row_index] > 0.0
                    for row_index in p_by_row
                    if row_index in baseline_p
                )
            ),
            "true_to_false": int(
                sum(
                    baseline_p[row_index] == 1.0 and p_by_row[row_index] < 1.0
                    for row_index in p_by_row
                    if row_index in baseline_p
                )
            ),
        }
    return {"by_condition": summary, "paired_vs_baseline": paired}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--prefix-trajectory-jsonl", type=Path, default=Path("results/stage2/decode_time/prefix_conditioned_margin_trajectory_gemma_manifest.jsonl"))
    parser.add_argument("--prefix-checkpoint", default="0")
    parser.add_argument("--selection-limit", type=int, default=None)
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layers", default="30,40,45")
    parser.add_argument("--option-seed", type=int, default=20260430)
    parser.add_argument("--foil-source", default="stage1_model_output")
    parser.add_argument("--min-block-tokens", type=int, default=32)
    parser.add_argument("--conditions", default="baseline,patch_recognition,patch_shuffled,noise_matched")
    parser.add_argument("--samples-per-row", type=int, default=1)
    parser.add_argument("--sample-seed", type=int, default=20260610)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-choice-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/recognition_state_patch_27b_property_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/recognition_state_patch_27b_property_manifest.json"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    load_dotenv()
    torch.set_grad_enabled(False)
    started = time.time()

    if args.samples_per_row < 1:
        raise ValueError("--samples-per-row must be >= 1")
    if args.samples_per_row > 1 and not args.do_sample:
        raise ValueError("--samples-per-row > 1 requires --do-sample")
    layers = parse_int_list(args.layers)
    condition_plan = make_condition_plan(parse_condition_kinds(args.conditions))
    dtype = torch_dtype(args.dtype)
    source_file = str(args.jsonl)

    print("Recognition-state cross-prompt patching", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"task={args.task}", flush=True)
    print(f"layers={layers}", flush=True)
    print(f"conditions={[condition.label for condition in condition_plan]}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)

    selected_rows, selection_summary = select_prefix_rows(
        prefix_jsonl=args.prefix_trajectory_jsonl,
        source_jsonl=args.jsonl,
        checkpoint=args.prefix_checkpoint,
        limit=args.selection_limit,
        prompt_gold_vs_foil_threshold=0.0,
    )
    if not selected_rows:
        raise ValueError(f"no rows selected: {selection_summary}")
    total_generations = len(selected_rows) * len(condition_plan) * args.samples_per_row
    print(
        f"selected_rows={len(selected_rows)} total_generations={total_generations}",
        flush=True,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "selection": selection_summary,
                    "conditions": [condition.__dict__ for condition in condition_plan],
                    "layers": layers,
                    "total_generations": total_generations,
                },
                indent=2,
                sort_keys=True,
                default=json_default,
            ),
            flush=True,
        )
        return 0

    bd_path = ensure_on_path()
    print(f"beyond_deduction_path={bd_path}", flush=True)

    model = load_tl_model(
        args.model,
        n_devices=args.n_devices,
        n_ctx=args.n_ctx,
        dtype=dtype,
        load_mode=args.load_mode,
    )
    hook_names = validate_hooks(model, layers)
    hook_name_by_layer = dict(zip(layers, hook_names))
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("loaded model has no tokenizer")
    print(f"using_hooks={hook_name_by_layer}", flush=True)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    with args.out_jsonl.open("w") as fout:
        for row_idx, stage1_row in enumerate(selected_rows, start=1):
            source_row_index = int(stage1_row["row_index"])
            receiver_text = render_chat_text(
                tokenizer,
                system=stage1_row["system_prompt"],
                user=stage1_row["prompt_text"],
                model_name=args.model,
                add_generation_prompt=True,
            )
            receiver_ids = tokenizer(receiver_text, add_special_tokens=False)["input_ids"]
            try:
                fc = build_margin_prompt(
                    stage1_row,
                    row_index=source_row_index,
                    option_seed=args.option_seed,
                    foil_source=args.foil_source,
                )
            except ValueError as exc:
                skipped.append({"source_row_index": source_row_index, "reason": str(exc)})
                print(f"row {row_idx}: skipped ({exc})", flush=True)
                continue
            donor_text = render_chat_text(
                tokenizer,
                system=fc.system,
                user=fc.user,
                model_name=args.model,
                add_generation_prompt=True,
            )
            donor_ids = tokenizer(donor_text, add_special_tokens=False)["input_ids"]
            if max(len(donor_ids), len(receiver_ids)) > args.n_ctx:
                raise ValueError(f"row {source_row_index} prompt exceeds n_ctx={args.n_ctx}")
            donor_start, receiver_start, block_len = longest_common_token_block(
                donor_ids, receiver_ids
            )
            if block_len < args.min_block_tokens:
                skipped.append(
                    {
                        "source_row_index": source_row_index,
                        "reason": f"matched block too short: {block_len}",
                    }
                )
                print(f"row {row_idx}: skipped (block_len={block_len})", flush=True)
                continue

            donor_cache, donor_choice = run_donor(
                model=model,
                token_ids=donor_ids,
                hook_names=hook_names,
                max_choice_tokens=args.max_choice_tokens,
                cache_dtype=dtype,
            )
            donor_selects_gold = donor_choice == fc.gold_choice
            print(
                f"row {row_idx}/{len(selected_rows)} source_row={source_row_index} "
                f"h={stage1_row.get('height')} block={block_len} tokens "
                f"(donor@{donor_start}, receiver@{receiver_start}) "
                f"donor_choice={donor_choice} gold={fc.gold_choice} selects_gold={donor_selects_gold}",
                flush=True,
            )

            donor_blocks = {
                layer: donor_cache[hook_name_by_layer[layer]][
                    donor_start : donor_start + block_len
                ]
                for layer in layers
            }
            for condition_index, condition in enumerate(condition_plan):
                for sample_index in range(args.samples_per_row):
                    if args.do_sample:
                        torch.manual_seed(
                            args.sample_seed
                            + source_row_index * 10007
                            + condition_index * 101
                            + sample_index
                        )
                    hook_states: dict[int, dict[str, Any]] = {}
                    fwd_hooks = []
                    if condition.patch_kind is not None:
                        for layer in layers:
                            hook_fn, hook_state = make_patch_hook(
                                donor_block=donor_blocks[layer],
                                receiver_start=receiver_start,
                                patch_kind=condition.patch_kind,
                                seed=args.sample_seed
                                + source_row_index * 10007
                                + condition_index * 101
                                + sample_index
                                + layer,
                            )
                            hook_states[layer] = hook_state
                            fwd_hooks.append((hook_name_by_layer[layer], hook_fn))
                    with model.hooks(fwd_hooks=fwd_hooks):
                        new_ids, reply = generate_one(
                            model=model,
                            token_ids=receiver_ids,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            stop_at_eos=args.stop_at_eos,
                            cache_dtype=dtype,
                        )
                    score = score_reply(stage1_row, reply)
                    output_row = {
                        "schema_version": 1,
                        "source_file": source_file,
                        "source_row_index": source_row_index,
                        "example_id": stage1_row.get("example_id"),
                        "task": stage1_row.get("task"),
                        "height": stage1_row.get("height"),
                        "model": args.model,
                        "original_is_correct_strong": bool(stage1_row.get("is_correct_strong")),
                        "condition": condition.label,
                        "sample_index": sample_index,
                        "method": "recognition_state_cross_prompt_patch",
                        "target_variable": "selected_hypothesis",
                        "representation_type": "patched_residual_state",
                        "patch_layers": layers,
                        "matched_block_tokens": block_len,
                        "donor_block_start": donor_start,
                        "receiver_block_start": receiver_start,
                        "donor_choice": donor_choice,
                        "donor_gold_choice": fc.gold_choice,
                        "donor_selects_gold": donor_selects_gold,
                        "gold_hypothesis": fc.gold_hypothesis,
                        "foil_hypothesis": fc.foil_hypothesis,
                        "prompt_token_count": len(receiver_ids),
                        "donor_prompt_token_count": len(donor_ids),
                        "generated_token_count": len(new_ids),
                        "model_output": reply,
                        "hook_summary": {
                            f"L{layer}": state for layer, state in sorted(hook_states.items())
                        },
                        **score,
                    }
                    rows.append(output_row)
                    fout.write(
                        json.dumps(output_row, ensure_ascii=False, default=json_default) + "\n"
                    )
                    fout.flush()
                    print(
                        f"  {condition.label}[{sample_index}]: strong={output_row['is_correct_strong']} "
                        f"weak={output_row['is_correct_weak']} parse_failed={output_row['parse_failed']} "
                        f"new_tokens={len(new_ids)}",
                        flush=True,
                    )
            del donor_cache, donor_blocks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    patch_summary = summarize_patch_rows(rows)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_recognition_state_patch.py",
        "model": args.model,
        "model_key": args.model_key,
        "task": args.task,
        "target_variable": "selected_hypothesis",
        "method": "recognition_state_cross_prompt_patch",
        "representation_type": "patched_residual_state",
        "layers": layers,
        "hook_names": hook_name_by_layer,
        "jsonl": str(args.jsonl),
        "prefix_trajectory_jsonl": str(args.prefix_trajectory_jsonl),
        "option_seed": args.option_seed,
        "foil_source": args.foil_source,
        "min_block_tokens": args.min_block_tokens,
        "out_jsonl": str(args.out_jsonl),
        "selection": selection_summary,
        "skipped_rows": skipped,
        "generation": {
            "conditions": [condition.__dict__ for condition in condition_plan],
            "samples_per_row": args.samples_per_row,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "stop_at_eos": args.stop_at_eos,
            "n_devices": args.n_devices,
            "n_ctx": args.n_ctx,
            "dtype": str(dtype),
            "load_mode": args.load_mode,
        },
        "summary": patch_summary,
        "baseline_metrics": patch_summary["by_condition"].get("baseline"),
        "intervention_metrics": {
            condition: metrics
            for condition, metrics in patch_summary["by_condition"].items()
            if condition != "baseline"
        },
        "paired_flips": patch_summary["paired_vs_baseline"],
        "n": len(rows),
        "controls": [
            "regenerated_baseline",
            "position_shuffled_donor",
            "magnitude_matched_gaussian_noise",
            "donor_choice_verification",
        ],
        "causal_abstraction_claim": (
            "Tests whether the recognition run's encoding of the shared ontology context "
            "carries the gold-preferring selection state into free-form generation. A repair "
            "claim requires false-to-true repairs above position-shuffled donor and "
            "magnitude-matched Gaussian controls on rows where the donor selects gold."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.out_jsonl}", flush=True)
    print(f"elapsed_seconds={report['elapsed_seconds']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
