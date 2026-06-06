#!/usr/bin/env python3
"""DAS-style low-rank subspace interchange preflight for matched h1/h4 pairs."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()
if "BD_PATH" not in os.environ:
    scratch_bd = Path(f"/scratch/scholar/{os.environ.get('USER', '')}/beyond-deduction")
    if scratch_bd.exists():
        os.environ["BD_PATH"] = str(scratch_bd)

from scripts.stage2_patch_clean_to_corrupt import (  # noqa: E402
    capture_vectors,
    find_landmarks,
    json_default,
    score_sequence_logprob,
    select_pairs,
    torch_dtype,
)
from src.activations import load_tl_model, render_chat_text, validate_hooks  # noqa: E402
from src.stage2_causal_abstraction import make_experiment_report  # noqa: E402
from src.stage2_probes import read_jsonl  # noqa: E402
from src.stage2_steering import parse_int_list  # noqa: E402


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def vector_norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector).item())


def make_delta_basis(deltas: list[torch.Tensor], rank: int) -> tuple[torch.Tensor, dict[str, Any]]:
    if rank <= 0 or not deltas:
        dim = int(deltas[0].numel()) if deltas else 0
        return torch.empty(dim, 0, dtype=torch.float32), {
            "basis_rank_requested": rank,
            "basis_rank_effective": 0,
            "basis_source_count": len(deltas),
            "singular_values": [],
        }
    matrix = torch.stack([delta.detach().cpu().float() for delta in deltas], dim=0)
    max_rank = min(rank, matrix.shape[0], matrix.shape[1])
    if max_rank <= 0:
        return torch.empty(matrix.shape[1], 0, dtype=torch.float32), {
            "basis_rank_requested": rank,
            "basis_rank_effective": 0,
            "basis_source_count": len(deltas),
            "singular_values": [],
        }
    _u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    basis = vh[:max_rank, :].T.contiguous().float()
    return basis, {
        "basis_rank_requested": rank,
        "basis_rank_effective": int(max_rank),
        "basis_source_count": len(deltas),
        "singular_values": [float(value) for value in singular_values[:max_rank].detach().cpu()],
    }


def random_basis(
    *,
    dim: int,
    rank: int,
    seed: int,
    orthogonal_to: torch.Tensor | None = None,
) -> torch.Tensor:
    if rank <= 0:
        return torch.empty(dim, 0, dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    extra = min(max(rank, 4), max(dim - rank, 0))
    columns = rank + extra
    for _attempt in range(8):
        matrix = torch.randn(dim, columns, generator=generator, dtype=torch.float32)
        if orthogonal_to is not None and orthogonal_to.numel() and orthogonal_to.shape[1] > 0:
            q = orthogonal_to.detach().cpu().float()
            matrix = matrix - q @ (q.T @ matrix)
        q_random, _r = torch.linalg.qr(matrix, mode="reduced")
        if q_random.shape[1] >= rank:
            return q_random[:, :rank].contiguous().float()
        columns += rank
    raise RuntimeError(f"failed to create random basis dim={dim} rank={rank}")


def project_delta(delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0 or basis.shape[1] == 0:
        return torch.zeros_like(delta)
    basis = basis.to(dtype=delta.dtype)
    return basis @ (basis.T @ delta)


def load_positive_control_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open() as f:
        payload = json.load(f)

    def find_first_key(value: Any, key: str) -> Any:
        if isinstance(value, dict):
            if key in value:
                return value[key]
            for child in value.values():
                found = find_first_key(child, key)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = find_first_key(child, key)
                if found is not None:
                    return found
        return None

    return {
        "path": str(path),
        "passed_positive_control_gate": find_first_key(payload, "passed_positive_control_gate"),
        "passed_toward_upper": find_first_key(payload, "passed_toward_upper"),
        "passed_toward_lower": find_first_key(payload, "passed_toward_lower"),
        "passed_toward_short": find_first_key(payload, "passed_toward_short"),
        "passed_toward_long": find_first_key(payload, "passed_toward_long"),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["intervention_mode"],
                int(row["layer"]),
                row["landmark"],
                int(row["basis_rank_effective"]),
            )
        ].append(row)

    out: dict[str, Any] = {}
    for (mode, layer, landmark, rank), group in sorted(groups.items()):
        key = f"{mode}_L{layer}_{landmark}_r{rank}"
        deltas = [float(row["margin_delta_vs_baseline"]) for row in group]
        recoveries = [float(row["recovery_fraction"]) for row in group]
        breakages = [
            float(row["breakage_fraction"])
            for row in group
            if row.get("breakage_fraction") is not None
        ]
        patch_l2 = [float(row["patch_delta_l2"]) for row in group]
        capture = [float(row["delta_capture_fraction"]) for row in group]
        out[key] = {
            "n": len(group),
            "mean_margin": float(np.mean([float(row["patched_margin_gold_minus_foil"]) for row in group])),
            "mean_margin_delta": float(np.mean(deltas)),
            "std_margin_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "mean_recovery_fraction": float(np.mean(recoveries)),
            "mean_breakage_fraction": float(np.mean(breakages)) if breakages else None,
            "mean_patch_delta_l2": float(np.mean(patch_l2)),
            "mean_delta_capture_fraction": float(np.mean(capture)),
            "margin_improved_count": int(sum(delta > 0 for delta in deltas)),
            "margin_decreased_count": int(sum(delta < 0 for delta in deltas)),
            "false_to_true_repair_count": int(sum(bool(row["false_to_true_repair"]) for row in group)),
            "true_to_false_disruption_count": int(sum(bool(row["true_to_false_disruption"]) for row in group)),
        }
    return out


def add_noise_comparisons(summary: dict[str, Any]) -> None:
    noise_by_site_rank: dict[tuple[int, str, int], dict[str, Any]] = {}
    for key, value in summary.items():
        if not key.startswith("matched_gaussian_"):
            continue
        parts = key.split("_")
        layer = int(parts[2][1:])
        landmark = "_".join(parts[3:-1])
        rank = int(parts[-1][1:])
        noise_by_site_rank[(layer, landmark, rank)] = value

    for key, value in summary.items():
        if not key.startswith("das_subspace_"):
            continue
        parts = key.split("_")
        layer = int(parts[2][1:])
        landmark = "_".join(parts[3:-1])
        rank = int(parts[-1][1:])
        noise = noise_by_site_rank.get((layer, landmark, rank))
        if not noise:
            continue
        noise_std = float(noise["std_margin_delta"])
        value["matched_gaussian_mean_margin_delta"] = noise["mean_margin_delta"]
        value["matched_gaussian_std_margin_delta"] = noise_std
        value["mean_delta_minus_matched_gaussian"] = (
            float(value["mean_margin_delta"]) - float(noise["mean_margin_delta"])
        )
        value["mean_delta_vs_noise_sigma"] = (
            value["mean_delta_minus_matched_gaussian"] / noise_std
            if noise_std > 1e-9
            else None
        )


def summarize_baselines(pair_baselines: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [float(row["clean_reference_margin_gold_minus_foil"]) for row in pair_baselines]
    corrupt = [float(row["corrupt_baseline_margin_gold_minus_foil"]) for row in pair_baselines]
    denominators = [float(row["recovery_denominator"]) for row in pair_baselines]
    return {
        "n_pairs": len(pair_baselines),
        "mean_clean_reference_margin_gold_minus_foil": float(np.mean(clean)),
        "mean_corrupt_baseline_margin_gold_minus_foil": float(np.mean(corrupt)),
        "mean_recovery_denominator": float(np.mean(denominators)),
        "clean_positive_margin_count": int(sum(value > 0 for value in clean)),
        "corrupt_positive_margin_count": int(sum(value > 0 for value in corrupt)),
        "corrupt_negative_margin_count": int(sum(value < 0 for value in corrupt)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-key", default="gemma3_27b")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--splits", type=Path, default=Path("results/stage2/splits.jsonl"))
    parser.add_argument("--split-family", default="s1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--pair-limit", type=int, default=8)
    parser.add_argument("--pair-seed", type=int, default=20260501)
    parser.add_argument("--layers", default="45")
    parser.add_argument("--landmarks", default="last_prompt")
    parser.add_argument("--ranks", default="1,2,4")
    parser.add_argument("--basis-scope", choices=("leave_one_out", "all_pairs"), default="leave_one_out")
    parser.add_argument(
        "--patch-direction",
        choices=("clean_to_corrupt", "corrupt_to_clean"),
        default="clean_to_corrupt",
    )
    parser.add_argument(
        "--intervention-modes",
        default="das_subspace,orthogonal_subspace,random_subspace,matched_gaussian,source",
    )
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--noise-seed", type=int, default=20260605)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--positive-control-report",
        type=Path,
        default=Path("docs/positive_control_format_gemma3_27b_l45.json"),
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("results/stage2/das/das_subspace_27b_l45_property_clean_to_corrupt_pilot.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/das_subspace_27b_l45_property_clean_to_corrupt_pilot.json"),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    started = time.time()
    torch.set_grad_enabled(False)
    rows = read_jsonl(args.jsonl)
    layers = parse_int_list(args.layers)
    landmarks = parse_csv(args.landmarks)
    ranks = parse_int_list(args.ranks)
    intervention_modes = parse_csv(args.intervention_modes)
    known_modes = {"das_subspace", "orthogonal_subspace", "random_subspace", "matched_gaussian", "source"}
    unknown_modes = sorted(set(intervention_modes) - known_modes)
    if unknown_modes:
        raise ValueError(f"unknown intervention mode(s): {unknown_modes}")
    if not ranks:
        raise ValueError("at least one rank is required")

    pairs, pair_summary = select_pairs(
        rows=rows,
        jsonl_path=args.jsonl,
        splits_path=args.splits,
        split_family=args.split_family,
        split=args.split,
        limit=args.pair_limit,
        seed=args.pair_seed,
    )
    if len(pairs) < args.pair_limit:
        print(f"warning: selected only {len(pairs)} pairs for requested limit {args.pair_limit}", flush=True)
    if not pairs:
        raise ValueError("no clean/corrupt pairs selected")

    positive_control = load_positive_control_summary(args.positive_control_report)

    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        dry_pairs = []
        missing_counts: dict[str, int] = defaultdict(int)
        for pair in pairs:
            clean_text = render_chat_text(
                tokenizer,
                system=pair.clean_row["system_prompt"],
                user=pair.clean_row["prompt_text"],
                model_name=args.model,
                add_generation_prompt=True,
            )
            corrupt_text = render_chat_text(
                tokenizer,
                system=pair.corrupt_row["system_prompt"],
                user=pair.corrupt_row["prompt_text"],
                model_name=args.model,
                add_generation_prompt=True,
            )
            clean_tokens = tokenizer(clean_text, add_special_tokens=False)["input_ids"]
            corrupt_tokens = tokenizer(corrupt_text, add_special_tokens=False)["input_ids"]
            clean_landmarks = find_landmarks(tokenizer=tokenizer, token_ids=clean_tokens, row=pair.clean_row)
            corrupt_landmarks = find_landmarks(tokenizer=tokenizer, token_ids=corrupt_tokens, row=pair.corrupt_row)
            for landmark in landmarks:
                if clean_landmarks.get(landmark) is None:
                    missing_counts[f"clean_{landmark}"] += 1
                if corrupt_landmarks.get(landmark) is None:
                    missing_counts[f"corrupt_{landmark}"] += 1
            dry_pairs.append(
                {
                    "pair_id": pair.pair_id,
                    "clean_row_index": pair.clean_row_index,
                    "corrupt_row_index": pair.corrupt_row_index,
                    "key": pair.key,
                    "gold_hypothesis": pair.gold_hypothesis,
                    "foil_hypothesis": pair.foil_hypothesis,
                    "clean_token_count": len(clean_tokens),
                    "corrupt_token_count": len(corrupt_tokens),
                    "clean_landmarks": clean_landmarks,
                    "corrupt_landmarks": corrupt_landmarks,
                }
            )
        print(
            json.dumps(
                {
                    "selection": pair_summary,
                    "layers": layers,
                    "landmarks": landmarks,
                    "ranks": ranks,
                    "basis_scope": args.basis_scope,
                    "patch_direction": args.patch_direction,
                    "intervention_modes": intervention_modes,
                    "positive_control": positive_control,
                    "missing_landmarks": dict(missing_counts),
                    "pairs": dry_pairs,
                },
                indent=2,
                sort_keys=True,
                default=json_default,
            )
        )
        return 0

    dtype = torch_dtype(args.dtype)
    print("Stage 2 DAS-style low-rank subspace interchange preflight", flush=True)
    print(f"model={args.model} task={args.task} direction={args.patch_direction}", flush=True)
    print(f"layers={layers} landmarks={landmarks} ranks={ranks} scope={args.basis_scope}", flush=True)
    print(f"pairs={len(pairs)} selection={pair_summary}", flush=True)
    print(f"positive_control={positive_control}", flush=True)
    print(f"transformer-lens={package_version('transformer-lens')} torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()}", flush=True)

    model = load_tl_model(
        args.model,
        n_devices=args.n_devices,
        n_ctx=args.n_ctx,
        dtype=dtype,
        load_mode=args.load_mode,
    )
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("loaded model has no tokenizer")
    hook_names = validate_hooks(model, layers)
    hook_names_by_layer = dict(zip(layers, hook_names, strict=True))

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    pair_baselines: list[dict[str, Any]] = []
    pair_payloads: dict[int, dict[str, Any]] = {}
    deltas_by_group: dict[tuple[int, str], dict[int, torch.Tensor]] = defaultdict(dict)
    missing_landmarks: dict[str, int] = defaultdict(int)

    for pair in pairs:
        clean_text = render_chat_text(
            tokenizer,
            system=pair.clean_row["system_prompt"],
            user=pair.clean_row["prompt_text"],
            model_name=args.model,
            add_generation_prompt=True,
        )
        corrupt_text = render_chat_text(
            tokenizer,
            system=pair.corrupt_row["system_prompt"],
            user=pair.corrupt_row["prompt_text"],
            model_name=args.model,
            add_generation_prompt=True,
        )
        clean_tokens = tokenizer(clean_text, add_special_tokens=False)["input_ids"]
        corrupt_tokens = tokenizer(corrupt_text, add_special_tokens=False)["input_ids"]
        clean_landmarks = find_landmarks(tokenizer=tokenizer, token_ids=clean_tokens, row=pair.clean_row)
        corrupt_landmarks = find_landmarks(tokenizer=tokenizer, token_ids=corrupt_tokens, row=pair.corrupt_row)

        corrupt_gold, _ = score_sequence_logprob(
            model=model,
            hook_name=None,
            patch_vector=None,
            patch_position=None,
            prompt_token_ids=corrupt_tokens,
            candidate_text=pair.gold_hypothesis,
        )
        corrupt_foil, _ = score_sequence_logprob(
            model=model,
            hook_name=None,
            patch_vector=None,
            patch_position=None,
            prompt_token_ids=corrupt_tokens,
            candidate_text=pair.foil_hypothesis,
        )
        clean_gold, _ = score_sequence_logprob(
            model=model,
            hook_name=None,
            patch_vector=None,
            patch_position=None,
            prompt_token_ids=clean_tokens,
            candidate_text=pair.gold_hypothesis,
        )
        clean_foil, _ = score_sequence_logprob(
            model=model,
            hook_name=None,
            patch_vector=None,
            patch_position=None,
            prompt_token_ids=clean_tokens,
            candidate_text=pair.foil_hypothesis,
        )
        corrupt_margin = corrupt_gold.logprob - corrupt_foil.logprob
        clean_margin = clean_gold.logprob - clean_foil.logprob
        denominator = clean_margin - corrupt_margin

        clean_vectors = capture_vectors(
            model=model,
            hook_names_by_layer=hook_names_by_layer,
            token_ids=clean_tokens,
            layers=layers,
            landmarks=clean_landmarks,
        )
        corrupt_vectors = capture_vectors(
            model=model,
            hook_names_by_layer=hook_names_by_layer,
            token_ids=corrupt_tokens,
            layers=layers,
            landmarks=corrupt_landmarks,
        )

        pair_baseline = {
            "pair_id": pair.pair_id,
            "clean_row_index": pair.clean_row_index,
            "corrupt_row_index": pair.corrupt_row_index,
            "key": pair.key,
            "gold_hypothesis": pair.gold_hypothesis,
            "foil_hypothesis": pair.foil_hypothesis,
            "clean_token_count": len(clean_tokens),
            "corrupt_token_count": len(corrupt_tokens),
            "clean_landmarks": clean_landmarks,
            "corrupt_landmarks": corrupt_landmarks,
            "corrupt_baseline_margin_gold_minus_foil": corrupt_margin,
            "clean_reference_margin_gold_minus_foil": clean_margin,
            "recovery_denominator": denominator,
        }
        pair_baselines.append(pair_baseline)
        pair_payloads[pair.pair_id] = {
            "pair": pair,
            "clean_tokens": clean_tokens,
            "corrupt_tokens": corrupt_tokens,
            "clean_landmarks": clean_landmarks,
            "corrupt_landmarks": corrupt_landmarks,
            "clean_vectors": clean_vectors,
            "corrupt_vectors": corrupt_vectors,
            "corrupt_margin": corrupt_margin,
            "clean_margin": clean_margin,
            "denominator": denominator,
        }

        print(
            f"pair {pair.pair_id}/{len(pairs)-1} clean={pair.clean_row_index} corrupt={pair.corrupt_row_index} "
            f"margin_corrupt={corrupt_margin:.3f} margin_clean={clean_margin:.3f}",
            flush=True,
        )

        for layer in layers:
            for landmark in landmarks:
                clean_pos = clean_landmarks.get(landmark)
                corrupt_pos = corrupt_landmarks.get(landmark)
                if clean_pos is None or corrupt_pos is None:
                    if clean_pos is None:
                        missing_landmarks[f"clean_{landmark}"] += 1
                    if corrupt_pos is None:
                        missing_landmarks[f"corrupt_{landmark}"] += 1
                    continue
                clean_vector = clean_vectors.get((layer, landmark))
                corrupt_vector = corrupt_vectors.get((layer, landmark))
                if clean_vector is None or corrupt_vector is None:
                    missing_landmarks[f"capture_L{layer}_{landmark}"] += 1
                    continue
                if args.patch_direction == "clean_to_corrupt":
                    deltas_by_group[(layer, landmark)][pair.pair_id] = clean_vector - corrupt_vector
                else:
                    deltas_by_group[(layer, landmark)][pair.pair_id] = corrupt_vector - clean_vector

    all_rows: list[dict[str, Any]] = []
    with args.out_jsonl.open("w") as fout:
        for pair in pairs:
            payload = pair_payloads[pair.pair_id]
            clean_tokens = payload["clean_tokens"]
            corrupt_tokens = payload["corrupt_tokens"]
            clean_landmarks = payload["clean_landmarks"]
            corrupt_landmarks = payload["corrupt_landmarks"]
            clean_vectors = payload["clean_vectors"]
            corrupt_vectors = payload["corrupt_vectors"]
            corrupt_margin = float(payload["corrupt_margin"])
            clean_margin = float(payload["clean_margin"])
            denominator = float(payload["denominator"])

            for layer in layers:
                for landmark in landmarks:
                    clean_pos = clean_landmarks.get(landmark)
                    corrupt_pos = corrupt_landmarks.get(landmark)
                    clean_vector = clean_vectors.get((layer, landmark))
                    corrupt_vector = corrupt_vectors.get((layer, landmark))
                    if (
                        clean_pos is None
                        or corrupt_pos is None
                        or clean_vector is None
                        or corrupt_vector is None
                    ):
                        continue
                    if args.patch_direction == "clean_to_corrupt":
                        source_vector = clean_vector
                        target_vector = corrupt_vector
                        patch_position = corrupt_pos
                        prompt_tokens = corrupt_tokens
                        baseline_margin = corrupt_margin
                        reference_margin = clean_margin
                        baseline_label = "corrupt"
                        reference_label = "clean"
                    else:
                        source_vector = corrupt_vector
                        target_vector = clean_vector
                        patch_position = clean_pos
                        prompt_tokens = clean_tokens
                        baseline_margin = clean_margin
                        reference_margin = corrupt_margin
                        baseline_label = "clean"
                        reference_label = "corrupt"

                    delta = (source_vector - target_vector).detach().cpu().float()
                    delta_norm = vector_norm(delta)
                    group_deltas = deltas_by_group[(layer, landmark)]

                    for rank in ranks:
                        if args.basis_scope == "leave_one_out":
                            basis_pair_ids = [pid for pid in sorted(group_deltas) if pid != pair.pair_id]
                        else:
                            basis_pair_ids = sorted(group_deltas)
                        basis_deltas = [group_deltas[pid] for pid in basis_pair_ids]
                        basis, basis_summary = make_delta_basis(basis_deltas, rank)
                        effective_rank = int(basis_summary["basis_rank_effective"])
                        if effective_rank <= 0:
                            continue
                        projected_delta = project_delta(delta, basis)
                        projected_norm = vector_norm(projected_delta)
                        dim = int(delta.numel())

                        patch_vectors: list[tuple[str, str, torch.Tensor, float, int | None]] = []
                        if "das_subspace" in intervention_modes:
                            patch_vectors.append(
                                (
                                    "das_subspace",
                                    "candidate",
                                    target_vector + projected_delta,
                                    projected_norm,
                                    None,
                                )
                            )
                        if "orthogonal_subspace" in intervention_modes:
                            orth_basis = random_basis(
                                dim=dim,
                                rank=effective_rank,
                                seed=args.noise_seed + pair.pair_id * 100003 + layer * 1009 + rank * 97,
                                orthogonal_to=basis,
                            )
                            orth_delta = project_delta(delta, orth_basis)
                            patch_vectors.append(
                                (
                                    "orthogonal_subspace",
                                    "orthogonal_direction",
                                    target_vector + orth_delta,
                                    vector_norm(orth_delta),
                                    None,
                                )
                            )
                        if "random_subspace" in intervention_modes:
                            rand_basis = random_basis(
                                dim=dim,
                                rank=effective_rank,
                                seed=args.noise_seed + pair.pair_id * 100003 + layer * 1009 + rank * 193 + 17,
                            )
                            random_delta = project_delta(delta, rand_basis)
                            patch_vectors.append(
                                (
                                    "random_subspace",
                                    "random_subspace",
                                    target_vector + random_delta,
                                    vector_norm(random_delta),
                                    None,
                                )
                            )
                        if "matched_gaussian" in intervention_modes:
                            for repeat in range(args.noise_repeats):
                                rng = np.random.default_rng(
                                    args.noise_seed
                                    + pair.pair_id * 100003
                                    + layer * 1009
                                    + rank * 389
                                    + repeat * 17
                                )
                                noise = torch.as_tensor(
                                    rng.standard_normal(dim),
                                    dtype=torch.float32,
                                )
                                noise = noise / torch.linalg.vector_norm(noise).clamp_min(1e-12)
                                noise = noise * max(projected_norm, 1e-12)
                                patch_vectors.append(
                                    (
                                        "matched_gaussian",
                                        "matched_gaussian_noise",
                                        target_vector + noise,
                                        vector_norm(noise),
                                        repeat,
                                    )
                                )
                        if "source" in intervention_modes:
                            patch_vectors.append(
                                (
                                    "source",
                                    "exact_patch_validation",
                                    source_vector,
                                    delta_norm,
                                    None,
                                )
                            )

                        for mode, control_type, patch_vector, patch_delta_l2, repeat in patch_vectors:
                            patched_gold, gold_hook = score_sequence_logprob(
                                model=model,
                                hook_name=hook_names_by_layer[layer],
                                patch_vector=patch_vector,
                                patch_position=patch_position,
                                prompt_token_ids=prompt_tokens,
                                candidate_text=pair.gold_hypothesis,
                            )
                            patched_foil, foil_hook = score_sequence_logprob(
                                model=model,
                                hook_name=hook_names_by_layer[layer],
                                patch_vector=patch_vector,
                                patch_position=patch_position,
                                prompt_token_ids=prompt_tokens,
                                candidate_text=pair.foil_hypothesis,
                            )
                            patched_margin = patched_gold.logprob - patched_foil.logprob
                            margin_delta = patched_margin - baseline_margin
                            recovery = margin_delta / denominator if abs(denominator) > 1e-9 else 0.0
                            breakage = (
                                -margin_delta / denominator
                                if args.patch_direction == "corrupt_to_clean" and abs(denominator) > 1e-9
                                else None
                            )
                            row: dict[str, Any] = {
                                "schema_version": 1,
                                "method": "low_rank_delta_subspace_interchange_preflight",
                                "representation_type": "das_subspace",
                                "patch_direction": args.patch_direction,
                                "basis_scope": args.basis_scope,
                                "pair_id": pair.pair_id,
                                "clean_row_index": pair.clean_row_index,
                                "corrupt_row_index": pair.corrupt_row_index,
                                "key": pair.key,
                                "layer": layer,
                                "hook_name": hook_names_by_layer[layer],
                                "landmark": landmark,
                                "intervention_mode": mode,
                                "control_type": control_type,
                                "noise_repeat": repeat,
                                "basis_rank_requested": int(rank),
                                "basis_rank_effective": effective_rank,
                                "basis_source_count": int(basis_summary["basis_source_count"]),
                                "basis_pair_ids": basis_pair_ids,
                                "basis_singular_values": basis_summary["singular_values"],
                                "clean_position": clean_pos,
                                "corrupt_position": corrupt_pos,
                                "patch_position": patch_position,
                                "clean_token_count": len(clean_tokens),
                                "corrupt_token_count": len(corrupt_tokens),
                                "gold_hypothesis": pair.gold_hypothesis,
                                "foil_hypothesis": pair.foil_hypothesis,
                                "corrupt_baseline_margin_gold_minus_foil": corrupt_margin,
                                "clean_reference_margin_gold_minus_foil": clean_margin,
                                "baseline_label": baseline_label,
                                "reference_label": reference_label,
                                "baseline_margin_gold_minus_foil": baseline_margin,
                                "reference_margin_gold_minus_foil": reference_margin,
                                "patched_gold_logprob": patched_gold.logprob,
                                "patched_foil_logprob": patched_foil.logprob,
                                "patched_margin_gold_minus_foil": patched_margin,
                                "margin_delta_vs_baseline": margin_delta,
                                "margin_delta_vs_corrupt": patched_margin - corrupt_margin,
                                "recovery_denominator": denominator,
                                "recovery_fraction": recovery,
                                "breakage_fraction": breakage,
                                "source_delta_l2": delta_norm,
                                "subspace_delta_l2": projected_norm,
                                "patch_delta_l2": patch_delta_l2,
                                "delta_capture_fraction": (
                                    (patch_delta_l2 * patch_delta_l2) / (delta_norm * delta_norm)
                                    if delta_norm > 1e-12
                                    else 0.0
                                ),
                                "false_to_true_repair": bool(
                                    args.patch_direction == "clean_to_corrupt"
                                    and baseline_margin < 0
                                    and patched_margin > 0
                                ),
                                "true_to_false_disruption": bool(
                                    args.patch_direction == "corrupt_to_clean"
                                    and baseline_margin > 0
                                    and patched_margin < 0
                                ),
                                "gold_hook_applications": gold_hook["applications"],
                                "foil_hook_applications": foil_hook["applications"],
                            }
                            all_rows.append(row)
                            fout.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
                            fout.flush()
                            print(
                                f"  pair={pair.pair_id} L{layer} {landmark} r{effective_rank} {mode}: "
                                f"delta={margin_delta:.3f} recovery={recovery:.3f}"
                                + (f" breakage={breakage:.3f}" if breakage is not None else ""),
                                flush=True,
                            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize_rows(all_rows)
    add_noise_comparisons(summary)
    matched_noise_summary = {
        key: value
        for key, value in summary.items()
        if key.startswith("matched_gaussian_")
    }
    paired_flips = {
        key: {
            "n": value["n"],
            "false_to_true_repair_count": value["false_to_true_repair_count"],
            "true_to_false_disruption_count": value["true_to_false_disruption_count"],
        }
        for key, value in summary.items()
    }
    controls = [
        "regenerated_baseline",
        "orthogonal_direction",
        "matched_gaussian_noise",
        "exact_patch_validation",
    ]
    if positive_control and positive_control.get("passed_positive_control_gate") is True:
        controls.append("positive_control")

    experiment_report = make_experiment_report(
        model=args.model,
        task=args.task,
        target_variable="gold_vs_foil_margin",
        split=args.split,
        site_or_layer=",".join(f"L{layer}:{','.join(landmarks)}" for layer in layers),
        method="low_rank_delta_subspace_interchange_preflight",
        representation_type="das_subspace",
        result_type="causal",
        controls=controls,
        n=len(pairs),
        baseline_metrics=summarize_baselines(pair_baselines),
        intervention_metrics=summary,
        paired_flips=paired_flips,
        parse_fail_rate=None,
        matched_noise_summary=matched_noise_summary,
        causal_abstraction_claim=(
            "Preflight causal-abstraction test for whether the InAbHyD gold_vs_foil_margin "
            "variable can be manipulated by interchanging a learned low-rank h1/h4 residual-state "
            "subspace. This is DAS-style distributed interchange, not a full trained DAS alignment; "
            "positive evidence requires the das_subspace mode to exceed orthogonal/random/matched-Gaussian "
            "controls and source exact-patch validation."
        ),
        notes=[
            "Basis is built from matched h1-correct/h4-incorrect residual deltas, with leave-one-out default.",
            "The working thesis remains provisional: if this fails with passing controls, say causal inaccessibility under tested methods rather than causally distributed.",
            "Teacher-forced gold-vs-foil margins are used here; no free-form parse rate is available.",
        ],
    )

    report = {
        **experiment_report,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_das_subspace_interchange.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "model_key": args.model_key,
        "jsonl": str(args.jsonl),
        "splits": str(args.splits),
        "selection": pair_summary,
        "patch_direction": args.patch_direction,
        "layers": layers,
        "landmarks": landmarks,
        "ranks": ranks,
        "basis_scope": args.basis_scope,
        "intervention_modes": intervention_modes,
        "positive_control": positive_control,
        "environment": {
            "transformer_lens": package_version("transformer-lens"),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "n_devices": args.n_devices,
            "n_ctx": args.n_ctx,
            "dtype": str(dtype),
            "load_mode": args.load_mode,
        },
        "missing_landmarks": dict(sorted(missing_landmarks.items())),
        "pair_baselines": pair_baselines,
        "out_jsonl": str(args.out_jsonl),
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
