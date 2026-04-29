#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import load_tl_model  # noqa: E402


TASKS = ("infer_property", "infer_subtype")


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dtype_str(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


def check_equivalence_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "gate_pass": False,
            "reason": f"missing file: {path}",
        }

    report = read_json(path)
    checks = {c.get("name"): c.get("status") for c in report.get("checks", [])}
    hard_expected_ok = [
        "stage1_jsonl_invariants",
        "model_tokenizer_invariants",
        "message_builder_contract",
        "prompt_encoding",
    ]
    hard_failures = [name for name in hard_expected_ok if checks.get(name) != "ok"]

    # In this repo's validator, top5/greedy are currently often "skipped".
    tier3 = checks.get("stage1_serving_top5_logits")
    tier4 = checks.get("greedy_output_byte_match")

    return {
        "exists": True,
        "report_status": report.get("status"),
        "hard_checks_expected_ok": hard_expected_ok,
        "hard_failures": hard_failures,
        "tier3_status": tier3,
        "tier4_status": tier4,
        "gate_pass": (len(hard_failures) == 0 and report.get("status") == "ok"),
        "note": "Current validator may mark tier3/tier4 as skipped without serving-stack access.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--model-key", default="gemma3_4b")
    parser.add_argument("--activations-dir", type=Path, default=Path("results/stage2/activations"))
    parser.add_argument("--jsonl-dir", type=Path, default=Path("results/full/with_errortype"))
    parser.add_argument("--layer-selection", type=Path, default=Path("docs/layer_selection.json"))
    parser.add_argument(
        "--equivalence-report",
        type=Path,
        default=Path("docs/stage2_full_validation_4b_infer_property.json"),
    )
    parser.add_argument("--spot-check-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--output", type=Path, default=Path("results/stage2/phase_a_verification.json"))
    args = parser.parse_args()

    selection = read_json(args.layer_selection)
    layers = [int(x) for x in selection["selected_layers"]]
    if len(layers) != 3:
        raise ValueError(f"expected 3 selected layers, got {layers}")
    l_last = max(layers)

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "model": args.model,
            "model_key": args.model_key,
            "layers": layers,
            "l_last": l_last,
            "activations_dir": str(args.activations_dir),
            "jsonl_dir": str(args.jsonl_dir),
            "equivalence_report": str(args.equivalence_report),
            "spot_check_n": args.spot_check_n,
            "seed": args.seed,
        },
        "artifact_checks": {},
        "spot_check": {},
        "equivalence_gate": {},
        "status": "running",
        "errors": [],
    }

    # 1) Artifact existence/shape/dtype/order checks
    task_rows: dict[str, list[dict[str, Any]]] = {}
    for task in TASKS:
        src_path = args.jsonl_dir / f"{args.model_key}_{task}.jsonl"
        if not src_path.exists():
            raise FileNotFoundError(src_path)
        task_rows[task] = read_jsonl(src_path)

    for task in TASKS:
        source_rows = task_rows[task]
        expected_n = len(source_rows)
        expected_ids = [row.get("example_id") for row in source_rows]
        report["artifact_checks"][task] = {}

        for layer in layers:
            prefix = args.activations_dir / f"{args.model_key}_{task}_L{layer}"
            safep = prefix.with_suffix(".safetensors")
            sidep = prefix.with_suffix(".example_ids.jsonl")
            metap = prefix.with_suffix(".meta.json")

            missing = [str(p) for p in (safep, sidep, metap) if not p.exists()]
            if missing:
                entry = {"ok": False, "missing": missing}
                report["artifact_checks"][task][f"L{layer}"] = entry
                report["errors"].append(f"{task} L{layer}: missing artifacts")
                continue

            tensor = load_file(safep)["activations"]
            sidecar = read_jsonl(sidep)
            meta = read_json(metap)

            observed_ids = [row.get("example_id") for row in sidecar]
            order_match = observed_ids == expected_ids

            shape_ok = list(tensor.shape) == [expected_n, 2560]
            dtype_ok = dtype_str(tensor) == "bfloat16"
            sidecar_count_ok = len(sidecar) == expected_n

            meta_shape_ok = meta.get("shape") == [expected_n, 2560]
            meta_dtype_ok = meta.get("dtype") in ("torch.bfloat16", "bfloat16")
            meta_count_ok = meta.get("row_count") == expected_n

            ok = all(
                [
                    shape_ok,
                    dtype_ok,
                    sidecar_count_ok,
                    order_match,
                    meta_shape_ok,
                    meta_dtype_ok,
                    meta_count_ok,
                ]
            )

            report["artifact_checks"][task][f"L{layer}"] = {
                "ok": ok,
                "tensor_shape": list(tensor.shape),
                "tensor_dtype": dtype_str(tensor),
                "sidecar_rows": len(sidecar),
                "expected_rows": expected_n,
                "example_id_order_match": order_match,
                "meta_shape": meta.get("shape"),
                "meta_dtype": meta.get("dtype"),
                "meta_row_count": meta.get("row_count"),
            }
            if not ok:
                report["errors"].append(f"{task} L{layer}: artifact mismatch")

    # 2) Spot-check logits at L_last
    model = load_tl_model(args.model, n_devices=1, n_ctx=4096, load_mode="no-processing")
    model.eval()
    device = next(model.unembed.parameters()).device
    tok = model.tokenizer

    rng = random.Random(args.seed)
    report["spot_check"] = {"layer": l_last, "tasks": {}}

    for task in TASKS:
        source_rows = task_rows[task]
        tensor = load_file(args.activations_dir / f"{args.model_key}_{task}_L{l_last}.safetensors")["activations"]
        n = len(source_rows)
        picks = sorted(rng.sample(range(n), min(args.spot_check_n, n)))
        task_results = []
        task_ok = True

        for idx in picks:
            row = source_rows[idx]
            out_text = row.get("model_output", "") or ""
            out_ids = tok(out_text, add_special_tokens=False)["input_ids"]
            if not out_ids:
                task_ok = False
                task_results.append(
                    {
                        "row_index": idx,
                        "example_id": row.get("example_id"),
                        "ok": False,
                        "reason": "empty tokenization for model_output",
                    }
                )
                continue

            expected_first = int(out_ids[0])

            with torch.inference_mode():
                h = tensor[idx].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                h = model.ln_final(h)
                logits = model.unembed(h)
                pred_first = int(torch.argmax(logits[0]).item())

            ok = pred_first == expected_first
            task_ok = task_ok and ok
            task_results.append(
                {
                    "row_index": idx,
                    "example_id": row.get("example_id"),
                    "expected_first_token_id": expected_first,
                    "pred_first_token_id": pred_first,
                    "ok": ok,
                }
            )

        report["spot_check"]["tasks"][task] = {
            "ok": task_ok,
            "sampled_indices": picks,
            "results": task_results,
        }
        if not task_ok:
            report["errors"].append(f"{task}: spot-check mismatch at L{l_last}")

    # 3) Equivalence gate
    report["equivalence_gate"] = check_equivalence_report(args.equivalence_report)
    if not report["equivalence_gate"]["gate_pass"]:
        report["errors"].append("equivalence gate failed")

    report["status"] = "ok" if not report["errors"] else "failed"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print(args.output)
    print(report["status"])
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()