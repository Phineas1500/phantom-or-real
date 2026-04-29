#!/usr/bin/env python3
"""Pre-handoff sanity checks for Stage 2 Phase B outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def check_probe_auc_present(probe_auc: dict[str, Any]) -> dict[str, Any]:
    tasks = probe_auc.get("results", {})
    ok = bool(tasks)
    return {
        "name": "probe_auc_has_results",
        "status": "pass" if ok else "fail",
        "detail": {"tasks_found": list(tasks.keys())},
    }


def check_transfer_present(probe_transfer: dict[str, Any]) -> dict[str, Any]:
    results = probe_transfer.get("results", {})
    ok = bool(results)
    return {
        "name": "probe_transfer_has_results",
        "status": "pass" if ok else "fail",
        "detail": {"directions_found": list(results.keys())},
    }


def check_nontrivial_probe_lift(probe_auc: dict[str, Any]) -> dict[str, Any]:
    found = []
    for task, by_layer in probe_auc.get("results", {}).items():
        for layer, by_probe in by_layer.items():
            for probe_type in ("logistic", "diffmeans"):
                for split in ("s1", "s3"):
                    cell = by_probe.get(probe_type, {}).get(split)
                    if not cell or cell.get("status") != "ok":
                        continue
                    delta = cell.get("delta_over_best_b0")
                    if delta is not None and delta > 0.0:
                        found.append(
                            {
                                "task": task,
                                "layer": layer,
                                "probe_type": probe_type,
                                "split": split,
                                "delta_over_best_b0": delta,
                            }
                        )
    return {
        "name": "nontrivial_probe_lift_over_b0",
        "status": "pass" if found else "fail",
        "detail": {"positive_delta_rows": found[:10], "count": len(found)},
    }


def check_transfer_gap(probe_transfer: dict[str, Any]) -> dict[str, Any]:
    ok_rows = []
    for direction, by_layer in probe_transfer.get("results", {}).items():
        for layer, by_split in by_layer.items():
            for split in ("s1", "s3"):
                cell = by_split.get(split)
                if not cell or cell.get("status") != "ok":
                    continue
                source_auc = cell.get("source_test_auc")
                target_auc = cell.get("target_test_auc")
                if source_auc is None or target_auc is None:
                    continue
                ok_rows.append(
                    {
                        "direction": direction,
                        "layer": layer,
                        "split": split,
                        "source_test_auc": source_auc,
                        "target_test_auc": target_auc,
                        "transfer_gap_abs": abs(float(source_auc) - float(target_auc)),
                    }
                )
    return {
        "name": "transfer_has_evaluable_rows",
        "status": "pass" if ok_rows else "fail",
        "detail": {"rows": ok_rows[:10], "count": len(ok_rows)},
    }


def check_pkl_artifacts(probe_auc: dict[str, Any], probes_dir: Path, model_key: str) -> dict[str, Any]:
    missing = []
    present = []
    for task, by_layer in probe_auc.get("results", {}).items():
        for layer, by_probe in by_layer.items():
            for probe_type in ("logistic", "diffmeans"):
                for split in ("s1", "s2", "s3"):
                    cell = by_probe.get(probe_type, {}).get(split)
                    if not cell:
                        continue
                    if cell.get("status") != "ok":
                        continue
                    expected = probes_dir / f"{model_key}_{task}_{layer}_{split}.{probe_type}.pkl"
                    if expected.exists():
                        present.append(str(expected))
                    else:
                        missing.append(str(expected))
    status = "pass" if not missing else "fail"
    return {
        "name": "probe_pkl_artifacts_exist_for_ok_rows",
        "status": status,
        "detail": {"present_count": len(present), "missing_count": len(missing), "missing": missing},
    }


def check_label_shuffle_files(label_shuffle_dir: Path, model_slug: str) -> dict[str, Any]:
    expected = [
        label_shuffle_dir / f"label_shuffle_{model_slug}_infer_property_L11.json",
        label_shuffle_dir / f"label_shuffle_{model_slug}_infer_subtype_L11.json",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    return {
        "name": "label_shuffle_outputs_present",
        "status": "pass" if not missing else "warn",
        "detail": {"expected": [str(p) for p in expected], "missing": missing},
    }


def check_s2_status(probe_auc: dict[str, Any]) -> dict[str, Any]:
    statuses = []
    for task, by_layer in probe_auc.get("results", {}).items():
        for layer, by_probe in by_layer.items():
            for probe_type in ("logistic", "diffmeans"):
                cell = by_probe.get(probe_type, {}).get("s2")
                if cell:
                    statuses.append(
                        {
                            "task": task,
                            "layer": layer,
                            "probe_type": probe_type,
                            "status": cell.get("status"),
                        }
                    )
    only_expected = all(s["status"] in ("skipped_no_evaluable_holdout", "ok") for s in statuses) if statuses else True
    return {
        "name": "s2_status_is_expected",
        "status": "pass" if only_expected else "warn",
        "detail": {"rows": statuses},
    }


def check_h1_signal(probe_auc: dict[str, Any]) -> dict[str, Any]:
    notes = []
    ok = False
    for task, by_layer in probe_auc.get("results", {}).items():
        for layer, by_probe in by_layer.items():
            cell = by_probe.get("logistic", {}).get("s1")
            if not cell or cell.get("status") != "ok":
                continue
            h1 = (cell.get("per_height") or {}).get("h1")
            if not h1:
                continue
            auc = h1.get("auc")
            n = h1.get("n")
            notes.append({"task": task, "layer": layer, "h1_auc": auc, "n": n})
            if auc is not None and auc > 0.5:
                ok = True
    return {
        "name": "h1_probe_signal_above_chance_exists",
        "status": "pass" if ok else "warn",
        "detail": {"rows": notes},
    }


def check_length_diagnostics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "name": "length_diagnostics_output_present",
            "status": "warn",
            "detail": {"missing": str(path)},
        }
    data = read_json(path)
    return {
        "name": "length_diagnostics_output_present",
        "status": "pass",
        "detail": {"path": str(path), "top_level_keys": list(data.keys())},
    }


def summarize(checks: list[dict[str, Any]]) -> dict[str, int]:
    out = {"pass": 0, "warn": 0, "fail": 0}
    for c in checks:
        out[c["status"]] += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-auc", type=Path, default=Path("results/stage2/probe_auc.json"))
    parser.add_argument("--probe-transfer", type=Path, default=Path("results/stage2/probe_transfer.json"))
    parser.add_argument("--probes-dir", type=Path, default=Path("results/stage2/probes"))
    parser.add_argument("--label-shuffle-dir", type=Path, default=Path("docs"))
    parser.add_argument("--model-key", default="gemma3_4b")
    parser.add_argument("--model-slug", default="4b")
    parser.add_argument(
        "--length-diagnostics",
        type=Path,
        default=Path("results/stage2/probe_diagnostics_length_4b.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/stage2/phase_b_sanity.json"))
    args = parser.parse_args()

    probe_auc = read_json(args.probe_auc)
    probe_transfer = read_json(args.probe_transfer)

    checks = [
        check_probe_auc_present(probe_auc),
        check_transfer_present(probe_transfer),
        check_nontrivial_probe_lift(probe_auc),
        check_transfer_gap(probe_transfer),
        check_pkl_artifacts(probe_auc, args.probes_dir, args.model_key),
        check_label_shuffle_files(args.label_shuffle_dir, args.model_slug),
        check_s2_status(probe_auc),
        check_h1_signal(probe_auc),
        check_length_diagnostics(args.length_diagnostics),
    ]

    summary = summarize(checks)
    overall = "fail" if summary["fail"] > 0 else ("warn" if summary["warn"] > 0 else "pass")
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "summary": summary,
        "checks": checks,
    }
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
