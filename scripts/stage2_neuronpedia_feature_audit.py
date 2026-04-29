#!/usr/bin/env python3
"""Fetch Neuronpedia dashboard metadata for top sparse probe features."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def fetch_feature(url: str, *, timeout: float, retries: int, sleep_seconds: float) -> dict[str, Any]:
    last_error = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "phantom-or-real-stage2-audit/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(sleep_seconds)
    return {"error": last_error or "unknown error"}


def short_list(values: Any, *, n: int) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values[:n]]


def explanations(feature: dict[str, Any], *, n: int) -> list[str]:
    rows = feature.get("explanations")
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows[:n]:
        if isinstance(row, dict):
            text = row.get("description") or row.get("explanation")
            if text:
                out.append(str(text))
    return out


def markdown_table(rows: list[dict[str, Any]], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "| Task | Rank | Feature | Assoc. | Probe density | Neuronpedia explanation | Positive logits | Negative logits | Link |",
        "| --- | ---: | ---: | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        np_data = row.get("neuronpedia", {})
        explanation = "; ".join(np_data.get("explanations", [])) or "No explanation found"
        positive = ", ".join(np_data.get("positive_logits", []))
        negative = ", ".join(np_data.get("negative_logits", []))
        density = row.get("activation_all", {}).get("density")
        density_text = "" if density is None else f"{float(density):.3f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['task']}`",
                    str(row["rank"]),
                    str(row["feature"]),
                    row["association"],
                    density_text,
                    explanation.replace("|", "\\|"),
                    positive.replace("|", "\\|"),
                    negative.replace("|", "\\|"),
                    f"[link]({row['url']})",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--sae-id", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--model-id", default="gemma-3-27b-it")
    parser.add_argument("--tasks", nargs="+", default=["infer_property", "infer_subtype"])
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = read_json(args.feature_report)
    rows = []
    for task in args.tasks:
        features = report["models"][args.sae_id][task]["top_abs_features"][: args.top_n]
        for feature in features:
            feature_index = int(feature["feature"])
            page_url = f"https://www.neuronpedia.org/{args.model_id}/{args.source_id}/{feature_index}"
            api_url = f"https://www.neuronpedia.org/api/feature/{args.model_id}/{args.source_id}/{feature_index}"
            fetched = fetch_feature(
                api_url,
                timeout=args.timeout,
                retries=args.retries,
                sleep_seconds=args.sleep_seconds,
            )
            rows.append(
                {
                    "task": task,
                    "rank": int(feature["rank"]),
                    "feature": feature_index,
                    "association": feature["association"],
                    "sign": feature["sign"],
                    "weight": feature["weight"],
                    "activation_all": feature.get("activation_all", {}),
                    "url": page_url,
                    "api_url": api_url,
                    "neuronpedia": {
                        "error": fetched.get("error"),
                        "explanations": explanations(fetched, n=3),
                        "positive_logits": short_list(fetched.get("pos_str"), n=8),
                        "negative_logits": short_list(fetched.get("neg_str"), n=8),
                        "frac_nonzero": fetched.get("frac_nonzero"),
                        "max_act_approx": fetched.get("maxActApprox"),
                        "hook_name": fetched.get("hookName"),
                        "source": fetched.get("source"),
                    },
                }
            )
            time.sleep(args.sleep_seconds)

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_report": str(args.feature_report),
        "model_id": args.model_id,
        "source_id": args.source_id,
        "sae_id": args.sae_id,
        "top_n": args.top_n,
        "rows": rows,
    }
    write_json(args.output_json, payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_table(rows, title=f"Neuronpedia Audit: {args.source_id}") + "\n")
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
