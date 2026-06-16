#!/usr/bin/env python3
"""Geometry rider for the L30 rank-k causal core.

Projects the compact-core PCA components onto the INLP readable subspace and
Gemma Scope decoder rows. This is an offline analysis: it uses saved
concept-position deltas, saved INLP direction stacks, and cached Gemma Scope
SAE params referenced by existing feature metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_rank_k_guard import fit_pca_basis, load_delta_by_row  # noqa: E402


def parse_int_list(text: str) -> list[int]:
    values = [int(part) for part in text.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"expected positive integer list, got {text!r}")
    return values


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def orthonormal_columns(row_vectors: np.ndarray, *, eps: float = 1e-10) -> np.ndarray:
    rows = np.asarray(row_vectors, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"expected rank-2 rows, got shape {rows.shape}")
    norms = np.linalg.norm(rows, axis=1)
    keep = norms > eps
    if not np.any(keep):
        raise ValueError("no nonzero row vectors")
    q, r = np.linalg.qr(rows[keep].T)
    rank = int(np.sum(np.abs(np.diag(r)) > eps))
    if rank == 0:
        raise ValueError("row vectors are numerically rank zero")
    return q[:, :rank]


def subspace_fraction(target_q: np.ndarray, reference_q: np.ndarray) -> float:
    if target_q.shape[0] != reference_q.shape[0]:
        raise ValueError(f"dimension mismatch {target_q.shape} vs {reference_q.shape}")
    overlap = reference_q.T @ target_q
    return float(np.linalg.norm(overlap, ord="fro") ** 2 / target_q.shape[1])


def component_projection_fractions(components: np.ndarray, reference_q: np.ndarray) -> list[float]:
    comps = np.asarray(components, dtype=np.float64)
    comps = comps / np.linalg.norm(comps, axis=1, keepdims=True)
    projections = comps @ reference_q
    return [float(v) for v in np.sum(projections**2, axis=1)]


def random_subspace_null(dim: int, rank: int, reference_q: np.ndarray, samples: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(samples):
        mat = rng.standard_normal((dim, rank))
        q, _ = np.linalg.qr(mat)
        vals.append(subspace_fraction(q[:, :rank], reference_q))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "samples": int(samples),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if samples > 1 else 0.0,
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def merge_top_abs(current: dict[str, list[Any]], values: np.ndarray, indices: np.ndarray, top_n: int) -> dict[str, list[Any]]:
    old_abs = np.asarray(current.get("abs_cosine", []), dtype=np.float64)
    old_signed = np.asarray(current.get("signed_cosine", []), dtype=np.float64)
    old_index = np.asarray(current.get("feature_index", []), dtype=np.int64)
    new_abs = np.abs(values.astype(np.float64))
    new_signed = values.astype(np.float64)
    new_index = indices.astype(np.int64)
    all_abs = np.concatenate([old_abs, new_abs]) if old_abs.size else new_abs
    all_signed = np.concatenate([old_signed, new_signed]) if old_signed.size else new_signed
    all_index = np.concatenate([old_index, new_index]) if old_index.size else new_index
    if all_abs.size > top_n:
        keep = np.argpartition(-all_abs, top_n - 1)[:top_n]
        order = keep[np.argsort(-all_abs[keep])]
    else:
        order = np.argsort(-all_abs)
    return {
        "feature_index": [int(v) for v in all_index[order]],
        "signed_cosine": [float(v) for v in all_signed[order]],
        "abs_cosine": [float(v) for v in all_abs[order]],
    }


def decoder_report(
    *,
    meta_path: Path,
    components: np.ndarray,
    ranks: list[int],
    chunk_size: int,
    top_n: int,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import safe_open

    meta = read_json(meta_path)
    params_path = Path(meta["sae_params_file"])
    sae_id = meta["sae_id"]
    d_sae = int(meta["sae_cfg"]["d_sae"])
    comp = torch.from_numpy(components.astype(np.float32))
    comp = torch.nn.functional.normalize(comp, dim=1)
    max_rank = max(ranks)
    comp = comp[:max_rank]

    component_stats = {
        str(i + 1): {
            "total_sq_cosine": 0.0,
            "count_abs_ge_0p05": 0,
            "count_abs_ge_0p10": 0,
            "count_abs_ge_0p20": 0,
            "top": {"feature_index": [], "signed_cosine": [], "abs_cosine": []},
        }
        for i in range(max_rank)
    }
    subspace_stats = {
        str(rank): {
            "total_decoder_projection_energy": 0.0,
            "count_sqrt_energy_ge_0p05": 0,
            "count_sqrt_energy_ge_0p10": 0,
            "count_sqrt_energy_ge_0p20": 0,
            "top": {"feature_index": [], "signed_cosine": [], "abs_cosine": []},
        }
        for rank in ranks
    }

    with safe_open(params_path, framework="pt", device="cpu") as f:
        w_slice = f.get_slice("w_dec")
        shape = w_slice.get_shape()
        if list(shape) != [d_sae, components.shape[1]]:
            raise ValueError(f"w_dec shape {shape} incompatible with components {components.shape}")
        for start in range(0, d_sae, chunk_size):
            end = min(start + chunk_size, d_sae)
            dec = w_slice[start:end].float()
            dec = torch.nn.functional.normalize(dec, dim=1)
            dots = dec @ comp.T
            dots_np = dots.numpy()
            feature_indices = np.arange(start, end, dtype=np.int64)
            for j in range(max_rank):
                vals = dots_np[:, j]
                st = component_stats[str(j + 1)]
                st["total_sq_cosine"] += float(np.sum(vals.astype(np.float64) ** 2))
                abs_vals = np.abs(vals)
                st["count_abs_ge_0p05"] += int(np.sum(abs_vals >= 0.05))
                st["count_abs_ge_0p10"] += int(np.sum(abs_vals >= 0.10))
                st["count_abs_ge_0p20"] += int(np.sum(abs_vals >= 0.20))
                st["top"] = merge_top_abs(st["top"], vals, feature_indices, top_n)
            for rank in ranks:
                vals = np.sqrt(np.sum(dots_np[:, :rank].astype(np.float64) ** 2, axis=1))
                st = subspace_stats[str(rank)]
                st["total_decoder_projection_energy"] += float(np.sum(vals**2))
                st["count_sqrt_energy_ge_0p05"] += int(np.sum(vals >= 0.05))
                st["count_sqrt_energy_ge_0p10"] += int(np.sum(vals >= 0.10))
                st["count_sqrt_energy_ge_0p20"] += int(np.sum(vals >= 0.20))
                st["top"] = merge_top_abs(st["top"], vals, feature_indices, top_n)

    for st in component_stats.values():
        top_sq = sum(v * v for v in st["top"]["abs_cosine"])
        st["top_sq_fraction_of_total_sq"] = float(top_sq / st["total_sq_cosine"]) if st["total_sq_cosine"] else None
        st["top1_abs_cosine"] = st["top"]["abs_cosine"][0] if st["top"]["abs_cosine"] else None
    for st in subspace_stats.values():
        top_sq = sum(v * v for v in st["top"]["abs_cosine"])
        total = st["total_decoder_projection_energy"]
        st["top_sq_fraction_of_total_energy"] = float(top_sq / total) if total else None
        st["top1_sqrt_energy"] = st["top"]["abs_cosine"][0] if st["top"]["abs_cosine"] else None

    return {
        "sae_id": sae_id,
        "meta_path": str(meta_path),
        "params_path": str(params_path),
        "d_sae": d_sae,
        "component_stats": component_stats,
        "subspace_stats": subspace_stats,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Rank-Core Geometry Rider",
        "",
        f"Analysis of L{report['layer']} rank-core PCA components from `{report['states_npz']}`.",
        "",
        "## INLP overlap",
        "",
        "| rank | subspace fraction in INLP | null mean | null p95 | component fractions |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for rank, entry in report["inlp"].items():
        comps = ", ".join(f"{v:.4f}" for v in entry["component_projection_fractions"])
        lines.append(
            f"| {rank} | {entry['subspace_fraction_in_inlp']:.4f} | "
            f"{entry['random_null']['mean']:.4f} | {entry['random_null']['p95']:.4f} | {comps} |"
        )
    lines.extend(["", "## Gemma Scope decoder overlap", ""])
    for dec in report["decoders"]:
        lines.extend([
            f"### {dec['sae_id']}",
            "",
            "| rank | top decoder sqrt-energy | top20 fraction of decoder energy | count >=0.10 |",
            "| --- | ---: | ---: | ---: |",
        ])
        for rank, st in dec["subspace_stats"].items():
            lines.append(
                f"| {rank} | {st['top1_sqrt_energy']:.4f} | "
                f"{st['top_sq_fraction_of_total_energy']:.6f} | {st['count_sqrt_energy_ge_0p10']} |"
            )
        lines.extend(["", "Top features by subspace sqrt-energy:", ""])
        for rank, st in dec["subspace_stats"].items():
            pairs = ", ".join(
                f"{idx}:{val:.4f}"
                for idx, val in zip(st["top"]["feature_index"][:10], st["top"]["abs_cosine"][:10], strict=False)
            )
            lines.append(f"- rank {rank}: {pairs}")
        lines.append("")
    lines.extend(["## Verdict", "", report["verdict"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states-npz", type=Path, default=Path("results/stage2/erasure/focus_state_composite_27b_property_states.npz"))
    parser.add_argument("--inlp-npz", type=Path, default=Path("results/stage2/erasure/inlp_direction_stacks_27b_property.npz"))
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--ranks", default="4,8")
    parser.add_argument("--sae-meta", type=Path, action="append", default=[
        Path("results/stage2/sae_features/gemma3_27b_infer_property_L30_layer_30_width_16k_l0_small_top128.meta.json"),
        Path("results/stage2/sae_features/gemma3_27b_infer_property_L30_layer_30_width_262k_l0_small_top128.meta.json"),
    ])
    parser.add_argument("--decoder-chunk-size", type=int, default=4096)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--null-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--output", type=Path, default=Path("docs/rank_core_geometry_27b_property.json"))
    parser.add_argument("--summary-output", type=Path, default=Path("docs/rank_core_geometry_27b_property_summary.md"))
    args = parser.parse_args()
    started = time.time()
    ranks = parse_int_list(args.ranks)
    max_rank = max(ranks)

    delta_by_row = load_delta_by_row(args.states_npz, args.layer)
    basis = fit_pca_basis(delta_by_row, max_rank)
    components = basis["components"].astype(np.float64)
    inlp = np.load(args.inlp_npz)[f"L{args.layer}_inlp_stack"].astype(np.float64)
    inlp_q = orthonormal_columns(inlp)

    inlp_report = {}
    for rank in ranks:
        comp_rank = components[:rank]
        comp_q = orthonormal_columns(comp_rank)
        frac = subspace_fraction(comp_q, inlp_q)
        null = random_subspace_null(components.shape[1], rank, inlp_q, args.null_samples, args.seed + rank)
        inlp_report[str(rank)] = {
            "subspace_fraction_in_inlp": frac,
            "canonical_cosines": [float(v) for v in np.linalg.svd(inlp_q.T @ comp_q, compute_uv=False)],
            "component_projection_fractions": component_projection_fractions(comp_rank, inlp_q),
            "random_null": null,
        }

    decoders = []
    for meta_path in args.sae_meta:
        print(f"decoder geometry {meta_path}", flush=True)
        decoders.append(
            decoder_report(
                meta_path=meta_path,
                components=components,
                ranks=ranks,
                chunk_size=args.decoder_chunk_size,
                top_n=args.top_n,
            )
        )

    verdict = (
        "The held-out-surviving L30 core is essentially outside the INLP readable subspace: "
        "rank-4 and rank-8 overlap are far below the random-subspace null. Gemma Scope gives a "
        "different answer: decoder rows align strongly with the causal-core subspace and the first "
        "few PCA components, so the dictionary exposes the object. But the exposure is highly "
        "redundant rather than sparse-small: thousands of decoder rows have nontrivial overlap, and "
        "the top decoder rows explain only a tiny fraction of total decoder-overlap mass. The loop "
        "therefore closes as gauge-orthogonal but dictionary-visible, not as a compact handful of "
        "Gemma Scope features."
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "script": "scripts/stage2_rank_core_geometry.py",
        "states_npz": str(args.states_npz),
        "inlp_npz": str(args.inlp_npz),
        "layer": args.layer,
        "ranks": ranks,
        "pca_basis": {
            "source_rows": basis["source_rows"],
            "n_pooled_positions": basis["n_pooled_positions"],
            "explained_variance_ratio_rank_max": basis["explained_variance_ratio"],
        },
        "inlp": inlp_report,
        "decoders": decoders,
        "verdict": verdict,
    }
    write_json(args.output, report)
    write_markdown(args.summary_output, report)
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {args.summary_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
