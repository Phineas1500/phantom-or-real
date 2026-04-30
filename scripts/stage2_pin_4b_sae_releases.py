#!/usr/bin/env python3
"""Pin Gemma Scope 2 4B SAE/transcoder releases in Stage 2 invariants."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_write_invariants import sha256_file  # noqa: E402
from src.stage2_sae import sae_file_name, snapshot_revision_from_path  # noqa: E402


HF_REPO_ID = "google/gemma-scope-2-4b-it"


@dataclass(frozen=True)
class SaePin:
    sae_release: str
    sae_id: str
    hf_subfolder: str
    architecture: str
    d_in: int
    d_sae: int
    d_out: int | None = None


PINS: tuple[SaePin, ...] = (
    SaePin("gemma-scope-2-4b-it-res-all", "layer_20_width_16k_l0_small", "resid_post_all", "jumprelu", 2560, 16384),
    SaePin("gemma-scope-2-4b-it-res-all", "layer_20_width_262k_l0_small", "resid_post_all", "jumprelu", 2560, 262144),
    SaePin("gemma-scope-2-4b-it-res-all", "layer_22_width_16k_l0_small", "resid_post_all", "jumprelu", 2560, 16384),
    SaePin("gemma-scope-2-4b-it-res-all", "layer_22_width_262k_l0_small", "resid_post_all", "jumprelu", 2560, 262144),
    SaePin("gemma-scope-2-4b-it-mlp-all", "layer_22_width_16k_l0_small", "mlp_out_all", "jumprelu", 2560, 16384),
    SaePin(
        "gemma-scope-2-4b-it-transcoders-all",
        "layer_22_width_16k_l0_small_affine",
        "transcoder_all",
        "jumprelu_skip_transcoder",
        2560,
        16384,
        2560,
    ),
    SaePin(
        "gemma-scope-2-4b-it-transcoders-all",
        "layer_22_width_262k_l0_small_affine",
        "transcoder_all",
        "jumprelu_skip_transcoder",
        2560,
        262144,
        2560,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--invariants", type=Path, default=Path("docs/stage2_invariants.4b.semani.json"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--cache-dir", type=Path, default=None, help="HF hub cache dir, e.g. $HF_HOME/hub")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local-files-only", dest="local_files_only", action="store_true", default=True)
    group.add_argument("--allow-downloads", dest="local_files_only", action="store_false")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def download_artifact(pin: SaePin, file_name: str, args: argparse.Namespace) -> Path:
    kwargs: dict[str, Any] = {
        "repo_id": HF_REPO_ID,
        "filename": sae_file_name(pin.hf_subfolder, pin.sae_id, file_name),
        "revision": args.hf_revision,
        "local_files_only": args.local_files_only,
    }
    if args.cache_dir is not None:
        kwargs["cache_dir"] = str(args.cache_dir)
    return Path(hf_hub_download(**kwargs))


def pin_entry(pin: SaePin, args: argparse.Namespace) -> dict[str, Any]:
    config_path = download_artifact(pin, "config.json", args)
    params_path = download_artifact(pin, "params.safetensors", args)
    snapshot = snapshot_revision_from_path(config_path)
    if snapshot is None:
        raise ValueError(f"could not infer HF snapshot revision from {config_path}")

    entry: dict[str, Any] = {
        "architecture": pin.architecture,
        "d_in": pin.d_in,
        "d_sae": pin.d_sae,
        "hf_repo_id": HF_REPO_ID,
        "hf_snapshot_revision": snapshot,
        "hf_subfolder": pin.hf_subfolder,
        "sae_config_sha256": sha256_file(config_path),
        "sae_id": pin.sae_id,
        "sae_params_sha256": sha256_file(params_path),
        "sae_release": pin.sae_release,
    }
    if pin.d_out is not None:
        entry["d_out"] = pin.d_out
    return entry


def main() -> None:
    args = parse_args()
    output = args.output or args.invariants
    invariants = load_json(args.invariants)
    sae_releases = invariants.setdefault("sae_releases", {})
    entries = sae_releases.setdefault("entries", {})

    for pin in PINS:
        key = f"{pin.sae_release}/{pin.sae_id}"
        print(f"pinning {key}", flush=True)
        entries[key] = pin_entry(pin, args)

    sae_releases["status"] = "pinned_4b_l20_l22_residual_l22_mlp_skip_transcoder_artifacts"
    write_json(output, invariants)
    print(output)


if __name__ == "__main__":
    main()
