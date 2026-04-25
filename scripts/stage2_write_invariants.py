#!/usr/bin/env python3
"""Write docs/stage2_invariants.json for Stage 2 artifact pinning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SYSTEM_PROMPT  # noqa: E402
from src.env_loader import load_env  # noqa: E402
from src.messages import build_messages  # noqa: E402


DEFAULT_MODELS = ("google/gemma-3-4b-it", "google/gemma-3-27b-it")


@dataclass(frozen=True)
class CachedSnapshot:
    repo_id: str
    repo_cache_dir: str | None
    refs_main: str | None
    selected_commit: str | None
    selected_snapshot_path: str | None
    available_snapshots: list[str]


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl_rows(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def normalize_hf_cache_root(path: Path | None) -> Path | None:
    if path is None:
        env_home = os.environ.get("HF_HOME")
        if env_home:
            return Path(env_home)
        return None
    return path.parent if path.name == "hub" else path


def hub_dir(cache_root: Path | None) -> Path | None:
    if cache_root is None:
        return None
    return cache_root if cache_root.name == "hub" else cache_root / "hub"


def repo_cache_dir(repo_id: str, cache_root: Path | None) -> Path | None:
    root = hub_dir(cache_root)
    if root is None:
        return None
    return root / f"models--{repo_id.replace('/', '--')}"


def discover_cached_snapshot(repo_id: str, cache_root: Path | None) -> CachedSnapshot:
    repo_dir = repo_cache_dir(repo_id, cache_root)
    if repo_dir is None or not repo_dir.exists():
        return CachedSnapshot(repo_id, str(repo_dir) if repo_dir is not None else None, None, None, None, [])

    snapshots_dir = repo_dir / "snapshots"
    snapshots = sorted(p.name for p in snapshots_dir.iterdir() if p.is_dir()) if snapshots_dir.exists() else []
    refs_main_path = repo_dir / "refs" / "main"
    refs_main = refs_main_path.read_text().strip() if refs_main_path.exists() else None
    selected = refs_main if refs_main in snapshots else None
    if selected is None and snapshots:
        selected = max((snapshots_dir / name for name in snapshots), key=lambda p: p.stat().st_mtime).name
    selected_path = str(snapshots_dir / selected) if selected else None
    return CachedSnapshot(
        repo_id=repo_id,
        repo_cache_dir=str(repo_dir),
        refs_main=refs_main,
        selected_commit=selected,
        selected_snapshot_path=selected_path,
        available_snapshots=snapshots,
    )


def hf_cache_dir_for_transformers(cache_root: Path | None) -> str | None:
    root = hub_dir(cache_root)
    if root is None:
        return None
    return str(root)


def _commit_from_obj(obj: Any) -> str | None:
    value = getattr(obj, "_commit_hash", None)
    if value:
        return str(value)
    init_kwargs = getattr(obj, "init_kwargs", None)
    if isinstance(init_kwargs, dict) and init_kwargs.get("_commit_hash"):
        return str(init_kwargs["_commit_hash"])
    return None


def inspect_model(
    model_name: str,
    *,
    cache_root: Path | None,
    local_files_only: bool,
    probe_system: str,
    probe_user: str,
    strict: bool,
) -> dict[str, Any]:
    cached = discover_cached_snapshot(model_name, cache_root)
    result: dict[str, Any] = {
        "hf_repo_id": model_name,
        "cache": asdict(cached),
        "local_files_only": local_files_only,
        "status": "unavailable",
    }

    try:
        from transformers import AutoConfig, AutoTokenizer

        kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        cache_dir = hf_cache_dir_for_transformers(cache_root)
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        if cached.selected_commit:
            kwargs["revision"] = cached.selected_commit

        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        config = AutoConfig.from_pretrained(model_name, **kwargs)

        messages = build_messages(probe_system, probe_user, model_name)
        rendered_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer(rendered_chat, add_special_tokens=False)["input_ids"]
        tokenizer_commit = _commit_from_obj(tokenizer) or cached.selected_commit
        config_commit = _commit_from_obj(config) or cached.selected_commit
        chat_template = tokenizer.chat_template or ""

        result.update(
            {
                "status": "ok",
                "model_revision_hash": config_commit,
                "tokenizer_revision_hash": tokenizer_commit,
                "tokenizer_class": tokenizer.__class__.__name__,
                "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
                "config_class": config.__class__.__name__,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "chat_template_sha256": sha256_text(chat_template),
                "rendered_chat_sha256": sha256_text(rendered_chat),
                "rendered_chat_char_count": len(rendered_chat),
                "token_ids_sha256": sha256_text(json.dumps(token_ids, separators=(",", ":"))),
                "token_count": len(token_ids),
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        if strict:
            raise

    return result


def build_invariants(
    *,
    jsonl_dir: Path,
    models: list[str],
    output: Path,
    hf_cache: Path | None,
    local_files_only: bool,
    judge_snapshot: str | None,
    strict: bool = False,
) -> dict[str, Any]:
    cache_root = normalize_hf_cache_root(hf_cache)
    probe_user = (
        "Q: Alex is a dax. We observe that: Alex is not noisy. "
        "Please come up with hypothesis to explain observations."
    )
    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
    if strict and not jsonl_files:
        raise FileNotFoundError(f"no JSONL files found in {jsonl_dir}")

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": display_path(Path(__file__)),
        "output_path": display_path(output),
        "python_packages": {
            "transformers": package_version("transformers"),
            "transformer-lens": package_version("transformer-lens"),
            "sae-lens": package_version("sae-lens"),
            "torch": package_version("torch"),
        },
        "hf_cache_root": str(cache_root) if cache_root is not None else None,
        "stage1_jsonls": {
            display_path(path): {
                "sha256": sha256_file(path),
                "rows": count_jsonl_rows(path),
                "bytes": path.stat().st_size,
            }
            for path in jsonl_files
        },
        "chat_template_probe": {
            "system": SYSTEM_PROMPT,
            "user": probe_user,
            "add_generation_prompt": True,
            "message_builder": "src.messages.build_messages",
        },
        "models": {
            model_name: inspect_model(
                model_name,
                cache_root=cache_root,
                local_files_only=local_files_only,
                probe_system=SYSTEM_PROMPT,
                probe_user=probe_user,
                strict=strict,
            )
            for model_name in models
        },
        "sae_releases": {
            "status": "pin_after_layer_selection",
            "entries": {},
        },
        "gpt_judge_snapshot": judge_snapshot,
    }


def main() -> None:
    load_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/stage2_invariants.json"))
    parser.add_argument("--jsonl-dir", type=Path, default=Path("results/full/with_errortype"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--hf-cache", type=Path, default=None, help="HF_HOME cache root, not the hub subdirectory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local-files-only", dest="local_files_only", action="store_true", default=True)
    group.add_argument("--allow-downloads", dest="local_files_only", action="store_false")
    parser.add_argument("--judge-snapshot", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    invariants = build_invariants(
        jsonl_dir=args.jsonl_dir,
        models=args.models,
        output=args.output,
        hf_cache=args.hf_cache,
        local_files_only=args.local_files_only,
        judge_snapshot=args.judge_snapshot,
        strict=args.strict,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(invariants, f, indent=2, sort_keys=True)
        f.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
