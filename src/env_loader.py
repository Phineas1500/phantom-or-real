"""Auto-load .env from the project root on import.

The project uses two OpenAI-compatible API keys side-by-side:
  OPENAI_API_KEY       — primary inference endpoint (Modal / OpenRouter / ...)
  OPENAI_API_KEY_GPT   — OpenAI proper, used only for Phase 5 error classification
We also load HF_TOKEN and OPENAI_BASE_URL from .env if present.
"""

from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_env(path: Path | None = None) -> None:
    """Load KEY=VAL lines from .env without overwriting already-set env vars."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover — dotenv is in environment.yml
        load_dotenv = None

    env_path = Path(path) if path is not None else _project_root() / ".env"
    if not env_path.exists():
        return
    if load_dotenv is not None:
        load_dotenv(env_path, override=False)
        return

    # Fallback parser — simple KEY=VAL, handles # comments and surrounding quotes.
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        os.environ.setdefault(k, v)


def get_openai_inference_credentials() -> tuple[str | None, str]:
    """Return (base_url, api_key) for the primary inference endpoint."""
    return os.environ.get("OPENAI_BASE_URL"), os.environ.get("OPENAI_API_KEY", "not-needed")


def get_openai_gpt_credentials() -> tuple[str, str | None]:
    """Return (base_url, api_key) for the Phase 5 GPT-as-judge path.

    Uses OPENAI_API_KEY_GPT if set, otherwise falls back to OPENAI_API_KEY.
    Base URL explicitly points at OpenAI unless OPENAI_BASE_URL_GPT overrides
    — we can't pass None because the openai SDK then falls back to the
    OPENAI_BASE_URL env var, which for this project points at Modal.
    """
    key = os.environ.get("OPENAI_API_KEY_GPT") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL_GPT") or "https://api.openai.com/v1"
    return base_url, key
