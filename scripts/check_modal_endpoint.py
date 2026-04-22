"""Quick health + first-prompt sanity check against a Modal (or any OpenAI-compatible) endpoint.

Reads OPENAI_BASE_URL / OPENAI_API_KEY from .env. Exits 0 if healthy and a
real response came back, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI  # noqa: E402

from src.config import SYSTEM_PROMPT  # noqa: E402
from src.env_loader import get_openai_inference_credentials, load_env  # noqa: E402
from src.inference import build_messages  # noqa: E402

load_env()


async def _run(model: str, base_url: str | None, api_key: str) -> int:
    print(f"base_url: {base_url}")
    print(f"model:    {model}")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    user = (
        "Q: Jerry is a dalpist. Dalpists are brown. We observe that: Jerry is brown. "
        "Please come up with hypothesis to explain observations."
    )
    messages = build_messages(SYSTEM_PROMPT, user, model)
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=128,
        )
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        await client.close()
        return 1

    text = r.choices[0].message.content
    print("-" * 40)
    print(text)
    print("-" * 40)
    await client.close()
    return 0 if text and text.strip() else 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="gemma3-27b",
        help="Served-model-name as registered by the endpoint. Modal: gemma3-27b",
    )
    args = p.parse_args()

    base_url, api_key = get_openai_inference_credentials()
    raise SystemExit(asyncio.run(_run(args.model, base_url, api_key)))


if __name__ == "__main__":
    main()
