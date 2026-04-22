"""Exercise analysis + sanity_check scripts on a small synthetic JSONL."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import load_examples, run_inference  # noqa: E402
from src.export import write_jsonl  # noqa: E402


class _FakeChoice:
    def __init__(self, text: str):
        class _M:
            pass

        self.message = _M()
        self.message.content = text


class _FakeCompletion:
    def __init__(self, text: str):
        self.choices = [_FakeChoice(text)]


async def _fake_create(self, *, model, messages, temperature, max_tokens):  # noqa: ARG001
    return _FakeCompletion(_fake_create.current_reply)  # type: ignore[attr-defined]


def build_rows(task: str, height: int) -> list[dict]:
    pkl = ROOT / "data" / "pilot" / f"examples_{task}_h{height}.pkl"
    examples, _, _ = load_examples(pkl)
    examples = examples[:10]

    rows = []
    for idx, ex in enumerate(examples):
        # alternate correct / gibberish so we exercise both code paths
        reply = ex.hypotheses if idx % 2 == 0 else "I cannot help with this."
        _fake_create.current_reply = reply  # type: ignore[attr-defined]
        with patch("openai.resources.chat.completions.AsyncCompletions.create", new=_fake_create):
            r = asyncio.run(
                run_inference(
                    [ex],
                    task_type=task,
                    height=height,
                    model_name="fake-model",
                    base_url=None,
                    api_key="dummy",
                    concurrency=1,
                    max_tokens=256,
                    temperature=0,
                    max_attempts=1,
                    example_id_prefix=f"{task}_h{height}",
                )
            )
            rows.extend(r)
    return rows


def main() -> None:
    rows = []
    for h in (1, 2, 3, 4):
        rows.extend(build_rows("property", h))
    out = ROOT / "results" / "smoke_analysis.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, out)

    # sanity_check exits 0 even with warnings; we just check it runs
    r = subprocess.run(
        [sys.executable, "scripts/sanity_check.py", "--jsonl", str(out)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert str(out) in data
    per_h = data[str(out)]["per_height"]
    assert set(per_h.keys()) == {"1", "2", "3", "4"}

    # make_plots
    figures_dir = ROOT / "docs" / "figures" / "smoke"
    r = subprocess.run(
        [
            sys.executable,
            "scripts/make_plots.py",
            "--results-dir",
            str(out.parent),
            "--output-dir",
            str(figures_dir),
            "--jsonl",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (figures_dir / "accuracy_vs_depth.png").exists()
    assert (figures_dir / "structural_slicing.png").exists()
    print("OK")

    out.unlink()
    for p in figures_dir.iterdir():
        p.unlink()
    figures_dir.rmdir()


if __name__ == "__main__":
    main()
