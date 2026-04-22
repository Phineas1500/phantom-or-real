"""End-to-end pipeline smoke test without network access.

Mocks the chat completions endpoint to always return the ground truth, then
exercises the inference driver on pilot data. Verifies:
  - JSONL rows contain every required field
  - Structural annotations attach
  - Scoring aggregates correctly
  - Summary JSON structure is as expected
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import load_examples, run_inference  # noqa: E402
from src.export import write_jsonl, read_jsonl  # noqa: E402
from src.analysis import build_summary  # noqa: E402


class _FakeChoice:
    def __init__(self, text: str):
        class _Msg:
            pass

        self.message = _Msg()
        self.message.content = text


class _FakeCompletion:
    def __init__(self, text: str):
        self.choices = [_FakeChoice(text)]


async def _fake_create(self, *, model, messages, temperature, max_tokens):  # noqa: ARG001
    # The user message contains theories + observations; we cheat by pulling
    # the ground truth from a closure attribute set below.
    gt = _fake_create.current_gt  # type: ignore[attr-defined]
    return _FakeCompletion(gt)


def run_one(task: str, height: int, limit: int = 5) -> list[dict]:
    pkl = ROOT / "data" / "pilot" / f"examples_{task}_h{height}.pkl"
    examples, task_type, h = load_examples(pkl)
    examples = examples[:limit]

    # Prime ground truths one at a time by iterating.
    rows = []
    for ex in examples:
        _fake_create.current_gt = ex.hypotheses  # type: ignore[attr-defined]
        with patch("openai.resources.chat.completions.AsyncCompletions.create", new=_fake_create):
            r = asyncio.run(
                run_inference(
                    [ex],
                    task_type=task,
                    height=h,
                    model_name="fake-model",
                    base_url=None,
                    api_key="dummy",
                    concurrency=1,
                    max_tokens=256,
                    temperature=0,
                    max_attempts=1,
                    example_id_prefix=f"{task}_h{h}",
                )
            )
            rows.extend(r)
    return rows


def test_pipeline() -> None:
    rows = run_one("property", 2, limit=5)
    assert len(rows) == 5
    required = {
        "example_id",
        "task",
        "height",
        "model",
        "prompt_text",
        "system_prompt",
        "ground_truth",
        "model_output",
        "is_correct_strong",
        "is_correct_weak",
        "quality_score",
        "parse_failed",
        "failure_mode",
        "error_type",
        "structural",
        "ontology_raw",
        "ontology_fol_string",
        "ontology_fol_structured",
    }
    for r in rows:
        missing = required - set(r.keys())
        assert not missing, f"missing fields: {missing}"
        assert r["is_correct_strong"] is True, "ground truth as reply should always score correct"
        assert "target_concept" in r["structural"]
        assert "membership" in r["ontology_fol_structured"]
        assert "hypothesis" in r["ontology_fol_structured"]

    # Exercise write + summary
    out = ROOT / "results" / "smoke_test.jsonl"
    write_jsonl(rows, out)
    loaded = read_jsonl(out)
    assert loaded == rows

    summary = build_summary(out)
    assert summary["by_height"]["h2"]["n"] == 5
    assert summary["by_height"]["h2"]["strong_accuracy"] == 1.0

    out.unlink()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    test_pipeline()
    print("OK")
