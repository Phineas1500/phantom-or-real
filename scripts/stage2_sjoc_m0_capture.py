#!/usr/bin/env python3
"""Item M0 capture (docs/causal_handle_directions.md, item M): one forward per
frame row on the original stage-1 prompt, no interventions; final-prompt-token
states at the stage-1 5-layer stack. Capture only — every gate and test in the
M0 battery is offline (scripts/stage2_sjoc_m0_battery.py), so nothing is
unblinded in-job."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_hint_delta import prompt_cache  # noqa: E402
from src.activations import load_tl_model, render_chat_text, validate_hooks  # noqa: E402
from src.env_loader import load_env  # noqa: E402


def torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--frame", type=Path, default=Path("docs/sjoc_frame_128.json"))
    parser.add_argument("--layers", type=int, nargs="+", default=[15, 30, 40, 45, 53])
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--out", type=Path, default=Path("results/stage2/sjoc_m0_states.npz"))
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", default="no-processing")
    args = parser.parse_args()

    frame = json.loads(args.frame.read_text())
    rows = frame["rows"]
    needed = {r["row_index"] for r in rows}
    stage1 = {}
    with args.jsonl.open() as f:
        for i, line in enumerate(f):
            if i in needed:
                stage1[i] = json.loads(line)
    assert len(stage1) == len(needed), (len(stage1), len(needed))
    print(f"frame rows: {len(rows)} (seed {frame['seed']})", flush=True)

    model = load_tl_model(args.model, n_devices=args.n_devices, n_ctx=args.n_ctx,
                          dtype=torch_dtype(args.dtype), load_mode=args.load_mode)
    hooks = validate_hooks(model, args.layers)
    tokenizer = model.tokenizer
    print(f"hooks: {hooks}", flush=True)

    t0 = time.time()
    states = np.zeros((len(rows), len(args.layers), model.cfg.d_model), dtype=np.float32)
    lens = []
    for k, r in enumerate(rows):
        srow = stage1[r["row_index"]]
        text = render_chat_text(tokenizer, system=srow["system_prompt"], user=srow["prompt_text"],
                                model_name=args.model, add_generation_prompt=True)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        cache = prompt_cache(model, ids, hooks)
        for li, h in enumerate(hooks):
            states[k, li] = cache[h][len(ids) - 1].numpy().astype(np.float32)
        lens.append(len(ids))
        if k % 16 == 0:
            print(f"{k + 1}/{len(rows)} rows, {time.time()-t0:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        states=states,
        row_index=np.array([r["row_index"] for r in rows], dtype=np.int64),
        oc=np.array([r["oc"] for r in rows], dtype=bool),
        sj_yes=np.array([r["sj_yes"] for r in rows], dtype=bool),
        layers=np.array(args.layers, dtype=np.int64),
        prompt_token_lens=np.array(lens, dtype=np.int64),
    )
    meta = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "model": args.model,
            "n_rows": len(rows), "layers": args.layers, "frame_seed": frame["seed"],
            "elapsed_seconds": round(time.time() - t0)}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
