"""Item M0 — SJ/OC factorization capture (registered 2026-08-14).

One forward per frame row (original stage-1 prompt, no interventions);
captures the final-prompt-token residual state at the stage-1 5-layer stack
(L15/L30/L40/L45/L53). Frame: docs/sjoc_m0_frame.json (32/cell, seed
20260816, soft-census SJ labels, ties excluded). The factorial battery
(gates, T1 identity test, differential erasure) runs offline on the npz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.activations import load_tl_model, render_chat_text  # noqa: E402

LAYERS = [15, 30, 40, 45, 53]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path,
                    default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    ap.add_argument("--frame", type=Path, default=Path("docs/sjoc_m0_frame.json"))
    ap.add_argument("--model", default="google/gemma-3-27b-it")
    ap.add_argument("--n-devices", type=int, default=2)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--load-mode", default="no-processing")
    ap.add_argument("--out", type=Path,
                    default=Path("results/stage2/erasure/sjoc_m0_capture_27b_property.npz"))
    args = ap.parse_args()

    frame = json.loads(args.frame.read_text())
    wanted: dict[int, tuple[int, str]] = {}
    for key, rows in frame["cells"].items():
        oc = int(key[2])
        sj = key.split("_sj")[1]
        for r in rows:
            wanted[int(r)] = (oc, sj)
    print(f"frame rows: {len(wanted)}", flush=True)

    stage1 = {}
    with open(args.jsonl) as f:
        for i, line in enumerate(f):
            if i in wanted:
                stage1[i] = json.loads(line)
    assert len(stage1) == len(wanted)

    model = load_tl_model(model_name=args.model, n_devices=args.n_devices,
                          n_ctx=args.n_ctx, dtype=args.dtype, load_mode=args.load_mode)
    tokenizer = model.tokenizer
    hooks = [f"blocks.{L}.hook_resid_post" for L in LAYERS]

    states = {L: [] for L in LAYERS}
    manifest = []
    t0 = time.time()
    for n, (row_id, (oc, sj)) in enumerate(sorted(wanted.items())):
        row = stage1[row_id]
        text = render_chat_text(tokenizer, system=row["system_prompt"],
                                user=row["prompt_text"], model_name=args.model,
                                add_generation_prompt=True)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        with torch.inference_mode():
            _, cache = model.run_with_cache(
                torch.tensor([ids]), names_filter=lambda name: name in hooks)
        for L in LAYERS:
            states[L].append(
                cache[f"blocks.{L}.hook_resid_post"][0, -1].float().cpu().numpy())
        manifest.append({"source_row_index": row_id, "oc": oc, "sj": sj,
                         "height": row.get("height"), "n_tokens": len(ids)})
        if n % 16 == 0:
            print(f"{n + 1}/{len(wanted)} rows, {time.time() - t0:.0f}s", flush=True)

    arrays = {f"L{L}_final": np.stack(states[L]).astype(np.float32) for L in LAYERS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)
    Path(str(args.out).replace(".npz", ".manifest.json")).write_text(
        json.dumps({"rows": manifest, "layers": LAYERS, "seed": frame["seed"],
                    "position": "final_prompt_token"}, indent=1))
    print(f"saved {args.out} shapes={{L: a.shape for L, a in arrays.items()}}"
          f" in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
