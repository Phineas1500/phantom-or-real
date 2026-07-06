#!/usr/bin/env python3
"""Item G0 (HF pathway): Qwen3.5-27B calibration — unhinted + hinted baselines
on the 16 claim-11 rows. Gate: unhinted within CI of 0.352; hint lift >= +0.30.
Pre-registered: docs/causal_handle_directions.md item G (+ 2026-07-06 amendment)."""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.stage2_qwen_patch_hf import load_hf_model, torch_dtype  # noqa: E402
from scripts.stage2_qwen_subspace_erasure import generate_one  # noqa: E402
from scripts.stage2_proposal_hints import make_user_prompt  # noqa: E402
from src.messages import render_chat_text  # noqa: E402
from src.stage2_steering import score_reply  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.env_loader import load_env  # noqa: E402
from scripts.stage2_subtype_discriminator import row_bootstrap_ci, json_default  # noqa: E402

G0_ROWS = [3154,3493,3927,4142,4719,4919,5671,5948,6182,7312,8249,9026,9187,9587,10780,10810]

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/qwen35_27b_infer_property.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--samples-per-row", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=20260706)
    p.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/qwen_g0_calibration.jsonl"))
    p.add_argument("--output", type=Path, default=Path("docs/qwen_g0_calibration.json"))
    args = p.parse_args()
    started = time.time()
    load_env(); ensure_on_path(); torch.set_grad_enabled(False)

    keep = set(G0_ROWS); rows = []
    with args.jsonl.open() as f:
        for i, line in enumerate(f):
            if i in keep:
                r = json.loads(line); r["row_index"] = i; rows.append(r)
    assert len(rows) == 16, len(rows)
    print(f"rows={len(rows)}", flush=True)

    ct = {"enable_thinking": False}
    model, tokenizer = load_hf_model(args.model, dtype=torch_dtype(args.dtype),
        device_map=args.device_map, device=None,
        attn_implementation=args.attn_implementation, trust_remote_code=args.trust_remote_code)

    out_rows = []
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as fout:
        for ridx, row in enumerate(rows, 1):
            for cidx, cond in enumerate(("unhinted_baseline", "hinted_baseline")):
                user = row["prompt_text"] if cond == "unhinted_baseline" else make_user_prompt(row, "hint_concept_first")
                text = render_chat_text(tokenizer, system=row["system_prompt"], user=user,
                    model_name=args.model, add_generation_prompt=True, chat_template_kwargs=ct)
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                for s in range(args.samples_per_row):
                    torch.manual_seed(args.seed + row["row_index"] * 10007 + cidx * 101 + s)
                    new_ids, reply = generate_one(model=model, token_ids=ids,
                        max_new_tokens=args.max_new_tokens, do_sample=True,
                        temperature=args.temperature, stop_at_eos=True)
                    score = score_reply(row, reply)
                    o = {"schema_version": 1, "source_row_index": row["row_index"],
                         "height": row.get("height"), "model": args.model, "condition": cond,
                         "sample_index": s, "method": "qwen_g0_calibration",
                         "original_is_correct_strong": bool(row.get("is_correct_strong")),
                         "generated_token_count": len(new_ids), "model_output": reply, **score}
                    out_rows.append(o); fout.write(json.dumps(o, ensure_ascii=False, default=json_default) + "\n")
                fout.flush()
            print(f"row {ridx}/16 done ({time.time()-started:.0f}s)", flush=True)

    per = {}
    for o in out_rows:
        per.setdefault(o["source_row_index"], {}).setdefault(o["condition"], []).append(o["is_correct_strong"])
    ub = np.array([np.mean(v["unhinted_baseline"]) for v in per.values()])
    hb = np.array([np.mean(v["hinted_baseline"]) for v in per.values()])
    lift = hb - ub
    lo, hi = row_bootstrap_ci(lift, np.random.default_rng(args.seed))
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": time.time() - started, "n_rows": len(per),
              "unhinted_p_strong": float(ub.mean()), "hinted_p_strong": float(hb.mean()),
              "hint_lift": float(lift.mean()), "hint_lift_ci95": [float(lo), float(hi)],
              "gate_reference_unhinted": 0.352, "gate_lift_threshold": 0.30,
              "seconds_per_generation": (time.time() - started) / max(len(out_rows), 1)}
    args.output.write_text(json.dumps(report, indent=2, default=json_default) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
