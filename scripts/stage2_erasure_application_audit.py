#!/usr/bin/env python3
"""Erasure application audit (verification addendum, registered 2026-08-18).

Answers the external review's concern 1 directly: prove the readable-stack
erasure was APPLIED, not silently a no-op. On 4 item-D rows, run (a) a clean
forward, recording per-direction projections onto the registered rank-9 INLP
stacks at all five layers; (b) an erased forward under the identical clamp
hooks used by the registered runs, with an audit hook appended at the same
hook points recording the post-hook stream. Report, per layer: the baseline
deviation of projections from the clamp targets (what the clamp has to move)
and the post-hook deviation (should be ~bf16 quantization noise). A no-op
would leave the two identical."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.stage2_subspace_erasure import build_layer_stacks, make_subspace_erasure_hook  # noqa: E402
from src.activations import load_tl_model, render_chat_text  # noqa: E402
from src.env_loader import load_env  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402

LAYERS = [15, 30, 40, 45, 53]


def main() -> int:
    load_env(); ensure_on_path(); torch.set_grad_enabled(False)
    t0 = time.time()
    stacks = build_layer_stacks(
        layers=LAYERS,
        inlp_npz=Path("results/stage2/erasure/inlp_direction_stacks_27b_property_5layer.npz"),
        stack_rank=9,
        activation_dir=Path("results/stage2/activations"),
        model_key="gemma3_27b",
        task="infer_property",
        splits_path=Path("results/stage2/splits.jsonl"),
        source_file="gemma3_27b_infer_property.jsonl",
        split_family="s1",
        random_draws=0,
        random_seed=0,
    )

    rows_wanted = []
    with open("results/stage2/erasure/readable_stack_erasure_27b_property_k8_shard0of2.jsonl") as f:
        for line in f:
            ri = json.loads(line)["source_row_index"]
            if ri not in rows_wanted:
                rows_wanted.append(ri)
            if len(rows_wanted) >= 4:
                break
    stage1 = {}
    with open("results/full/with_errortype/gemma3_27b_infer_property.jsonl") as f:
        for i, line in enumerate(f):
            if i in set(rows_wanted):
                stage1[i] = json.loads(line)

    model = load_tl_model(model_name="google/gemma-3-27b-it", n_devices=2,
                          n_ctx=4096, dtype="bfloat16", load_mode="no-processing")
    tokenizer = model.tokenizer
    hooks_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]

    report = {"rows": rows_wanted, "layers": {}}
    per_layer = {L: {"base_dev": [], "erased_dev": [], "moved": []} for L in LAYERS}
    for ri in rows_wanted:
        row = stage1[ri]
        text = render_chat_text(tokenizer, system=row["system_prompt"],
                                user=row["prompt_text"], model_name="google/gemma-3-27b-it",
                                add_generation_prompt=True)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        toks = torch.tensor([ids])

        with torch.inference_mode():
            _, cache = model.run_with_cache(toks, names_filter=lambda n: n in hooks_names)
        base_proj = {}
        for L in LAYERS:
            entry = stacks[L]["readable_stack"]
            act = cache[f"blocks.{L}.hook_resid_post"][0].float().cpu().numpy()
            proj = act @ entry["basis"]
            base_proj[L] = proj
            per_layer[L]["base_dev"].append(float(np.abs(proj - entry["means"]).mean()))

        fwd_hooks = []
        audit_store = {}
        for L in LAYERS:
            entry = stacks[L]["readable_stack"]
            fn, _state = make_subspace_erasure_hook(
                basis=entry["basis"], projection_means=entry["means"],
                projection_stds=entry["stds"])
            name = f"blocks.{L}.hook_resid_post"
            def make_audit(Lc):
                def audit(act, hook):
                    audit_store[Lc] = act[0].detach().float().cpu().numpy()
                    return act
                return audit
            fwd_hooks.append((name, fn))
            fwd_hooks.append((name, make_audit(L)))
        with torch.inference_mode(), model.hooks(fwd_hooks=fwd_hooks):
            model(toks)
        for L in LAYERS:
            entry = stacks[L]["readable_stack"]
            proj = audit_store[L] @ entry["basis"]
            per_layer[L]["erased_dev"].append(float(np.abs(proj - entry["means"]).max()))
            per_layer[L]["moved"].append(float(np.abs(proj - base_proj[L]).mean()))
        print(f"row {ri} audited ({time.time()-t0:.0f}s)", flush=True)

    for L in LAYERS:
        b = float(np.mean(per_layer[L]["base_dev"]))
        e = float(np.max(per_layer[L]["erased_dev"]))
        report["layers"][f"L{L}"] = {
            "baseline_mean_abs_dev_from_target": b,
            "erased_max_abs_dev_from_target": e,
            "mean_abs_projection_moved": float(np.mean(per_layer[L]["moved"])),
            "suppression_ratio": b / max(e, 1e-9),
        }
        print(f"L{L}: base_dev={b:.4f} erased_max_dev={e:.6f} ratio={b/max(e,1e-9):.1f}", flush=True)
    Path("docs/erasure_application_audit.json").write_text(json.dumps(report, indent=1))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
