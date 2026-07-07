#!/usr/bin/env python3
"""Item J1 (HF pathway): J-aware F(i)-analog on Qwen 3.5.

Directional-response probe: does injecting epsilon x (the row's own staged
L43 concept-position states) produce downstream responses that separate
natural correct from incorrect, where linear probes on the raw states read
chance (G6: 0.504)? No generation; deterministic forwards only.
Pre-registered: docs/causal_handle_directions.md item J1."""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.stage2_qwen_patch_hf import (  # noqa: E402
    load_hf_model, torch_dtype, hidden_from_output, replace_hidden_in_output, validate_hf_layers)
from scripts.stage2_hint_delta import concept_positions  # noqa: E402
from scripts.stage2_natural_state_capture import select_balanced_rows  # noqa: E402
from scripts.stage2_rank_k_guard_v2 import select_fresh_rows  # noqa: E402
from scripts.stage2_qwen_g0_hf import G0_ROWS  # noqa: E402
from scripts.stage2_qwen_g6_hf import cv_auc  # noqa: E402
from scripts.stage2_subtype_discriminator import json_default  # noqa: E402
from src.messages import render_chat_text  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.env_loader import load_env  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/qwen35_27b_infer_property.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--layer", type=int, default=43)
    p.add_argument("--control-layer", type=int, default=53)
    p.add_argument("--per-height", type=int, default=8)
    p.add_argument("--extra-per-height", type=int, default=5)
    p.add_argument("--source-per-cell", type=int, default=24)
    p.add_argument("--selection-seed", type=int, default=20260706)
    p.add_argument("--extra-seed", type=int, default=20260708)
    p.add_argument("--source-seed", type=int, default=20260708)
    p.add_argument("--tangent-seed", type=int, default=20260709)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--null-draws", type=int, default=10)
    p.add_argument("--label-shuffles", type=int, default=200)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--output", type=Path, default=Path("docs/qwen_j1_probe.json"))
    args = p.parse_args()
    started = time.time()
    load_env(); ensure_on_path(); torch.set_grad_enabled(False)

    core_rows = select_fresh_rows(args.jsonl, exclude=set(G0_ROWS), heights=[3, 4],
                                  per_height=args.per_height, seed=args.selection_seed)
    core_indices = {r["row_index"] for r in core_rows}
    extra_rows = select_fresh_rows(args.jsonl, exclude=set(G0_ROWS) | core_indices, heights=[3, 4],
                                   per_height=args.extra_per_height, seed=args.extra_seed)
    test_indices = core_indices | {r["row_index"] for r in extra_rows}
    source_rows = select_balanced_rows(args.jsonl, exclude=set(G0_ROWS) | test_indices,
                                       heights=[3, 4], per_cell=args.source_per_cell,
                                       seed=args.source_seed)
    print(f"sources: {len(source_rows)}", flush=True)

    ct = {"enable_thinking": False}
    model, tokenizer = load_hf_model(args.model, dtype=torch_dtype(args.dtype),
        device_map="auto", device=None, attn_implementation="sdpa", trust_remote_code=True)
    n_layers = len(model.model.layers)
    pen_layer = n_layers - 2
    validate_hf_layers(model, [args.layer, args.control_layer, pen_layer])
    L = args.layer
    dev = next(model.parameters()).device
    print(f"n_layers={n_layers} pen_layer={pen_layer}", flush=True)

    def run_forward(ids, capture: dict[int, list[int]], inject=None):
        """capture: {layer: positions}; inject: (positions, matrix) added at L during prefill."""
        store: dict[tuple[int, int], np.ndarray] = {}
        handles = []
        for lyr, poss in capture.items():
            def fn(_m, _i, out, _lyr=lyr, _poss=poss):
                h = hidden_from_output(out)
                if h.shape[1] >= len(ids):
                    for pp in _poss:
                        store[(_lyr, pp)] = h[0, pp, :].detach().cpu().float().numpy()
            handles.append(model.model.layers[lyr].register_forward_hook(fn))
        if inject is not None:
            poss, mat = inject
            vec = torch.from_numpy(mat)
            def inj(_m, _i, out):
                h = hidden_from_output(out)
                if h.shape[1] > 1:
                    for j, pp in enumerate(poss):
                        if pp < h.shape[1]:
                            h[0, pp, :] += vec[j].to(h.device, h.dtype)
                    return replace_hidden_in_output(out, h)
                return out
            handles.append(model.model.layers[L].register_forward_hook(inj))
        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
        finally:
            for h in handles: h.remove()
        return store

    feats = {k: [] for k in ["raw_concept", "raw_L43_final", "raw_ctrl_final", "raw_pen_final",
                             "resp_real_pen", "resp_real_ctrl"]}
    resp_null_pen = [[] for _ in range(args.null_draws)]
    y_labels = []
    for row in source_rows:
        ri = row["row_index"]
        rt = render_chat_text(tokenizer, system=row["system_prompt"], user=row["prompt_text"],
                              model_name=args.model, add_generation_prompt=True, chat_template_kwargs=ct)
        ids = tokenizer(rt, add_special_tokens=False)["input_ids"]
        gold = row["ontology_fol_structured"]["hypothesis"]["subject"]
        pos = concept_positions(tokenizer, rt, gold, 0, len(ids))
        if not pos:
            continue
        last = len(ids) - 1
        base = run_forward(ids, {L: pos + [last], args.control_layer: [last], pen_layer: [last]})
        states = np.stack([base[(L, q)] for q in pos])
        norms = np.linalg.norm(states, axis=1, keepdims=True)
        base_pen = base[(pen_layer, last)]
        base_ctrl = base[(args.control_layer, last)]

        def response(tangent):
            pert = run_forward(ids, {args.control_layer: [last], pen_layer: [last]},
                               inject=(pos, (args.epsilon * tangent).astype(np.float32)))
            return ((pert[(pen_layer, last)] - base_pen) / args.epsilon,
                    (pert[(args.control_layer, last)] - base_ctrl) / args.epsilon)

        r_pen, r_ctrl = response(states)
        for d in range(args.null_draws):
            rng = np.random.default_rng(args.tangent_seed + d + 1 + ri * 100003)
            noise = rng.standard_normal(states.shape).astype(np.float32)
            noise *= norms / np.maximum(np.linalg.norm(noise, axis=1, keepdims=True), 1e-8)
            n_pen, _ = response(noise)
            resp_null_pen[d].append(n_pen)
        feats["raw_concept"].append(states.mean(axis=0))
        feats["raw_L43_final"].append(base[(L, last)])
        feats["raw_ctrl_final"].append(base_ctrl)
        feats["raw_pen_final"].append(base_pen)
        feats["resp_real_pen"].append(r_pen)
        feats["resp_real_ctrl"].append(r_ctrl)
        y_labels.append(bool(row["is_correct_strong"]))
        print(f"row {ri} done ({len(pos)} pos, {time.time()-started:.0f}s)", flush=True)

    y = np.array(y_labels, dtype=int)
    print(f"rows={len(y)} ({y.sum()} correct / {len(y)-y.sum()} incorrect)", flush=True)

    def auc_of(mat):
        X = np.stack(mat).astype(np.float64)
        return cv_auc(X - X.mean(axis=0, keepdims=True), y)

    report = {"n_rows": int(len(y)), "epsilon": args.epsilon,
              "pen_layer": pen_layer, "control_layer": args.control_layer}
    for k, v in feats.items():
        report[k] = auc_of(v)
        print(f"AUC {k}: {report[k]:.3f}", flush=True)
    null_aucs = [auc_of(resp_null_pen[d]) for d in range(args.null_draws)]
    report["resp_null_pen_aucs"] = null_aucs
    report["resp_null_pen_p95"] = float(np.percentile(null_aucs, 95))
    report["resp_null_pen_median"] = float(np.median(null_aucs))
    print(f"random-tangent null AUCs: {[round(a,3) for a in null_aucs]}", flush=True)

    rng = np.random.default_rng(args.tangent_seed)
    X = np.stack(feats["resp_real_pen"]).astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    shuf = [cv_auc(Xc, y[rng.permutation(len(y))]) for _ in range(args.label_shuffles)]
    report["resp_real_pen_labelshuffle_p95"] = float(np.percentile(shuf, 95))
    print(f"label-shuffle p95: {report['resp_real_pen_labelshuffle_p95']:.3f}", flush=True)

    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_seconds"] = time.time() - started
    args.output.write_text(json.dumps(report, indent=2, default=json_default) + "\n")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
