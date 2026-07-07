#!/usr/bin/env python3
"""Item J2 (HF pathway): position-separated staging — fresh-row confirmation
+ staging-site steering.

Part A: fresh 96-row capture; A-PRIMARY = fresh raw_L43_final AUC >= 0.70 and
above label-shuffle p95. Part B: final-token class-mean steering on the 24 G6
test rows; registered prediction = does NOT repair (gauge, not lever, at the
staging site). Pre-registered: docs/causal_handle_directions.md item J2."""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.stage2_qwen_patch_hf import (  # noqa: E402
    load_hf_model, torch_dtype, hidden_from_output, replace_hidden_in_output, validate_hf_layers)
from scripts.stage2_qwen_subspace_erasure import generate_one  # noqa: E402
from scripts.stage2_hint_delta import concept_positions  # noqa: E402
from scripts.stage2_natural_state_capture import select_balanced_rows  # noqa: E402
from scripts.stage2_rank_k_guard_v2 import Arm, select_fresh_rows  # noqa: E402
from scripts.stage2_qwen_g0_hf import G0_ROWS  # noqa: E402
from scripts.stage2_qwen_g6_hf import cv_auc  # noqa: E402
from scripts.stage2_subtype_discriminator import json_default  # noqa: E402
from src.messages import render_chat_text  # noqa: E402
from src.stage2_steering import score_reply  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.env_loader import load_env  # noqa: E402


def build_j2_arms(layer: int) -> list[tuple[Arm, int, float]]:
    """(arm, seed_ai, norm_scale); staging-site steering at the final token."""
    return [
        (Arm("unhinted_baseline", "none", "none"), 0, 0.0),
        (Arm(f"finaltok_classmean_050_L{layer}", "finaltok_classmean", "unhinted_baseline", (layer,), None, "final_token_class_mean"), 50, 0.5),
        (Arm(f"finaltok_classmean_100_L{layer}", "finaltok_classmean", "unhinted_baseline", (layer,), None, "final_token_class_mean"), 51, 1.0),
        (Arm(f"finaltok_shuffled_100_L{layer}", "finaltok_shuffled", "unhinted_baseline", (layer,), None, "final_token_shuffled_class_mean"), 52, 1.0),
    ]


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
    p.add_argument("--g6-source-seed", type=int, default=20260708)
    p.add_argument("--fresh-seed", type=int, default=20260710)
    p.add_argument("--rand-pos-count", type=int, default=8)
    p.add_argument("--rand-pos-draws", type=int, default=2)
    p.add_argument("--label-shuffles", type=int, default=200)
    p.add_argument("--samples-per-row", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/qwen_j2_steering.jsonl"))
    p.add_argument("--output", type=Path, default=Path("docs/qwen_j2_probe.json"))
    args = p.parse_args()
    started = time.time()
    load_env(); ensure_on_path(); torch.set_grad_enabled(False)

    core_rows = select_fresh_rows(args.jsonl, exclude=set(G0_ROWS), heights=[3, 4],
                                  per_height=args.per_height, seed=args.selection_seed)
    core_indices = {r["row_index"] for r in core_rows}
    extra_rows = select_fresh_rows(args.jsonl, exclude=set(G0_ROWS) | core_indices, heights=[3, 4],
                                   per_height=args.extra_per_height, seed=args.extra_seed)
    test_rows = core_rows + extra_rows
    test_indices = {r["row_index"] for r in test_rows}
    g6_sources = select_balanced_rows(args.jsonl, exclude=set(G0_ROWS) | test_indices,
                                      heights=[3, 4], per_cell=args.source_per_cell,
                                      seed=args.g6_source_seed)
    g6_source_indices = {r["row_index"] for r in g6_sources}
    fresh_sources = select_balanced_rows(args.jsonl,
                                         exclude=set(G0_ROWS) | test_indices | g6_source_indices,
                                         heights=[3, 4], per_cell=args.source_per_cell,
                                         seed=args.fresh_seed)
    assert not ({r["row_index"] for r in fresh_sources} & (g6_source_indices | test_indices))
    print(f"fresh sources: {len(fresh_sources)} (disjoint from {len(g6_source_indices)} J1/G6 sources + {len(test_indices)} test rows)", flush=True)

    ct = {"enable_thinking": False}
    model, tokenizer = load_hf_model(args.model, dtype=torch_dtype(args.dtype),
        device_map="auto", device=None, attn_implementation="sdpa", trust_remote_code=True)
    validate_hf_layers(model, [args.layer, args.control_layer])
    L = args.layer
    dev = next(model.parameters()).device

    def render_ids(row):
        rt = render_chat_text(tokenizer, system=row["system_prompt"], user=row["prompt_text"],
                              model_name=args.model, add_generation_prompt=True, chat_template_kwargs=ct)
        return rt, tokenizer(rt, add_special_tokens=False)["input_ids"]

    def capture(ids, capture_map):
        store = {}
        handles = []
        for lyr, poss in capture_map.items():
            def fn(_m, _i, out, _lyr=lyr, _poss=poss):
                h = hidden_from_output(out)
                if h.shape[1] >= len(ids):
                    for pp in _poss:
                        store[(_lyr, pp)] = h[0, pp, :].detach().cpu().float().numpy()
            handles.append(model.model.layers[lyr].register_forward_hook(fn))
        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
        finally:
            for h in handles: h.remove()
        return store

    feats = {k: [] for k in ["concept_L43", "final_L43", "final_L53", "concept_L53"]}
    for d in range(1, args.rand_pos_draws + 1):
        feats[f"randpos_L43_d{d}"] = []
    finaltok_states, y_labels = [], []
    for row in fresh_sources:
        ri = row["row_index"]
        rt, ids = render_ids(row)
        gold = row["ontology_fol_structured"]["hypothesis"]["subject"]
        pos = concept_positions(tokenizer, rt, gold, 0, len(ids))
        if not pos:
            continue
        last = len(ids) - 1
        rand_pos = {}
        candidates = [q for q in range(4, last) if q not in set(pos)]
        for d in range(1, args.rand_pos_draws + 1):
            rng = np.random.default_rng(args.fresh_seed + d * 7919 + ri)
            rand_pos[d] = list(rng.choice(candidates, size=min(args.rand_pos_count, len(candidates)), replace=False))
        all_l43 = sorted(set(pos) | {last} | set().union(*rand_pos.values()))
        st = capture(ids, {L: all_l43, args.control_layer: sorted(set(pos) | {last})})
        feats["concept_L43"].append(np.stack([st[(L, q)] for q in pos]).mean(axis=0))
        feats["final_L43"].append(st[(L, last)])
        feats["final_L53"].append(st[(args.control_layer, last)])
        feats["concept_L53"].append(np.stack([st[(args.control_layer, q)] for q in pos]).mean(axis=0))
        for d in range(1, args.rand_pos_draws + 1):
            feats[f"randpos_L43_d{d}"].append(np.stack([st[(L, q)] for q in rand_pos[d]]).mean(axis=0))
        finaltok_states.append(st[(L, last)])
        y_labels.append(bool(row["is_correct_strong"]))
        print(f"captured {ri} ({len(pos)} pos)", flush=True)

    y = np.array(y_labels, dtype=int)
    print(f"fresh rows captured: {len(y)} ({y.sum()} correct / {len(y)-y.sum()} incorrect)", flush=True)

    def auc_of(mat):
        X = np.stack(mat).astype(np.float64)
        return cv_auc(X - X.mean(axis=0, keepdims=True), y)

    report = {"n_fresh_rows": int(len(y))}
    for k, v in feats.items():
        report[f"auc_{k}"] = auc_of(v)
        print(f"AUC {k}: {report[f'auc_{k}']:.3f}", flush=True)
    rng = np.random.default_rng(args.fresh_seed)
    X = np.stack(feats["final_L43"]).astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    shuf = [cv_auc(Xc, y[rng.permutation(len(y))]) for _ in range(args.label_shuffles)]
    report["final_L43_labelshuffle_p95"] = float(np.percentile(shuf, 95))
    print(f"final_L43 label-shuffle p95: {report['final_L43_labelshuffle_p95']:.3f}", flush=True)

    ft = np.stack(finaltok_states).astype(np.float64)
    class_mean = ft[y == 1].mean(axis=0) - ft[y == 0].mean(axis=0)
    rng_s = np.random.default_rng(args.fresh_seed + 1)
    y_shuf = y[rng_s.permutation(len(y))]
    shuffled_mean = ft[y_shuf == 1].mean(axis=0) - ft[y_shuf == 0].mean(axis=0)
    report["class_mean_norm"] = float(np.linalg.norm(class_mean))
    print(f"final-token class-mean norm: {report['class_mean_norm']:.3f}", flush=True)

    prepared = []
    for row in test_rows:
        rt, ids = render_ids(row)
        prepared.append({"row": row, "ri": row["row_index"], "ids": ids,
                         "core": row["row_index"] in core_indices})
    print(f"test rows prepared: {len(prepared)}", flush=True)

    def add_hook(vec_np, position):
        vec = torch.from_numpy(vec_np.astype(np.float32))
        def fn(_m, _i, out):
            h = hidden_from_output(out)
            if h.shape[1] > 1 and position < h.shape[1]:
                h[0, position, :] += vec.to(h.device, h.dtype)
                return replace_hidden_in_output(out, h)
            return out
        return fn

    arms = build_j2_arms(L)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as fout:
        for arm, seed_ai, scale in arms:
            for prep in prepared:
                handles = []
                if arm.kind != "none":
                    last = len(prep["ids"]) - 1
                    st = capture(prep["ids"], {L: [last]})
                    state_norm = float(np.linalg.norm(st[(L, last)]))
                    src = shuffled_mean if arm.kind == "finaltok_shuffled" else class_mean
                    vec = src / max(np.linalg.norm(src), 1e-8) * (scale * state_norm)
                    handles = [model.model.layers[L].register_forward_hook(add_hook(vec, last))]
                try:
                    for s in range(args.samples_per_row):
                        torch.manual_seed(args.selection_seed + prep["ri"] * 10007 + seed_ai * 101 + s)
                        new_ids, reply = generate_one(model=model, token_ids=prep["ids"],
                            max_new_tokens=args.max_new_tokens, do_sample=True,
                            temperature=args.temperature, stop_at_eos=True)
                        sc = score_reply(prep["row"], reply)
                        o = {"schema_version": 1, "source_row_index": prep["ri"],
                             "height": prep["row"].get("height"), "model": args.model,
                             "condition": arm.label, "sample_index": s,
                             "method": "qwen_j2_staging_site_steering", "patch_layer": L,
                             "seed_arm_index": seed_ai, "core_row": prep["core"],
                             "generated_token_count": len(new_ids), "model_output": reply, **sc}
                        fout.write(json.dumps(o, ensure_ascii=False, default=json_default) + "\n")
                finally:
                    for h in handles: h.remove()
                fout.flush()
            print(f"ARM DONE {arm.label} ({time.time()-started:.0f}s)", flush=True)

    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_seconds"] = time.time() - started
    args.output.write_text(json.dumps(report, indent=2, default=json_default) + "\n")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
