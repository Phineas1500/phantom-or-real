#!/usr/bin/env python3
"""KV/hint-span job: does the hint act through decode-time attention or writes?

Nine arms in kill-safe order on the 14-row prefix rowset: unhinted and hinted
in-job baselines; decode-only attention masking of the hint-span key positions;
the exhaustive-necessity combination (masking x concept-position reversion);
gold and wrong hint-span KV transplants into the unhinted cache (manual decode
loop, attention telemetry over the spliced slots); and the additive ladder
rungs (own per-position delta, restricted mean x2, rank-k L30 reconstruction
from the composite-job states). Spec: docs/causal_handle_directions.md
KV/HINT-SPAN JOB SPEC v2 + KV spec v2 additions.

Positional-convention deviation note: the spec pinned "append-with-donor-
phases", but TL 3.0 caches PRE-rotary keys and re-applies rotary from
mask-cumsum positions on every forward, so unmodified tail-appended donor K
acquires the phases of its splice slots [r_len, r_len+span_len), not the donor
prompt's original positions; values carry no positions and are exact. Decode
query positions shift by +span_len (offset is computed from the cache/mask
length), which is pre-registered and acceptable.

Symmetry note (review): the first of the 96 sampled tokens in the masking
arms (3-4) comes from the full-prompt forward's logits, which see the hint
span unmasked — mirroring kv_decode.first_sampled_token_pre_splice for arms
5-6. Telemetry fires on decode steps only, so arm-3's mask-check mean is
unaffected; both reference arms sample their first token identically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.stage2_decode_time_correction import json_default, torch_dtype  # noqa: E402
from scripts.stage2_focus_state_composite import make_position_add_hook, make_replace_hook  # noqa: E402
from scripts.stage2_hint_delta import concept_positions, prompt_cache  # noqa: E402
from scripts.stage2_hint_state_interchange import generate_sample_batch  # noqa: E402
from scripts.stage2_interchange_concept_analysis import canon, subjects_of  # noqa: E402
from scripts.stage2_prompt_margin_gated_decode_correction import select_prefix_rows  # noqa: E402
from scripts.stage2_proposal_hints import make_user_prompt, wrong_concept  # noqa: E402
from scripts.stage2_recognition_state_patch import longest_common_token_block  # noqa: E402
from src.activations import input_device_for_model, load_tl_model, render_chat_text, validate_hooks  # noqa: E402
from src.bd_path import ensure_on_path  # noqa: E402
from src.stage2_steering import parse_int_list, score_reply  # noqa: E402

SPAN_MIN = 2
SPAN_MAX = 48
RANK_K_READING_RULE = (
    "k<=4 recovering >=70% of the subset-patch effect (+0.250) = compactly structured; "
    "anything else = distributed-with-structure"
)


@dataclass(frozen=True)
class Arm:
    label: str
    kind: str
    receiver: str  # "unhinted" | "hinted"
    reference: str
    scale: float = 1.0


ARMS = (
    Arm("unhinted_baseline", "none", "unhinted", "none"),
    Arm("hinted_baseline", "telemetry_only", "hinted", "none"),
    Arm("hint_span_masking", "mask_span", "hinted", "hinted_baseline"),
    Arm("masking_x_reversion", "mask_span_and_revert", "hinted", "hinted_baseline"),
    Arm("gold_kv_transplant", "kv_transplant_gold", "unhinted", "unhinted_baseline"),
    Arm("wrong_kv_transplant", "kv_transplant_wrong", "unhinted", "unhinted_baseline"),
    Arm("perpos_add_own", "perpos_add", "unhinted", "unhinted_baseline"),
    Arm("restricted_add_x2", "restricted_add", "unhinted", "unhinted_baseline", 2.0),
    Arm("rank_k_L30", "rank_k_add", "unhinted", "unhinted_baseline"),
)

KV_KINDS = {"kv_transplant_gold", "kv_transplant_wrong"}
HINT_TELEMETRY_KINDS = {"telemetry_only", "mask_span", "mask_span_and_revert"}
MASK_KINDS = {"mask_span", "mask_span_and_revert"}


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def make_span_mask_hook(span_start: int, span_end: int):
    def hook_fn(scores: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        if scores.shape[-2] != 1 or scores.shape[-1] < span_end:
            return scores
        scores[..., span_start:span_end] = torch.finfo(scores.dtype).min
        return scores

    return hook_fn


def make_span_attention_telemetry(span_start: int, span_end: int):
    state: dict[str, Any] = {"sum": None, "max": None, "steps": 0}

    def hook_fn(pattern: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        if pattern.shape[-2] != 1 or pattern.shape[-1] < span_end:
            return pattern
        mass = pattern[:, :, 0, span_start:span_end].sum(dim=-1).detach().float().cpu()
        if state["sum"] is None:
            state["sum"] = mass.mean(dim=1)
            state["max"] = mass.max(dim=1).values
        else:
            state["sum"] = state["sum"] + mass.mean(dim=1)
            state["max"] = torch.maximum(state["max"], mass.max(dim=1).values)
        state["steps"] += 1
        return pattern

    return hook_fn, state


def summarize_telemetry(state: dict[str, Any] | None, n_samples: int) -> tuple[list, list]:
    if state is None or not state["steps"]:
        return [None] * n_samples, [None] * n_samples
    means = [float(v) for v in (state["sum"] / state["steps"])]
    maxes = [float(v) for v in state["max"]]
    return means, maxes


def donor_kv_spans(*, model, token_ids: list[int], span_start: int, span_end: int, cache_dtype: torch.dtype) -> list[tuple[torch.Tensor, torch.Tensor]]:
    from transformer_lens import TransformerLensKeyValueCache

    device = input_device_for_model(model)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    previous_default_dtype = torch.get_default_dtype()
    with torch.inference_mode():
        try:
            torch.set_default_dtype(cache_dtype)
            cache = TransformerLensKeyValueCache.init_cache(model.cfg, model.cfg.device, batch_size=1)
            model.forward(tokens, past_kv_cache=cache, return_type=None)
        finally:
            torch.set_default_dtype(previous_default_dtype)
    assert int(cache.previous_attention_mask.shape[1]) == len(token_ids)
    spans = []
    for entry in cache.entries:
        assert entry.past_keys.shape[1] == len(token_ids)
        spans.append((
            entry.past_keys[:, span_start:span_end].detach().to("cpu"),
            entry.past_values[:, span_start:span_end].detach().to("cpu"),
        ))
    return spans


def splice_donor_kv(cache, donor_layers: list[tuple[torch.Tensor, torch.Tensor]], *, n_samples: int, receiver_len: int) -> int:
    assert cache.frozen is False
    span_len = int(donor_layers[0][0].shape[1])
    for entry, (donor_k, donor_v) in zip(cache.entries, donor_layers, strict=True):
        assert entry.frozen is False
        assert entry.past_keys.shape[:2] == (n_samples, receiver_len)
        assert donor_k.shape == (1, span_len, *entry.past_keys.shape[2:])
        assert donor_v.shape == (1, span_len, *entry.past_values.shape[2:])
        entry.past_keys = torch.cat(
            [entry.past_keys, donor_k.to(device=entry.past_keys.device, dtype=entry.past_keys.dtype).expand(n_samples, -1, -1, -1)],
            dim=1,
        )
        entry.past_values = torch.cat(
            [entry.past_values, donor_v.to(device=entry.past_values.device, dtype=entry.past_values.dtype).expand(n_samples, -1, -1, -1)],
            dim=1,
        )
    mask = cache.previous_attention_mask
    assert int(mask.shape[1]) == receiver_len
    cache.previous_attention_mask = torch.cat(
        [mask, torch.ones((mask.shape[0], span_len), dtype=mask.dtype, device=mask.device)],
        dim=1,
    )
    return span_len


def sample_next_tokens(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    probs = torch.softmax(logits[:, -1].float() / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def kv_transplant_sample_batch(
    *,
    model,
    receiver_ids: list[int],
    donor_layers: list[tuple[torch.Tensor, torch.Tensor]],
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    cache_dtype: torch.dtype,
    fwd_hooks: list[tuple[str, Any]],
) -> list[tuple[list[int], str]]:
    """Batched manual decode with donor hint-span K/V appended at the cache tail.

    Pinned positional convention: donor K/V are appended unmodified after the
    receiver prompt's cache entries, occupying slots [r_len, r_len+span_len).
    TL 3.0 stores pre-rotary keys and re-applies rotary from mask-cumsum
    positions on every forward, so the spliced keys take the phases of their
    tail slots rather than the donor prompt's original positions (deviation
    from the spec sketch's "donor phases" wording; values carry no positions
    and are exact). Every decode query position is computed from the cache/
    mask length and therefore shifts by +span_len relative to the unhinted
    baseline — pre-registered and acceptable. The first of max_new_tokens
    sampled tokens comes from the receiver prompt forward's logits and so
    precedes the splice; the remaining max_new_tokens-1 decode forwards attend
    over the spliced slots. All samples run the full loop; replies are
    post-hoc truncated at the first tokenizer.eos_token_id (<eos>=1 only, NOT
    <end_of_turn>=106) to match TL generate/generate_sample_batch semantics.
    """
    tokenizer = model.tokenizer
    device = input_device_for_model(model)
    tokens = torch.tensor([receiver_ids] * n_samples, dtype=torch.long, device=device)
    previous_default_dtype = torch.get_default_dtype()
    generated = []
    from transformer_lens import TransformerLensKeyValueCache

    with torch.inference_mode(), model.hooks(fwd_hooks=fwd_hooks):
        try:
            torch.set_default_dtype(cache_dtype)
            cache = TransformerLensKeyValueCache.init_cache(model.cfg, model.cfg.device, batch_size=n_samples)
            logits = model.forward(tokens, past_kv_cache=cache, return_type="logits")
            splice_donor_kv(cache, donor_layers, n_samples=n_samples, receiver_len=len(receiver_ids))
            next_tokens = sample_next_tokens(logits, temperature)
            generated.append(next_tokens)
            for _ in range(max_new_tokens - 1):
                logits = model.forward(next_tokens[:, None].to(device), past_kv_cache=cache, return_type="logits")
                next_tokens = sample_next_tokens(logits, temperature)
                generated.append(next_tokens)
        finally:
            torch.set_default_dtype(previous_default_dtype)
    all_ids = torch.stack(generated, dim=1).cpu().tolist()
    eos_id = int(tokenizer.eos_token_id)
    results = []
    for sample_ids in all_ids:
        cut = sample_ids.index(eos_id) if eos_id in sample_ids else len(sample_ids)
        reply = tokenizer.decode(sample_ids[:cut], skip_special_tokens=True).strip()
        results.append((sample_ids, reply))
    return results


def rank_k_basis(states_npz: Path, layer: int, rank_k: int) -> dict[str, Any]:
    data = np.load(states_npz)
    keys = sorted(k for k in data.files if k.startswith(f"L{layer}_row") and k.endswith("_concept_delta"))
    if not keys:
        raise ValueError(f"no L{layer}_row*_concept_delta arrays in {states_npz}")
    pooled = np.concatenate([data[k] for k in keys], axis=0).astype(np.float64)
    mean = pooled.mean(axis=0)
    _, singular, vt = np.linalg.svd(pooled - mean, full_matrices=False)
    return {
        "mean": mean,
        "components": vt[:rank_k],
        "n_pooled_arrays": len(keys),
        "n_pooled_positions": int(pooled.shape[0]),
        "explained_variance_ratio": float((singular[:rank_k] ** 2).sum() / (singular**2).sum()),
    }


def rank_k_reconstruction(delta: torch.Tensor, basis: dict[str, Any]) -> torch.Tensor:
    centered = delta.numpy().astype(np.float64) - basis["mean"][None, :]
    coords = centered @ basis["components"].T
    recon = basis["mean"][None, :] + coords @ basis["components"]
    return torch.from_numpy(recon.astype(np.float32))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("results/full/with_errortype/gemma3_27b_infer_property.jsonl"))
    parser.add_argument("--prefix-trajectory-jsonl", type=Path, default=Path("results/stage2/decode_time/prefix_conditioned_margin_trajectory_gemma_manifest.jsonl"))
    parser.add_argument("--prefix-checkpoint", default="0")
    parser.add_argument("--selection-limit", type=int, default=None)
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--task", default="infer_property")
    parser.add_argument("--layers", default="30,40,45")
    parser.add_argument("--min-block-tokens", type=int, default=32)
    parser.add_argument("--samples-per-row", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=20260616)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--states-npz", type=Path, default=Path("results/stage2/erasure/focus_state_composite_27b_property_states.npz"))
    parser.add_argument("--rank-k", type=int, default=4)
    parser.add_argument("--n-devices", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-mode", choices=("no-processing", "default"), default="no-processing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/stage2/erasure/kv_hint_span_27b_property.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/kv_hint_span_27b_property.json"))
    args = parser.parse_args()
    load_dotenv()
    torch.set_grad_enabled(False)
    started = time.time()

    layers = parse_int_list(args.layers)
    rank_layer = layers[0]
    arms = list(ARMS)
    selected_rows, selection_summary = select_prefix_rows(
        prefix_jsonl=args.prefix_trajectory_jsonl,
        source_jsonl=args.jsonl,
        checkpoint=args.prefix_checkpoint,
        limit=args.selection_limit,
        prompt_gold_vs_foil_threshold=0.0,
    )
    total = len(selected_rows) * len(arms) * args.samples_per_row
    print(f"selected_rows={len(selected_rows)} arms={[arm.label for arm in arms]} total_generations={total}", flush=True)
    if args.dry_run:
        print(json.dumps({"selection": selection_summary, "layers": layers, "arms": [arm.__dict__ for arm in arms], "total_generations": total}, indent=2, default=json_default), flush=True)
        return 0

    ensure_on_path()
    basis = rank_k_basis(args.states_npz, rank_layer, args.rank_k)
    print(f"rank_k basis: L{rank_layer} k={args.rank_k} pooled_positions={basis['n_pooled_positions']} explained_var={basis['explained_variance_ratio']:.3f}", flush=True)
    model = load_tl_model(args.model, n_devices=args.n_devices, n_ctx=args.n_ctx, dtype=torch_dtype(args.dtype), load_mode=args.load_mode)
    dtype = torch_dtype(args.dtype)
    hook_names = validate_hooks(model, layers)
    hook_by_layer = dict(zip(layers, hook_names))
    all_layers = list(range(model.cfg.n_layers))
    pattern_hook_names = validate_hooks(model, all_layers, hook_template="blocks.{layer}.attn.hook_pattern")
    score_hook_names = validate_hooks(model, all_layers, hook_template="blocks.{layer}.attn.hook_attn_scores")
    tokenizer = model.tokenizer
    print(f"using_hooks={hook_by_layer} n_layers={model.cfg.n_layers}", flush=True)

    prepared = []
    for stage1_row in selected_rows:
        source_row_index = int(stage1_row["row_index"])
        gold_concept = stage1_row["ontology_fol_structured"]["hypothesis"]["subject"]
        wrong = wrong_concept(stage1_row)
        receiver_text = render_chat_text(tokenizer, system=stage1_row["system_prompt"], user=stage1_row["prompt_text"], model_name=args.model, add_generation_prompt=True)
        receiver_ids = tokenizer(receiver_text, add_special_tokens=False)["input_ids"]
        hinted_text = render_chat_text(tokenizer, system=stage1_row["system_prompt"], user=make_user_prompt(stage1_row, "hint_concept_first"), model_name=args.model, add_generation_prompt=True)
        hinted_ids = tokenizer(hinted_text, add_special_tokens=False)["input_ids"]
        wrong_text = render_chat_text(tokenizer, system=stage1_row["system_prompt"], user=make_user_prompt(stage1_row, "hint_wrong_concept_first"), model_name=args.model, add_generation_prompt=True)
        wrong_ids = tokenizer(wrong_text, add_special_tokens=False)["input_ids"]

        h_start, r_start, block_len = longest_common_token_block(hinted_ids, receiver_ids)
        if block_len < args.min_block_tokens:
            print(f"skip row {source_row_index}: block={block_len}", flush=True)
            continue
        gold_prefix = common_prefix_len(hinted_ids, receiver_ids)
        gold_span_len = h_start - gold_prefix
        if not (SPAN_MIN <= gold_span_len <= SPAN_MAX):
            print(f"skip row {source_row_index}: gold span_len={gold_span_len} outside [{SPAN_MIN},{SPAN_MAX}]", flush=True)
            continue
        w_start, _, wrong_block_len = longest_common_token_block(wrong_ids, receiver_ids)
        wrong_prefix = common_prefix_len(wrong_ids, receiver_ids)
        wrong_span_len = w_start - wrong_prefix
        if wrong_block_len < args.min_block_tokens or not (SPAN_MIN <= wrong_span_len <= SPAN_MAX):
            print(f"skip row {source_row_index}: wrong span_len={wrong_span_len} block={wrong_block_len}", flush=True)
            continue

        positions_r = concept_positions(tokenizer, receiver_text, gold_concept, r_start, block_len)
        if not positions_r:
            print(f"skip row {source_row_index}: no concept positions", flush=True)
            continue
        rel = [p - r_start for p in positions_r]
        positions_h = [p + h_start for p in rel]
        assert max(positions_r) < len(receiver_ids)
        assert max(positions_h) < len(hinted_ids)

        unhinted_cache = prompt_cache(model, receiver_ids, hook_names)
        hinted_cache = prompt_cache(model, hinted_ids, hook_names)
        per_layer = {}
        for layer in layers:
            u_block = unhinted_cache[hook_by_layer[layer]][r_start : r_start + block_len]
            h_block = hinted_cache[hook_by_layer[layer]][h_start : h_start + block_len]
            concept_delta = h_block[rel] - u_block[rel]
            per_layer[layer] = {
                "u_block": u_block,
                "h_block": h_block,
                "concept_delta": concept_delta,
                "restricted_mean": concept_delta.mean(dim=0),
            }
        gold_kv = donor_kv_spans(model=model, token_ids=hinted_ids, span_start=gold_prefix, span_end=h_start, cache_dtype=dtype)
        wrong_kv = donor_kv_spans(model=model, token_ids=wrong_ids, span_start=wrong_prefix, span_end=w_start, cache_dtype=dtype)
        prepared.append({
            "row": stage1_row,
            "source_row_index": source_row_index,
            "gold_concept": gold_concept,
            "wrong_concept": wrong,
            "receiver_ids": receiver_ids,
            "hinted_ids": hinted_ids,
            "r_start": r_start,
            "h_start": h_start,
            "block_len": block_len,
            "gold_span": (gold_prefix, h_start),
            "gold_span_len": gold_span_len,
            "wrong_span_len": wrong_span_len,
            "rel": rel,
            "positions_r": positions_r,
            "positions_h": positions_h,
            "per_layer": per_layer,
            "gold_kv": gold_kv,
            "wrong_kv": wrong_kv,
            "rank_recon": rank_k_reconstruction(per_layer[rank_layer]["concept_delta"], basis),
        })
        print(f"prepared row {source_row_index}: block={block_len} concept_tokens={len(rel)} gold_span={gold_span_len} wrong_span={wrong_span_len}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n_rows = len(prepared)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    with args.out_jsonl.open("w") as fout:
        for arm_index, arm in enumerate(arms):
            arm_started = time.time()
            for prep_index, prep in enumerate(prepared):
                torch.manual_seed(args.sample_seed + prep["source_row_index"] * 10007 + arm_index * 101)
                telemetry_state = None
                span_len_used = prep["wrong_span_len"] if arm.kind == "kv_transplant_wrong" else prep["gold_span_len"]
                if arm.kind in KV_KINDS:
                    donor_layers = prep["gold_kv"] if arm.kind == "kv_transplant_gold" else prep["wrong_kv"]
                    r_len = len(prep["receiver_ids"])
                    telemetry_fn, telemetry_state = make_span_attention_telemetry(r_len, r_len + span_len_used)
                    fwd_hooks = [(name, telemetry_fn) for name in pattern_hook_names]
                    batch = kv_transplant_sample_batch(
                        model=model,
                        receiver_ids=prep["receiver_ids"],
                        donor_layers=donor_layers,
                        n_samples=args.samples_per_row,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        cache_dtype=dtype,
                        fwd_hooks=fwd_hooks,
                    )
                else:
                    token_ids = prep["hinted_ids"] if arm.receiver == "hinted" else prep["receiver_ids"]
                    fwd_hooks = []
                    if arm.kind in HINT_TELEMETRY_KINDS:
                        telemetry_fn, telemetry_state = make_span_attention_telemetry(*prep["gold_span"])
                        fwd_hooks += [(name, telemetry_fn) for name in pattern_hook_names]
                    if arm.kind in MASK_KINDS:
                        mask_fn = make_span_mask_hook(*prep["gold_span"])
                        fwd_hooks += [(name, mask_fn) for name in score_hook_names]
                    if arm.kind == "mask_span_and_revert":
                        for layer in layers:
                            fwd_hooks.append((hook_by_layer[layer], make_replace_hook(prep["per_layer"][layer]["u_block"][prep["rel"]], prep["positions_h"])))
                    elif arm.kind == "perpos_add":
                        for layer in layers:
                            fwd_hooks.append((hook_by_layer[layer], make_position_add_hook(prep["per_layer"][layer]["concept_delta"], prep["positions_r"], arm.scale)))
                    elif arm.kind == "restricted_add":
                        for layer in layers:
                            matrix = prep["per_layer"][layer]["restricted_mean"].unsqueeze(0).expand(len(prep["rel"]), -1)
                            fwd_hooks.append((hook_by_layer[layer], make_position_add_hook(matrix, prep["positions_r"], arm.scale)))
                    elif arm.kind == "rank_k_add":
                        fwd_hooks.append((hook_by_layer[rank_layer], make_position_add_hook(prep["rank_recon"], prep["positions_r"], arm.scale)))
                    with model.hooks(fwd_hooks=fwd_hooks):
                        batch = generate_sample_batch(
                            model=model,
                            token_ids=token_ids,
                            n_samples=args.samples_per_row,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            stop_at_eos=True,
                            cache_dtype=dtype,
                        )
                tel_means, tel_maxes = summarize_telemetry(telemetry_state, args.samples_per_row)
                if arm.kind == "mask_span_and_revert":
                    patch_layers = layers
                elif arm.kind in ("perpos_add", "restricted_add"):
                    patch_layers = layers
                elif arm.kind == "rank_k_add":
                    patch_layers = [rank_layer]
                else:
                    patch_layers = None
                for sample_index, (new_ids, reply) in enumerate(batch):
                    score = score_reply(prep["row"], reply)
                    subjects = subjects_of(reply)
                    out = {
                        "schema_version": 1,
                        "source_row_index": prep["source_row_index"],
                        "example_id": prep["row"].get("example_id"),
                        "height": prep["row"].get("height"),
                        "model": args.model,
                        "condition": arm.label,
                        "arm_kind": arm.kind,
                        "receiver": arm.receiver,
                        "reference": arm.reference,
                        "scale": arm.scale,
                        "sample_index": sample_index,
                        "method": "kv_hint_span",
                        "target_variable": "target_concept",
                        "representation_type": "kv_cache_state" if arm.kind in KV_KINDS else "patched_residual_state",
                        "patch_layers": patch_layers,
                        "gold_concept": prep["gold_concept"],
                        "wrong_concept": prep["wrong_concept"],
                        "span_len": span_len_used,
                        "n_concept_positions": len(prep["rel"]),
                        "attn_to_span_mean": tel_means[sample_index],
                        "attn_to_span_max": tel_maxes[sample_index],
                        "targets_gold_concept": canon(prep["gold_concept"]) in subjects,
                        "targets_wrong_concept": canon(prep["wrong_concept"]) in subjects,
                        "generated_token_count": len(new_ids),
                        "model_output": reply,
                        **score,
                    }
                    rows_out.append(out)
                    fout.write(json.dumps(out, ensure_ascii=False, default=json_default) + "\n")
                    fout.flush()
                strong_rate = float(np.mean([r["is_correct_strong"] for r in rows_out[-args.samples_per_row:]]))
                tel_note = "" if tel_means[0] is None else f" attn_mean={float(np.mean(tel_means)):.4f} attn_max={float(np.max(tel_maxes)):.4f}"
                print(f"arm {arm_index + 1}/{len(arms)} {arm.label} row {prep_index + 1}/{n_rows}: P(strong)={strong_rate:.2f}{tel_note}", flush=True)
            print(f"ARM DONE {arm.label}: {time.time() - arm_started:.0f}s elapsed_total={time.time() - started:.0f}s", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    by = defaultdict(lambda: defaultdict(list))
    for r in rows_out:
        by[r["condition"]][r["source_row_index"]].append(r)
    rng = np.random.default_rng(args.sample_seed)
    refs = {arm.label: arm.reference for arm in arms}
    summary = {}
    for cond in sorted(by):
        per_row = {i: float(np.mean([s["is_correct_strong"] for s in v])) for i, v in by[cond].items()}
        flat = [s for v in by[cond].values() for s in v]
        entry: dict[str, Any] = {
            "n_rows": len(per_row),
            "strong_accuracy": float(np.mean([s["is_correct_strong"] for s in flat])),
            "targets_gold_concept_rate": float(np.mean([s["targets_gold_concept"] for s in flat])),
            "targets_wrong_concept_rate": float(np.mean([s["targets_wrong_concept"] for s in flat])),
            "reference": refs.get(cond, "none"),
        }
        tel_vals = [s["attn_to_span_mean"] for s in flat if s["attn_to_span_mean"] is not None]
        if tel_vals:
            entry["attn_to_span_mean"] = float(np.mean(tel_vals))
            entry["attn_to_span_max"] = float(np.max([s["attn_to_span_max"] for s in flat if s["attn_to_span_max"] is not None]))
        ref = refs.get(cond, "none")
        if ref != "none" and ref in by:
            ref_rows = {i: float(np.mean([s["is_correct_strong"] for s in v])) for i, v in by[ref].items()}
            deltas = np.array([per_row[i] - ref_rows[i] for i in sorted(ref_rows) if i in per_row])
            boots = [float(np.mean(deltas[rng.integers(0, len(deltas), len(deltas))])) for _ in range(10000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            entry["paired_delta_vs_reference"] = float(np.mean(deltas))
            entry["paired_ci95"] = [float(lo), float(hi)]
        summary[cond] = entry

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": "scripts/stage2_kv_hint_span.py",
        "model": args.model,
        "task": args.task,
        "target_variable": "target_concept",
        "method": "kv_hint_span",
        "representation_type": "patched_residual_state (arms 1-4, 7-9) / kv_cache_state (arms 5-6)",
        "layers": layers,
        "selection": selection_summary,
        "arms": [arm.__dict__ for arm in arms],
        "generation": {"samples_per_row": args.samples_per_row, "temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
        "summary": summary,
        "n": len(rows_out),
        "rank_k": {
            "k": args.rank_k,
            "layer": rank_layer,
            "states_npz": str(args.states_npz),
            "n_pooled_arrays": basis["n_pooled_arrays"],
            "n_pooled_positions": basis["n_pooled_positions"],
            "explained_variance_ratio": basis["explained_variance_ratio"],
        },
        "rank_k_reading_rule": RANK_K_READING_RULE,
        "kv_decode": {
            "convention": (
                "append-at-tail, unmodified donor K/V at slots [r_len, r_len+span_len). Deviation from the "
                "'append-with-donor-phases' spec wording: TL 3.0 caches pre-rotary keys and re-applies rotary "
                "from mask-cumsum positions on every forward, so spliced keys acquire the phases of their tail "
                "slots, not the donor prompt's positions; values are position-free and exact."
            ),
            "decode_position_shift": "+span_len (offset computed from cache/mask length; pre-registered, acceptable)",
            "first_sampled_token_pre_splice": True,
            "decode_forwards_per_sample": args.max_new_tokens - 1,
            "eos_truncation_ids": [int(tokenizer.eos_token_id)],
        },
        "span_bounds": [SPAN_MIN, SPAN_MAX],
        "controls": [
            "in_job_unhinted_and_hinted_baselines",
            "decode_only_masking_preserves_prompt_encoding",
            "masking_x_reversion_exhaustive_necessity_headline",
            "wrong_kv_concept_specificity",
            "arm3_telemetry_mean_must_be_zero_mask_check",
            "gold_kv_telemetry_disambiguates_splice_bug_vs_insufficiency",
        ],
        "causal_abstraction_claim": (
            "Pre-registered bins: collapse-only-under-combination = pathways jointly exhaustive; "
            "survives-both = unpatched-layer carrier; collapses-under-masking-alone = decode-time "
            "attention dominant; gold-KV-positive = both pathways individually sufficient; gold-KV-null "
            "disambiguated by telemetry: no-attention = splice bug, attention-without-repair = genuine "
            "insufficiency. Rank-k rung read by rule: " + RANK_K_READING_RULE + "."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
