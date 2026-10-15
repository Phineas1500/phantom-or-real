#!/usr/bin/env python3
"""Item NX — does the frozen hint-delta write decompose into a few MLP neurons?
givemeanode H100 entrypoint. Rows, donors and the frozen direction u exactly
as item WX (cross-fit on the 59 WF rows; u = unit mean donor gold-mention
hint-delta at L30, norm target N = donor mean per-position |delta|).

NX_STAGE=source (stages 1-2): the donor forwards that fit u also record, per
layer 0..L30 at the gold-mention positions, every MLP neuron's post-norm
contribution change (a_i * s, s = the post-feedforward RMSNorm scale) and every
attention head's (o_proj input slice * the post-attention scale); projected on
u they decompose the mean donor delta exactly (reconstruction reported). Test
arms at the gold address, k=4 each: U (WX write), U_NOMASSIVE, NEURON_k for
k in NX_KS (the top-k neurons by projection, each neuron's donor-mean
contribution vector added at its own layer's output, all scaled by one alpha
so the summed projection on u equals rung*N), RANDOM_64 x 2 seeded draws
(norm-matched to NEURON_64); NEURON_64 also at three seeded non-gold
addresses.

NX_STAGE=read (stage 3): donors with the U write at their gold mentions rank
neurons in NX_READ_LAYERS at the written positions by |donor-mean change of
the post-norm contribution| * |column|; test arms at the gold address: U,
FREEZE_MLP (U, with the MLP output at the written positions in the read
layers clamped to its no-write value), CLAMP_TOP (U, the top NX_CLAMP_K
neurons' activations at the written positions clamped to no-write values),
CLAMP_RAND (U, NX_CLAMP_K seeded random neurons from the same layers).

Every arm's hooks wrap generate; each hook counts prefill calls and positions
touched, and a zero is execution-invalid (raised).
Registered: docs/causal_handle_directions.md item NX."""
from __future__ import annotations
import gzip, json, os, re, sys, time
import numpy as np
import torch

APP = os.environ.get("NX_APP_DIR", "/app")
sys.path.insert(0, APP)
from wikihop_common import SYSTEM, hint_first_prompt, normalize_answer, std_closed_prompts  # noqa: E402

MODEL = os.environ.get("NX_MODEL", "google/gemma-3-27b-it")


def layers_of(model):
    return model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers


def hidden_of(out):
    return out[0] if isinstance(out, tuple) else out


def with_hidden(out, h):
    return (h, *out[1:]) if isinstance(out, tuple) else h


def mention_positions(text, offsets, cand):
    spans = [m.span() for m in re.finditer(r"(?<!\w)" + re.escape(cand) + r"(?!\w)", text, re.IGNORECASE)]
    return sorted({i for (a, b) in spans for i, (s0, s1) in enumerate(offsets) if s0 < b and s1 > a and s1 > s0})


def int_list(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


class Counted:
    def __init__(self):
        self.prefill_calls = 0
        self.positions_touched = 0

    def ok(self):
        return self.prefill_calls >= 1 and self.positions_touched >= 1


class ResidualAdd(Counted):
    """Adds a fixed vector at the given positions of a decoder layer's output (prefill only)."""

    def __init__(self, vec, positions):
        super().__init__()
        self.vec = vec
        self.positions = positions

    def __call__(self, _m, _i, out):
        h = hidden_of(out)
        if h.shape[1] <= 1:
            return out
        v = self.vec.to(h.device, h.dtype)
        pp = [p for p in self.positions if p < h.shape[1]]
        h[:, pp, :] += v
        self.prefill_calls += 1
        self.positions_touched += len(pp)
        return with_hidden(out, h)


class OutputClamp(Counted):
    """Replaces a module's output at the given positions with stored values (prefill only)."""

    def __init__(self, stored, positions):
        super().__init__()
        self.stored = stored
        self.positions = positions

    def __call__(self, _m, _i, out):
        h = hidden_of(out)
        if h.shape[1] <= 1:
            return out
        h[:, self.positions, :] = self.stored.to(h.device, h.dtype).unsqueeze(0)
        self.prefill_calls += 1
        self.positions_touched += len(self.positions)
        return with_hidden(out, h)


class NeuronClamp(Counted):
    """forward_pre hook on down_proj: sets the chosen neurons' activations at the
    given positions to stored values (prefill only)."""

    def __init__(self, idx, stored, positions):
        super().__init__()
        self.idx = idx
        self.stored = stored  # (n_pos, n_idx)
        self.positions = positions

    def __call__(self, _m, args):
        a = args[0]
        if a.shape[1] <= 1:
            return None
        a = a.clone()
        ix = self.idx.to(a.device)
        st = self.stored.to(a.device, a.dtype)
        for j, p in enumerate(self.positions):
            a[:, p, ix] = st[j]
        self.prefill_calls += 1
        self.positions_touched += len(self.positions)
        return (a, *args[1:])


def main():
    t0 = time.time()
    outdir = os.environ.get("GMN_OUTPUT_DIR", "/tmp")
    stage = os.environ.get("NX_STAGE", "source")
    wx_job = os.environ.get("NX_WX_JOB", "A")
    seed0 = int(os.environ.get("NX_SEED", "20261016"))
    write_layer = int(os.environ.get("NX_WRITE_LAYER", "30"))
    rung = float(os.environ.get("NX_RUNG", "2.0"))
    ks = int_list(os.environ.get("NX_KS", "16,64,256,1024"))
    rand_seeds = int_list(os.environ.get("NX_RANDOM_SEEDS", "20261017,20261018"))
    massive = int_list(os.environ.get("NX_MASSIVE", "104,2733"))
    read_layers = int_list(os.environ.get("NX_READ_LAYERS", "31-36"))
    clamp_k = int(os.environ.get("NX_CLAMP_K", "64"))
    clamp_seed = int(os.environ.get("NX_CLAMP_SEED", "20261019"))
    n_nongold = int(os.environ.get("NX_NONGOLD", "3"))
    k_base, k_write = int(os.environ.get("NX_K_BASE", "8")), int(os.environ.get("NX_K_WRITE", "4"))
    max_new = int(os.environ.get("NX_MAX_NEW_TOKENS", "32"))
    max_rows = int(os.environ.get("NX_MAX_ROWS", "10000"))
    fake = os.environ.get("NX_FAKE_MODEL") == "1"

    pins = json.load(open(os.path.join(APP, os.environ.get("NX_PINS_FILE", "wikihop_wx_pinned.json"))))
    row_ids = pins["jobs"][wx_job]["test_rows"][:max_rows]
    donor_ids = pins["jobs"][wx_job]["donor_rows"][:max_rows]
    frame = {}
    for line in gzip.open(os.path.join(APP, "wikihop_port_input.jsonl.gz"), "rt"):
        r = json.loads(line)
        frame[r["id"]] = r
    print(f"stage={stage} wx_job={wx_job} rows={len(row_ids)} donors={len(donor_ids)} write=L{write_layer} rung={rung}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if fake:
        from fake_gemma import FakeGemma
        model = FakeGemma(n_layers=int(os.environ.get("NX_FAKE_LAYERS", "40")), d=int(os.environ.get("NX_FAKE_D", "64")),
                          vocab=len(tok), inter=int(os.environ.get("NX_FAKE_INTER", "128")), n_heads=4, head_dim=8)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    model.eval()
    layers = layers_of(model)
    dev = next(model.parameters()).device
    n_inter = layers[0].mlp.down_proj.in_features
    head_dim = layers[0].self_attn.head_dim
    n_heads = layers[0].self_attn.o_proj.in_features // head_dim
    d_model = layers[0].mlp.down_proj.out_features
    turn_end = tok.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = sorted({tok.eos_token_id, turn_end} - {None, tok.unk_token_id})
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"model loaded ({time.time()-t0:.0f}s) n_layers={len(layers)} d={d_model} inter={n_inter} heads={n_heads}x{head_dim}", flush=True)

    def render(user_text):
        text = tok.apply_chat_template([{"role": "user", "content": SYSTEM + "\n\n" + user_text}], tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        return text, enc["input_ids"], enc["offset_mapping"]

    def run(ids, hooks=()):
        handles = [mod.register_forward_pre_hook(fn) if kind == "pre" else mod.register_forward_hook(fn) for kind, mod, fn in hooks]
        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
        finally:
            for h in handles:
                h.remove()

    def generate(ids, k, seed, hooks=()):
        handles = [mod.register_forward_pre_hook(fn) if kind == "pre" else mod.register_forward_hook(fn) for kind, mod, fn in hooks]
        try:
            torch.manual_seed(seed)
            with torch.inference_mode():
                out = model.generate(input_ids=torch.tensor([ids], device=dev), attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=dev),
                                     do_sample=True, temperature=0.7, max_new_tokens=max_new, num_return_sequences=k,
                                     eos_token_id=eos_ids, pad_token_id=pad_id)
        finally:
            for h in handles:
                h.remove()
        return [tok.decode(seq, skip_special_tokens=True).strip() for seq in out[:, len(ids):].detach().cpu().tolist()]

    def capture_hooks(store, positions, layer_ids, mlp=True, attn=False, resid=None):
        """Hooks recording, at `positions`, each layer's down_proj input a, the
        post-feedforward RMSNorm scale s, and optionally o_proj input z with the
        post-attention scale; resid=L also records that layer's output."""
        hooks = []
        pos = list(positions)
        for L in layer_ids:
            lay = layers[L]
            if mlp:
                def a_pre(_m, args, L=L):
                    store[("a", L)] = args[0][0, pos].detach().float()
                def m_out(_m, _i, out, L=L):
                    m = out[0, pos].detach().float()
                    store[("s", L)] = torch.rsqrt(m.pow(2).mean(-1) + lay_eps(L, "post_feedforward_layernorm"))
                hooks += [("pre", lay.mlp.down_proj, a_pre), ("fwd", lay.mlp.down_proj, m_out)]
            if attn:
                def z_pre(_m, args, L=L):
                    store[("z", L)] = args[0][0, pos].detach().float()
                def o_out(_m, _i, out, L=L):
                    o = out[0, pos].detach().float()
                    store[("t", L)] = torch.rsqrt(o.pow(2).mean(-1) + lay_eps(L, "post_attention_layernorm"))
                hooks += [("pre", lay.self_attn.o_proj, z_pre), ("fwd", lay.self_attn.o_proj, o_out)]
        if resid is not None:
            def r_out(_m, _i, out):
                store["resid"] = hidden_of(out)[0, pos].detach().float()
            hooks.append(("fwd", layers[resid], r_out))
        return hooks

    def lay_eps(L, name):
        return float(getattr(layers[L], name).eps)

    def norm_gain(L, name):
        return 1.0 + getattr(layers[L], name).weight.detach().float()

    def paired_positions(r, cand):
        text_s, ids_s, off_s = render(std_closed_prompts(r)["std"])
        pos_s = mention_positions(text_s, off_s, cand)
        text_h, ids_h, off_h = render(hint_first_prompt(r, cand))
        pos_h_all = mention_positions(text_h, off_h, cand)
        pos_h = pos_h_all[len(pos_h_all) - len(pos_s):] if len(pos_h_all) > len(pos_s) else None
        if not pos_s or pos_h is None or len(pos_h) != len(pos_s) or any(ids_h[a] != ids_s[b] for a, b in zip(pos_h, pos_s)):
            return None
        return ids_s, pos_s, ids_h, pos_h

    src_layers = list(range(write_layer + 1))
    # ---- donors: fit u (as WX) and, in the source stage, the exact decomposition ----
    acc_delta, n_pos, norms, n_donors = None, 0, [], 0
    acc_a = {L: torch.zeros(n_inter, dtype=torch.float64, device=dev) for L in src_layers} if stage == "source" else {}
    acc_z = {L: torch.zeros(n_heads * head_dim, dtype=torch.float64, device=dev) for L in src_layers} if stage == "source" else {}
    donor_rows_used = []
    for did in donor_ids:
        r = frame[did]
        pp = paired_positions(r, r["answer"])
        if pp is None:
            print(f"  donor skip {did}", flush=True)
            continue
        ids_s, pos_s, ids_h, pos_h = pp
        st_s, st_h = {}, {}
        do_src = stage == "source"
        run(ids_s, capture_hooks(st_s, pos_s, src_layers if do_src else [], attn=do_src, resid=write_layer))
        run(ids_h, capture_hooks(st_h, pos_h, src_layers if do_src else [], attn=do_src, resid=write_layer))
        d = (st_h["resid"] - st_s["resid"]).double()
        acc_delta = d.sum(0) if acc_delta is None else acc_delta + d.sum(0)
        n_pos += len(pos_s)
        norms.extend(d.norm(dim=1).tolist())
        n_donors += 1
        donor_rows_used.append(did)
        if do_src:
            for L in src_layers:
                acc_a[L] += (st_h[("a", L)] * st_h[("s", L)][:, None] - st_s[("a", L)] * st_s[("s", L)][:, None]).double().sum(0)
                acc_z[L] += (st_h[("z", L)] * st_h[("t", L)][:, None] - st_s[("z", L)] * st_s[("t", L)][:, None]).double().sum(0)
    massive = [m for m in massive if m < d_model]
    mean_delta = acc_delta / n_pos
    u = (mean_delta / mean_delta.norm()).float()
    N = float(np.mean(norms))
    frozen = {"n_donors": n_donors, "n_positions": n_pos, "mean_delta_norm": float(mean_delta.norm()), "norm_target": N,
              "massive_dims": massive, "u_massive_share": float(u[massive].pow(2).sum()) if massive else None}
    print(f"frozen u from {n_donors} donors / {n_pos} positions: |mean delta| {frozen['mean_delta_norm']:.1f}, N {N:.1f}, "
          f"massive share {frozen['u_massive_share']} ({time.time()-t0:.0f}s)", flush=True)

    import transformers
    summary = {"transformers": transformers.__version__, "torch": torch.__version__, "stage": stage, "wx_job": wx_job, "seed": seed0, "write_layer": write_layer, "rung": rung, "frozen": frozen,
               "donor_rows": donor_rows_used, "model_dims": {"d": d_model, "inter": n_inter, "heads": n_heads, "head_dim": head_dim}}
    u_write = (u * rung * N)

    if stage == "source":
        # projections of each unit's donor-mean contribution onto u
        cbar_mlp, cbar_head, vecs = {}, {}, {}
        for L in src_layers:
            lay = layers[L]
            wcol = lay.mlp.down_proj.weight.detach().float() * norm_gain(L, "post_feedforward_layernorm")[:, None]  # (d, inter)
            r_m = wcol.T @ u.to(wcol.device)
            cbar_mlp[L] = (acc_a[L].float() / n_pos) * r_m
            wo = lay.self_attn.o_proj.weight.detach().float() * norm_gain(L, "post_attention_layernorm")[:, None]  # (d, H*hd)
            r_o = wo.T @ u.to(wo.device)
            cbar_head[L] = ((acc_z[L].float() / n_pos) * r_o).view(n_heads, head_dim).sum(-1)
        mlp_total = float(sum(cbar_mlp[L].sum() for L in src_layers))
        head_total = float(sum(cbar_head[L].sum() for L in src_layers))
        target = float(mean_delta.float().to(u.device) @ u)
        flat = torch.cat([cbar_mlp[L].cpu() for L in src_layers])
        order = torch.argsort(flat, descending=True)
        pos_mass = float(flat.clamp(min=0).sum())
        curve = {}
        for k in [1, 4, 16, 64, 256, 1024, 4096, 16384]:
            curve[str(k)] = {"share_of_net_projection": float(flat[order[:k]].sum()) / target,
                             "share_of_positive_mlp_mass": float(flat[order[:k]].clamp(min=0).sum()) / pos_mass}
        head_flat = torch.cat([cbar_head[L].cpu() for L in src_layers])
        h_order = torch.argsort(head_flat, descending=True)
        summary["decomposition"] = {
            "projection_target_mean_delta_dot_u": target, "mlp_total": mlp_total, "head_total": head_total,
            "reconstruction_ratio": (mlp_total + head_total) / target, "mlp_share": mlp_total / target, "head_share": head_total / target,
            "positive_mlp_mass": pos_mass, "concentration": curve,
            "per_layer_mlp": {str(L): float(cbar_mlp[L].sum()) for L in src_layers},
            "per_layer_heads": {str(L): float(cbar_head[L].sum()) for L in src_layers},
            "top_heads": [{"layer": int(i // n_heads), "head": int(i % n_heads), "c": float(head_flat[i])} for i in h_order[:32].tolist()]}
        print(f"decomposition: target {target:.1f}, mlp {mlp_total:.1f}, heads {head_total:.1f}, "
              f"reconstruction {(mlp_total + head_total) / target:.4f}; top-64 share {curve['64']['share_of_net_projection']:.3f}", flush=True)

        def unit(i):
            return src_layers[int(i) // n_inter], int(i) % n_inter

        def neuron_vectors(flat_idx):
            """Per-layer sums of the chosen neurons' donor-mean post-norm contribution vectors."""
            by_layer = {}
            for i in flat_idx:
                L, j = unit(i)
                by_layer.setdefault(L, []).append(j)
            out = {}
            for L, js in by_layer.items():
                lay = layers[L]
                jj = torch.tensor(js, device=lay.mlp.down_proj.weight.device)
                wcol = lay.mlp.down_proj.weight.detach().float()[:, jj] * norm_gain(L, "post_feedforward_layernorm")[:, None]
                out[L] = (wcol @ (acc_a[L].float()[jj.to(acc_a[L].device)] / n_pos).to(wcol.device)).cpu()
            return out

        arm_vecs, arm_meta = {}, {}
        for k in ks:
            idx = order[:k].tolist()
            v = neuron_vectors(idx)
            proj = float(flat[order[:k]].sum())
            alpha = rung * N / proj
            arm_vecs[f"NEURON_{k}"] = {L: alpha * x for L, x in v.items()}
            tot = sum(arm_vecs[f"NEURON_{k}"].values())
            arm_meta[f"NEURON_{k}"] = {"k": k, "alpha": alpha, "direct_projection": proj, "n_layers": len(v),
                                       "summed_norm": float(tot.norm()), "cos_sum_u": float(tot @ u.cpu() / tot.norm())}
        summary["top_neurons"] = [{"layer": unit(i)[0], "neuron": unit(i)[1], "c": float(flat[i]),
                                   "dact_mean": float(acc_a[unit(i)[0]][unit(i)[1]] / n_pos)} for i in order[:1024].tolist()]
        ref = sum(arm_vecs["NEURON_64"].values()).norm() if "NEURON_64" in arm_vecs else None
        for s in rand_seeds:
            rng = np.random.default_rng(s)
            idx = rng.choice(len(flat), size=64, replace=False).tolist()
            v = neuron_vectors(idx)
            tot = sum(v.values())
            scale = float(ref / tot.norm()) if ref is not None else 1.0
            arm_vecs[f"RANDOM_64_s{s}"] = {L: scale * x for L, x in v.items()}
            arm_meta[f"RANDOM_64_s{s}"] = {"seed": s, "scale": scale, "direct_projection": float(flat[idx].sum()),
                                          "cos_sum_u": float(tot @ u.cpu() / tot.norm()),
                                          "neurons": [list(unit(i)) for i in idx]}
        def head_vectors(flat_idx):
            """Per-layer sums of the chosen heads' donor-mean post-norm contribution vectors."""
            out = {}
            for i in flat_idx:
                L, h = src_layers[int(i) // n_heads], int(i) % n_heads
                lay = layers[L]
                sl = slice(h * head_dim, (h + 1) * head_dim)
                wo = lay.self_attn.o_proj.weight.detach().float()[:, sl] * norm_gain(L, "post_attention_layernorm")[:, None]
                v = (wo @ (acc_z[L].float()[sl] / n_pos).to(wo.device)).cpu()
                out[L] = out.get(L, 0) + v
            return out

        h_idx = h_order[:8].tolist()
        hv = head_vectors(h_idx)
        h_proj = float(head_flat[h_order[:8]].sum())
        h_alpha = rung * N / h_proj
        arm_vecs["HEAD_8"] = {L: h_alpha * x for L, x in hv.items()}
        h_tot = sum(arm_vecs["HEAD_8"].values())
        arm_meta["HEAD_8"] = {"alpha": h_alpha, "direct_projection": h_proj, "summed_norm": float(h_tot.norm()),
                              "cos_sum_u": float(h_tot @ u.cpu() / h_tot.norm()),
                              "heads": [[src_layers[int(i) // n_heads], int(i) % n_heads] for i in h_idx]}
        for s in int_list(os.environ.get("NX_HEAD_SEEDS", "20261021,20261022")):
            idx = np.random.default_rng(s).choice(len(head_flat), size=8, replace=False).tolist()
            v = head_vectors(idx)
            tot = sum(v.values())
            scale = float(h_tot.norm() / tot.norm())
            arm_vecs[f"RANDOM_HEAD_8_s{s}"] = {L: scale * x for L, x in v.items()}
            arm_meta[f"RANDOM_HEAD_8_s{s}"] = {"seed": s, "scale": scale, "direct_projection": float(head_flat[idx].sum()),
                                              "cos_sum_u": float(tot @ u.cpu() / tot.norm()),
                                              "heads": [[src_layers[int(i) // n_heads], int(i) % n_heads] for i in idx]}
        if "NEURON_64" in arm_vecs:
            n64 = sum(arm_vecs["NEURON_64"].values())
            arm_vecs["NEURON_64_L30"] = {write_layer: n64}
            arm_meta["NEURON_64_L30"] = {"summed_norm": float(n64.norm()), "cos_sum_u": float(n64 @ u.cpu() / n64.norm())}
        u_nm = u.clone()
        u_nm[massive] = 0
        u_nm = u_nm / u_nm.norm()
        arm_vecs["U"] = {write_layer: u_write.cpu()}
        arm_vecs["U_NOMASSIVE"] = {write_layer: (u_nm * rung * N).cpu()}
        summary["arms"] = arm_meta
        nongold_arms = {"NEURON_64", "HEAD_8"} & set(arm_vecs)
        np.savez_compressed(os.path.join(outdir, "nx_source.npz"), u=u.cpu().numpy(), flat_c=flat.numpy().astype(np.float32),
                            head_c=head_flat.numpy().astype(np.float32), n_inter=np.array([n_inter]), src_layers=np.array(src_layers))

        def hooks_for(arm, positions):
            hs = []
            for L, v in sorted(arm_vecs[arm].items()):
                hs.append(("fwd", layers[L], ResidualAdd(v, positions)))
            return hs
        arm_names = ["U", "U_NOMASSIVE"] + [f"NEURON_{k}" for k in ks] + [f"RANDOM_64_s{s}" for s in rand_seeds]
        if os.environ.get("NX_ARMS"):
            arm_names = [a for a in os.environ["NX_ARMS"].split(",") if a]
            assert all(a in arm_vecs for a in arm_names), f"unknown arm in NX_ARMS: {arm_names}"
    else:
        # ---- read stage: rank downstream neurons at the written positions on donors ----
        acc_r = {L: torch.zeros(n_inter, dtype=torch.float64, device=dev) for L in read_layers}
        npos_r = 0
        for did in donor_rows_used:
            r = frame[did]
            text_s, ids_s, off_s = render(std_closed_prompts(r)["std"])
            pos = mention_positions(text_s, off_s, r["answer"])
            st0, st1 = {}, {}
            run(ids_s, capture_hooks(st0, pos, read_layers))
            w = ResidualAdd(u_write, pos)
            run(ids_s, [("fwd", layers[write_layer], w)] + capture_hooks(st1, pos, read_layers))
            assert w.ok(), f"{did}: donor write did not fire"
            for L in read_layers:
                acc_r[L] += (st1[("a", L)] * st1[("s", L)][:, None] - st0[("a", L)] * st0[("s", L)][:, None]).double().sum(0)
            npos_r += len(pos)
        scores = []
        for L in read_layers:
            lay = layers[L]
            colnorm = (lay.mlp.down_proj.weight.detach().float() * norm_gain(L, "post_feedforward_layernorm")[:, None]).norm(dim=0)
            scores.append(((acc_r[L].float() / npos_r).abs() * colnorm.to(acc_r[L].device)).cpu())
        flat_r = torch.cat(scores)
        order_r = torch.argsort(flat_r, descending=True)

        def unit_r(i):
            return read_layers[int(i) // n_inter], int(i) % n_inter
        top = [unit_r(i) for i in order_r[:clamp_k].tolist()]
        rnd = [unit_r(i) for i in np.random.default_rng(clamp_seed).choice(len(flat_r), size=clamp_k, replace=False).tolist()]
        summary["read_ranking"] = {"n_positions": npos_r, "top": [{"layer": L, "neuron": j, "score": float(flat_r[order_r[n]]),
                                                                    "dact_mean": float(acc_r[L][j] / npos_r)}
                                                                   for n, (L, j) in enumerate([unit_r(i) for i in order_r[:256].tolist()])],
                                   "clamp_top": [list(x) for x in top], "clamp_rand": [list(x) for x in rnd],
                                   "top_share_of_score": float(flat_r[order_r[:clamp_k]].sum() / flat_r.sum())}
        arm_names = ["U", "FREEZE_MLP", "CLAMP_TOP", "CLAMP_RAND"]
        nongold_arms = set()

    # ---- test rows ----
    fout = open(os.path.join(outdir, f"nx_{stage}.jsonl"), "w")
    audit = {"n_arm_rows": 0, "n_invalid": 0}
    for ri, rid in enumerate(row_ids):
        r = frame[rid]
        gold = r["answer"]
        text_s, ids_s, off_s = render(std_closed_prompts(r)["std"])
        row_seed = seed0 + ri * 10007
        cands_pos = {c: mention_positions(text_s, off_s, c) for c in r["candidates"]}
        cands_pos = {c: p for c, p in cands_pos.items() if p}
        assert gold in cands_pos, f"{rid}: gold has no mentions"
        rng = np.random.default_rng(row_seed)
        ng_all = sorted(c for c in cands_pos if c != gold)
        nongold = [ng_all[i] for i in rng.choice(len(ng_all), size=min(n_nongold, len(ng_all)), replace=False)]

        def emit(arm, fired, outputs, hook_objs):
            valid = all(h.ok() for h in hook_objs)
            audit["n_arm_rows"] += 1
            if not valid:
                audit["n_invalid"] += 1
                raise RuntimeError(f"{rid}/{arm}/{fired}: a hook did not fire inside generate — execution-invalid")
            for s, o in enumerate(outputs):
                fout.write(json.dumps({"id": rid, "row_ordinal": ri, "stage": stage, "condition": arm, "fired_candidate": fired,
                                       "fired_is_gold": None if fired is None else fired == gold, "sample_index": s, "model_output": o,
                                       "normalized_output": normalize_answer(o), "correct": normalize_answer(o) == normalize_answer(gold),
                                       "answers_fired": None if fired is None else normalize_answer(o) == normalize_answer(fired),
                                       "n_fired_positions": None if fired is None else len(cands_pos[fired]),
                                       "hook_prefill_calls": [h.prefill_calls for h in hook_objs],
                                       "hook_positions_touched": [h.positions_touched for h in hook_objs]}, ensure_ascii=False) + "\n")

        base = []
        for c0 in range(0, k_base, k_write):
            base += generate(ids_s, min(k_write, k_base - c0), row_seed + c0)
        emit("baseline", None, base, [])

        gpos = cands_pos[gold]
        if stage == "read":
            st0 = {}
            run(ids_s, capture_hooks(st0, gpos, read_layers) + [("fwd", layers[L].mlp, (lambda L: (lambda _m, _i, out: st0.__setitem__(("m", L), out[0, gpos].detach())))(L)) for L in read_layers])

        for ai, arm in enumerate(arm_names):
            targets = [gold] + (nongold if arm in nongold_arms else [])
            for ci, cand in enumerate(targets):
                pos = cands_pos[cand]
                if stage == "source":
                    hooks = hooks_for(arm, pos)
                else:
                    hooks = [("fwd", layers[write_layer], ResidualAdd(u_write, pos))]
                    if arm == "FREEZE_MLP":
                        hooks += [("fwd", layers[L].mlp, OutputClamp(st0[("m", L)], pos)) for L in read_layers]
                    elif arm in ("CLAMP_TOP", "CLAMP_RAND"):
                        chosen = top if arm == "CLAMP_TOP" else rnd
                        for L in read_layers:
                            js = [j for (LL, j) in chosen if LL == L]
                            if js:
                                stored = st0[("a", L)][:, js]
                                hooks.append(("pre", layers[L].mlp.down_proj, NeuronClamp(torch.tensor(js), stored, pos)))
                outs = generate(ids_s, k_write, row_seed + 1000 + ai * 100 + ci * 10, hooks)
                emit(arm, cand, outs, [h for _, _, h in hooks])
        fout.flush()
        print(f"row {ri+1}/{len(row_ids)} {rid}: {len(cands_pos)} cands, {len(ids_s)} tokens, gold pos {len(gpos)} ({time.time()-t0:.0f}s)", flush=True)
    fout.close()
    summary["audit"] = audit
    summary["n_rows"] = len(row_ids)
    summary["seconds"] = round(time.time() - t0)
    with open(os.path.join(outdir, f"nx_{stage}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    rp = os.environ.get("GMN_RESULT_PATH")
    if rp:
        small = {k: v for k, v in summary.items() if k in ("stage", "wx_job", "frozen", "audit", "n_rows", "seconds")}
        if "decomposition" in summary:
            small["decomposition"] = {k: v for k, v in summary["decomposition"].items() if k not in ("per_layer_mlp", "per_layer_heads", "top_heads")}
        with open(rp, "w") as f:
            json.dump(small, f)
    print(json.dumps({k: v for k, v in summary.items() if k in ("stage", "frozen", "audit", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
