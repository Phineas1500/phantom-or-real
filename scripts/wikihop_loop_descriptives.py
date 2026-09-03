#!/usr/bin/env python3
"""Descriptive (post-hoc, unregistered) analyses of the frozen-write +
output-first loop on data already on disk.

Part A — abstention rule sweep on the WD/WO rows: for each rule (top
answers-fired rate unique and >= t; or top minus runner-up >= m) the
yield on the doc-dependent strata, the collateral on correct rows, and a
frame-level net (pool shares of the WO frame).

Part B — where the selector loses to the oracle: per row, was the gold
branch accepted at the top, beaten by a wrong branch, or never fired at
all; and among wrong selections, how often the chosen candidate is a
string cousin of the gold answer (containment / token overlap on the
original strings, pseudonyms mapped back), against the cousin base rate
among all non-gold candidates of the same rows."""
from __future__ import annotations
import argparse, gzip, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikihop_w1_gates import boot_ci  # noqa: E402
from wikihop_common import normalize_answer  # noqa: E402

RS = Path("results/loop_screen")
STOP = {"the", "of", "and", "de", "la", "le", "del", "da", "di", "in", "on", "a", "an", "du", "des", "el", "los", "las"}


def load(files):
    return [json.loads(l) for f in files for l in open(f)]


def frame(path):
    return {r["id"]: r for r in (json.loads(l) for l in gzip.open(path, "rt"))}


def rows_from(recs, rung=2.0, tie_key="second_L38"):
    out = {}
    for i in sorted({r["id"] for r in recs}):
        brec = [r for r in recs if r["id"] == i and r["condition"] == "baseline"]
        base = [r["correct"] for r in brec]
        base_modal = Counter(r["normalized_output"] for r in brec).most_common(1)[0][0]
        base_gauge = brec[0].get("base_gauge_scores") or {}
        br = defaultdict(list)
        for r in recs:
            if r["id"] == i and r["condition"] == "delta_write" and r["rung"] == rung:
                br[r["fired_candidate"]].append(r)
        cands = {}
        for c, xs in br.items():
            g = (xs[0].get("gauge_scores") or {}).get(tie_key, xs[0]["gauge_score"])
            cands[c] = {"af": float(np.mean([x["answers_fired"] for x in xs])), "corr": float(np.mean([x["correct"] for x in xs])),
                        "gauge": float(g), "gold": bool(xs[0]["fired_is_gold"])}
        out[i] = {"base": float(np.mean(base)), "cands": cands, "base_modal": base_modal, "base_gauge": base_gauge}
    return out


def select(cands):
    return max(cands, key=lambda c: (cands[c]["af"], cands[c]["gauge"]))


def rule_answer(row, kind, thr):
    cands = row["cands"]; best = select(cands)
    afs = sorted((v["af"] for v in cands.values()), reverse=True)
    top = afs[0]; second = afs[1] if len(afs) > 1 else 0.0
    uniq = sum(1 for v in cands.values() if v["af"] == top) == 1
    if kind == "always":
        fire = True
    elif kind == "unique_top":
        fire = uniq and top >= thr
    elif kind == "margin":
        fire = (top - second) >= thr and top > 0
    else:
        raise ValueError(kind)
    return (cands[best]["corr"] if fire else row["base"]), (not fire)


def part_a(pins):
    R = rows_from(load([RS / "wikihop_wo_a.jsonl", RS / "wikihop_wo_b.jsonl"]))
    U = rows_from(load([RS / "wikihop_wd_y1.jsonl", RS / "wikihop_wd_y2.jsonl"]))
    C = rows_from(load([RS / "wikihop_wd_c.jsonl"]))
    w_rep, w_unrep = pins["pool"]["weight_repairable"], pins["pool"]["weight_unrepairable"]
    n_dd, n_cor, n_tot = pins["pool"]["n_doc_dependent"], pins["pool"]["n_correct_majority"], 507
    rng = np.random.default_rng(20260847)
    rules = [("always", 0.0)] + [("unique_top", t) for t in (0.25, 0.5, 0.75, 1.0)] + [("margin", m) for m in (0.25, 0.5, 0.75, 1.0)]
    table = []
    for kind, thr in rules:
        def stratum(d):
            vals = [rule_answer(d[i], kind, thr) for i in sorted(d)]
            dP = [v[0] - d[i]["base"] for v, i in zip(vals, sorted(d))]
            return np.array(dP), float(np.mean([v[1] for v in vals]))
        d_rep, ab_rep = stratum(R); d_unrep, ab_unrep = stratum(U); d_col, ab_col = stratum(C)
        draws = []
        for _ in range(4000):
            draws.append(w_rep * d_rep[rng.integers(0, len(d_rep), len(d_rep))].mean() + w_unrep * d_unrep[rng.integers(0, len(d_unrep), len(d_unrep))].mean())
        y = float(w_rep * d_rep.mean() + w_unrep * d_unrep.mean()); yci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
        col = float(d_col.mean()); cci = boot_ci(list(d_col))
        table.append({"rule": kind, "threshold": thr, "yield_weighted": y, "yield_ci": yci,
                      "repairable_dP": float(d_rep.mean()), "repairable_abstained": ab_rep,
                      "unrepairable_dP": float(d_unrep.mean()), "unrepairable_abstained": ab_unrep,
                      "collateral_dP": col, "collateral_ci": cci, "collateral_abstained": ab_col,
                      "frame_net": float(n_dd / n_tot * y + n_cor / n_tot * col)})
    return table


def toks(s):
    return {t for t in re.findall(r"[a-z0-9]+", s) if t not in STOP}


def cousin(a, b):
    a, b = normalize_answer(a), normalize_answer(b)
    if not a or not b or a == b:
        return "same" if a == b else "none"
    if a in b or b in a:
        return "strict"
    ta, tb = toks(a), toks(b)
    if ta and tb:
        j = len(ta & tb) / len(ta | tb)
        if j >= 0.5:
            return "strict"
        if ta & tb:
            return "loose"
    return "none"


def part_b():
    settings = [
        ("WX real text (59, frozen)", [RS / "wikihop_wx_a.jsonl", RS / "wikihop_wx_b.jsonl"], RS / "wikihop_fresh_input.jsonl.gz"),
        ("WA rider anonymized (60, frozen)", [RS / "wikihop_wa_frozen_a.jsonl", RS / "wikihop_wa_frozen_b.jsonl"], RS / "wikihop_anon_input.jsonl.gz"),
        ("WO anonymized (47, frozen)", [RS / "wikihop_wo_a.jsonl", RS / "wikihop_wo_b.jsonl"], RS / "wikihop_anon2_input.jsonl.gz"),
        ("WD unrepairable (100, frozen)", [RS / "wikihop_wd_y1.jsonl", RS / "wikihop_wd_y2.jsonl"], RS / "wikihop_anon2_input.jsonl.gz"),
    ]
    out = {}
    for name, files, fpath in settings:
        rows = rows_from(load(files)); fr = frame(fpath)
        cat = defaultdict(int); gap = defaultdict(float); n = 0
        wrong_sel = []; nongold_all = []; top_wrong = []
        oracle = loop = 0.0
        for i, row in rows.items():
            cands = row["cands"]; gold = [c for c in cands if cands[c]["gold"]]
            if not gold:
                cat["gold_branch_missing"] += 1; continue
            g = gold[0]; n += 1
            sel = select(cands); top = max(v["af"] for v in cands.values())
            oracle += cands[g]["corr"]; loop += cands[sel]["corr"]
            row_gap = cands[g]["corr"] - cands[sel]["corr"]
            if cands[g]["af"] == 0.0:
                k = "gold_never_fired"
            elif cands[g]["af"] < top:
                k = "gold_beaten"
            elif sel == g:
                k = "gold_selected"
            else:
                k = "gold_tied_lost"
            cat[k] += 1; gap[k] += row_gap
            f = fr[i]; rev = {v.lower(): kk for kk, v in (f.get("anon_map") or {}).items()}
            orig = lambda c: rev.get(c, c)
            gold_s = orig(f["answer"])
            for c in cands:
                if c != g:
                    nongold_all.append(cousin(orig(c), gold_s))
                    if cands[c]["af"] >= top and top > 0:
                        top_wrong.append(cousin(orig(c), gold_s))
            if sel != g:
                wrong_sel.append({"id": i, "selected": orig(sel), "gold": gold_s, "kind": cousin(orig(sel), gold_s),
                                  "af_sel": cands[sel]["af"], "af_gold": cands[g]["af"], "sel_corr": cands[sel]["corr"]})
        def rate(xs, lvl):
            return (sum(1 for x in xs if x in lvl) / len(xs)) if xs else None
        out[name] = {"n_rows": n, "oracle": oracle / n, "loop": loop / n, "gap": (oracle - loop) / n,
                     "categories": {k: {"rows": cat[k], "gap_share": (gap[k] / (oracle - loop)) if oracle > loop else None} for k in ("gold_selected", "gold_tied_lost", "gold_beaten", "gold_never_fired")},
                     "cousin": {"wrong_selections": len(wrong_sel),
                                "strict_among_wrong_selections": rate([w["kind"] for w in wrong_sel], {"strict"}),
                                "loose_or_strict_among_wrong_selections": rate([w["kind"] for w in wrong_sel], {"strict", "loose"}),
                                "strict_among_wrong_top_branches": rate(top_wrong, {"strict"}),
                                "loose_or_strict_among_wrong_top_branches": rate(top_wrong, {"strict", "loose"}),
                                "strict_base_rate_nongold": rate(nongold_all, {"strict"}),
                                "loose_or_strict_base_rate_nongold": rate(nongold_all, {"strict", "loose"}),
                                "n_nongold_candidates": len(nongold_all), "n_wrong_top_branches": len(top_wrong)},
                     "wrong_selection_examples": sorted(wrong_sel, key=lambda w: -w["af_sel"])[:12], "wrong_selections_all": wrong_sel}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pins", type=Path, default=Path("docs/wikihop_wd_pinned.json"))
    p.add_argument("--out", type=Path, default=Path("docs/wikihop_loop_descriptives.json"))
    args = p.parse_args()
    pins = json.load(open(args.pins))
    A = part_a(pins); B = part_b()
    args.out.write_text(json.dumps({"abstention_sweep": A, "selector_failures": B}, indent=1) + "\n")
    print("| rule | yield (weighted) | 95% CI | repairable dP / abst | unrepairable dP / abst | collateral | 95% CI | abst | frame net |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t in A:
        print(f"| {t['rule']} ≥ {t['threshold']} | {t['yield_weighted']:+.3f} | [{t['yield_ci'][0]:+.3f}, {t['yield_ci'][1]:+.3f}] | "
              f"{t['repairable_dP']:+.3f} / {t['repairable_abstained']:.0%} | {t['unrepairable_dP']:+.3f} / {t['unrepairable_abstained']:.0%} | "
              f"{t['collateral_dP']:+.3f} | [{t['collateral_ci'][0]:+.3f}, {t['collateral_ci'][1]:+.3f}] | {t['collateral_abstained']:.0%} | {t['frame_net']:+.3f} |")
    print()
    for name, b in B.items():
        c = b["categories"]; k = b["cousin"]
        print(f"## {name}: rows {b['n_rows']}, oracle {b['oracle']:.3f}, loop {b['loop']:.3f}, gap {b['gap']:.3f}")
        for kk, v in c.items():
            gs = f"{v['gap_share']:.0%}" if v["gap_share"] is not None else "—"
            print(f"  {kk:18s} rows {v['rows']:3d}  share of gap {gs}")
        fmt = lambda x: "—" if x is None else f"{x:.0%}"
        print(f"  cousins: wrong selections {k['wrong_selections']}: strict {fmt(k['strict_among_wrong_selections'])}, loose+ {fmt(k['loose_or_strict_among_wrong_selections'])}"
              f" | wrong top branches {k['n_wrong_top_branches']}: strict {fmt(k['strict_among_wrong_top_branches'])}, loose+ {fmt(k['loose_or_strict_among_wrong_top_branches'])}"
              f" | base rate over {k['n_nongold_candidates']} non-gold candidates: strict {fmt(k['strict_base_rate_nongold'])}, loose+ {fmt(k['loose_or_strict_base_rate_nongold'])}")
        for w in b["wrong_selection_examples"][:6]:
            print(f"    {w['id']}: picked '{w['selected']}' (af {w['af_sel']:.2f}) over gold '{w['gold']}' (af {w['af_gold']:.2f}) — {w['kind']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
