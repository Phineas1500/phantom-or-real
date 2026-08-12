#!/usr/bin/env python3
"""Paper figures 1 and 4 for gauge_not_lever.tex.

Every number is transcribed from a pooled verdict doc, cited per block;
nothing is recomputed here. Outputs paper/fig_erasure.pdf and
paper/fig_fseries.pdf.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2563ab"    # treatment / real-vector arms
RED = "#d23f3f"     # destructive / content-destroying arms
AMBER = "#d99114"   # intermediate arms
SLATE = "#75879c"   # matched controls (deliberately recessive)
INK = "#1a2433"
MUTED = "#5b6b7f"
GRID = "#e3e9f1"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
    }
)


def whisker(ax, x, lo, hi, y, color):
    ax.plot([lo, hi], [y, y], color=color, lw=1.1, zorder=3)
    for v in (lo, hi):
        ax.plot([v, v], [y - 0.13, y + 0.13], color=color, lw=1.1, zorder=3)


def fig_erasure() -> None:
    # docs/readable_stack_erasure_27b_property_pooled_summary.md (item D):
    # erase_raw            -0.016 [-0.094, +0.070]
    # erase_readable_stack +0.047 [-0.070, +0.203]
    # random family        -0.380 [-0.586, -0.182]; draws -0.383/-0.375/-0.383
    rows = [
        ("erase raw correctness axis", -0.016, -0.094, 0.070, SLATE),
        ("erase readable stack\n(rank 9, 5 layers, all tokens)", 0.047, -0.070, 0.203, BLUE),
        ("erase matched-rank\nrandom stacks (3 draws)", -0.380, -0.586, -0.182, RED),
    ]
    draws = [-0.383, -0.375, -0.383]

    fig = plt.figure(figsize=(5.5, 2.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.0], wspace=0.42)
    ax = fig.add_subplot(gs[0])
    for i, (label, point, lo, hi, color) in enumerate(rows):
        y = len(rows) - 1 - i
        whisker(ax, None, lo, hi, y, color)
        ax.plot(point, y, "o", ms=5, color=color, zorder=4)
        ax.text(hi + 0.025, y, f"{point:+.3f}", va="center", fontsize=7.5, color=color)
    ax.plot(draws, [0.22] * 3, "o", ms=3, mfc="none", mec=RED, mew=0.9, zorder=4)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.5)
    ax.set_xlim(-0.68, 0.42)
    ax.set_ylim(-0.5, len(rows) - 0.4)
    ax.set_xlabel(r"$\Delta P(\mathrm{strong})$ vs baseline (95% CI)", fontsize=7.5)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # Inset: within-run projection variance exemplar at L15 (claims table row 2 /
    # docs/next_paper_claims_table.md: 10.65 vs 818/382 sd^2); pooled summary:
    # readable stack carries 10-1300x less across layers/stacks.
    ax2 = fig.add_subplot(gs[1])
    vals = [10.65, 818.0, 382.0]
    colors = [BLUE, RED, RED]
    bars = ax2.bar(range(3), vals, width=0.62, color=colors)
    bars[2].set_alpha(0.65)
    ax2.set_yscale("log")
    ax2.set_ylim(3, 3000)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["readable", "rand d1", "rand d2"], fontsize=6.8, rotation=20)
    ax2.set_ylabel("within-run projection\nvariance at L15 (sd$^2$)", fontsize=6.8)
    ax2.text(0.02, 0.965, "ratio spans 10--1300$\\times$\nacross layers/stacks", transform=ax2.transAxes,
             fontsize=6.4, va="top", color=MUTED)
    ax2.grid(axis="y", color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(axis="y", labelsize=6.5)

    fig.savefig("paper/fig_erasure.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_fseries() -> None:
    fig = plt.figure(figsize=(5.5, 2.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.30)

    # Left: one vector, five fates (26 failing rows, baseline 0.120).
    # docs/classmean_repair_27b_property_pooled_summary.md (F(ii)):
    #   raw +0.043 [-0.120,+0.207]; projected +0.341 [+0.202,+0.495]
    # docs/classmean_b_controls_27b_property_pooled_summary.md (F(ii)-b):
    #   shuffled family -0.043 [-0.103,+0.006]; sign-flip -0.120 [-0.245,-0.024];
    #   donor-free +0.399 [+0.245,+0.558]
    # Baselines (ledger 1.5): self-consistency 0.000 abs (dP -0.120);
    #   best-of-8 0.192 abs (dP +0.072).
    ax = fig.add_subplot(gs[0])
    arms = [
        ("raw", 0.043, -0.120, 0.207, BLUE),
        ("proj-\nected", 0.341, 0.202, 0.495, BLUE),
        ("shuffled\nlabels", -0.043, -0.103, 0.006, SLATE),
        ("sign-\nflip", -0.120, -0.245, -0.024, RED),
        ("donor-\nfree", 0.399, 0.245, 0.558, BLUE),
    ]
    for i, (label, point, lo, hi, color) in enumerate(arms):
        ax.bar(i, point, width=0.62, color=color, zorder=2,
               alpha=1.0 if color != BLUE or i != 0 else 0.55)
        ax.plot([i, i], [lo, hi], color=INK, lw=0.9, zorder=3)
        ax.plot([i - 0.10, i + 0.10], [lo, lo], color=INK, lw=0.9, zorder=3)
        ax.plot([i - 0.10, i + 0.10], [hi, hi], color=INK, lw=0.9, zorder=3)
    ax.axhline(0, color=INK, lw=0.9)
    ax.axhline(0.072, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.text(2.5, 0.098, "best-of-8 ceiling", fontsize=6.2, color=MUTED, ha="center")
    ax.axhline(-0.120, color=MUTED, lw=0.8, ls=(0, (1, 2)))
    ax.text(1.0, -0.185, "self-consistency (abs 0.000)", fontsize=6.2, color=MUTED, ha="center")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0] for a in arms], fontsize=6.6)
    ax.set_ylim(-0.30, 0.62)
    ax.set_ylabel(r"$\Delta P(\mathrm{strong})$ vs baseline 0.120", fontsize=7.2)
    ax.set_title("One vector, five fates (failing rows)", fontsize=7.8, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Right: the removal family on naturally-correct rows (46 rows, baseline
    # 0.764). docs/necessity_27b_property_pooled.json and
    # docs/necessity_prime_27b_property_pooled.json:
    #   ablate rank-8   -0.666 [-0.769,-0.557]
    #   rand-8 family   -0.072 [-0.124,-0.024]
    #   permuted basis  -0.038 [-0.079,-0.003]
    #   mean-ablate     -0.764 [-0.851,-0.668]
    #   state-PCA top-8 -0.764 [-0.848,-0.671]
    #   keep-only-8     -0.764 [-0.851,-0.671]
    #   shrink a=0.12   -0.103 [-0.158,-0.054]
    ax2 = fig.add_subplot(gs[1])
    removals = [
        ("ablate\nrank-8", -0.666, -0.769, -0.557, RED),
        ("random\n8-dim", -0.072, -0.124, -0.024, SLATE),
        ("perm.\n8-dim", -0.038, -0.079, -0.003, SLATE),
        ("mean-\nablate", -0.764, -0.851, -0.668, RED),
        ("state-\nPCA-8", -0.764, -0.848, -0.671, RED),
        ("keep-\nonly-8", -0.764, -0.851, -0.671, RED),
        ("shrink\n$\\alpha$=.12", -0.103, -0.158, -0.054, AMBER),
    ]
    for i, (label, point, lo, hi, color) in enumerate(removals):
        ax2.bar(i, point, width=0.62, color=color, zorder=2)
        ax2.plot([i, i], [lo, hi], color=INK, lw=0.9, zorder=3)
        ax2.plot([i - 0.10, i + 0.10], [lo, lo], color=INK, lw=0.9, zorder=3)
        ax2.plot([i - 0.10, i + 0.10], [hi, hi], color=INK, lw=0.9, zorder=3)
    ax2.axhline(0, color=INK, lw=0.9)
    ax2.axhline(-0.764, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    ax2.text(1.55, -0.812, "floor (zero survivors)", fontsize=6.2, color=MUTED, ha="center")
    ax2.set_xticks(range(len(removals)))
    ax2.set_xticklabels([r[0] for r in removals], fontsize=6.4)
    ax2.set_ylim(-0.90, 0.10)
    ax2.set_ylabel(r"$\Delta P(\mathrm{strong})$ vs baseline 0.764", fontsize=7.2)
    ax2.set_title("The removal family (naturally-correct rows)", fontsize=7.8, color=INK)
    ax2.grid(axis="y", color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.savefig("paper/fig_fseries.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_erasure()
    fig_fseries()
    print("wrote paper/fig_erasure.pdf, paper/fig_fseries.pdf")
