# -*- coding: utf-8 -*-
"""Generate publication-quality figures for 22-05-26 experiments."""
from __future__ import annotations

import json, os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = str(Path(__file__).resolve().parent.parent)
OUT = os.path.join(_ROOT, "22-05-26", "output")

BLUE = "#2196F3"
GREEN = "#4CAF50"
ORANGE = "#FF9800"
RED = "#E91E63"
PURPLE = "#9C27B0"
GREY = "#607D8B"


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


# ═══ Figure 1: Fair-capital CVaR comparison ═══════════════════════════

def fig1_fair_capital():
    data = load("exp_a_fair_capital.json")
    dims = sorted(data.keys(), key=int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: own V₀ (biased)
    x = np.arange(len(dims))
    w = 0.25
    for i, algo in enumerate(["LSM", "RLSM", "NLSM"]):
        vals = [data[d][algo]["cvar95_own_v0"] for d in dims]
        color = [BLUE, GREEN, ORANGE][i]
        ax1.bar(x + (i - 1) * w, vals, w, label=algo, color=color, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["d={}".format(d) for d in dims])
    ax1.set_ylabel("CVaR₉₅")
    ax1.set_title("(a) Each model uses own V₀ (biased)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Right: fair V₀ (unbiased)
    for i, algo in enumerate(["LSM", "RLSM", "NLSM"]):
        vals = [data[d][algo]["cvar95_fair"] for d in dims]
        color = [BLUE, GREEN, ORANGE][i]
        ax2.bar(x + (i - 1) * w, vals, w, label=algo, color=color, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["d={}".format(d) for d in dims])
    ax2.set_ylabel("CVaR₉₅")
    ax2.set_title("(b) Same V₀ for all (fair comparison)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Highlight best
    for d_idx, d in enumerate(dims):
        vals = {a: data[d][a]["cvar95_fair"] for a in ["LSM","RLSM","NLSM"]}
        best = min(vals, key=vals.get)
        if best != "LSM":
            ax2.annotate("✓", xy=(d_idx + (["LSM","RLSM","NLSM"].index(best)-1)*w,
                                  vals[best]),
                        ha='center', va='bottom', fontsize=14, color='green',
                        fontweight='bold')

    plt.suptitle("Experiment A: Fair-Capital Hedge Comparison on BTC", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig1_fair_capital.png"), dpi=150)
    plt.close()
    print("  fig1_fair_capital.png")


# ═══ Figure 2: Bias decomposition ════════════════════════════════════

def fig2_bias_decomposition():
    data = load("exp_a_fair_capital.json")
    dims = sorted(data.keys(), key=int)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for algo, color, marker in [("LSM", BLUE, "s"), ("RLSM", GREEN, "o"),
                                 ("NLSM", ORANGE, "^")]:
        own = [data[d][algo]["cvar95_own_v0"] for d in dims]
        fair = [data[d][algo]["cvar95_fair"] for d in dims]
        ds = [int(d) for d in dims]
        ax.plot(ds, own, marker + "--", color=color, alpha=0.5, linewidth=1.5,
                label="{} (own V₀)".format(algo))
        ax.plot(ds, fair, marker + "-", color=color, linewidth=2.5,
                label="{} (fair V₀)".format(algo))

    ax.set_xlabel("Dimension d", fontsize=12)
    ax.set_ylabel("CVaR₉₅", fontsize=12)
    ax.set_title("How Initial Capital Bias Masks NN Advantage", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig2_bias_decomposition.png"), dpi=150)
    plt.close()
    print("  fig2_bias_decomposition.png")


# ═══ Figure 3: NLSM tuning effect ═══════════════════════════════════

def fig3_nlsm_tuning():
    data = load("exp_b_nlsm_tuned.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, d in enumerate(["5", "10"]):
        ax = [ax1, ax2][idx]
        dd = data[d]
        labels = sorted(dd.keys())
        cvars = [dd[l]["cvar95"] for l in labels]
        colors_map = {
            "LSM-128": BLUE,
            "RLSM-128": GREEN, "RLSM-256": "#81C784",
            "NLSM-128-ep50": "#FFE0B2", "NLSM-128-ep200": ORANGE,
            "NLSM-256-ep200": RED,
        }
        cs = [colors_map.get(l, GREY) for l in labels]

        bars = ax.barh(range(len(labels)), cvars, color=cs, alpha=0.85,
                       edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("CVaR₉₅ (lower = better hedge)")
        ax.set_title("d = {}".format(d))
        ax.grid(axis="x", alpha=0.3)

        # Annotate values
        for bar, val in zip(bars, cvars):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                   "{:.1f}".format(val), va="center", fontsize=9)

    plt.suptitle("Experiment B: Effect of NLSM Tuning on Hedge Quality", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig3_nlsm_tuning.png"), dpi=150)
    plt.close()
    print("  fig3_nlsm_tuning.png")


# ═══ Figure 4: Delta speed vs CVaR scatter ══════════════════════════

def fig4_speed_vs_quality():
    data = load("exp_b_nlsm_tuned.json")

    fig, ax = plt.subplots(figsize=(10, 6))

    markers = {"LSM-128": ("s", BLUE), "RLSM-128": ("o", GREEN),
               "RLSM-256": ("D", "#81C784"),
               "NLSM-128-ep50": ("v", "#FFB74D"),
               "NLSM-128-ep200": ("^", ORANGE),
               "NLSM-256-ep200": ("*", RED)}

    for d in ["5", "10"]:
        for label, (marker, color) in markers.items():
            if label not in data[d]:
                continue
            r = data[d][label]
            sz = 120 if d == "10" else 80
            alpha = 0.9 if d == "10" else 0.6
            ax.scatter(r["delta_us"] / 1000, r["cvar95"],
                      marker=marker, color=color, s=sz, alpha=alpha,
                      edgecolors="black", linewidths=0.5,
                      label="{} d={}".format(label, d) if d == "10" else "")

    ax.set_xlabel("Delta computation time (ms)", fontsize=12)
    ax.set_ylabel("CVaR₉₅ (lower = better hedge)", fontsize=12)
    ax.set_title("Trade-off: Delta Speed vs Hedge Quality (d=10)", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # Ideal corner
    ax.annotate("← Better", xy=(0.5, 10), fontsize=11, color="green",
               fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig4_speed_vs_quality.png"), dpi=150)
    plt.close()
    print("  fig4_speed_vs_quality.png")


# ═══ Figure 5: Rebalancing frequency ════════════════════════════════

def fig5_rebalance():
    data = load("exp_c_rebalance.json")

    freqs_order = ["daily", "8h", "4h"]
    available = [f for f in freqs_order if f in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    x = np.arange(len(available))
    w = 0.25
    for i, algo in enumerate(["LSM", "RLSM", "NLSM"]):
        color = [BLUE, GREEN, ORANGE][i]
        cvars = [data[f][algo]["cvar95"] for f in available]
        means = [data[f][algo]["mean"] for f in available]
        ax1.bar(x + (i - 1) * w, cvars, w, label=algo, color=color, alpha=0.85)
        ax2.bar(x + (i - 1) * w, means, w, label=algo, color=color, alpha=0.85)

    for ax, title, ylabel in [(ax1, "CVaR₉₅ vs Rebalancing Frequency", "CVaR₉₅"),
                               (ax2, "Mean Shortfall vs Rebalancing Frequency",
                                "Mean Shortfall")]:
        ax.set_xticks(x)
        ax.set_xticklabels(available)
        ax.set_xlabel("Rebalancing Frequency")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Experiment C: Impact of Rebalancing Frequency (d=5, BTC)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig5_rebalance.png"), dpi=150)
    plt.close()
    print("  fig5_rebalance.png")


# ═══ Figure 6: Hedge error distributions ════════════════════════════

def fig6_distributions():
    data = load("exp_a_fair_capital.json")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, d in enumerate(["1", "5", "10", "15"]):
        ax = axes[idx // 2][idx % 2]
        for algo, color in [("LSM", BLUE), ("RLSM", GREEN), ("NLSM", ORANGE)]:
            losses = data[d][algo]["losses_fair"]
            ax.hist(losses, bins=20, alpha=0.5, color=color, label=algo,
                    edgecolor="white", linewidth=0.5)
            # VaR line
            var95 = np.percentile(losses, 95)
            ax.axvline(var95, color=color, linestyle="--", alpha=0.7, linewidth=1.5)

        ax.set_xlabel("Max Discounted Shortfall")
        ax.set_ylabel("Count")
        ax.set_title("d = {} (fair V₀)".format(d))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

    plt.suptitle("Hedge Error Distributions (Fair Capital)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig6_distributions.png"), dpi=150)
    plt.close()
    print("  fig6_distributions.png")


# ═══ Figure 7: Summary comparison table as figure ═══════════════════

def fig7_summary_table():
    data_a = load("exp_a_fair_capital.json")
    data_b = load("exp_b_nlsm_tuned.json")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    rows = [
        ["d=5",
         "{:.1f}".format(data_a["5"]["LSM"]["cvar95_fair"]),
         "{:.1f}".format(data_a["5"]["RLSM"]["cvar95_fair"]),
         "{:.1f}".format(data_a["5"]["NLSM"]["cvar95_fair"]),
         "{:.1f}".format(data_b["5"]["NLSM-128-ep200"]["cvar95"]),
         "{:.1f}".format(data_b["5"]["NLSM-256-ep200"]["cvar95"])],
        ["d=10",
         "{:.1f}".format(data_a["10"]["LSM"]["cvar95_fair"]),
         "{:.1f}".format(data_a["10"]["RLSM"]["cvar95_fair"]),
         "{:.1f}".format(data_a["10"]["NLSM"]["cvar95_fair"]),
         "{:.1f}".format(data_b["10"]["NLSM-128-ep200"]["cvar95"]),
         "{:.1f}".format(data_b["10"]["NLSM-256-ep200"]["cvar95"])],
    ]

    cols = ["Dim", "LSM", "RLSM", "NLSM\n(50 ep)", "NLSM\n(200 ep)", "NLSM-256\n(200 ep)"]
    table = ax.table(cellText=rows, colLabels=cols, loc="center",
                     cellLoc="center", colColours=["#E3F2FD"]*6)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Color best cells
    for i in range(len(rows)):
        vals = [float(v) for v in rows[i][1:]]
        best_j = vals.index(min(vals)) + 1
        table[i + 1, best_j].set_facecolor("#C8E6C9")

    ax.set_title("CVaR₉₅ Hedge Comparison (Fair Capital, BTC Historical)", fontsize=13,
                pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig7_summary_table.png"), dpi=150)
    plt.close()
    print("  fig7_summary_table.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Generating figures for 22-05-26 ...")
    fig1_fair_capital()
    fig2_bias_decomposition()
    fig3_nlsm_tuning()
    fig4_speed_vs_quality()
    fig5_rebalance()
    fig6_distributions()
    fig7_summary_table()
    print("All figures generated!")


if __name__ == "__main__":
    main()
