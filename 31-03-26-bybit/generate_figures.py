# -*- coding: utf-8 -*-
"""Generate all figures for the BTC experiments."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
_ROOT = str(Path(__file__).resolve().parent.parent)
for p in (_SCRIPT_DIR, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = os.path.join(_ROOT, "31-03-26-bybit", "output")


def _safe_load_json(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def plot_exp3_price_comparison():
    data = _safe_load_json("exp3_details.json")
    if data is None:
        return
    algos = ["LSM", "RLSM", "NLSM"]
    prices = [data[a]["price"] for a in algos]
    fit_times = [data[a]["fit_time"] for a in algos]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    bars1 = ax1.bar(algos, prices, color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Option Price (normalised)")
    ax1.set_title("American Put Price on BTC Historical Paths")
    for bar, v in zip(bars1, prices):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 "{:.3f}".format(v), ha="center", va="bottom", fontsize=10)
    ax1.set_ylim(0, max(prices) * 1.25)

    bars2 = ax2.bar(algos, fit_times, color=colors, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("Fit Time (s)")
    ax2.set_title("Training Time")
    for bar, v in zip(bars2, fit_times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 "{:.2f}s".format(v), ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp3_price_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved exp3_price_comparison.png")


def plot_exp3_hedge_error_distribution():
    data = _safe_load_json("exp3_details.json")
    if data is None:
        return
    algos = ["LSM", "RLSM", "NLSM"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i, algo in enumerate(algos):
        losses = np.array(data[algo]["losses_intr"])
        ax = axes[i]
        ax.hist(losses, bins=30, color=colors[i], edgecolor="black",
                linewidth=0.5, alpha=0.85)
        cvar = data[algo]["var_cvar_intr"]["cvar"]
        var = data[algo]["var_cvar_intr"]["var"]
        ax.axvline(var, color="red", linestyle="--", linewidth=1.5, label="VaR95={:.2f}".format(var))
        ax.axvline(cvar, color="darkred", linestyle="-", linewidth=1.5, label="CVaR95={:.2f}".format(cvar))
        ax.set_title("{} — Hedge Shortfall".format(algo))
        ax.set_xlabel("Max Discounted Shortfall")
        if i == 0:
            ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle("Exp 3: Hedge Error Distribution (vs Intrinsic) — BTC Historical", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "exp3_hedge_error_dist.png"), dpi=150)
    plt.close(fig)
    print("  Saved exp3_hedge_error_dist.png")


def plot_exp3_risk_metrics_bar():
    data = _safe_load_json("exp3_details.json")
    if data is None:
        return
    algos = ["LSM", "RLSM", "NLSM"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    metrics_model = {a: data[a]["var_cvar_model"] for a in algos}
    metrics_intr = {a: data[a]["var_cvar_intr"] for a in algos}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(algos))
    w = 0.35
    var_m = [metrics_model[a]["var"] for a in algos]
    cvar_m = [metrics_model[a]["cvar"] for a in algos]
    ax1.bar(x - w / 2, var_m, w, color=[c + "88" for c in colors], edgecolor="black",
            linewidth=0.5, label="VaR95")
    ax1.bar(x + w / 2, cvar_m, w, color=colors, edgecolor="black",
            linewidth=0.5, label="CVaR95")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos)
    ax1.set_ylabel("Shortfall (norm. units)")
    ax1.set_title("Risk Metrics vs Model Value")
    ax1.legend()

    var_i = [metrics_intr[a]["var"] for a in algos]
    cvar_i = [metrics_intr[a]["cvar"] for a in algos]
    ax2.bar(x - w / 2, var_i, w, color=[c + "88" for c in colors], edgecolor="black",
            linewidth=0.5, label="VaR95")
    ax2.bar(x + w / 2, cvar_i, w, color=colors, edgecolor="black",
            linewidth=0.5, label="CVaR95")
    ax2.set_xticks(x)
    ax2.set_xticklabels(algos)
    ax2.set_ylabel("Shortfall (norm. units)")
    ax2.set_title("Risk Metrics vs Intrinsic Value")
    ax2.legend()

    fig.suptitle("Exp 3: VaR95 / CVaR95 Comparison — BTC Historical", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "exp3_risk_metrics.png"), dpi=150)
    plt.close(fig)
    print("  Saved exp3_risk_metrics.png")


def plot_exp3_summary_table():
    """Render a summary table as an image."""
    data = _safe_load_json("exp3_details.json")
    if data is None:
        return
    algos = ["LSM", "RLSM", "NLSM"]

    rows = []
    for a in algos:
        d = data[a]
        rows.append([
            a,
            "{:.3f}".format(d["price"]),
            "{:.2f}s".format(d["fit_time"]),
            "{:.2f}".format(d["var_cvar_intr"]["var"]),
            "{:.2f}".format(d["var_cvar_intr"]["cvar"]),
            "{:.2f}".format(d["stats_intr"]["mean"]),
            "{:.2f}s".format(d["hedge_time"]),
        ])
    col_labels = ["Algo", "Price", "Fit Time", "VaR95", "CVaR95", "Mean Shortfall", "Hedge Time"]

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#E0E0E0")
            cell.set_text_props(fontweight="bold")
    ax.set_title("Exp 3: Historical Bootstrap Results — BTC American Put (normalised)",
                 fontsize=12, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp3_summary_table.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved exp3_summary_table.png")


def plot_btc_price_context():
    """Plot BTC daily close price with train/test split annotation."""
    from btc_data_loader import load_btc_daily
    from config_btc import DEFAULT as CFG

    csv_path = os.path.join(_ROOT, CFG.csv_path)
    daily = load_btc_daily(csv_path)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily.index, daily.values, color="#333", linewidth=0.7)

    import datetime
    train_end = datetime.datetime(2023, 12, 31)
    ax.axvline(train_end, color="red", linestyle="--", linewidth=1.2,
               label="Train / Test split")
    ax.fill_between(daily.index, daily.values, alpha=0.15,
                    where=daily.index <= train_end, color="#2196F3")
    ax.fill_between(daily.index, daily.values, alpha=0.15,
                    where=daily.index > train_end, color="#FF9800")

    ax.set_ylabel("BTC/USDT close")
    ax.set_title("BTC Daily Close — Train (blue) / Test (orange)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "btc_price_context.png"), dpi=150)
    plt.close(fig)
    print("  Saved btc_price_context.png")


def plot_exp3_augmentation_comparison():
    """Side-by-side: original vs augmented CVaR95."""
    orig = _safe_load_json("exp3_details.json")
    aug = _safe_load_json("exp3_augmented_details.json")
    if orig is None or aug is None:
        return
    algos = ["LSM", "RLSM", "NLSM"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    cvar_orig = [orig[a]["var_cvar_intr"]["cvar"] for a in algos]
    cvar_aug = [aug[a]["var_cvar_intr"]["cvar"] for a in algos]

    x = np.arange(len(algos))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w / 2, cvar_orig, w, label="879 paths (original)",
                color=[c + "66" for c in colors], edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w / 2, cvar_aug, w, label="4879 paths (augmented)",
                color=colors, edgecolor="black", linewidth=0.6)
    for bar, v in zip(list(b1) + list(b2), cvar_orig + cvar_aug):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                "{:.1f}".format(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(algos)
    ax.set_ylabel("CVaR95 (norm. units)")
    ax.set_title("Effect of Data Augmentation on Hedge Quality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp3_augmentation_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved exp3_augmentation_comparison.png")


def plot_exp4_rolling(fname="exp4_rolling_details.json"):
    data = _safe_load_json(fname)
    if data is None:
        return
    algos = [a for a in ["LSM", "RLSM", "NLSM"] if a in data]
    if not algos:
        return
    colors_map = {"LSM": "#2196F3", "RLSM": "#4CAF50", "NLSM": "#FF9800"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for a in algos:
        wins = data[a]["windows"]
        dates = list(range(len(wins)))
        shortfalls = [w["max_shortfall_intr"] for w in wins]
        ax.plot(dates, shortfalls, label=a, color=colors_map.get(a, "gray"), linewidth=1.2)
    ax.set_xlabel("Rolling window index")
    ax.set_ylabel("Max shortfall (intrinsic)")
    ax.set_title("Rolling Backtest: Per-Window Max Shortfall")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    cvar_vals = []
    for a in algos:
        losses = [w["max_shortfall_intr"] for w in data[a]["windows"]]
        cvar_vals.append(float(np.mean(sorted(losses)[int(0.95 * len(losses)):])))
    ax.bar(algos, cvar_vals, color=[colors_map.get(a, "gray") for a in algos],
           edgecolor="black", linewidth=0.6)
    for i, v in enumerate(cvar_vals):
        ax.text(i, v + 0.1, "{:.2f}".format(v), ha="center", fontsize=10)
    ax.set_ylabel("CVaR95")
    ax.set_title("Rolling Backtest: CVaR95 across all windows")

    ax = axes[1, 0]
    for a in algos:
        wins = data[a]["windows"]
        prices = [w["price"] for w in wins]
        ax.plot(range(len(prices)), prices, label=a, color=colors_map.get(a, "gray"), linewidth=1.2)
    ax.set_xlabel("Rolling window index")
    ax.set_ylabel("Option price (norm.)")
    ax.set_title("Rolling Backtest: Option Price per Window")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for a in algos:
        losses = [w["max_shortfall_intr"] for w in data[a]["windows"]]
        ax.hist(losses, bins=20, alpha=0.6, label=a, color=colors_map.get(a, "gray"),
                edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Max shortfall (intrinsic)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Per-Window Shortfalls")
    ax.legend()

    fig.suptitle("Exp 4: Rolling Backtest — BTC American Put", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "exp4_rolling_backtest.png"), dpi=150)
    plt.close(fig)
    print("  Saved exp4_rolling_backtest.png")


def plot_exp5_speed(fname="exp5_speed.json"):
    data = _safe_load_json(fname)
    if data is None:
        return
    algos = [a for a in ["LSM", "RLSM", "NLSM"] if a in data]
    colors_map = {"LSM": "#2196F3", "RLSM": "#4CAF50", "NLSM": "#FF9800"}

    times_us = [data[a]["mean_delta_us"] for a in algos]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(algos, times_us,
                  color=[colors_map.get(a, "gray") for a in algos],
                  edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, times_us):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                "{:.0f} µs".format(v), ha="center", fontsize=10)
    ax.set_ylabel("Mean delta computation time (µs)")
    ax.set_title("Exp 5: Delta Inference Speed Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp5_delta_speed.png"), dpi=150)
    plt.close(fig)
    print("  Saved exp5_delta_speed.png")


def plot_exp7_multidim(fname="exp7_multidim_details.json"):
    data = _safe_load_json(fname)
    if data is None:
        return
    dims = sorted([int(k) for k in data.keys() if k.isdigit()])
    if not dims:
        return
    algos = ["LSM", "RLSM"]
    colors_map = {"LSM": "#2196F3", "RLSM": "#4CAF50", "NLSM": "#FF9800"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for a in algos:
        cvars = []
        for d in dims:
            entry = data[str(d)].get(a)
            cvars.append(entry["cvar95_intr"] if entry else float("nan"))
        ax1.plot(dims, cvars, "o-", label=a, color=colors_map.get(a, "gray"), linewidth=1.5)
    ax1.set_xlabel("Feature dimension d")
    ax1.set_ylabel("CVaR95 (intrinsic)")
    ax1.set_title("Hedge Quality vs Dimension")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for a in algos:
        times = []
        for d in dims:
            entry = data[str(d)].get(a)
            times.append(entry["fit_time"] if entry else float("nan"))
        ax2.plot(dims, times, "s-", label=a, color=colors_map.get(a, "gray"), linewidth=1.5)
    ax2.set_xlabel("Feature dimension d")
    ax2.set_ylabel("Fit time (s)")
    ax2.set_title("Training Time vs Dimension")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Exp 7: Multi-Dimensional Features — LSM vs RLSM", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "exp7_multidim.png"), dpi=150)
    plt.close(fig)
    print("  Saved exp7_multidim.png")


def plot_exp6_ablation(fname="exp6_ablation.json"):
    data = _safe_load_json(fname)
    if data is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # RLSM hidden size
    hs_data = data.get("rlsm_hidden", {})
    hs_keys = sorted(hs_data.keys(), key=lambda k: int(k))
    hs_vals = [int(k) for k in hs_keys]
    cvar_rlsm = []
    for k in hs_keys:
        c = hs_data[k]["cvar95"]
        cvar_rlsm.append(min(c, 100))  # clip for display
    ax1.plot(hs_vals, cvar_rlsm, "o-", color="#4CAF50", linewidth=1.5, markersize=8)
    lsm_cvar = data.get("lsm_baseline", {}).get("cvar95", 0)
    if lsm_cvar > 0:
        ax1.axhline(lsm_cvar, color="#2196F3", linestyle="--", linewidth=1.2,
                     label="LSM baseline ({:.1f})".format(lsm_cvar))
    ax1.set_xlabel("RLSM hidden_size")
    ax1.set_ylabel("CVaR95")
    ax1.set_title("RLSM: Effect of Reservoir Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    for x, y in zip(hs_vals, cvar_rlsm):
        ax1.annotate("{:.1f}".format(y), (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)

    # NLSM epochs
    ep_data = data.get("nlsm_epochs", {})
    ep_keys = sorted(ep_data.keys(), key=lambda k: int(k))
    ep_vals = [int(k) for k in ep_keys]
    cvar_nlsm = [ep_data[k]["cvar95"] for k in ep_keys]
    ax2.plot(ep_vals, cvar_nlsm, "s-", color="#FF9800", linewidth=1.5, markersize=8)
    if lsm_cvar > 0:
        ax2.axhline(lsm_cvar, color="#2196F3", linestyle="--", linewidth=1.2,
                     label="LSM baseline ({:.1f})".format(lsm_cvar))
    ax2.set_xlabel("NLSM training epochs")
    ax2.set_ylabel("CVaR95")
    ax2.set_title("NLSM: Effect of Training Duration")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    for x, y in zip(ep_vals, cvar_nlsm):
        ax2.annotate("{:.1f}".format(y), (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)

    fig.suptitle("Exp 6: Ablation Study — Hyperparameter Sensitivity", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "exp6_ablation.png"), dpi=150)
    plt.close(fig)
    print("  Saved exp6_ablation.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Generating figures ...")
    plot_btc_price_context()
    plot_exp3_price_comparison()
    plot_exp3_hedge_error_distribution()
    plot_exp3_risk_metrics_bar()
    plot_exp3_summary_table()
    plot_exp3_augmentation_comparison()
    plot_exp4_rolling()
    plot_exp5_speed()
    plot_exp6_ablation()
    plot_exp7_multidim()
    print("Done.")


if __name__ == "__main__":
    main()
