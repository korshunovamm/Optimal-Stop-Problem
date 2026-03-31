# -*- coding: utf-8 -*-
"""
Generate all figures from experiment CSVs/JSONs.

Usage:
  conda-activate OptStopRandNN
  cd Optimal-Stop-Problem
  PYTHONPATH=. python 30-03-26/generate_all_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PKG = str(Path(__file__).resolve().parent)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

try:
    import pandas as pd
except ImportError:
    pd = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parent / "output"


def plot_block1():
    csv_path = OUT / "block1_1d_validation.csv"
    if not csv_path.exists():
        print("Skipping block1 plot: no data")
        return
    if pd is None:
        return

    df = pd.read_csv(str(csv_path))
    ref = df["ref_price"].iloc[0]
    agg = df.groupby("algo").agg(
        price_mean=("price", "mean"),
        price_std=("price", "std"),
        time_mean=("fit_time_s", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(agg))
    bars = ax.bar(x, agg["price_mean"], yerr=agg["price_std"],
                  capsize=5, alpha=0.8)
    ax.axhline(ref, color="red", ls="--", lw=1.5, label="Binomial ref={:.4f}".format(ref))
    ax.set_xticks(x)
    ax.set_xticklabels(agg["algo"])
    ax.set_ylabel("Price")
    ax.set_title("Block 1: 1D American Put (all methods match reference)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(OUT / "block1_validation.png"), dpi=150)
    plt.close(fig)
    print("Saved block1_validation.png")


def plot_block2():
    csv_path = OUT / "block2_maxcall_pricing.csv"
    if not csv_path.exists():
        print("Skipping block2 plot: no data")
        return
    if pd is None:
        return

    df = pd.read_csv(str(csv_path))
    df = df.dropna(subset=["price"])

    dims = sorted(df["nb_stocks"].unique())
    algos = ["LSM", "NLSM", "RLSM"]

    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 5), sharey=False)
    if len(dims) == 1:
        axes = [axes]

    for ax, d in zip(axes, dims):
        sub = df[df["nb_stocks"] == d]
        agg = sub.groupby("algo").agg(
            price_mean=("price", "mean"),
            price_std=("price", "std"),
        ).reindex(algos)
        x = np.arange(len(algos))
        ax.bar(x, agg["price_mean"], yerr=agg["price_std"],
               capsize=5, alpha=0.8, color=["#4C72B0", "#DD8452", "#55A868"])
        ax.set_xticks(x)
        ax.set_xticklabels(algos)
        ax.set_title("MaxCall d={}".format(d))
        ax.set_ylabel("Price")

    fig.suptitle("Block 2: Multi-Asset MaxCall Pricing", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(str(OUT / "block2_maxcall_pricing.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved block2_maxcall_pricing.png")

    # Time comparison
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    agg_time = df.groupby(["nb_stocks", "algo"]).agg(
        time_mean=("fit_time_s", "mean")).reset_index()
    for algo in algos:
        sub = agg_time[agg_time["algo"] == algo]
        ax2.plot(sub["nb_stocks"], sub["time_mean"], "o-", label=algo)
    ax2.set_xlabel("Number of assets (d)")
    ax2.set_ylabel("Fit time (s)")
    ax2.set_title("Block 2: Training Time vs Dimension")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(str(OUT / "block2_time_vs_dim.png"), dpi=150)
    plt.close(fig2)
    print("Saved block2_time_vs_dim.png")


def plot_block4():
    csv_path = OUT / "block4_heston.csv"
    if not csv_path.exists():
        print("Skipping block4 plot: no data")
        return
    if pd is None:
        return

    df = pd.read_csv(str(csv_path))
    models = df["stock_model"].unique()
    algos = ["LSM", "NLSM", "RLSM"]

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, m in zip(axes, models):
        sub = df[df["stock_model"] == m]
        x = np.arange(len(algos))
        prices = [sub[sub["algo"] == a]["price"].values[0] if len(sub[sub["algo"] == a]) > 0
                  else 0 for a in algos]
        ax.bar(x, prices, alpha=0.8, color=["#4C72B0", "#DD8452", "#55A868"])
        ax.set_xticks(x)
        ax.set_xticklabels(algos)
        ax.set_title("{} d=5".format(m))
        ax.set_ylabel("Price")

    fig.suptitle("Block 4: Heston Model Pricing", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(str(OUT / "block4_heston.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved block4_heston.png")


def plot_block3():
    csv_path = OUT / "block3_scaling.csv"
    if not csv_path.exists():
        print("Skipping block3 plot: no data")
        return
    if pd is None:
        return

    df = pd.read_csv(str(csv_path))
    # Combine block2 data if available
    b2_path = OUT / "block2_maxcall_pricing.csv"
    if b2_path.exists():
        b2 = pd.read_csv(str(b2_path))
        b2_agg = b2.groupby(["nb_stocks", "algo"]).agg(
            price=("price", "mean"),
            fit_time_s=("fit_time_s", "mean"),
        ).reset_index()
        b2_agg["payoff"] = "MaxCall"
        b2_agg["path_gen_s"] = 0.0
        combined = pd.concat([b2_agg, df], ignore_index=True, sort=False)
    else:
        combined = df

    algos = ["LSM", "RLSM"]
    present = [a for a in algos if a in combined["algo"].values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for algo in present:
        sub = combined[combined["algo"] == algo].sort_values("nb_stocks")
        ax1.plot(sub["nb_stocks"], sub["price"], "o-", label=algo, markersize=8)
    ax1.set_xlabel("Number of assets (d)")
    ax1.set_ylabel("Price")
    ax1.set_title("MaxCall Price vs Dimension")
    ax1.legend()

    for algo in present:
        sub = combined[combined["algo"] == algo].sort_values("nb_stocks")
        ax2.plot(sub["nb_stocks"], sub["fit_time_s"], "o-", label=algo, markersize=8)
    ax2.set_xlabel("Number of assets (d)")
    ax2.set_ylabel("Fit time (s)")
    ax2.set_title("Training Time vs Dimension")
    ax2.set_yscale("log")
    ax2.legend()

    fig.suptitle("Scalability: RLSM vs LSM", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(str(OUT / "block3_scaling.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved block3_scaling.png")

    # Speedup chart
    dims = sorted(combined["nb_stocks"].unique())
    speedups = []
    dim_labels = []
    for d in dims:
        lsm_sub = combined[(combined["algo"] == "LSM") & (combined["nb_stocks"] == d)]
        rlsm_sub = combined[(combined["algo"] == "RLSM") & (combined["nb_stocks"] == d)]
        if len(lsm_sub) > 0 and len(rlsm_sub) > 0:
            sp = float(lsm_sub["fit_time_s"].values[0]) / max(float(rlsm_sub["fit_time_s"].values[0]), 0.01)
            speedups.append(sp)
            dim_labels.append(str(int(d)))

    if speedups:
        fig2, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(speedups)), speedups, alpha=0.8, color="#55A868")
        ax.set_xticks(range(len(speedups)))
        ax.set_xticklabels(dim_labels)
        ax.set_xlabel("Number of assets (d)")
        ax.set_ylabel("Speedup (LSM time / RLSM time)")
        ax.set_title("RLSM Speedup over LSM")
        for i, v in enumerate(speedups):
            ax.text(i, v + 0.5, "{:.1f}x".format(v), ha="center", fontweight="bold")
        fig2.tight_layout()
        fig2.savefig(str(OUT / "block3_speedup.png"), dpi=150)
        plt.close(fig2)
        print("Saved block3_speedup.png")


def plot_block6():
    json_path = OUT / "block6_hedge_results.json"
    if not json_path.exists():
        print("Skipping block6 plot: no data")
        return

    with open(str(json_path)) as f:
        data = json.load(f)

    algos = ["LSM", "NLSM", "RLSM"]
    existing = [a for a in algos if a in data]
    if not existing:
        return

    cvars = [data[a]["var_cvar_95"]["cvar"] for a in existing]
    vars_ = [data[a]["var_cvar_95"]["var"] for a in existing]
    means = [data[a]["hedge_loss_summary"]["mean"] for a in existing]
    prices = [data[a]["price"] for a in existing]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(existing))
    width = 0.25
    ax1.bar(x - width, means, width, label="Mean Loss", alpha=0.8)
    ax1.bar(x, vars_, width, label="VaR 95%", alpha=0.8)
    ax1.bar(x + width, cvars, width, label="CVaR 95%", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(existing)
    ax1.set_ylabel("Loss")
    ax1.set_title("Hedge Risk Metrics (d=5 MaxCall)")
    ax1.legend()

    ax2.bar(x, prices, alpha=0.8, color=["#4C72B0", "#DD8452", "#55A868"][:len(existing)])
    ax2.set_xticks(x)
    ax2.set_xticklabels(existing)
    ax2.set_ylabel("Price")
    ax2.set_title("Option Price (d=5 MaxCall)")

    fig.tight_layout()
    fig.savefig(str(OUT / "block6_hedge_comparison.png"), dpi=150)
    plt.close(fig)
    print("Saved block6_hedge_comparison.png")


def main():
    plot_block1()
    plot_block2()
    plot_block3()
    plot_block4()
    plot_block6()
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
