#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate report figures from CSV/JSON outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    base = Path(__file__).resolve().parent
    out = base / "output"
    out.mkdir(exist_ok=True, parents=True)

    bs = _read_csv(out / "report_prices_bs.csv")
    heston = _read_csv(out / "report_prices_heston.csv")
    hedge = _read_csv(out / "report_hedge_summary.csv")
    sweep = _read_csv(out / "report_vol_sweep.csv")

    # 1) Price bars (BS + Heston)
    for rows, title, fname in [
        (bs, "Black-Scholes: option price by algorithm", "report_prices_bs_bar.png"),
        (heston, "Heston: option price by algorithm", "report_prices_heston_bar.png"),
    ]:
        algos = [r["algo"] for r in rows]
        prices = [float(r["price"]) for r in rows]
        x = np.arange(len(algos))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x, prices, color=["#264653", "#2a9d8f", "#e9c46a"], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(algos)
        ax.set_ylabel("Price")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=140)
        plt.close(fig)

    # 2) Hedge risk bars
    algos = [r["algo"] for r in hedge]
    var95 = [float(r["var95"]) for r in hedge]
    cvar95 = [float(r["cvar95"]) for r in hedge]
    x = np.arange(len(algos))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, var95, width, label="VaR95", color="#457b9d")
    ax.bar(x + width / 2, cvar95, width, label="CVaR95", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(algos)
    ax.set_ylabel("Loss")
    ax.set_title("Hedging risk: VaR/CVaR by algorithm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "report_hedge_var_cvar_bar.png", dpi=140)
    plt.close(fig)

    # 3) Volatility sweep (CVaR95)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for algo, color in [("LSM", "#264653"), ("NLSM", "#e76f51")]:
        pts = [r for r in sweep if r["algo"] == algo]
        xs = [float(r["volatility"]) for r in pts]
        ys = [float(r["cvar95"]) for r in pts]
        ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=algo)
    ax.set_xlabel("Volatility sigma")
    ax.set_ylabel("CVaR95")
    ax.set_title("Hedge risk sensitivity to volatility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "report_sweep_cvar.png", dpi=140)
    plt.close(fig)

    # 4) Build synthetic histograms using summary moments for visual support
    # (raw per-path losses were not saved in CSV; this figure is approximate)
    with open(out / "report_hedge_risk.json", "r", encoding="utf-8") as f:
        risk = json.load(f)
    for algo, color in [("LSM", "#2a6f97"), ("NLSM", "#f4a261"), ("RLSM", "#8d99ae")]:
        st = risk[algo]["hedge_loss_model_summary"]
        mu = float(st["mean"])
        sd = max(float(st["std"]), 1e-6)
        # gamma-like positive sample (method-of-moments approximation)
        k = (mu / sd) ** 2 if sd > 0 else 10.0
        theta = (sd ** 2) / mu if mu > 0 else 1.0
        samples = np.random.default_rng(123).gamma(shape=k, scale=theta, size=5000)
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        ax.hist(samples, bins=45, density=True, color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.set_title(f"Approximate loss distribution ({algo})")
        ax.set_xlabel("Loss")
        ax.set_ylabel("Density")
        fig.tight_layout()
        fig.savefig(out / f"report_hedge_hist_bs_main_{algo}.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()

