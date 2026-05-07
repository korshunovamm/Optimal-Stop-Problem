# -*- coding: utf-8 -*-
"""Generate figures for Experiment 8 market validation."""
from __future__ import annotations

import json, os, sys
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
OUT = os.path.join(_ROOT, "6-05-26", "output")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("WARNING: matplotlib not available, skipping figures")


def plot_market_comparison():
    """Bar chart: model price vs market price for each option."""
    fpath = os.path.join(OUT, "exp8_market_validation.json")
    if not os.path.exists(fpath):
        return
    with open(fpath) as f:
        data = json.load(f)

    names = [d['full_name'][:18] for d in data]
    market = [d['market_mark_usd'] for d in data]
    lsm = [d.get('LSM_usd', 0) for d in data]
    rlsm = [d.get('RLSM_usd', 0) for d in data]
    nlsm = [d.get('NLSM_usd', 0) for d in data]

    x = np.arange(len(names))
    w = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 1.5*w, market, w, label='Market', color='black', alpha=0.7)
    ax.bar(x - 0.5*w, lsm, w, label='LSM', color='#2196F3')
    ax.bar(x + 0.5*w, rlsm, w, label='RLSM', color='#4CAF50')
    ax.bar(x + 1.5*w, nlsm, w, label='NLSM', color='#FF9800')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Option Price (USD)')
    ax.set_title('Model vs Market BTC Put Prices (Historical Vol Training)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_market_comparison.png"), dpi=150)
    plt.close()
    print("  Saved fig_market_comparison.png")


def plot_vol_adjusted_comparison():
    """Bar chart for vol-adjusted results."""
    fpath = os.path.join(OUT, "exp8b_vol_adjusted.json")
    if not os.path.exists(fpath):
        return
    with open(fpath) as f:
        data = json.load(f)

    names = [d['full_name'][:18] for d in data]
    market = [d['market_mark_usd'] for d in data]
    lsm = [d.get('LSM_usd', 0) for d in data]
    rlsm = [d.get('RLSM_usd', 0) for d in data]
    nlsm = [d.get('NLSM_usd', 0) for d in data]
    bids = [d.get('market_bid_usd', 0) or 0 for d in data]
    asks = [d.get('market_ask_usd', 0) or 0 for d in data]

    x = np.arange(len(names))
    w = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    # Bid-ask range as shaded area
    for i in range(len(names)):
        ax.axhspan(bids[i], asks[i], xmin=(i-0.4)/len(names),
                   xmax=(i+0.4)/len(names), alpha=0.1, color='gray')

    ax.bar(x - 1.5*w, market, w, label='Market Mark', color='black', alpha=0.7)
    ax.bar(x - 0.5*w, lsm, w, label='LSM', color='#2196F3')
    ax.bar(x + 0.5*w, rlsm, w, label='RLSM', color='#4CAF50')
    ax.bar(x + 1.5*w, nlsm, w, label='NLSM', color='#FF9800')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Option Price (USD)')
    ax.set_title('Vol-Adjusted Model vs Market BTC Put Prices')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_vol_adjusted_comparison.png"), dpi=150)
    plt.close()
    print("  Saved fig_vol_adjusted_comparison.png")


def plot_error_by_tte():
    """Scatter: pricing error vs TTE for each algorithm."""
    fpath = os.path.join(OUT, "exp8_market_validation.json")
    if not os.path.exists(fpath):
        return
    with open(fpath) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'LSM': '#2196F3', 'RLSM': '#4CAF50', 'NLSM': '#FF9800'}
    for algo in ['LSM', 'RLSM', 'NLSM']:
        ttes = [d['tte_days'] for d in data]
        errs = [d.get('{}_error_pct'.format(algo), 0) for d in data]
        ax.scatter(ttes, errs, label=algo, color=colors[algo], s=80, alpha=0.8)

    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time to Expiry (days)')
    ax.set_ylabel('Pricing Error vs Market (%)')
    ax.set_title('Model Pricing Error by Time to Expiry')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_error_vs_tte.png"), dpi=150)
    plt.close()
    print("  Saved fig_error_vs_tte.png")


def plot_scaling_btc():
    """Delta speed and CVaR scaling from 21-04-26 experiment."""
    fpath = os.path.join(_ROOT, "21-04-26-bybit", "output", "scaling_btc.json")
    if not os.path.exists(fpath):
        return
    with open(fpath) as f:
        data = json.load(f)

    dims = sorted(data.keys(), key=int)
    algos = ['LSM', 'RLSM', 'NLSM']
    colors = {'LSM': '#2196F3', 'RLSM': '#4CAF50', 'NLSM': '#FF9800'}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Delta speed
    for algo in algos:
        vals = [data[d][algo]['delta_us'] for d in dims if algo in data[d]]
        ds = [int(d) for d in dims if algo in data[d]]
        ax1.semilogy(ds, vals, 'o-', label=algo, color=colors[algo], linewidth=2)
    ax1.set_xlabel('Dimension d')
    ax1.set_ylabel('Delta computation (µs, log scale)')
    ax1.set_title('Delta Speed vs Dimension')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # CVaR95
    for algo in algos:
        vals = [data[d][algo]['cvar95'] for d in dims if algo in data[d]]
        ds = [int(d) for d in dims if algo in data[d]]
        ax2.plot(ds, vals, 'o-', label=algo, color=colors[algo], linewidth=2)
    ax2.set_xlabel('Dimension d')
    ax2.set_ylabel('CVaR₉₅ Hedge Shortfall')
    ax2.set_title('Hedge Quality vs Dimension (BTC)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Fit time
    for algo in algos:
        vals = [data[d][algo]['fit_time'] for d in dims if algo in data[d]]
        ds = [int(d) for d in dims if algo in data[d]]
        ax3.semilogy(ds, vals, 'o-', label=algo, color=colors[algo], linewidth=2)
    ax3.set_xlabel('Dimension d')
    ax3.set_ylabel('Training Time (s, log scale)')
    ax3.set_title('Training Time vs Dimension')
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_scaling_btc.png"), dpi=150)
    plt.close()
    print("  Saved fig_scaling_btc.png")


def plot_scaling_synthetic():
    """Synthetic scaling results."""
    fpath = os.path.join(_ROOT, "21-04-26-bybit", "output", "scaling_synthetic.json")
    if not os.path.exists(fpath):
        return
    with open(fpath) as f:
        data = json.load(f)

    dims = sorted(data.keys(), key=int)
    algos = ['LSM', 'RLSM']
    colors = {'LSM': '#2196F3', 'RLSM': '#4CAF50'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Delta speed
    for algo in algos:
        vals = [data[d][algo]['delta_us'] for d in dims if algo in data[d]]
        ds = [int(d) for d in dims if algo in data[d]]
        ax1.semilogy(ds, vals, 'o-', label=algo, color=colors[algo],
                     linewidth=2, markersize=8)
    ax1.set_xlabel('Dimension d')
    ax1.set_ylabel('Delta computation (µs, log scale)')
    ax1.set_title('Delta Speed: Synthetic BS MaxCall')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # CVaR
    for algo in algos:
        vals = [data[d][algo]['cvar95'] for d in dims if algo in data[d]]
        ds = [int(d) for d in dims if algo in data[d]]
        ax2.plot(ds, vals, 'o-', label=algo, color=colors[algo],
                 linewidth=2, markersize=8)
    ax2.set_xlabel('Dimension d')
    ax2.set_ylabel('CVaR₉₅ Hedge Shortfall')
    ax2.set_title('Hedge Quality: Synthetic BS MaxCall')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_scaling_synthetic.png"), dpi=150)
    plt.close()
    print("  Saved fig_scaling_synthetic.png")


def plot_combined_delta_speed():
    """Combined delta speed comparison for the thesis."""
    btc_path = os.path.join(_ROOT, "21-04-26-bybit", "output", "scaling_btc.json")
    syn_path = os.path.join(_ROOT, "21-04-26-bybit", "output", "scaling_synthetic.json")
    if not os.path.exists(btc_path) or not os.path.exists(syn_path):
        return

    with open(btc_path) as f:
        btc = json.load(f)
    with open(syn_path) as f:
        syn = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    # BTC
    dims_btc = sorted(btc.keys(), key=int)
    lsm_btc = [btc[d]['LSM']['delta_us'] for d in dims_btc]
    rlsm_btc = [btc[d]['RLSM']['delta_us'] for d in dims_btc]
    ds_btc = [int(d) for d in dims_btc]

    # Synthetic
    dims_syn = sorted(syn.keys(), key=int)
    lsm_syn = [syn[d]['LSM']['delta_us'] for d in dims_syn]
    rlsm_syn = [syn[d]['RLSM']['delta_us'] for d in dims_syn]
    ds_syn = [int(d) for d in dims_syn]

    ax.semilogy(ds_btc, lsm_btc, 's-', color='#2196F3', linewidth=2,
                label='LSM FD (BTC)', markersize=7)
    ax.semilogy(ds_btc, rlsm_btc, 'o-', color='#4CAF50', linewidth=2,
                label='RLSM autograd (BTC)', markersize=7)
    ax.semilogy(ds_syn, lsm_syn, 's--', color='#2196F3', linewidth=1.5,
                alpha=0.6, label='LSM FD (Synthetic)', markersize=5)
    ax.semilogy(ds_syn, rlsm_syn, 'o--', color='#4CAF50', linewidth=1.5,
                alpha=0.6, label='RLSM autograd (Synthetic)', markersize=5)

    ax.axhline(1000, color='red', linestyle=':', alpha=0.5, label='1ms threshold')
    ax.set_xlabel('Dimension d', fontsize=12)
    ax.set_ylabel('Per-delta time (µs, log scale)', fontsize=12)
    ax.set_title('Delta Computation Speed: LSM O(d³) vs RLSM O(1)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Annotate speedups
    for d_str in ['30']:
        if d_str in btc:
            ratio = btc[d_str]['LSM']['delta_us'] / btc[d_str]['RLSM']['delta_us']
            ax.annotate('{:.0f}x'.format(ratio),
                       xy=(int(d_str), btc[d_str]['LSM']['delta_us']),
                       xytext=(int(d_str)+2, btc[d_str]['LSM']['delta_us']*1.5),
                       fontsize=10, color='red')
    for d_str in ['100']:
        if d_str in syn:
            ratio = syn[d_str]['LSM']['delta_us'] / syn[d_str]['RLSM']['delta_us']
            ax.annotate('{:.0f}x'.format(ratio),
                       xy=(int(d_str), syn[d_str]['LSM']['delta_us']),
                       xytext=(int(d_str)-15, syn[d_str]['LSM']['delta_us']*2),
                       fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_delta_speed_combined.png"), dpi=150)
    plt.close()
    print("  Saved fig_delta_speed_combined.png")


def main():
    if not HAS_PLT:
        return
    os.makedirs(OUT, exist_ok=True)
    print("Generating figures...")
    plot_market_comparison()
    plot_vol_adjusted_comparison()
    plot_error_by_tte()
    plot_scaling_btc()
    plot_scaling_synthetic()
    plot_combined_delta_speed()
    print("Done!")


if __name__ == "__main__":
    main()
