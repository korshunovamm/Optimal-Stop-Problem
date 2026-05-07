# -*- coding: utf-8 -*-
"""Generate figures for 21-04-26 high-dim scaling experiments."""
from __future__ import annotations

import json, os, sys
from pathlib import Path
import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
OUT = os.path.join(_ROOT, "21-04-26-bybit", "output")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_btc_scaling():
    fpath = os.path.join(OUT, "scaling_btc.json")
    with open(fpath) as f:
        data = json.load(f)

    dims = sorted(data.keys(), key=int)
    algos = ['LSM', 'RLSM', 'NLSM']
    colors = {'LSM': '#2196F3', 'RLSM': '#4CAF50', 'NLSM': '#FF9800'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Delta speed
    ax = axes[0, 0]
    for algo in algos:
        vals = [data[d][algo]['delta_us'] for d in dims]
        ds = [int(d) for d in dims]
        ax.semilogy(ds, vals, 'o-', label=algo, color=colors[algo], linewidth=2)
    ax.axhline(1000, color='red', ls=':', alpha=0.5, label='1ms threshold')
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('Delta time (µs)')
    ax.set_title('Delta Speed vs Dimension (BTC)')
    ax.legend()
    ax.grid(alpha=0.3)

    # CVaR95
    ax = axes[0, 1]
    for algo in algos:
        vals = [data[d][algo]['cvar95'] for d in dims]
        ds = [int(d) for d in dims]
        ax.plot(ds, vals, 'o-', label=algo, color=colors[algo], linewidth=2)
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('CVaR₉₅')
    ax.set_title('Hedge Quality (CVaR₉₅) vs Dimension')
    ax.legend()
    ax.grid(alpha=0.3)

    # Fit time
    ax = axes[1, 0]
    for algo in algos:
        vals = [data[d][algo]['fit_time'] for d in dims]
        ds = [int(d) for d in dims]
        ax.semilogy(ds, vals, 'o-', label=algo, color=colors[algo], linewidth=2)
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('Training time (s)')
    ax.set_title('Training Time vs Dimension')
    ax.legend()
    ax.grid(alpha=0.3)

    # Price
    ax = axes[1, 1]
    for algo in algos:
        vals = [data[d][algo]['price'] for d in dims]
        ds = [int(d) for d in dims]
        ax.plot(ds, vals, 'o-', label=algo, color=colors[algo], linewidth=2)
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('Option Price')
    ax.set_title('Model Price vs Dimension')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_btc_scaling_4panel.png"), dpi=150)
    plt.close()
    print("  Saved fig_btc_scaling_4panel.png")


def plot_synthetic_scaling():
    fpath = os.path.join(OUT, "scaling_synthetic.json")
    with open(fpath) as f:
        data = json.load(f)

    dims = sorted(data.keys(), key=int)
    algos = ['LSM', 'RLSM']
    colors = {'LSM': '#2196F3', 'RLSM': '#4CAF50'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Delta speed
    ax = axes[0]
    for algo in algos:
        vals = [data[d][algo]['delta_us'] for d in dims]
        ds = [int(d) for d in dims]
        ax.semilogy(ds, vals, 'o-', label=algo, color=colors[algo],
                    linewidth=2, markersize=8)
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('Delta time (µs)')
    ax.set_title('Delta Speed (BS MaxCall)')
    ax.legend()
    ax.grid(alpha=0.3)
    # Annotate speedup at d=100
    if '100' in data:
        r = data['100']['LSM']['delta_us'] / data['100']['RLSM']['delta_us']
        ax.annotate('{:.0f}x faster'.format(r), xy=(100, data['100']['RLSM']['delta_us']),
                   xytext=(60, 5000), fontsize=10, color='green',
                   arrowprops=dict(arrowstyle='->', color='green'))

    # CVaR
    ax = axes[1]
    for algo in algos:
        vals = [data[d][algo]['cvar95'] for d in dims]
        ds = [int(d) for d in dims]
        ax.plot(ds, vals, 'o-', label=algo, color=colors[algo],
                linewidth=2, markersize=8)
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('CVaR₉₅')
    ax.set_title('Hedge Quality (BS MaxCall)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Fit time
    ax = axes[2]
    for algo in algos:
        vals = [data[d][algo]['fit_time'] for d in dims]
        ds = [int(d) for d in dims]
        ax.semilogy(ds, vals, 'o-', label=algo, color=colors[algo],
                    linewidth=2, markersize=8)
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('Training time (s)')
    ax.set_title('Training Time (BS MaxCall)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_synthetic_scaling_3panel.png"), dpi=150)
    plt.close()
    print("  Saved fig_synthetic_scaling_3panel.png")


def plot_speedup_ratio():
    """Bar chart showing RLSM speedup factor over LSM."""
    btc_path = os.path.join(OUT, "scaling_btc.json")
    syn_path = os.path.join(OUT, "scaling_synthetic.json")

    with open(btc_path) as f:
        btc = json.load(f)
    with open(syn_path) as f:
        syn = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 5))

    # BTC speedups
    btc_dims = sorted(btc.keys(), key=int)
    btc_ratios = [btc[d]['LSM']['delta_us'] / btc[d]['RLSM']['delta_us'] for d in btc_dims]
    btc_labels = ['BTC d={}'.format(d) for d in btc_dims]

    # Synthetic speedups
    syn_dims = sorted(syn.keys(), key=int)
    syn_ratios = [syn[d]['LSM']['delta_us'] / syn[d]['RLSM']['delta_us'] for d in syn_dims]
    syn_labels = ['Synth d={}'.format(d) for d in syn_dims]

    labels = btc_labels + syn_labels
    ratios = btc_ratios + syn_ratios
    colors = ['#2196F3'] * len(btc_dims) + ['#FF9800'] * len(syn_dims)

    x = range(len(labels))
    bars = ax.bar(x, ratios, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('RLSM speedup over LSM (x times)')
    ax.set_title('Delta Computation Speedup: RLSM vs LSM')
    ax.axhline(1, color='black', ls='--', alpha=0.3)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, ratios):
        if val > 1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.1,
                   '{:.0f}x'.format(val), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_speedup_ratio.png"), dpi=150)
    plt.close()
    print("  Saved fig_speedup_ratio.png")


def main():
    print("Generating 21-04-26 figures...")
    plot_btc_scaling()
    plot_synthetic_scaling()
    plot_speedup_ratio()
    print("Done!")


if __name__ == "__main__":
    main()
