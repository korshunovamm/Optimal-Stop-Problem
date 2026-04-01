# -*- coding: utf-8 -*-
"""
Experiment 6: Ablation study — compare RLSM with different hidden_size
and NLSM with different nb_epochs.

Usage (from project root):
    PYTHONUNBUFFERED=1 /home/kulikoval/.conda/envs/OptStopRandNN/bin/python -u \
        31-03-26-bybit/run_exp6_ablation.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
_ROOT = str(Path(__file__).resolve().parent.parent)
for p in (_SCRIPT_DIR, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from config_btc import DEFAULT as CFG
from btc_data_loader import prepare_btc_paths
from historical_model import HistoricalModel
from backward_snapshots import fit_backward_snapshots
from hedge_simulation import hedge_one_path
from risk_metrics import var_cvar, summary_stats

from optimal_stopping.payoffs.payoff import Put1Dim

RLSM_HIDDEN_SIZES = [32, 64, 128, 256, 512]
NLSM_EPOCHS = [20, 50, 100, 200]
NB_HEDGE = 100
AUGMENT = 2000


def main():
    csv_path = os.path.join(_ROOT, CFG.csv_path)
    out_dir = os.path.join(_ROOT, CFG.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Experiment 6: Ablation Study")
    print("=" * 70)
    sys.stdout.flush()

    train_paths, _ = prepare_btc_paths(
        csv_path, CFG.train_start, CFG.train_end,
        window_size=CFG.window_size, step=CFG.window_step,
        reference_spot=CFG.reference_spot, augment=AUGMENT,
        noise_sigma=0.008, seed=42)
    test_paths, _ = prepare_btc_paths(
        csv_path, CFG.test_start, CFG.test_end,
        window_size=CFG.window_size, step=CFG.window_step,
        reference_spot=CFG.reference_spot, augment=0)

    np.random.seed(42)
    train_paths = train_paths[np.random.permutation(train_paths.shape[0])]

    print("  Train: {}, Test: {}".format(train_paths.shape, test_paths.shape))
    sys.stdout.flush()

    maturity = CFG.maturity_days / 365.0
    payoff_f = Put1Dim(CFG.strike)
    all_results = {"rlsm_hidden": {}, "nlsm_epochs": {}}

    # --- RLSM hidden size sweep ---------------------------------------------
    print("\n--- RLSM hidden_size sweep ---")
    sys.stdout.flush()
    for hs in RLSM_HIDDEN_SIZES:
        print("  RLSM hidden_size={} ...".format(hs))
        sys.stdout.flush()
        model = HistoricalModel(train_paths, CFG.rate, maturity, CFG.nb_dates)
        t0 = time.time()
        bundle = fit_backward_snapshots(
            algo="RLSM", model=model, payoff_f=payoff_f,
            hidden_size=hs,
            nb_epochs_nlsm=CFG.nb_epochs_nlsm,
            rlsm_factors=CFG.rlsm_factors,
            train_itm_only=True,
            train_eval_split=CFG.train_eval_split,
            seed=CFG.seed_train)
        fit_time = time.time() - t0

        n_h = min(NB_HEDGE, test_paths.shape[0])
        losses = np.zeros(n_h)
        t0 = time.time()
        for j in range(n_h):
            out = hedge_one_path(bundle, payoff_f, test_paths[j],
                                 hidden_size=hs, bump=CFG.bump_spot)
            losses[j] = float(np.max(out["disc_shortfall_vs_intrinsic"]))
        hedge_time = time.time() - t0
        vc = var_cvar(losses)

        all_results["rlsm_hidden"][str(hs)] = {
            "hidden_size": hs,
            "price": float(bundle.price_terminal_discounted),
            "fit_time": fit_time, "hedge_time": hedge_time,
            "cvar95": vc["cvar"], "var95": vc["var"],
            "mean_shortfall": float(np.mean(losses)),
        }
        print("    hs={} — price={:.3f}, CVaR95={:.3f}, fit={:.2f}s".format(
            hs, bundle.price_terminal_discounted, vc["cvar"], fit_time))
        sys.stdout.flush()

    # --- NLSM epochs sweep ---------------------------------------------------
    print("\n--- NLSM nb_epochs sweep ---")
    sys.stdout.flush()
    for ep in NLSM_EPOCHS:
        print("  NLSM epochs={} ...".format(ep))
        sys.stdout.flush()
        model = HistoricalModel(train_paths, CFG.rate, maturity, CFG.nb_dates)
        t0 = time.time()
        bundle = fit_backward_snapshots(
            algo="NLSM", model=model, payoff_f=payoff_f,
            hidden_size=CFG.hidden_size,
            nb_epochs_nlsm=ep,
            rlsm_factors=CFG.rlsm_factors,
            train_itm_only=True,
            train_eval_split=CFG.train_eval_split,
            seed=CFG.seed_train)
        fit_time = time.time() - t0

        n_h = min(NB_HEDGE, test_paths.shape[0])
        losses = np.zeros(n_h)
        t0 = time.time()
        for j in range(n_h):
            out = hedge_one_path(bundle, payoff_f, test_paths[j],
                                 hidden_size=CFG.hidden_size, bump=CFG.bump_spot)
            losses[j] = float(np.max(out["disc_shortfall_vs_intrinsic"]))
        hedge_time = time.time() - t0
        vc = var_cvar(losses)

        all_results["nlsm_epochs"][str(ep)] = {
            "nb_epochs": ep,
            "price": float(bundle.price_terminal_discounted),
            "fit_time": fit_time, "hedge_time": hedge_time,
            "cvar95": vc["cvar"], "var95": vc["var"],
            "mean_shortfall": float(np.mean(losses)),
        }
        print("    ep={} — price={:.3f}, CVaR95={:.3f}, fit={:.2f}s".format(
            ep, bundle.price_terminal_discounted, vc["cvar"], fit_time))
        sys.stdout.flush()

    # --- LSM baseline --------------------------------------------------------
    print("\n--- LSM baseline ---")
    model = HistoricalModel(train_paths, CFG.rate, maturity, CFG.nb_dates)
    t0 = time.time()
    bundle = fit_backward_snapshots(
        algo="LSM", model=model, payoff_f=payoff_f,
        hidden_size=CFG.hidden_size, nb_epochs_nlsm=100,
        rlsm_factors=(1.0,), train_itm_only=True,
        train_eval_split=CFG.train_eval_split, seed=CFG.seed_train)
    fit_time = time.time() - t0
    n_h = min(NB_HEDGE, test_paths.shape[0])
    losses = np.zeros(n_h)
    for j in range(n_h):
        out = hedge_one_path(bundle, payoff_f, test_paths[j],
                             hidden_size=CFG.hidden_size, bump=CFG.bump_spot)
        losses[j] = float(np.max(out["disc_shortfall_vs_intrinsic"]))
    vc = var_cvar(losses)
    all_results["lsm_baseline"] = {
        "price": float(bundle.price_terminal_discounted),
        "fit_time": fit_time,
        "cvar95": vc["cvar"], "var95": vc["var"],
        "mean_shortfall": float(np.mean(losses)),
    }
    print("  LSM — price={:.3f}, CVaR95={:.3f}".format(
        bundle.price_terminal_discounted, vc["cvar"]))

    # Save
    json_out = os.path.join(out_dir, "exp6_ablation.json")
    with open(json_out, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved", json_out)

    # Summary
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print("\nRLSM hidden_size:")
    for hs in RLSM_HIDDEN_SIZES:
        r = all_results["rlsm_hidden"][str(hs)]
        print("  hs={:<4} price={:.3f}  CVaR95={:.3f}  fit={:.2f}s".format(
            hs, r["price"], r["cvar95"], r["fit_time"]))
    print("\nNLSM nb_epochs:")
    for ep in NLSM_EPOCHS:
        r = all_results["nlsm_epochs"][str(ep)]
        print("  ep={:<4} price={:.3f}  CVaR95={:.3f}  fit={:.2f}s".format(
            ep, r["price"], r["cvar95"], r["fit_time"]))
    r = all_results["lsm_baseline"]
    print("\nLSM baseline: price={:.3f}  CVaR95={:.3f}".format(r["price"], r["cvar95"]))
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
