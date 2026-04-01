# -*- coding: utf-8 -*-
"""
Experiment 7: Multi-dimensional feature space.
Show that as dimensionality grows, LSM degrades while RLSM stays stable.

Uses BTC price + auxiliary features (realized vol, volume, returns) to build
multi-dimensional state paths.  The payoff is still a 1D Put on the first
coordinate (normalised BTC price).

Usage (from project root):
    PYTHONUNBUFFERED=1 /home/kulikoval/.conda/envs/OptStopRandNN/bin/python -u \
        31-03-26-bybit/run_exp7_multidim_features.py
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
from btc_feature_builder import load_btc_daily_full, add_features, build_multidim_paths
from btc_data_loader import augment_with_noise
from historical_model import HistoricalModel
from backward_snapshots import fit_backward_snapshots
from btc_hedge import btc_hedge_one_path
from risk_metrics import var_cvar, summary_stats

from btc_payoff import BtcPut

DIMS_TO_TEST = [1, 2, 3, 5]
ALGOS = ["LSM", "RLSM"]
AUGMENT = 2000
NB_HEDGE_PATHS = 100
REFERENCE_SPOT = 100.0
RATE = 0.05


def main():
    csv_path = os.path.join(_ROOT, CFG.csv_path)
    out_dir = os.path.join(_ROOT, CFG.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Experiment 7: Multi-Dimensional Features")
    print("  Dimensions: {}".format(DIMS_TO_TEST))
    print("  Algorithms: {}".format(ALGOS))
    print("=" * 70)
    sys.stdout.flush()

    print("\n[1] Loading BTC data ...")
    df_train = load_btc_daily_full(csv_path, CFG.train_start, CFG.train_end)
    df_train = add_features(df_train)
    df_test = load_btc_daily_full(csv_path, CFG.test_start, CFG.test_end)
    df_test = add_features(df_test)
    print("  Train days: {}, Test days: {}".format(len(df_train), len(df_test)))
    sys.stdout.flush()

    maturity = CFG.maturity_days / 365.0
    payoff_f = BtcPut(REFERENCE_SPOT)

    all_results = {}

    for d in DIMS_TO_TEST:
        print("\n" + "-" * 50)
        print("  Dimension d={}".format(d))
        print("-" * 50)
        sys.stdout.flush()

        train_paths = build_multidim_paths(df_train, CFG.window_size, 1, d, REFERENCE_SPOT)
        test_paths = build_multidim_paths(df_test, CFG.window_size, 1, d, REFERENCE_SPOT)

        if train_paths.shape[0] == 0:
            print("  No valid train paths for d={}, skipping".format(d))
            continue

        if AUGMENT > 0 and d == 1:
            extra = augment_with_noise(train_paths, AUGMENT, 0.008, seed=42)
            train_paths = np.concatenate([train_paths, extra], axis=0)
        elif AUGMENT > 0 and d > 1:
            n_aug = min(AUGMENT, train_paths.shape[0] * 3)
            idx = np.random.choice(train_paths.shape[0], n_aug, replace=True)
            extra = train_paths[idx].copy()
            noise_price = np.random.normal(0, 0.008, extra[:, 0:1, 1:].shape)
            log_ret = np.diff(np.log(np.maximum(extra[:, 0:1, :], 1e-8)), axis=2) + noise_price
            cumret = np.cumsum(log_ret, axis=2)
            extra[:, 0:1, 1:] = extra[:, 0:1, 0:1] * np.exp(cumret)
            for k in range(1, d):
                feat_noise = np.random.normal(0, 0.1, extra[:, k:k+1, :].shape)
                extra[:, k:k+1, :] = extra[:, k:k+1, :] + feat_noise
            train_paths = np.concatenate([train_paths, extra], axis=0)

        np.random.seed(42)
        train_paths = train_paths[np.random.permutation(train_paths.shape[0])]

        print("  Train paths: {}, Test paths: {}".format(train_paths.shape, test_paths.shape))
        sys.stdout.flush()

        dim_results = {}
        for algo in ALGOS:
            print("\n    >>> {} (d={}) ...".format(algo, d))
            sys.stdout.flush()

            model = HistoricalModel(train_paths, RATE, maturity, CFG.nb_dates)
            t0 = time.time()
            bundle = fit_backward_snapshots(
                algo=algo, model=model, payoff_f=payoff_f,
                hidden_size=CFG.hidden_size,
                nb_epochs_nlsm=CFG.nb_epochs_nlsm,
                rlsm_factors=CFG.rlsm_factors,
                train_itm_only=True,
                train_eval_split=CFG.train_eval_split,
                seed=CFG.seed_train)
            fit_time = time.time() - t0
            price = bundle.price_terminal_discounted

            n_hedge = min(NB_HEDGE_PATHS, test_paths.shape[0])
            losses = np.zeros(n_hedge)
            t0 = time.time()
            for j in range(n_hedge):
                out = btc_hedge_one_path(
                    bundle, payoff_f, test_paths[j],
                    hidden_size=CFG.hidden_size, bump=CFG.bump_spot)
                losses[j] = float(np.max(out["disc_shortfall_vs_intrinsic"]))
                if (j + 1) % 25 == 0:
                    print("      hedged {}/{}".format(j + 1, n_hedge))
                    sys.stdout.flush()
            hedge_time = time.time() - t0

            vc = var_cvar(losses)
            st = summary_stats(losses)

            dim_results[algo] = {
                "price": float(price),
                "fit_time": fit_time,
                "hedge_time": hedge_time,
                "n_hedge": n_hedge,
                "cvar95_intr": vc["cvar"],
                "var95_intr": vc["var"],
                "mean_shortfall": st["mean"],
                "losses": losses.tolist(),
            }
            print("    {} d={} — price={:.3f}, CVaR95={:.3f}, fit={:.2f}s, hedge={:.2f}s".format(
                algo, d, price, vc["cvar"], fit_time, hedge_time))
            sys.stdout.flush()

        all_results[str(d)] = dim_results

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-DIM SUMMARY")
    print("=" * 70)
    hdr = "{:<4} {:<8} {:>10} {:>12} {:>10} {:>10}".format(
        "d", "Algo", "Price", "CVaR95", "FitTime", "HedgeTime")
    print(hdr)
    print("-" * len(hdr))
    for d in DIMS_TO_TEST:
        for algo in ALGOS:
            entry = all_results.get(str(d), {}).get(algo)
            if entry is None:
                continue
            print("{:<4} {:<8} {:>10.3f} {:>12.3f} {:>10.2f}s {:>10.2f}s".format(
                d, algo, entry["price"], entry["cvar95_intr"],
                entry["fit_time"], entry["hedge_time"]))
    print("=" * 70)

    json_out = os.path.join(out_dir, "exp7_multidim_details.json")
    with open(json_out, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved", json_out)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
