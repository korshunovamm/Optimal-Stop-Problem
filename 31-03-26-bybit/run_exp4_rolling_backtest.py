# -*- coding: utf-8 -*-
"""
Experiment 4: Rolling-window backtest.
For each 30-day window in the test period, train on past 365 days of BTC,
then hedge the next 30 days on real BTC prices.

Usage (from project root):
    PYTHONUNBUFFERED=1 /home/kulikoval/.conda/envs/OptStopRandNN/bin/python -u \
        31-03-26-bybit/run_exp4_rolling_backtest.py
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
from btc_data_loader import load_btc_daily, create_sliding_window_paths, augment_with_noise
from historical_model import HistoricalModel
from backward_snapshots import fit_backward_snapshots
from hedge_simulation import hedge_one_path
from risk_metrics import var_cvar, summary_stats

from optimal_stopping.payoffs.payoff import Put1Dim

ALGOS = ["LSM", "RLSM", "NLSM"]
TRAIN_LOOKBACK_DAYS = 365
WINDOW_DAYS = 30
ROLL_STEP_DAYS = 30       # non-overlapping 30-day windows
AUGMENT = 2000
NOISE_SIGMA = 0.008
REFERENCE_SPOT = 100.0
RATE = 0.05


def _maturity_years(days):
    return days / 365.0


def main():
    csv_path = os.path.join(_ROOT, CFG.csv_path)
    out_dir = os.path.join(_ROOT, CFG.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Experiment 4: Rolling Backtest on BTC")
    print("=" * 70)
    sys.stdout.flush()

    daily = load_btc_daily(csv_path)
    daily_arr = daily.values.astype(np.float64)
    daily_idx = daily.index

    test_mask = daily_idx >= "2023-07-01"
    test_start_pos = int(np.argmax(test_mask))

    all_results = {a: {"windows": []} for a in ALGOS}

    window_count = 0
    pos = test_start_pos
    while pos + WINDOW_DAYS < len(daily_arr):
        window_count += 1
        test_start = daily_idx[pos]
        test_end = daily_idx[min(pos + WINDOW_DAYS, len(daily_arr) - 1)]

        train_start_pos = max(0, pos - TRAIN_LOOKBACK_DAYS)
        train_prices = daily_arr[train_start_pos:pos + 1]

        test_prices = daily_arr[pos:pos + WINDOW_DAYS + 1]
        if len(test_prices) < WINDOW_DAYS + 1:
            break

        test_path_norm = test_prices / test_prices[0] * REFERENCE_SPOT
        test_path = test_path_norm[np.newaxis, np.newaxis, :]  # (1, 1, 31)

        train_paths = create_sliding_window_paths(
            train_prices, WINDOW_DAYS, step=1, reference_spot=REFERENCE_SPOT)

        if train_paths.shape[0] < 50:
            print("  Window {}: too few train paths ({}), skipping".format(
                window_count, train_paths.shape[0]))
            pos += ROLL_STEP_DAYS
            continue

        if AUGMENT > 0:
            extra = augment_with_noise(train_paths, AUGMENT, NOISE_SIGMA, seed=42 + window_count)
            train_paths = np.concatenate([train_paths, extra], axis=0)

        np.random.seed(42)
        train_paths = train_paths[np.random.permutation(train_paths.shape[0])]

        print("\n  Window {} — test: {} to {} | train paths: {} | BTC: {:.0f} -> {:.0f}".format(
            window_count, str(test_start.date()), str(test_end.date()),
            train_paths.shape[0], test_prices[0], test_prices[-1]))
        sys.stdout.flush()

        maturity = _maturity_years(WINDOW_DAYS)
        payoff_f = Put1Dim(REFERENCE_SPOT)

        for algo in ALGOS:
            model = HistoricalModel(train_paths, RATE, maturity, WINDOW_DAYS)

            t0 = time.time()
            bundle = fit_backward_snapshots(
                algo=algo, model=model, payoff_f=payoff_f,
                hidden_size=CFG.hidden_size,
                nb_epochs_nlsm=CFG.nb_epochs_nlsm,
                rlsm_factors=CFG.rlsm_factors,
                train_itm_only=True,
                train_eval_split=CFG.train_eval_split,
                seed=CFG.seed_train,
            )
            fit_time = time.time() - t0

            t0 = time.time()
            out = hedge_one_path(
                bundle, payoff_f, test_path[0],
                hidden_size=CFG.hidden_size, bump=CFG.bump_spot)
            hedge_time = time.time() - t0

            ms_model = float(np.max(out["disc_shortfall_vs_model"]))
            ms_intr = float(np.max(out["disc_shortfall_vs_intrinsic"]))

            all_results[algo]["windows"].append({
                "window": window_count,
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "btc_start": float(test_prices[0]),
                "btc_end": float(test_prices[-1]),
                "price": float(bundle.price_terminal_discounted),
                "fit_time": fit_time,
                "hedge_time": hedge_time,
                "max_shortfall_model": ms_model,
                "max_shortfall_intr": ms_intr,
            })
            print("    {} — price={:.3f}, shortfall_intr={:.3f}, fit={:.2f}s".format(
                algo, bundle.price_terminal_discounted, ms_intr, fit_time))
            sys.stdout.flush()

        pos += ROLL_STEP_DAYS

    # Aggregate metrics
    print("\n" + "=" * 70)
    print("ROLLING BACKTEST SUMMARY ({} windows)".format(window_count))
    print("=" * 70)
    hdr = "{:<8} {:>10} {:>12} {:>12} {:>12} {:>10}".format(
        "Algo", "Avg Price", "Mean SF", "VaR95 SF", "CVaR95 SF", "Avg Fit")
    print(hdr)
    print("-" * len(hdr))
    for algo in ALGOS:
        wins = all_results[algo]["windows"]
        if not wins:
            continue
        sfs = [w["max_shortfall_intr"] for w in wins]
        prices = [w["price"] for w in wins]
        fits = [w["fit_time"] for w in wins]
        vc = var_cvar(np.array(sfs))
        all_results[algo]["aggregate"] = {
            "mean_price": float(np.mean(prices)),
            "mean_shortfall": float(np.mean(sfs)),
            "var95": vc["var"], "cvar95": vc["cvar"],
            "mean_fit_time": float(np.mean(fits)),
            "n_windows": len(wins),
        }
        print("{:<8} {:>10.3f} {:>12.3f} {:>12.3f} {:>12.3f} {:>10.2f}s".format(
            algo, np.mean(prices), np.mean(sfs), vc["var"], vc["cvar"], np.mean(fits)))
    print("=" * 70)

    json_out = os.path.join(out_dir, "exp4_rolling_details.json")
    with open(json_out, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved", json_out)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
