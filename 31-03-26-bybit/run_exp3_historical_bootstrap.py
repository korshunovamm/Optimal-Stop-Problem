# -*- coding: utf-8 -*-
"""
Experiment 3: Historical Bootstrap — train LSM / RLSM / NLSM on real BTC
daily paths, then evaluate pricing accuracy and hedge quality on held-out
test paths.

Usage (from project root):
    PYTHONUNBUFFERED=1 /home/kulikoval/.conda/envs/OptStopRandNN/bin/python -u \
        31-03-26-bybit/run_exp3_historical_bootstrap.py
"""
from __future__ import annotations

import csv
import json
import math
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

from config_btc import BTCExperimentParams, DEFAULT as CFG
from btc_data_loader import prepare_btc_paths
from historical_model import HistoricalModel
from backward_snapshots import fit_backward_snapshots
from american_value import get_delta, american_value
from hedge_simulation import hedge_one_path
from risk_metrics import var_cvar, summary_stats

from optimal_stopping.payoffs.payoff import Put1Dim


ALGOS = ["LSM", "RLSM", "NLSM"]


def _maturity_years(days):
    return days / 365.0


def run_pricing(algo, train_paths, params):
    # type: (str, np.ndarray, BTCExperimentParams) -> dict
    """Train one algorithm on the given paths and return the snapshot bundle + timing."""
    maturity = _maturity_years(params.maturity_days)
    model = HistoricalModel(
        paths=train_paths,
        rate=params.rate,
        maturity=maturity,
        nb_dates=params.nb_dates,
    )
    payoff_f = Put1Dim(params.strike)

    t0 = time.time()
    bundle = fit_backward_snapshots(
        algo=algo,
        model=model,
        payoff_f=payoff_f,
        hidden_size=params.hidden_size,
        nb_epochs_nlsm=params.nb_epochs_nlsm,
        rlsm_factors=params.rlsm_factors,
        train_itm_only=True,
        train_eval_split=params.train_eval_split,
        seed=params.seed_train,
    )
    fit_time = time.time() - t0

    return {
        "bundle": bundle,
        "price": bundle.price_terminal_discounted,
        "fit_time": fit_time,
    }


def run_hedge(bundle, test_paths, params):
    # type: (...) -> dict
    """Run hedge simulation on test paths and compute risk metrics."""
    payoff_f = Put1Dim(params.strike)
    n_test = min(params.nb_test_hedge_paths, test_paths.shape[0])
    test_sub = test_paths[:n_test]

    losses_model = np.zeros(n_test)
    losses_intr = np.zeros(n_test)

    t0 = time.time()
    for j in range(n_test):
        path_j = test_sub[j]  # (1, nb_dates+1)
        out = hedge_one_path(
            bundle, payoff_f, path_j,
            hidden_size=params.hidden_size,
            bump=params.bump_spot,
        )
        losses_model[j] = float(np.max(out["disc_shortfall_vs_model"]))
        losses_intr[j] = float(np.max(out["disc_shortfall_vs_intrinsic"]))
        if (j + 1) % 50 == 0:
            print("    hedged {}/{} paths".format(j + 1, n_test))
            sys.stdout.flush()
    hedge_time = time.time() - t0

    vc_model = var_cvar(losses_model, alpha=0.95)
    vc_intr = var_cvar(losses_intr, alpha=0.95)
    stats_model = summary_stats(losses_model)
    stats_intr = summary_stats(losses_intr)

    return {
        "hedge_time": hedge_time,
        "n_test": n_test,
        "losses_model": losses_model,
        "losses_intr": losses_intr,
        "var_cvar_model": vc_model,
        "var_cvar_intr": vc_intr,
        "stats_model": stats_model,
        "stats_intr": stats_intr,
    }


def main():
    params = CFG
    csv_path = os.path.join(_ROOT, params.csv_path)
    out_dir = os.path.join(_ROOT, params.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Experiment 3: Historical Bootstrap on BTC")
    print("=" * 70)
    sys.stdout.flush()

    # --- Load data -----------------------------------------------------------
    print("\n[1/4] Loading BTC daily data ...")
    sys.stdout.flush()

    train_paths, daily_train = prepare_btc_paths(
        csv_path,
        start_date=params.train_start,
        end_date=params.train_end,
        window_size=params.window_size,
        step=params.window_step,
        reference_spot=params.reference_spot,
        augment=0,
    )
    test_paths, daily_test = prepare_btc_paths(
        csv_path,
        start_date=params.test_start,
        end_date=params.test_end,
        window_size=params.window_size,
        step=params.window_step,
        reference_spot=params.reference_spot,
        augment=0,
    )
    print("  Train paths: {}  (daily prices {} -> {})".format(
        train_paths.shape, daily_train.index[0].date(), daily_train.index[-1].date()))
    print("  Test  paths: {}  (daily prices {} -> {})".format(
        test_paths.shape, daily_test.index[0].date(), daily_test.index[-1].date()))
    sys.stdout.flush()

    # --- Train & price -------------------------------------------------------
    print("\n[2/4] Training algorithms on historical BTC paths ...")
    sys.stdout.flush()

    results = {}
    for algo in ALGOS:
        print("\n  >>> {} ...".format(algo))
        sys.stdout.flush()
        res = run_pricing(algo, train_paths, params)
        results[algo] = res
        print("  {} — price={:.4f}, fit_time={:.2f}s".format(
            algo, res["price"], res["fit_time"]))
        sys.stdout.flush()

    # --- Hedge on test paths -------------------------------------------------
    print("\n[3/4] Running hedge simulations on test BTC paths ...")
    sys.stdout.flush()

    for algo in ALGOS:
        print("\n  >>> Hedging {} ...".format(algo))
        sys.stdout.flush()
        hedge_res = run_hedge(results[algo]["bundle"], test_paths, params)
        results[algo].update(hedge_res)
        print("  {} — CVaR95_model={:.4f}, CVaR95_intr={:.4f}, hedge_time={:.2f}s".format(
            algo,
            hedge_res["var_cvar_model"]["cvar"],
            hedge_res["var_cvar_intr"]["cvar"],
            hedge_res["hedge_time"],
        ))
        sys.stdout.flush()

    # --- Save results --------------------------------------------------------
    print("\n[4/4] Saving results ...")
    sys.stdout.flush()

    # CSV summary
    csv_path_out = os.path.join(out_dir, "exp3_summary.csv")
    with open(csv_path_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algo", "price", "fit_time",
            "var95_model", "cvar95_model", "mean_shortfall_model",
            "var95_intr", "cvar95_intr", "mean_shortfall_intr",
            "hedge_time", "n_test",
        ])
        for algo in ALGOS:
            r = results[algo]
            writer.writerow([
                algo,
                "{:.4f}".format(r["price"]),
                "{:.2f}".format(r["fit_time"]),
                "{:.4f}".format(r["var_cvar_model"]["var"]),
                "{:.4f}".format(r["var_cvar_model"]["cvar"]),
                "{:.4f}".format(r["stats_model"]["mean"]),
                "{:.4f}".format(r["var_cvar_intr"]["var"]),
                "{:.4f}".format(r["var_cvar_intr"]["cvar"]),
                "{:.4f}".format(r["stats_intr"]["mean"]),
                "{:.2f}".format(r["hedge_time"]),
                r["n_test"],
            ])
    print("  Saved", csv_path_out)

    # JSON with full details
    json_path = os.path.join(out_dir, "exp3_details.json")
    details = {}
    for algo in ALGOS:
        r = results[algo]
        details[algo] = {
            "price": r["price"],
            "fit_time": r["fit_time"],
            "hedge_time": r["hedge_time"],
            "n_test": r["n_test"],
            "var_cvar_model": r["var_cvar_model"],
            "var_cvar_intr": r["var_cvar_intr"],
            "stats_model": r["stats_model"],
            "stats_intr": r["stats_intr"],
            "losses_model": r["losses_model"].tolist(),
            "losses_intr": r["losses_intr"].tolist(),
        }
    details["params"] = {
        "train_start": params.train_start,
        "train_end": params.train_end,
        "test_start": params.test_start,
        "test_end": params.test_end,
        "reference_spot": params.reference_spot,
        "strike": params.strike,
        "maturity_days": params.maturity_days,
        "nb_dates": params.nb_dates,
        "rate": params.rate,
        "hidden_size": params.hidden_size,
        "nb_epochs_nlsm": params.nb_epochs_nlsm,
        "train_paths": int(train_paths.shape[0]),
        "test_paths": int(test_paths.shape[0]),
    }
    with open(json_path, "w") as f:
        json.dump(details, f, indent=2)
    print("  Saved", json_path)

    # --- Summary table -------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    header = "{:<8} {:>10} {:>10} {:>12} {:>12} {:>10}".format(
        "Algo", "Price", "FitTime", "CVaR95_mdl", "CVaR95_intr", "HedgeTime")
    print(header)
    print("-" * len(header))
    for algo in ALGOS:
        r = results[algo]
        print("{:<8} {:>10.4f} {:>10.2f}s {:>12.4f} {:>12.4f} {:>10.2f}s".format(
            algo, r["price"], r["fit_time"],
            r["var_cvar_model"]["cvar"],
            r["var_cvar_intr"]["cvar"],
            r["hedge_time"],
        ))
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
