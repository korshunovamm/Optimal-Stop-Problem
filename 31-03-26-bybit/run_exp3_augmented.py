# -*- coding: utf-8 -*-
"""
Experiment 3b: Same as Exp 3 but with data augmentation (noise-perturbed copies
of training paths) to give NN methods more data to learn from.

Usage (from project root):
    PYTHONUNBUFFERED=1 /home/kulikoval/.conda/envs/OptStopRandNN/bin/python -u \
        31-03-26-bybit/run_exp3_augmented.py
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
from hedge_simulation import hedge_one_path
from risk_metrics import var_cvar, summary_stats

from optimal_stopping.payoffs.payoff import Put1Dim


ALGOS = ["LSM", "RLSM", "NLSM"]
AUGMENT_COUNT = 4000
NOISE_SIGMA = 0.008


def _maturity_years(days):
    return days / 365.0


def run_pricing(algo, train_paths, params):
    maturity = _maturity_years(params.maturity_days)
    model = HistoricalModel(
        paths=train_paths, rate=params.rate,
        maturity=maturity, nb_dates=params.nb_dates,
    )
    payoff_f = Put1Dim(params.strike)
    t0 = time.time()
    bundle = fit_backward_snapshots(
        algo=algo, model=model, payoff_f=payoff_f,
        hidden_size=params.hidden_size,
        nb_epochs_nlsm=params.nb_epochs_nlsm,
        rlsm_factors=params.rlsm_factors,
        train_itm_only=True,
        train_eval_split=params.train_eval_split,
        seed=params.seed_train,
    )
    return {"bundle": bundle, "price": bundle.price_terminal_discounted,
            "fit_time": time.time() - t0}


def run_hedge(bundle, test_paths, params):
    payoff_f = Put1Dim(params.strike)
    n_test = min(params.nb_test_hedge_paths, test_paths.shape[0])
    test_sub = test_paths[:n_test]
    losses_model = np.zeros(n_test)
    losses_intr = np.zeros(n_test)
    t0 = time.time()
    for j in range(n_test):
        out = hedge_one_path(bundle, payoff_f, test_sub[j],
                             hidden_size=params.hidden_size,
                             bump=params.bump_spot)
        losses_model[j] = float(np.max(out["disc_shortfall_vs_model"]))
        losses_intr[j] = float(np.max(out["disc_shortfall_vs_intrinsic"]))
        if (j + 1) % 50 == 0:
            print("    hedged {}/{} paths".format(j + 1, n_test))
            sys.stdout.flush()
    return {
        "hedge_time": time.time() - t0,
        "n_test": n_test,
        "losses_model": losses_model, "losses_intr": losses_intr,
        "var_cvar_model": var_cvar(losses_model),
        "var_cvar_intr": var_cvar(losses_intr),
        "stats_model": summary_stats(losses_model),
        "stats_intr": summary_stats(losses_intr),
    }


def main():
    params = CFG
    csv_path = os.path.join(_ROOT, params.csv_path)
    out_dir = os.path.join(_ROOT, params.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Experiment 3b: Historical Bootstrap + Augmentation")
    print("  augment={}, noise_sigma={}".format(AUGMENT_COUNT, NOISE_SIGMA))
    print("=" * 70)
    sys.stdout.flush()

    print("\n[1/4] Loading & augmenting BTC data ...")
    sys.stdout.flush()
    train_paths, _ = prepare_btc_paths(
        csv_path, params.train_start, params.train_end,
        window_size=params.window_size, step=params.window_step,
        reference_spot=params.reference_spot,
        augment=AUGMENT_COUNT, noise_sigma=NOISE_SIGMA, seed=42)
    test_paths, _ = prepare_btc_paths(
        csv_path, params.test_start, params.test_end,
        window_size=params.window_size, step=params.window_step,
        reference_spot=params.reference_spot, augment=0)
    print("  Train paths: {} (original + augmented)".format(train_paths.shape))
    print("  Test  paths: {}".format(test_paths.shape))
    sys.stdout.flush()

    np.random.seed(params.seed_train)
    idx = np.random.permutation(train_paths.shape[0])
    train_paths = train_paths[idx]

    print("\n[2/4] Training ...")
    sys.stdout.flush()
    results = {}
    for algo in ALGOS:
        print("  >>> {} ...".format(algo))
        sys.stdout.flush()
        res = run_pricing(algo, train_paths, params)
        results[algo] = res
        print("  {} — price={:.4f}, fit={:.2f}s".format(algo, res["price"], res["fit_time"]))
        sys.stdout.flush()

    print("\n[3/4] Hedging ...")
    sys.stdout.flush()
    for algo in ALGOS:
        print("  >>> Hedging {} ...".format(algo))
        sys.stdout.flush()
        h = run_hedge(results[algo]["bundle"], test_paths, params)
        results[algo].update(h)
        print("  {} — CVaR95_intr={:.4f}, hedge_time={:.2f}s".format(
            algo, h["var_cvar_intr"]["cvar"], h["hedge_time"]))
        sys.stdout.flush()

    print("\n[4/4] Saving ...")
    csv_out = os.path.join(out_dir, "exp3_augmented_summary.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "price", "fit_time", "var95_intr", "cvar95_intr",
                     "mean_shortfall_intr", "hedge_time", "n_test"])
        for algo in ALGOS:
            r = results[algo]
            w.writerow([algo, "{:.4f}".format(r["price"]),
                        "{:.2f}".format(r["fit_time"]),
                        "{:.4f}".format(r["var_cvar_intr"]["var"]),
                        "{:.4f}".format(r["var_cvar_intr"]["cvar"]),
                        "{:.4f}".format(r["stats_intr"]["mean"]),
                        "{:.2f}".format(r["hedge_time"]), r["n_test"]])
    print("  Saved", csv_out)

    json_out = os.path.join(out_dir, "exp3_augmented_details.json")
    details = {}
    for algo in ALGOS:
        r = results[algo]
        details[algo] = {
            "price": r["price"], "fit_time": r["fit_time"],
            "hedge_time": r["hedge_time"], "n_test": r["n_test"],
            "var_cvar_model": r["var_cvar_model"],
            "var_cvar_intr": r["var_cvar_intr"],
            "stats_model": r["stats_model"],
            "stats_intr": r["stats_intr"],
            "losses_model": r["losses_model"].tolist(),
            "losses_intr": r["losses_intr"].tolist(),
        }
    details["params"] = {
        "augment": AUGMENT_COUNT, "noise_sigma": NOISE_SIGMA,
        "train_paths": int(train_paths.shape[0]),
        "test_paths": int(test_paths.shape[0]),
    }
    with open(json_out, "w") as f:
        json.dump(details, f, indent=2)
    print("  Saved", json_out)

    print("\n" + "=" * 70)
    print("RESULTS (augmented)")
    print("=" * 70)
    hdr = "{:<8} {:>10} {:>10} {:>12} {:>12} {:>10}".format(
        "Algo", "Price", "FitTime", "VaR95_intr", "CVaR95_intr", "HedgeTime")
    print(hdr)
    print("-" * len(hdr))
    for algo in ALGOS:
        r = results[algo]
        print("{:<8} {:>10.4f} {:>10.2f}s {:>12.4f} {:>12.4f} {:>10.2f}s".format(
            algo, r["price"], r["fit_time"],
            r["var_cvar_intr"]["var"], r["var_cvar_intr"]["cvar"],
            r["hedge_time"]))
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
