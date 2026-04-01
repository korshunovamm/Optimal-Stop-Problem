# -*- coding: utf-8 -*-
"""
Experiment 5: Delta inference speed benchmark.
Measure per-point delta computation time for LSM (FD), RLSM (analytical), NLSM (autograd).

Usage (from project root):
    PYTHONUNBUFFERED=1 /home/kulikoval/.conda/envs/OptStopRandNN/bin/python -u \
        31-03-26-bybit/run_exp5_delta_speed.py
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
from american_value import get_delta

from optimal_stopping.payoffs.payoff import Put1Dim

ALGOS = ["LSM", "RLSM", "NLSM"]
N_DELTA_EVALS = 2000
AUGMENT = 2000


def main():
    csv_path = os.path.join(_ROOT, CFG.csv_path)
    out_dir = os.path.join(_ROOT, CFG.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Experiment 5: Delta Speed Benchmark")
    print("=" * 70)
    sys.stdout.flush()

    train_paths, _ = prepare_btc_paths(
        csv_path, CFG.train_start, CFG.train_end,
        window_size=CFG.window_size, step=CFG.window_step,
        reference_spot=CFG.reference_spot, augment=AUGMENT,
        noise_sigma=0.008, seed=42)

    print("Train paths: {}".format(train_paths.shape))
    np.random.seed(42)
    train_paths = train_paths[np.random.permutation(train_paths.shape[0])]

    maturity = CFG.maturity_days / 365.0
    payoff_f = Put1Dim(CFG.strike)

    bundles = {}
    for algo in ALGOS:
        print("\n  Training {} ...".format(algo))
        sys.stdout.flush()
        model = HistoricalModel(train_paths, CFG.rate, maturity, CFG.nb_dates)
        bundle = fit_backward_snapshots(
            algo=algo, model=model, payoff_f=payoff_f,
            hidden_size=CFG.hidden_size,
            nb_epochs_nlsm=CFG.nb_epochs_nlsm,
            rlsm_factors=CFG.rlsm_factors,
            train_itm_only=True,
            train_eval_split=CFG.train_eval_split,
            seed=CFG.seed_train)
        bundles[algo] = bundle
        print("  {} trained, price={:.4f}".format(algo, bundle.price_terminal_discounted))

    np.random.seed(123)
    spot_samples = np.random.uniform(80, 120, N_DELTA_EVALS)
    date_samples = np.random.randint(1, CFG.nb_dates, N_DELTA_EVALS)

    results = {}
    for algo in ALGOS:
        bundle = bundles[algo]
        print("\n  Benchmarking {} delta ({} evaluations) ...".format(algo, N_DELTA_EVALS))
        sys.stdout.flush()

        # Warmup
        for i in range(min(10, N_DELTA_EVALS)):
            s = np.array([spot_samples[i]])
            get_delta(bundle, payoff_f, int(date_samples[i]), s,
                      hidden_size=CFG.hidden_size, bump=CFG.bump_spot)

        t0 = time.time()
        for i in range(N_DELTA_EVALS):
            s = np.array([spot_samples[i]])
            get_delta(bundle, payoff_f, int(date_samples[i]), s,
                      hidden_size=CFG.hidden_size, bump=CFG.bump_spot)
        elapsed = time.time() - t0

        mean_us = elapsed / N_DELTA_EVALS * 1e6
        results[algo] = {
            "total_time_s": elapsed,
            "n_evals": N_DELTA_EVALS,
            "mean_delta_us": mean_us,
        }
        print("  {} — total={:.3f}s, mean={:.0f} µs/delta".format(
            algo, elapsed, mean_us))
        sys.stdout.flush()

    # Speedup ratios
    lsm_time = results["LSM"]["mean_delta_us"]
    for algo in ALGOS:
        results[algo]["speedup_vs_lsm"] = lsm_time / results[algo]["mean_delta_us"]

    json_out = os.path.join(out_dir, "exp5_speed.json")
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved", json_out)

    print("\n" + "=" * 70)
    print("DELTA SPEED SUMMARY")
    print("=" * 70)
    hdr = "{:<8} {:>15} {:>15}".format("Algo", "µs / delta", "Speedup vs LSM")
    print(hdr)
    print("-" * len(hdr))
    for algo in ALGOS:
        r = results[algo]
        print("{:<8} {:>15.0f} {:>15.2f}x".format(
            algo, r["mean_delta_us"], r["speedup_vs_lsm"]))
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
