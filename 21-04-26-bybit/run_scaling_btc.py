# -*- coding: utf-8 -*-
"""
High-dimensional scaling experiment on BTC historical paths.

For each d in {1, 5, 10, 15, 20, 30}:
  - Train LSM / RLSM / NLSM on augmented BTC paths with d features
  - Measure pricing, hedge quality (CVaR95), training time
  - Benchmark per-delta computation speed

Usage (from project root):
    PYTHONUNBUFFERED=1 /home/kulikoval/.conda/envs/OptStopRandNN/bin/python -u \
        21-04-26-bybit/run_scaling_btc.py
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path

import numpy as np

_DIR = str(Path(__file__).resolve().parent)
_ROOT = str(Path(__file__).resolve().parent.parent)
for p in (_DIR, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from feature_builder import load_btc_daily, add_all_features, build_paths, augment_paths
from historical_model import HistoricalModel
from backward_snapshots import fit_backward_snapshots
from btc_payoff import BtcPut
from btc_hedge import btc_hedge_one_path
from american_value import get_delta
from risk_metrics import var_cvar, summary_stats

CSV = os.path.join(_ROOT, "data", "bybit_BTCUSDT_ohlc_interval_1min.csv")
OUT = os.path.join(_ROOT, "21-04-26-bybit", "output")

DIMS = [1, 5, 10, 15, 20, 30]
ALGOS = ["LSM", "RLSM", "NLSM"]
AUGMENT = 3000
NOISE_SIGMA = 0.008
REFERENCE_SPOT = 100.0
STRIKE = 100.0
MATURITY_DAYS = 30
NB_DATES = 30
RATE = 0.05
HIDDEN = 128
NB_EPOCHS = 50          # shorter for NLSM since there are many dim combos
NB_HEDGE = 80           # test hedge paths per dim
NB_DELTA_BENCH = 500    # delta speed benchmark evals
SEED = 42


def train_one(algo, train_paths, d):
    maturity = MATURITY_DAYS / 365.0
    model = HistoricalModel(train_paths, RATE, maturity, NB_DATES)
    payoff_f = BtcPut(STRIKE)
    t0 = time.time()
    bundle = fit_backward_snapshots(
        algo=algo, model=model, payoff_f=payoff_f,
        hidden_size=HIDDEN, nb_epochs_nlsm=NB_EPOCHS,
        rlsm_factors=(1.0,), train_itm_only=True,
        train_eval_split=2, seed=SEED)
    fit_time = time.time() - t0
    return bundle, fit_time


def hedge_eval(bundle, test_paths, d):
    payoff_f = BtcPut(STRIKE)
    n = min(NB_HEDGE, test_paths.shape[0])
    losses = np.zeros(n)
    t0 = time.time()
    for j in range(n):
        out = btc_hedge_one_path(bundle, payoff_f, test_paths[j],
                                 hidden_size=HIDDEN, bump=0.5)
        losses[j] = float(np.max(out["disc_shortfall_vs_intrinsic"]))
    ht = time.time() - t0
    vc = var_cvar(losses)
    st = summary_stats(losses)
    return {"losses": losses, "var_cvar": vc, "stats": st,
            "hedge_time": ht, "n": n}


def delta_speed(bundle, d):
    payoff_f = BtcPut(STRIKE)
    np.random.seed(123)
    spots = np.random.uniform(85, 115, (NB_DELTA_BENCH, d))
    spots[:, 0] = np.random.uniform(85, 115, NB_DELTA_BENCH)
    for k in range(1, d):
        spots[:, k] = REFERENCE_SPOT + np.random.normal(0, REFERENCE_SPOT * 0.1, NB_DELTA_BENCH)
    dates = np.random.randint(1, NB_DATES, NB_DELTA_BENCH)
    # warmup
    for i in range(min(5, NB_DELTA_BENCH)):
        get_delta(bundle, payoff_f, int(dates[i]), spots[i],
                  hidden_size=HIDDEN, bump=0.5)
    t0 = time.time()
    for i in range(NB_DELTA_BENCH):
        get_delta(bundle, payoff_f, int(dates[i]), spots[i],
                  hidden_size=HIDDEN, bump=0.5)
    elapsed = time.time() - t0
    return {"total_s": elapsed, "n": NB_DELTA_BENCH,
            "mean_us": elapsed / NB_DELTA_BENCH * 1e6}


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 70)
    print("High-Dim Scaling Experiment on BTC Historical Paths")
    print("  dims={}, algos={}".format(DIMS, ALGOS))
    print("=" * 70)
    sys.stdout.flush()

    print("\n[1] Loading data ...")
    df_train = load_btc_daily(CSV, "2021-07-06", "2023-12-31")
    df_train = add_all_features(df_train)
    df_test = load_btc_daily(CSV, "2024-01-01", "2024-07-22")
    df_test = add_all_features(df_test)
    print("  Train days: {}, Test days: {}".format(len(df_train), len(df_test)))
    sys.stdout.flush()

    all_results = {}

    for d in DIMS:
        print("\n" + "=" * 60)
        print("  DIMENSION d={}".format(d))
        print("=" * 60)
        sys.stdout.flush()

        train_p = build_paths(df_train, 30, 1, d, REFERENCE_SPOT)
        test_p = build_paths(df_test, 30, 1, d, REFERENCE_SPOT)

        if train_p.shape[0] == 0:
            print("  No valid paths, skipping")
            continue

        extra = augment_paths(train_p, AUGMENT, NOISE_SIGMA, seed=SEED)
        train_p = np.concatenate([train_p, extra], axis=0)
        np.random.seed(SEED)
        train_p = train_p[np.random.permutation(train_p.shape[0])]

        print("  Train: {}, Test: {}".format(train_p.shape, test_p.shape))
        sys.stdout.flush()

        dim_res = {}
        for algo in ALGOS:
            print("\n    >>> {} (d={}) ...".format(algo, d))
            sys.stdout.flush()

            bundle, fit_time = train_one(algo, train_p, d)
            price = float(bundle.price_terminal_discounted)
            print("      price={:.3f}, fit={:.2f}s".format(price, fit_time))
            sys.stdout.flush()

            print("      hedging ({} paths) ...".format(NB_HEDGE))
            sys.stdout.flush()
            h = hedge_eval(bundle, test_p, d)
            print("      CVaR95={:.3f}, hedge_time={:.2f}s".format(
                h["var_cvar"]["cvar"], h["hedge_time"]))
            sys.stdout.flush()

            print("      delta speed ({} evals) ...".format(NB_DELTA_BENCH))
            sys.stdout.flush()
            sp = delta_speed(bundle, d)
            print("      mean_delta={:.0f} µs".format(sp["mean_us"]))
            sys.stdout.flush()

            dim_res[algo] = {
                "price": price, "fit_time": fit_time,
                "cvar95": h["var_cvar"]["cvar"],
                "var95": h["var_cvar"]["var"],
                "mean_shortfall": h["stats"]["mean"],
                "hedge_time": h["hedge_time"],
                "delta_us": sp["mean_us"],
                "n_hedge": h["n"],
                "n_delta": sp["n"],
                "losses": h["losses"].tolist(),
            }

        all_results[str(d)] = dim_res

    # ── Summary ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("SCALING SUMMARY (BTC historical)")
    print("=" * 80)
    hdr = "{:<4} {:<6} {:>8} {:>8} {:>10} {:>10} {:>10}".format(
        "d", "Algo", "Price", "CVaR95", "FitTime", "HedgeTime", "δ µs")
    print(hdr)
    print("-" * len(hdr))
    for d in DIMS:
        for algo in ALGOS:
            r = all_results.get(str(d), {}).get(algo)
            if r is None:
                continue
            print("{:<4} {:<6} {:>8.3f} {:>8.3f} {:>10.2f}s {:>10.2f}s {:>10.0f}".format(
                d, algo, r["price"], r["cvar95"], r["fit_time"],
                r["hedge_time"], r["delta_us"]))
    print("=" * 80)

    fout = os.path.join(OUT, "scaling_btc.json")
    with open(fout, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved", fout)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
