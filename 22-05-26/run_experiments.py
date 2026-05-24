# -*- coding: utf-8 -*-
"""
Comprehensive NN hedging advantage experiments on BTC historical data.

Experiment A: Fair-capital hedge — same V₀ for all algos, d=1,5,10,15
Experiment B: Properly tuned NLSM — hidden=256, epochs=200
Experiment C: Rebalancing frequency — 4h vs daily at d=10

Usage:
    PYTHONUNBUFFERED=1 /home/kulikoval/.conda/envs/OptStopRandNN/bin/python -u \
        22-05-26/run_experiments.py
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

from feature_builder import (load_btc_daily, add_all_features, build_paths,
                              augment_paths)
from historical_model import HistoricalModel
from backward_snapshots import fit_backward_snapshots
from btc_payoff import BtcPut
from fair_hedge import fair_hedge_one_path
from american_value import get_delta
from risk_metrics import var_cvar, summary_stats

CSV = os.path.join(_ROOT, "data", "bybit_BTCUSDT_ohlc_interval_1min.csv")
OUT = os.path.join(_ROOT, "22-05-26", "output")

REFERENCE_SPOT = 100.0
STRIKE = 100.0
RATE = 0.05
SEED = 42


# ═══════════════════════════════════════════════════════════════════════
#  Experiment A: Fair-capital hedge at d=1,5,10,15
# ═══════════════════════════════════════════════════════════════════════

def experiment_a():
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Fair-Capital Hedge Comparison")
    print("  Same V₀ for all algorithms → isolate delta quality")
    print("=" * 70)
    sys.stdout.flush()

    DIMS = [1, 5, 10, 15]
    ALGOS = ["LSM", "RLSM", "NLSM"]
    HIDDEN = 128
    NB_EPOCHS = 50
    AUGMENT = 3000
    NB_HEDGE = 80
    MATURITY_DAYS = 30

    df_train = load_btc_daily(CSV, "2021-07-06", "2023-12-31")
    df_train = add_all_features(df_train)
    df_test = load_btc_daily(CSV, "2024-01-01", "2024-07-22")
    df_test = add_all_features(df_test)
    print("  Train: {} days, Test: {} days".format(len(df_train), len(df_test)))
    sys.stdout.flush()

    results = {}
    for d in DIMS:
        print("\n  --- d={} ---".format(d))
        sys.stdout.flush()

        train_p = build_paths(df_train, 30, 1, d, REFERENCE_SPOT)
        test_p = build_paths(df_test, 30, 1, d, REFERENCE_SPOT)
        extra = augment_paths(train_p, AUGMENT, 0.008, seed=SEED)
        train_p = np.concatenate([train_p, extra], axis=0)
        np.random.seed(SEED)
        train_p = train_p[np.random.permutation(train_p.shape[0])]

        maturity = MATURITY_DAYS / 365.0
        payoff_f = BtcPut(STRIKE)

        # Phase 1: train all models
        bundles = {}
        prices = {}
        for algo in ALGOS:
            model = HistoricalModel(train_p, RATE, maturity, MATURITY_DAYS)
            t0 = time.time()
            b = fit_backward_snapshots(
                algo=algo, model=model, payoff_f=payoff_f,
                hidden_size=HIDDEN, nb_epochs_nlsm=NB_EPOCHS,
                rlsm_factors=(1.0,), train_itm_only=True,
                train_eval_split=2, seed=SEED)
            ft = time.time() - t0
            bundles[algo] = b
            prices[algo] = float(b.price_terminal_discounted)
            print("    {} price={:.3f} fit={:.2f}s".format(algo, prices[algo], ft))
            sys.stdout.flush()

        # Phase 2: hedge with COMMON V₀ = max of all model prices
        common_v0 = max(prices.values())
        print("    Common V₀ = {:.3f}".format(common_v0))
        sys.stdout.flush()

        n_h = min(NB_HEDGE, test_p.shape[0])
        dim_res = {}
        for algo in ALGOS:
            losses_own = np.zeros(n_h)   # each model's own V₀
            losses_fair = np.zeros(n_h)  # common V₀
            all_deltas = []
            t0 = time.time()
            for j in range(n_h):
                out_own = fair_hedge_one_path(
                    bundles[algo], payoff_f, test_p[j],
                    hidden_size=HIDDEN, bump=0.5, external_v0=None)
                out_fair = fair_hedge_one_path(
                    bundles[algo], payoff_f, test_p[j],
                    hidden_size=HIDDEN, bump=0.5, external_v0=common_v0)
                losses_own[j] = out_own["max_shortfall"]
                losses_fair[j] = out_fair["max_shortfall"]
                all_deltas.append(out_fair["deltas"].tolist())
            ht = time.time() - t0

            vc_own = var_cvar(losses_own)
            vc_fair = var_cvar(losses_fair)

            dim_res[algo] = {
                "price": prices[algo],
                "cvar95_own_v0": vc_own["cvar"],
                "var95_own_v0": vc_own["var"],
                "mean_own_v0": float(losses_own.mean()),
                "cvar95_fair": vc_fair["cvar"],
                "var95_fair": vc_fair["var"],
                "mean_fair": float(losses_fair.mean()),
                "common_v0": common_v0,
                "hedge_time": ht,
                "losses_fair": losses_fair.tolist(),
                "losses_own": losses_own.tolist(),
            }
            print("    {} CVaR95(own)={:.2f} CVaR95(fair)={:.2f} hedge={:.1f}s".format(
                algo, vc_own["cvar"], vc_fair["cvar"], ht))
            sys.stdout.flush()

        results[str(d)] = dim_res

    with open(os.path.join(OUT, "exp_a_fair_capital.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved exp_a_fair_capital.json")
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Experiment B: Properly tuned NLSM at d=5,10
# ═══════════════════════════════════════════════════════════════════════

def experiment_b():
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Properly Tuned NLSM")
    print("  hidden=256, epochs=200 → full convergence")
    print("=" * 70)
    sys.stdout.flush()

    DIMS = [5, 10]
    AUGMENT = 4000
    NB_HEDGE = 80
    MATURITY_DAYS = 30

    configs = [
        ("LSM",  128, 0,   "LSM-128"),
        ("RLSM", 128, 0,   "RLSM-128"),
        ("RLSM", 256, 0,   "RLSM-256"),
        ("NLSM", 128, 50,  "NLSM-128-ep50"),
        ("NLSM", 128, 200, "NLSM-128-ep200"),
        ("NLSM", 256, 200, "NLSM-256-ep200"),
    ]

    df_train = load_btc_daily(CSV, "2021-07-06", "2023-12-31")
    df_train = add_all_features(df_train)
    df_test = load_btc_daily(CSV, "2024-01-01", "2024-07-22")
    df_test = add_all_features(df_test)

    results = {}
    for d in DIMS:
        print("\n  --- d={} ---".format(d))
        sys.stdout.flush()

        train_p = build_paths(df_train, 30, 1, d, REFERENCE_SPOT)
        test_p = build_paths(df_test, 30, 1, d, REFERENCE_SPOT)
        extra = augment_paths(train_p, AUGMENT, 0.008, seed=SEED)
        train_p = np.concatenate([train_p, extra], axis=0)
        np.random.seed(SEED)
        train_p = train_p[np.random.permutation(train_p.shape[0])]

        maturity = MATURITY_DAYS / 365.0
        payoff_f = BtcPut(STRIKE)

        bundles = {}
        all_prices = {}
        for algo, hid, ep, label in configs:
            model = HistoricalModel(train_p, RATE, maturity, MATURITY_DAYS)
            t0 = time.time()
            b = fit_backward_snapshots(
                algo=algo, model=model, payoff_f=payoff_f,
                hidden_size=hid, nb_epochs_nlsm=ep,
                rlsm_factors=(1.0,), train_itm_only=True,
                train_eval_split=2, seed=SEED)
            ft = time.time() - t0
            bundles[label] = (b, hid)
            p = float(b.price_terminal_discounted)
            all_prices[label] = p
            print("    {} price={:.3f} fit={:.2f}s".format(label, p, ft))
            sys.stdout.flush()

        common_v0 = max(all_prices.values())
        print("    Common V₀ = {:.3f}".format(common_v0))

        n_h = min(NB_HEDGE, test_p.shape[0])
        dim_res = {}
        for algo, hid, ep, label in configs:
            bundle, hidden = bundles[label]
            losses = np.zeros(n_h)
            t0 = time.time()
            for j in range(n_h):
                out = fair_hedge_one_path(
                    bundle, payoff_f, test_p[j],
                    hidden_size=hidden, bump=0.5,
                    external_v0=common_v0)
                losses[j] = out["max_shortfall"]
            ht = time.time() - t0

            vc = var_cvar(losses)
            # delta speed
            np.random.seed(777)
            spots = np.random.uniform(85, 115, (200, d))
            for k in range(1, d):
                spots[:, k] = REFERENCE_SPOT + np.random.normal(
                    0, REFERENCE_SPOT * 0.1, 200)
            dates = np.random.randint(1, MATURITY_DAYS, 200)
            for i in range(3):
                get_delta(bundle, payoff_f, int(dates[i]), spots[i],
                          hidden_size=hidden, bump=0.5)
            dt0 = time.time()
            for i in range(200):
                get_delta(bundle, payoff_f, int(dates[i]), spots[i],
                          hidden_size=hidden, bump=0.5)
            delta_us = (time.time() - dt0) / 200 * 1e6

            dim_res[label] = {
                "price": all_prices[label],
                "cvar95": vc["cvar"],
                "var95": vc["var"],
                "mean": float(losses.mean()),
                "hedge_time": ht,
                "delta_us": delta_us,
                "losses": losses.tolist(),
            }
            print("    {} CVaR95={:.2f} δ={:.0f}µs hedge={:.1f}s".format(
                label, vc["cvar"], delta_us, ht))
            sys.stdout.flush()

        results[str(d)] = dim_res

    with open(os.path.join(OUT, "exp_b_nlsm_tuned.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved exp_b_nlsm_tuned.json")
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Experiment C: Rebalancing frequency — hourly data, 4h steps at d=10
# ═══════════════════════════════════════════════════════════════════════

def experiment_c():
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Rebalancing Frequency Advantage")
    print("  Compare daily vs 8h vs 4h rebalancing at d=5")
    print("  NN can rebalance more often → better hedge")
    print("=" * 70)
    sys.stdout.flush()

    import pandas as pd

    d = 5
    AUGMENT = 3000
    HIDDEN = 128
    NB_EPOCHS = 100
    MATURITY_DAYS = 30

    # Load hourly BTC data
    print("  Loading hourly data ...")
    sys.stdout.flush()
    df_raw = pd.read_csv(CSV, parse_dates=["datetime"])
    df_raw = df_raw.set_index("datetime").sort_index()

    freqs = {
        "daily":  ("1D",  30),
        "8h":     ("8H",  90),
        "4h":     ("4H",  180),
    }

    results = {}
    for freq_name, (resample_rule, nb_dates) in freqs.items():
        print("\n  --- {} rebalancing (nb_dates={}) ---".format(
            freq_name, nb_dates))
        sys.stdout.flush()

        df_resampled = df_raw.resample(resample_rule).agg({
            "price_open": "first",
            "price_high": "max",
            "price_low": "min",
            "price_close": "last",
            "volume": "sum",
            "turnover": "sum",
        }).dropna()

        df_tr = df_resampled[df_resampled.index < "2024-01-01"]
        df_te = df_resampled[df_resampled.index >= "2024-01-01"]
        df_te = df_te[df_te.index < "2024-07-22"]

        # Compute features
        for name, func in __import__('feature_builder', fromlist=['FEATURE_CATALOGUE']).FEATURE_CATALOGUE[:d-1]:
            df_tr[name] = func(df_tr)
            df_te[name] = func(df_te)

        train_p = build_paths(df_tr, nb_dates, max(1, nb_dates // 10), d,
                              REFERENCE_SPOT)
        test_p = build_paths(df_te, nb_dates, max(1, nb_dates // 10), d,
                             REFERENCE_SPOT)

        if train_p.shape[0] < 50:
            print("    Skip: not enough paths ({})".format(train_p.shape[0]))
            continue

        extra = augment_paths(train_p, AUGMENT, 0.008, seed=SEED)
        train_p = np.concatenate([train_p, extra], axis=0)
        np.random.seed(SEED)
        train_p = train_p[np.random.permutation(train_p.shape[0])]

        print("    Train: {}, Test: {}".format(train_p.shape, test_p.shape))
        sys.stdout.flush()

        maturity = MATURITY_DAYS / 365.0
        payoff_f = BtcPut(STRIKE)

        freq_res = {}
        all_prices = {}
        bundles = {}
        for algo in ["LSM", "RLSM", "NLSM"]:
            model = HistoricalModel(train_p, RATE, maturity, nb_dates)
            t0 = time.time()
            b = fit_backward_snapshots(
                algo=algo, model=model, payoff_f=payoff_f,
                hidden_size=HIDDEN, nb_epochs_nlsm=NB_EPOCHS,
                rlsm_factors=(1.0,), train_itm_only=True,
                train_eval_split=2, seed=SEED)
            ft = time.time() - t0
            bundles[algo] = b
            all_prices[algo] = float(b.price_terminal_discounted)
            print("    {} price={:.3f} fit={:.2f}s".format(
                algo, all_prices[algo], ft))
            sys.stdout.flush()

        common_v0 = max(all_prices.values())
        n_h = min(60, test_p.shape[0])

        for algo in ["LSM", "RLSM", "NLSM"]:
            losses = np.zeros(n_h)
            t0 = time.time()
            for j in range(n_h):
                out = fair_hedge_one_path(
                    bundles[algo], payoff_f, test_p[j],
                    hidden_size=HIDDEN, bump=0.5,
                    external_v0=common_v0)
                losses[j] = out["max_shortfall"]
            ht = time.time() - t0

            vc = var_cvar(losses)
            freq_res[algo] = {
                "price": all_prices[algo],
                "cvar95": vc["cvar"],
                "var95": vc["var"],
                "mean": float(losses.mean()),
                "hedge_time": ht,
                "nb_dates": nb_dates,
                "losses": losses.tolist(),
            }
            print("    {} CVaR95={:.2f} mean={:.2f} hedge={:.1f}s".format(
                algo, vc["cvar"], float(losses.mean()), ht))
            sys.stdout.flush()

        results[freq_name] = freq_res

    with open(os.path.join(OUT, "exp_c_rebalance.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved exp_c_rebalance.json")
    return results


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 70)
    print("  22-05-26: NN Hedging Advantage on BTC — Full Suite")
    print("=" * 70)
    sys.stdout.flush()

    ra = experiment_a()
    rb = experiment_b()
    rc = experiment_c()

    # ── Final summary ────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\nA) Fair-Capital Hedge (CVaR95 with same V₀):")
    for d in sorted(ra.keys(), key=int):
        print("  d={}:".format(d))
        for algo in ["LSM", "RLSM", "NLSM"]:
            r = ra[d][algo]
            print("    {:<6} own={:.2f}  fair={:.2f}".format(
                algo, r["cvar95_own_v0"], r["cvar95_fair"]))

    print("\nB) Tuned NLSM vs baseline:")
    for d in sorted(rb.keys(), key=int):
        print("  d={}:".format(d))
        for label in sorted(rb[d].keys()):
            r = rb[d][label]
            print("    {:<20} CVaR95={:.2f}  δ={:.0f}µs".format(
                label, r["cvar95"], r["delta_us"]))

    print("\nC) Rebalancing frequency (d=5):")
    for freq in ["daily", "8h", "4h"]:
        if freq not in rc:
            continue
        print("  {}:".format(freq))
        for algo in ["LSM", "RLSM", "NLSM"]:
            r = rc[freq][algo]
            print("    {:<6} CVaR95={:.2f}  mean={:.2f}".format(
                algo, r["cvar95"], r["mean"]))

    print("\n" + "=" * 80)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
