# -*- coding: utf-8 -*-
"""
Synthetic high-dimensional scaling experiment (Black-Scholes MaxCall).

Clean environment to show the curse of dimensionality for LSM vs RLSM.
d = 10, 20, 50, 100.

Usage (from project root):
    PYTHONUNBUFFERED=1 /home/kulikoval/.conda/envs/OptStopRandNN/bin/python -u \
        21-04-26-bybit/run_scaling_synthetic.py
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

from backward_snapshots import fit_backward_snapshots
from american_value import get_delta
from hedge_simulation import hedge_one_path
from risk_metrics import var_cvar, summary_stats

from optimal_stopping.data.stock_model import BlackScholes
from optimal_stopping.payoffs.payoff import MaxCall

OUT = os.path.join(_ROOT, "21-04-26-bybit", "output")

DIMS = [10, 20, 50, 100]
ALGOS = ["LSM", "RLSM"]    # skip NLSM for speed at very high d
NB_PATHS = 40000
NB_HEDGE = 60
NB_DELTA_BENCH = 300
STRIKE = 100.0
SPOT = 100.0
VOL = 0.2
RATE = 0.05
MATURITY = 1.0
NB_DATES = 10
HIDDEN = 200
SEED = 42


def make_model(d):
    return BlackScholes(
        drift=RATE, volatility=VOL, nb_paths=NB_PATHS,
        nb_stocks=d, nb_dates=NB_DATES, spot=SPOT,
        maturity=MATURITY, dividend=0.0)


def train(algo, model, d):
    payoff_f = MaxCall(STRIKE)
    t0 = time.time()
    bundle = fit_backward_snapshots(
        algo=algo, model=model, payoff_f=payoff_f,
        hidden_size=HIDDEN, nb_epochs_nlsm=30,
        rlsm_factors=(1.0,), train_itm_only=True,
        train_eval_split=2, seed=SEED)
    return bundle, time.time() - t0


def hedge_eval(bundle, d):
    payoff_f = MaxCall(STRIKE)
    model2 = make_model(d)
    np.random.seed(SEED + 1000)
    test_paths, _ = model2.generate_paths(nb_paths=NB_HEDGE)
    losses = np.zeros(NB_HEDGE)
    t0 = time.time()
    for j in range(NB_HEDGE):
        out = hedge_one_path(bundle, payoff_f, test_paths[j],
                             hidden_size=HIDDEN, bump=0.5)
        losses[j] = float(np.max(out["disc_shortfall_vs_intrinsic"]))
    ht = time.time() - t0
    return {"losses": losses, "var_cvar": var_cvar(losses),
            "stats": summary_stats(losses), "hedge_time": ht}


def delta_speed(bundle, d):
    payoff_f = MaxCall(STRIKE)
    np.random.seed(777)
    spots = np.random.uniform(90, 110, (NB_DELTA_BENCH, d))
    dates = np.random.randint(1, NB_DATES, NB_DELTA_BENCH)
    for i in range(min(3, NB_DELTA_BENCH)):
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
    print("Synthetic High-Dim Scaling (BS MaxCall)")
    print("  dims={}, algos={}".format(DIMS, ALGOS))
    print("=" * 70)
    sys.stdout.flush()

    all_results = {}

    for d in DIMS:
        print("\n" + "=" * 60)
        print("  DIMENSION d={}".format(d))
        print("=" * 60)
        sys.stdout.flush()

        np.random.seed(SEED)
        model = make_model(d)

        dim_res = {}
        for algo in ALGOS:
            print("\n    >>> {} (d={}) ...".format(algo, d))
            sys.stdout.flush()

            bundle, fit_time = train(algo, model, d)
            price = float(bundle.price_terminal_discounted)
            print("      price={:.3f}, fit={:.2f}s".format(price, fit_time))
            sys.stdout.flush()

            print("      hedging ({} paths) ...".format(NB_HEDGE))
            sys.stdout.flush()
            h = hedge_eval(bundle, d)
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
                "losses": h["losses"].tolist(),
            }

        all_results[str(d)] = dim_res

    # ── Summary ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("SYNTHETIC SCALING SUMMARY")
    print("=" * 80)
    hdr = "{:<4} {:<6} {:>8} {:>8} {:>10} {:>10} {:>10}".format(
        "d", "Algo", "Price", "CVaR95", "FitTime", "HedgeTime", "δ µs")
    print(hdr)
    print("-" * len(hdr))
    for d in DIMS:
        for algo in ALGOS:
            r = all_results.get(str(d), {}).get(algo)
            if not r:
                continue
            print("{:<4} {:<6} {:>8.3f} {:>8.3f} {:>10.2f}s {:>10.2f}s {:>10.0f}".format(
                d, algo, r["price"], r["cvar95"], r["fit_time"],
                r["hedge_time"], r["delta_us"]))
    print("=" * 80)

    fout = os.path.join(OUT, "scaling_synthetic.json")
    with open(fout, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved", fout)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
