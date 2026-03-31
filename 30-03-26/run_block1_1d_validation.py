# -*- coding: utf-8 -*-
"""
Block 1: 1D American Put validation. All methods should converge to the same
reference price (binomial tree).

Usage:
  conda-activate OptStopRandNN
  cd Optimal-Stop-Problem
  PYTHONPATH=. python 30-03-26/run_block1_1d_validation.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_PKG = str(Path(__file__).resolve().parent)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import numpy as np

from config import ExperimentParams
from build_model import build_payoff, build_market_model
from backward_snapshots import fit_backward_snapshots


def binomial_american_put(S0, K, T, r, sigma, N=5000):
    """CRR binomial tree for American put price."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    prices = S0 * d ** np.arange(N, -1, -1) * u ** np.arange(0, N + 1, 1)
    V = np.maximum(K - prices, 0.0)

    for i in range(N - 1, -1, -1):
        prices_i = S0 * d ** np.arange(i, -1, -1) * u ** np.arange(0, i + 1, 1)
        V = disc * (p * V[1:] + (1 - p) * V[:-1])
        exercise = np.maximum(K - prices_i, 0.0)
        V = np.maximum(V, exercise)

    return float(V[0])


ALGOS = ["LSM", "NLSM", "RLSM"]
NB_SEEDS = 3


def main():
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)

    params = ExperimentParams(
        payoff_name="Put1Dim",
        nb_stocks=1,
        strike=100.0,
        spot=100.0,
        maturity=1.0,
        drift=0.05,
        volatility=0.2,
        dividend=0.0,
        nb_dates=50,
        nb_paths_train=50000,
        hidden_size=128,
        nb_epochs_nlsm=80,
    )

    ref_price = binomial_american_put(
        params.spot, params.strike, params.maturity,
        params.drift, params.volatility, N=5000)
    print("Reference (binomial N=5000): {:.6f}".format(ref_price))

    rows = []
    for algo in ALGOS:
        for s in range(NB_SEEDS):
            seed = 42 + s * 1000
            print("[Block1] algo={} seed={}".format(algo, seed))
            payoff_f = build_payoff(params)
            model = build_market_model(params)

            t0 = time.time()
            bundle = fit_backward_snapshots(
                algo=algo, model=model, payoff_f=payoff_f,
                hidden_size=params.hidden_size,
                nb_epochs_nlsm=params.nb_epochs_nlsm,
                rlsm_factors=params.rlsm_factors,
                train_itm_only=True,
                train_eval_split=params.train_eval_split,
                seed=seed,
            )
            elapsed = time.time() - t0
            price = bundle.price_terminal_discounted or 0.0
            rel_err = abs(price - ref_price) / ref_price * 100
            row = {
                "algo": algo, "seed": seed,
                "price": price, "ref_price": ref_price,
                "rel_error_pct": rel_err,
                "fit_time_s": elapsed,
            }
            rows.append(row)
            print("  price={:.6f} ref={:.6f} rel_err={:.2f}% time={:.2f}s".format(
                price, ref_price, rel_err, elapsed))

    csv_path = out_dir / "block1_1d_validation.csv"
    with open(str(csv_path), "w") as f:
        w = csv.DictWriter(f, fieldnames=[
            "algo", "seed", "price", "ref_price", "rel_error_pct", "fit_time_s"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n=== Block 1 Summary ===")
    print("Reference price: {:.6f}".format(ref_price))
    for algo in ALGOS:
        subset = [r for r in rows if r["algo"] == algo]
        prices = [r["price"] for r in subset]
        errs = [r["rel_error_pct"] for r in subset]
        print("  {}: price={:.4f} +/- {:.4f}, rel_err={:.2f}%".format(
            algo, np.mean(prices), np.std(prices), np.mean(errs)))
    print("Saved to {}".format(csv_path))


if __name__ == "__main__":
    main()
