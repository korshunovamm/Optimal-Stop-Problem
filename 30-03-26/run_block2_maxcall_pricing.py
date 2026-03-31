# -*- coding: utf-8 -*-
"""
Block 2: Multi-Asset MaxCall pricing comparison LSM vs NLSM vs RLSM
for d = 2, 5, 10.

Usage:
  conda-activate OptStopRandNN
  cd Optimal-Stop-Problem
  PYTHONPATH=. python 30-03-26/run_block2_maxcall_pricing.py
"""
from __future__ import annotations

import csv
import json
import os
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

ALGOS = ["LSM", "NLSM", "RLSM"]
DIMS = [2, 5, 10]
NB_SEEDS = 3

PARAMS_TEMPLATE = ExperimentParams(
    payoff_name="MaxCall",
    strike=100.0,
    spot=100.0,
    maturity=1.0,
    drift=0.05,
    volatility=0.2,
    dividend=0.0,
    nb_dates=10,
    nb_paths_train=40000,
    hidden_size=128,
    nb_epochs_nlsm=80,
    rlsm_factors=(1.0,),
    train_eval_split=2,
)


def run_one(algo, nb_stocks, seed):
    params = ExperimentParams(
        payoff_name=PARAMS_TEMPLATE.payoff_name,
        strike=PARAMS_TEMPLATE.strike,
        spot=PARAMS_TEMPLATE.spot,
        maturity=PARAMS_TEMPLATE.maturity,
        drift=PARAMS_TEMPLATE.drift,
        volatility=PARAMS_TEMPLATE.volatility,
        dividend=PARAMS_TEMPLATE.dividend,
        nb_dates=PARAMS_TEMPLATE.nb_dates,
        nb_stocks=nb_stocks,
        nb_paths_train=PARAMS_TEMPLATE.nb_paths_train,
        hidden_size=PARAMS_TEMPLATE.hidden_size,
        nb_epochs_nlsm=PARAMS_TEMPLATE.nb_epochs_nlsm,
        rlsm_factors=PARAMS_TEMPLATE.rlsm_factors,
        train_eval_split=PARAMS_TEMPLATE.train_eval_split,
        seed_train=seed,
    )
    payoff_f = build_payoff(params)
    model = build_market_model(params)

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
        seed=seed,
    )
    elapsed = time.time() - t0
    price = bundle.price_terminal_discounted or 0.0
    return {
        "algo": algo,
        "nb_stocks": nb_stocks,
        "seed": seed,
        "price": price,
        "fit_time_s": elapsed,
        "path_gen_s": bundle.path_gen_seconds,
    }


def main():
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "block2_maxcall_pricing.csv"

    rows = []
    for d in DIMS:
        for algo in ALGOS:
            for s in range(NB_SEEDS):
                seed = 42 + s * 1000
                print("[Block2] algo={} d={} seed={}".format(algo, d, seed))
                try:
                    r = run_one(algo, d, seed)
                    rows.append(r)
                    print("  price={:.4f}  time={:.2f}s".format(r["price"], r["fit_time_s"]))
                except Exception as e:
                    print("  FAILED: {}".format(e))
                    rows.append({
                        "algo": algo, "nb_stocks": d, "seed": seed,
                        "price": float("nan"), "fit_time_s": float("nan"),
                        "path_gen_s": float("nan"),
                    })

    with open(str(csv_path), "w") as f:
        w = csv.DictWriter(f, fieldnames=["algo", "nb_stocks", "seed", "price", "fit_time_s", "path_gen_s"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n=== Block 2 Summary ===")
    for d in DIMS:
        print("\nd = {}:".format(d))
        for algo in ALGOS:
            subset = [r for r in rows if r["algo"] == algo and r["nb_stocks"] == d
                       and not np.isnan(r["price"])]
            if subset:
                prices = [r["price"] for r in subset]
                times = [r["fit_time_s"] for r in subset]
                print("  {}: price={:.4f} +/- {:.4f}, time={:.2f}s".format(
                    algo, np.mean(prices), np.std(prices), np.mean(times)))
            else:
                print("  {}: FAILED".format(algo))

    print("\nSaved to {}".format(csv_path))


if __name__ == "__main__":
    main()
