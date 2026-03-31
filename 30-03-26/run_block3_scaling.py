# -*- coding: utf-8 -*-
"""
Block 3: Scalability test. MaxCall/BasketCall at d=50, 100.
LSM should fail or give poor results; RLSM should scale.

Usage:
  conda-activate OptStopRandNN
  cd Optimal-Stop-Problem
  PYTHONPATH=. python 30-03-26/run_block3_scaling.py
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

ALGOS = ["LSM", "RLSM"]
DIMS = [50, 100]


def run_one(algo, nb_stocks, payoff_name="MaxCall", nb_paths=60000,
            hidden_size=500, nb_epochs=50, seed=42):
    params = ExperimentParams(
        payoff_name=payoff_name,
        nb_stocks=nb_stocks,
        strike=100.0, spot=100.0, maturity=1.0,
        drift=0.05, volatility=0.2, dividend=0.0,
        nb_dates=10,
        nb_paths_train=nb_paths,
        hidden_size=hidden_size,
        nb_epochs_nlsm=nb_epochs,
    )
    payoff_f = build_payoff(params)
    model = build_market_model(params)

    t0 = time.time()
    bundle = fit_backward_snapshots(
        algo=algo, model=model, payoff_f=payoff_f,
        hidden_size=hidden_size,
        nb_epochs_nlsm=nb_epochs,
        rlsm_factors=params.rlsm_factors,
        train_itm_only=True,
        train_eval_split=params.train_eval_split,
        seed=seed,
    )
    elapsed = time.time() - t0
    price = bundle.price_terminal_discounted or 0.0
    return {
        "payoff": payoff_name,
        "algo": algo,
        "nb_stocks": nb_stocks,
        "price": price,
        "fit_time_s": elapsed,
        "path_gen_s": bundle.path_gen_seconds,
    }


def main():
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)

    rows = []
    for d in DIMS:
        for algo in ALGOS:
            print("[Block3] algo={} d={}".format(algo, d))
            sys.stdout.flush()
            try:
                r = run_one(algo, d, nb_paths=40000, hidden_size=200, nb_epochs=30)
                rows.append(r)
                print("  price={:.4f} time={:.2f}s".format(r["price"], r["fit_time_s"]))
            except Exception as e:
                print("  FAILED: {}".format(e))
                rows.append({
                    "payoff": "MaxCall", "algo": algo, "nb_stocks": d,
                    "price": float("nan"), "fit_time_s": float("nan"),
                    "path_gen_s": float("nan"),
                })

    csv_path = out_dir / "block3_scaling.csv"
    with open(str(csv_path), "w") as f:
        w = csv.DictWriter(f, fieldnames=[
            "payoff", "algo", "nb_stocks", "price", "fit_time_s", "path_gen_s"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n=== Block 3 Summary ===")
    for d in DIMS:
        print("\nd = {}:".format(d))
        for algo in ALGOS:
            subset = [r for r in rows if r["algo"] == algo and r["nb_stocks"] == d]
            if subset and not np.isnan(subset[0]["price"]):
                print("  {}: price={:.4f} time={:.2f}s".format(
                    algo, subset[0]["price"], subset[0]["fit_time_s"]))
            else:
                print("  {}: FAILED".format(algo))
    print("\nSaved to {}".format(csv_path))


if __name__ == "__main__":
    main()
