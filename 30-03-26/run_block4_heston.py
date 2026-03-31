# -*- coding: utf-8 -*-
"""
Block 4: Heston model experiments.
MaxCall d=5 under Heston (standard) and HestonWithVar.

Usage:
  conda-activate OptStopRandNN
  cd Optimal-Stop-Problem
  PYTHONPATH=. python 30-03-26/run_block4_heston.py
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

ALGOS = ["LSM", "NLSM", "RLSM"]


def run_heston_suite(stock_model_name, nb_stocks=5, nb_paths=30000,
                     hidden_size=128, nb_epochs=80, nb_dates=10):
    params = ExperimentParams(
        payoff_name="MaxCall",
        nb_stocks=nb_stocks,
        strike=100.0,
        spot=100.0,
        maturity=1.0,
        drift=0.05,
        volatility=0.2,
        dividend=0.0,
        nb_dates=nb_dates,
        stock_model=stock_model_name,
        heston_mean=0.04,
        heston_speed=2.0,
        heston_corr=-0.7,
        nb_paths_train=nb_paths,
        hidden_size=hidden_size,
        nb_epochs_nlsm=nb_epochs,
    )
    payoff_f = build_payoff(params)
    results = []

    for algo in ALGOS:
        print("[Block4] model={} algo={} d={}".format(stock_model_name, algo, nb_stocks))
        model = build_market_model(params)
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
        elapsed = time.time() - t0
        price = bundle.price_terminal_discounted or 0.0
        results.append({
            "stock_model": stock_model_name,
            "algo": algo,
            "nb_stocks": nb_stocks,
            "price": price,
            "fit_time_s": elapsed,
            "path_gen_s": bundle.path_gen_seconds,
        })
        print("  price={:.4f} time={:.2f}s".format(price, elapsed))

    return results


def main():
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)

    all_rows = []

    print("=== Heston (standard, state=S only) ===")
    all_rows.extend(run_heston_suite("Heston", nb_stocks=5))

    print("\n=== HestonWithVar (state=(S,V)) ===")
    all_rows.extend(run_heston_suite("HestonWithVar", nb_stocks=5))

    csv_path = out_dir / "block4_heston.csv"
    with open(str(csv_path), "w") as f:
        w = csv.DictWriter(f, fieldnames=[
            "stock_model", "algo", "nb_stocks", "price", "fit_time_s", "path_gen_s"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print("\n=== Block 4 Summary ===")
    for model_name in ["Heston", "HestonWithVar"]:
        print("\n{}:".format(model_name))
        for algo in ALGOS:
            subset = [r for r in all_rows
                      if r["stock_model"] == model_name and r["algo"] == algo]
            if subset:
                print("  {}: price={:.4f} time={:.2f}s".format(
                    algo, subset[0]["price"], subset[0]["fit_time_s"]))
    print("Saved to {}".format(csv_path))


if __name__ == "__main__":
    main()
