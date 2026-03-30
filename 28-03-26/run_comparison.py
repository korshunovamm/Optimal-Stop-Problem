#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Блок C: сравнение цен LSM / NLSM / RLSM на синтетике (BS или Heston).

Запуск (окружение conda OptStopRandNN, Python 3.7):
  conda-activate OptStopRandNN
  cd .../Optimal-Stop-Problem
  PYTHONPATH=. python 28-03-26/run_comparison.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(PKG)]

from backward_snapshots import fit_backward_snapshots
from build_model import build_market_model, build_payoff
from config import ExperimentParams


def main():
    params = ExperimentParams(
        nb_paths_train=8000,
        nb_dates=30,
        stock_model="BlackScholes",
        hidden_size=24,
        nb_epochs_nlsm=15,
    )
    payoff_f = build_payoff(params)
    rows = []
    for algo in ("LSM", "NLSM", "RLSM"):
        model = build_market_model(params)
        t0 = time.time()
        bundle = fit_backward_snapshots(
            algo,
            model,
            payoff_f,
            hidden_size=params.hidden_size,
            nb_epochs_nlsm=params.nb_epochs_nlsm,
            rlsm_factors=params.rlsm_factors,
            train_eval_split=params.train_eval_split,
            seed=params.seed_train,
        )
        elapsed = time.time() - t0
        rows.append(
            {
                "algo": algo,
                "model": params.stock_model,
                "price": bundle.price_terminal_discounted,
                "seconds": round(elapsed, 3),
                "path_gen_s": round(bundle.path_gen_seconds, 4),
            }
        )
        print(
            f"{algo:4}  price={bundle.price_terminal_discounted:.6f}  "
            f"time={elapsed:.2f}s  (paths {params.nb_paths_train})"
        )

    outdir = PKG / params.output_subdir
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "comparison_prices.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
