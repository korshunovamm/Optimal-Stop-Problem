#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Доп. к блоку B/D: несколько распределений ошибок хеджа при разных σ (или nb_dates).
Сохраняет CSV в output/. Перед запуском: conda-activate OptStopRandNN.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PKG = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(PKG)]

from backward_snapshots import fit_backward_snapshots
from build_model import build_payoff, clone_model_same_params
from build_model import build_black_scholes
from config import ExperimentParams
from hedge_simulation import max_hedge_losses
import risk_metrics as rm


def main():
    base = ExperimentParams(
        nb_paths_train=10_000,
        nb_paths_hedge_test=1500,
        nb_dates=25,
        stock_model="BlackScholes",
        hidden_size=24,
        nb_epochs_nlsm=10,
        bump_spot=0.5,
    )
    vol_grid = [0.2, 0.3, 0.4]
    outdir = PKG / base.output_subdir
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for vol in vol_grid:
        for algo in ("LSM", "NLSM"):
            p = ExperimentParams(**{**base.__dict__, "volatility": vol})
            payoff_f = build_payoff(p)
            model = build_black_scholes(p)
            bundle = fit_backward_snapshots(
                algo,
                model,
                payoff_f,
                hidden_size=p.hidden_size,
                nb_epochs_nlsm=p.nb_epochs_nlsm,
                train_eval_split=p.train_eval_split,
                seed=p.seed_train,
            )
            np.random.seed(p.seed_hedge + int(vol * 100))
            test_m = clone_model_same_params(model, p.nb_paths_hedge_test)
            paths, _ = test_m.generate_paths()
            m_model, _ = max_hedge_losses(
                bundle, payoff_f, paths, p.hidden_size, p.bump_spot
            )
            st = rm.summary_stats(m_model)
            q = rm.var_cvar(m_model, 0.95)
            rows.append(
                {
                    "volatility": vol,
                    "algo": algo,
                    "price": bundle.price_terminal_discounted,
                    "loss_mean": st["mean"],
                    "loss_p95": st["p95"],
                    "var95": q["var"],
                    "cvar95": q["cvar"],
                }
            )
            print(vol, algo, st["mean"], q["cvar"])

    path_csv = outdir / "sweep_volatility_hedge.csv"
    with open(path_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved", path_csv)


if __name__ == "__main__":
    main()
