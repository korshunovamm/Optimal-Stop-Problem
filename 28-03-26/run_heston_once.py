#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Один прогон обучения + хеджа для Heston (блок B, марковский 1D без v в регрессоре).
Активируйте: conda-activate OptStopRandNN
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(PKG)]

from backward_snapshots import fit_backward_snapshots
from build_model import build_heston, build_payoff, clone_model_same_params
from config import ExperimentParams
from hedge_simulation import max_hedge_losses
import risk_metrics as rm


def main():
    params = ExperimentParams(
        stock_model="Heston",
        nb_paths_train=8000,
        nb_paths_hedge_test=2000,
        nb_dates=20,
        hidden_size=20,
        nb_epochs_nlsm=10,
        bump_spot=0.5,
    )
    payoff_f = build_payoff(params)
    model = build_heston(params)
    bundle = fit_backward_snapshots(
        "NLSM",
        model,
        payoff_f,
        hidden_size=params.hidden_size,
        nb_epochs_nlsm=params.nb_epochs_nlsm,
        train_eval_split=params.train_eval_split,
        seed=params.seed_train,
    )
    np.random.seed(params.seed_hedge)
    test_m = clone_model_same_params(model, params.nb_paths_hedge_test)
    paths, _ = test_m.generate_paths()
    m1, _ = max_hedge_losses(
        bundle, payoff_f, paths, params.hidden_size, params.bump_spot
    )
    print("Heston NLSM price", bundle.price_terminal_discounted)
    print("Hedge loss stats", rm.summary_stats(m1))
    print("VaR/CVaR 95%", rm.var_cvar(m1, 0.95))


if __name__ == "__main__":
    main()
