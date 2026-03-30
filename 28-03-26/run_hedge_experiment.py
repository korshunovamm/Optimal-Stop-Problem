#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Блоки D–E: распределение ошибки хеджа, VaR/CVaR, сохранение гистограмм.
Запуск: conda-activate OptStopRandNN, затем из корня репозитория:
  PYTHONPATH=. python 28-03-26/run_hedge_experiment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PKG = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(PKG)]

from backward_snapshots import fit_backward_snapshots
from build_model import build_market_model, build_payoff, clone_model_same_params
from config import ExperimentParams
from hedge_simulation import max_hedge_losses
import risk_metrics as rm_module

# matplotlib optional for headless
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def main():
    params = ExperimentParams(
        nb_paths_train=12_000,
        nb_paths_hedge_test=3000,
        nb_dates=25,
        stock_model="BlackScholes",
        hidden_size=24,
        nb_epochs_nlsm=12,
        bump_spot=0.5,
    )
    payoff_f = build_payoff(params)
    outdir = PKG / params.output_subdir
    outdir.mkdir(parents=True, exist_ok=True)

    report = {}

    for algo in ("LSM", "NLSM", "RLSM"):
        np.random.seed(params.seed_train)
        train_model = build_market_model(params)
        bundle = fit_backward_snapshots(
            algo,
            train_model,
            payoff_f,
            hidden_size=params.hidden_size,
            nb_epochs_nlsm=params.nb_epochs_nlsm,
            rlsm_factors=params.rlsm_factors,
            train_eval_split=params.train_eval_split,
            seed=params.seed_train,
        )

        np.random.seed(params.seed_hedge)
        test_model = clone_model_same_params(train_model, params.nb_paths_hedge_test)
        paths, _ = test_model.generate_paths()

        m_model, m_intr = max_hedge_losses(
            bundle,
            payoff_f,
            paths,
            params.hidden_size,
            params.bump_spot,
        )
        st_m = rm_module.summary_stats(m_model)
        st_i = rm_module.summary_stats(m_intr)
        rv95 = rm_module.var_cvar(m_model, 0.95)
        report[algo] = {
            "price_train": bundle.price_terminal_discounted,
            "hedge_loss_model": st_m,
            "hedge_loss_intrinsic": st_i,
            "var_cvar_95_model": rv95,
        }
        print(algo, "price", bundle.price_terminal_discounted)
        print("  hedge vs model:", st_m)
        print("  VaR/CVaR (95%):", rv95)

        if plt is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(m_model, bins=40, density=True, alpha=0.75, color="steelblue")
            ax.set_title(f"Hedging error ({algo}) — max disc. shortfall vs model")
            ax.set_xlabel("Loss")
            ax.set_ylabel("Density")
            fig.tight_layout()
            fig.savefig(outdir / f"hedge_hist_{algo}.png", dpi=120)
            plt.close(fig)

    with open(outdir / "hedge_risk_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report -> {outdir / 'hedge_risk_report.json'}")


if __name__ == "__main__":
    main()
