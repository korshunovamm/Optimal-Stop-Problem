# -*- coding: utf-8 -*-
"""
Block 6: Multi-dimensional hedging comparison (THE NOVELTY).
MaxCall d=5, BS model. Compare hedge error distributions across LSM/NLSM/RLSM.

Usage:
  conda-activate OptStopRandNN
  cd Optimal-Stop-Problem
  PYTHONPATH=. python 30-03-26/run_block6_multidim_hedge.py
"""
from __future__ import annotations

import csv
import json
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
from build_model import build_payoff, build_market_model, clone_model
from backward_snapshots import fit_backward_snapshots
from hedge_simulation import max_hedge_losses
from risk_metrics import var_cvar, summary_stats

ALGOS = ["LSM", "NLSM", "RLSM"]


def run_hedge_experiment(nb_stocks=5, nb_paths_train=40000,
                         nb_paths_hedge=2000, nb_dates=10,
                         hidden_size=128, nb_epochs=80):
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
        nb_paths_train=nb_paths_train,
        nb_paths_hedge_test=nb_paths_hedge,
        hidden_size=hidden_size,
        nb_epochs_nlsm=nb_epochs,
        bump_spot=0.5,
    )
    payoff_f = build_payoff(params)
    results = {}

    for algo in ALGOS:
        print("\n[Block6] algo={} d={}".format(algo, nb_stocks))
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
        fit_time = time.time() - t0
        price = bundle.price_terminal_discounted or 0.0
        print("  price={:.4f} fit_time={:.2f}s".format(price, fit_time))

        test_model = clone_model(model, params.nb_paths_hedge_test)
        np.random.seed(params.seed_hedge)
        test_paths, _ = test_model.generate_paths()
        # test_paths shape: (nb_paths, nb_stocks, nb_dates+1)

        print("  Running hedge on {} paths...".format(params.nb_paths_hedge_test))
        t1 = time.time()
        losses_model, losses_intr = max_hedge_losses(
            bundle, payoff_f, test_paths, params.hidden_size, params.bump_spot)
        hedge_time = time.time() - t1
        print("  Hedge done in {:.1f}s".format(hedge_time))

        ss = summary_stats(losses_model)
        vc = var_cvar(losses_model, 0.95)

        results[algo] = {
            "price": price,
            "fit_time_s": fit_time,
            "hedge_time_s": hedge_time,
            "hedge_loss_summary": ss,
            "var_cvar_95": vc,
        }
        print("  loss_mean={:.4f} CVaR95={:.4f}".format(ss["mean"], vc["cvar"]))

    return results


def main():
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)

    results = run_hedge_experiment(
        nb_stocks=5,
        nb_paths_train=40000,
        nb_paths_hedge=500,
        nb_dates=10,
        hidden_size=128,
        nb_epochs=80,
    )

    json_path = out_dir / "block6_hedge_results.json"
    with open(str(json_path), "w") as f:
        json.dump(results, f, indent=2)

    csv_path = out_dir / "block6_hedge_summary.csv"
    with open(str(csv_path), "w") as f:
        w = csv.DictWriter(f, fieldnames=[
            "algo", "price", "fit_time_s", "hedge_time_s",
            "loss_mean", "loss_p95", "var95", "cvar95", "n_paths"])
        w.writeheader()
        for algo in ALGOS:
            r = results[algo]
            w.writerow({
                "algo": algo,
                "price": r["price"],
                "fit_time_s": r["fit_time_s"],
                "hedge_time_s": r["hedge_time_s"],
                "loss_mean": r["hedge_loss_summary"]["mean"],
                "loss_p95": r["hedge_loss_summary"]["p95"],
                "var95": r["var_cvar_95"]["var"],
                "cvar95": r["var_cvar_95"]["cvar"],
                "n_paths": r["hedge_loss_summary"]["n"],
            })

    print("\n=== Block 6 Hedge Summary (d=5 MaxCall) ===")
    for algo in ALGOS:
        r = results[algo]
        print("  {}: price={:.4f} CVaR95={:.4f} fit={:.1f}s hedge={:.1f}s".format(
            algo, r["price"], r["var_cvar_95"]["cvar"],
            r["fit_time_s"], r["hedge_time_s"]))
    print("Saved to {} and {}".format(csv_path, json_path))


if __name__ == "__main__":
    main()
