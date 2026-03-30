# -*- coding: utf-8 -*-
"""
Параметры экспериментов (блок B): американский пут, BS/Heston, сетка времени.

Запуск (окружение, conda, Python 3.7):
  conda-activate OptStopRandNN
  cd /path/to/Optimal-Stop-Problem
  PYTHONPATH=. python 28-03-26/run_comparison.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ExperimentParams:
    # --- опцион (американский пут, 1 актив) ---
    strike: float = 100.0
    spot: float = 100.0
    maturity: float = 1.0
    dividend: float = 0.0
    # nb_dates = число шагов на [0, T]; на сетке n+1 точек включая S_0,...,S_n
    nb_dates: int = 50

    # --- Black–Scholes ---
    drift: float = 0.05
    volatility: float = 0.30

    # --- Heston (если stock_model == "Heston") ---
    heston_mean: float = 0.04
    heston_speed: float = 2.0
    heston_corr: float = -0.7

    # "BlackScholes" или "Heston" (см. build_model.build_market_model).
    stock_model: str = "BlackScholes"

    # --- Monte Carlo ---
    nb_paths_train: int = 20_000
    nb_paths_hedge_test: int = 5_000
    train_eval_split: int = 2

    # --- нейросети (NLSM / RLSM) ---
    hidden_size: int = 32
    nb_epochs_nlsm: int = 30
    rlsm_factors: Tuple[float, ...] = (1.0,)

    # --- численное дифференцирование цены V(S) ---
    bump_spot: float = 0.01

    seed_train: int = 42
    seed_hedge: int = 12345

    # папка для фигур относительно этого модуля
    output_subdir: str = "output"


DEFAULT = ExperimentParams()
