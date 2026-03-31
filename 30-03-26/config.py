# -*- coding: utf-8 -*-
"""
Multi-dimensional experiment parameters for NN vs LSM comparison.

Usage:
  conda-activate OptStopRandNN
  cd /path/to/Optimal-Stop-Problem
  PYTHONPATH=. python 30-03-26/<script>.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ExperimentParams:
    strike: float = 100.0
    spot: float = 100.0
    maturity: float = 1.0
    dividend: float = 0.0
    nb_dates: int = 10
    nb_stocks: int = 5

    # payoff: "MaxCall", "BasketCall", "Put1Dim", "GeometricPut", "MaxPut"
    payoff_name: str = "MaxCall"

    drift: float = 0.05
    volatility: float = 0.2

    # "BlackScholes", "Heston", "HestonWithVar"
    stock_model: str = "BlackScholes"

    heston_mean: float = 0.04
    heston_speed: float = 2.0
    heston_corr: float = -0.7

    nb_paths_train: int = 50000
    nb_paths_hedge_test: int = 5000
    train_eval_split: int = 2

    hidden_size: int = 128
    nb_epochs_nlsm: int = 100
    rlsm_factors: Tuple[float, ...] = (1.0,)

    bump_spot: float = 0.5

    seed_train: int = 42
    seed_hedge: int = 12345

    output_subdir: str = "output"


DEFAULT = ExperimentParams()
