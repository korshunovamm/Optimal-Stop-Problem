# -*- coding: utf-8 -*-
"""BTC experiment parameters."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BTCExperimentParams:
    # Data paths
    csv_path: str = "data/bybit_BTCUSDT_ohlc_interval_1min.csv"

    # Train / test split dates
    train_start: str = "2021-07-06"
    train_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2024-07-22"

    # Option parameters
    reference_spot: float = 100.0
    strike: float = 100.0       # ATM put
    maturity_days: int = 30
    nb_dates: int = 30          # daily rebalancing
    rate: float = 0.05          # annualised risk-free rate proxy

    # Sliding window
    window_size: int = 30       # days in one path
    window_step: int = 1        # overlap stride

    # Algorithm parameters
    hidden_size: int = 128
    nb_epochs_nlsm: int = 100
    rlsm_factors: tuple = (1.0,)
    train_eval_split: int = 2

    # Hedge
    bump_spot: float = 0.5
    nb_test_hedge_paths: int = 200

    # Reproducibility
    seed_train: int = 42
    seed_hedge: int = 12345

    output_dir: str = "31-03-26-bybit/output"


DEFAULT = BTCExperimentParams()
