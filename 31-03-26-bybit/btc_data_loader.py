# -*- coding: utf-8 -*-
"""Load BTC OHLCV from CSV, resample to daily, create sliding-window paths."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def load_btc_daily(csv_path, start_date=None, end_date=None):
    # type: (str, str, str) -> pd.Series
    """Load 1-min BTC OHLCV → daily close. Returns pd.Series indexed by date."""
    df = pd.read_csv(csv_path, parse_dates=["datetime"], usecols=["datetime", "price_close"])
    df = df.set_index("datetime").sort_index()
    if start_date is not None:
        df = df[df.index >= start_date]
    if end_date is not None:
        df = df[df.index <= end_date]
    daily = df["price_close"].resample("1D").last().dropna()
    return daily


def create_sliding_window_paths(prices_array, window_size, step=1,
                                reference_spot=100.0):
    # type: (np.ndarray, int, int, float) -> np.ndarray
    """Build overlapping normalised paths from a 1-D price array.

    Each path of length ``window_size + 1`` is scaled so that the first value
    equals ``reference_spot``.  This way the payoff K = reference_spot means ATM.

    Returns
    -------
    paths : np.ndarray, shape (nb_paths, 1, window_size + 1)
    """
    n = len(prices_array)
    paths = []
    for i in range(0, n - window_size, step):
        segment = prices_array[i: i + window_size + 1].astype(np.float64)
        s0 = segment[0]
        if s0 <= 0:
            continue
        normalised = segment / s0 * reference_spot
        paths.append(normalised)
    arr = np.array(paths, dtype=np.float64)          # (nb_paths, window_size+1)
    return arr[:, np.newaxis, :]                      # (nb_paths, 1, window_size+1)


def augment_with_noise(paths, n_augmented, sigma_noise=0.005, seed=None):
    # type: (np.ndarray, int, float, int) -> np.ndarray
    """Create additional paths by adding small Gaussian noise to returns.

    Preserves the first value of each path.
    """
    if seed is not None:
        np.random.seed(seed)
    nb, ns, nd = paths.shape
    indices = np.random.randint(0, nb, size=n_augmented)
    base = paths[indices].copy()
    noise = np.random.normal(0, sigma_noise, base[:, :, 1:].shape)
    log_ret = np.diff(np.log(base), axis=2) + noise
    cumret = np.cumsum(log_ret, axis=2)
    base[:, :, 1:] = base[:, :, 0:1] * np.exp(cumret)
    return base


def prepare_btc_paths(csv_path, start_date, end_date, window_size=30,
                      step=1, reference_spot=100.0, augment=0,
                      noise_sigma=0.005, seed=None):
    # type: (...) -> tuple
    """End-to-end: CSV → daily → sliding window → optional augmentation.

    Returns
    -------
    paths : np.ndarray (nb_paths, 1, window_size+1)
    daily_series : pd.Series  (full daily close for the date range)
    """
    daily = load_btc_daily(csv_path, start_date, end_date)
    prices = daily.values.astype(np.float64)
    paths = create_sliding_window_paths(prices, window_size, step, reference_spot)
    if augment > 0 and len(paths) > 0:
        extra = augment_with_noise(paths, augment, noise_sigma, seed)
        paths = np.concatenate([paths, extra], axis=0)
    return paths, daily


if __name__ == "__main__":
    csv = os.path.join(_ROOT, "data", "bybit_BTCUSDT_ohlc_interval_1min.csv")
    paths_train, daily_train = prepare_btc_paths(
        csv, "2021-07-06", "2023-12-31", window_size=30, step=1,
        reference_spot=100.0, augment=0)
    print("Train paths shape:", paths_train.shape)
    print("Daily prices: {} days, from {} to {}".format(
        len(daily_train), daily_train.index[0].date(), daily_train.index[-1].date()))
    paths_test, daily_test = prepare_btc_paths(
        csv, "2024-01-01", "2024-07-22", window_size=30, step=1,
        reference_spot=100.0, augment=0)
    print("Test  paths shape:", paths_test.shape)
