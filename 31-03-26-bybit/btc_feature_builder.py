# -*- coding: utf-8 -*-
"""Build multi-dimensional feature paths from BTC data.

Dimension mapping:
  d=1 : price only
  d=2 : price + realized volatility (30d rolling)
  d=3 : price + realized vol + log volume
  d=5 : + 7d return + 14d return
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def load_btc_daily_full(csv_path, start_date=None, end_date=None):
    # type: (str, str, str) -> pd.DataFrame
    """Load 1-min BTC → daily OHLCV DataFrame."""
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    daily = df.resample("1D").agg({
        "price_open": "first",
        "price_high": "max",
        "price_low": "min",
        "price_close": "last",
        "volume": "sum",
        "turnover": "sum",
    }).dropna()
    if start_date is not None:
        daily = daily[daily.index >= start_date]
    if end_date is not None:
        daily = daily[daily.index <= end_date]
    return daily


def add_features(df, rv_window=30):
    # type: (pd.DataFrame, int) -> pd.DataFrame
    """Compute additional features and add them as columns."""
    df = df.copy()
    df["log_ret"] = np.log(df["price_close"] / df["price_close"].shift(1))
    df["realized_vol"] = df["log_ret"].rolling(rv_window).std() * np.sqrt(252)
    df["log_volume"] = np.log1p(df["volume"])
    df["ret_7d"] = df["price_close"].pct_change(7)
    df["ret_14d"] = df["price_close"].pct_change(14)
    return df


def build_multidim_paths(df, window_size, step, dims, reference_spot=100.0):
    # type: (pd.DataFrame, int, int, int, float) -> np.ndarray
    """Create sliding-window paths with ``dims`` features.

    dims=1 → price
    dims=2 → price, realized_vol
    dims=3 → price, realized_vol, log_volume
    dims=5 → price, realized_vol, log_volume, ret_7d, ret_14d

    The first feature (price) is normalised to ``reference_spot``.
    Other features are normalised to zero-mean unit-variance per window.

    Returns shape (nb_paths, dims, window_size + 1).
    """
    feature_cols = ["price_close"]
    if dims >= 2:
        feature_cols.append("realized_vol")
    if dims >= 3:
        feature_cols.append("log_volume")
    if dims >= 5:
        feature_cols.extend(["ret_7d", "ret_14d"])

    actual_dims = len(feature_cols)
    data = df[feature_cols].values.astype(np.float64)
    valid_mask = ~np.isnan(data).any(axis=1)
    first_valid = int(np.argmax(valid_mask))
    data = data[first_valid:]

    n = len(data)
    paths = []
    for i in range(0, n - window_size, step):
        segment = data[i: i + window_size + 1].copy()
        if np.isnan(segment).any():
            continue

        s0 = segment[0, 0]
        if s0 <= 0:
            continue
        segment[:, 0] = segment[:, 0] / s0 * reference_spot

        for k in range(1, actual_dims):
            col = segment[:, k]
            mu, sigma = col.mean(), col.std()
            if sigma > 1e-12:
                normed = (col - mu) / sigma
            else:
                normed = np.zeros_like(col)
            segment[:, k] = reference_spot + normed * (reference_spot * 0.1)

        paths.append(segment.T)  # (dims, window_size+1)

    return np.array(paths, dtype=np.float64)  # (nb_paths, dims, window_size+1)


if __name__ == "__main__":
    csv = os.path.join(_ROOT, "data", "bybit_BTCUSDT_ohlc_interval_1min.csv")
    df = load_btc_daily_full(csv, "2021-07-06", "2023-12-31")
    df = add_features(df)
    print("Daily shape:", df.shape)
    print("Columns:", list(df.columns))
    for d in [1, 2, 3, 5]:
        p = build_multidim_paths(df, 30, 1, d)
        print("  d={} -> paths shape: {}".format(d, p.shape))
