# -*- coding: utf-8 -*-
"""
Extended feature builder for high-dimensional BTC experiments.

Supports d up to 30 using price + technical/statistical features.
All auxiliary features are scaled to the same range as the normalised
price (~reference_spot ± 10%) so the reservoir sees balanced inputs.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Feature catalogue ────────────────────────────────────────────────
# Each entry: (column_name, builder_func(df) -> pd.Series)
# Price is always index 0 and handled separately.

def _ret(df, n):
    return df["price_close"].pct_change(n)

def _rvol(df, w):
    lr = np.log(df["price_close"] / df["price_close"].shift(1))
    return lr.rolling(w).std() * np.sqrt(252)

def _rsi(df, w=14):
    delta = df["price_close"].diff()
    gain = delta.clip(lower=0).rolling(w).mean()
    loss = (-delta.clip(upper=0)).rolling(w).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - 100 / (1 + rs)

def _ma_ratio(df, short, long):
    return df["price_close"].rolling(short).mean() / df["price_close"].rolling(long).mean()

def _boll_width(df, w=20):
    ma = df["price_close"].rolling(w).mean()
    std = df["price_close"].rolling(w).std()
    return 2 * std / ma.replace(0, 1e-10)

def _vol_of_vol(df, w=20):
    lr = np.log(df["price_close"] / df["price_close"].shift(1))
    rv = lr.rolling(w).std()
    return rv.rolling(w).std()

def _hl_range(df):
    return (df["price_high"] - df["price_low"]) / df["price_close"]

def _oc_range(df):
    return (df["price_close"] - df["price_open"]) / df["price_close"]

def _log_volume(df):
    return np.log1p(df["volume"])

def _volume_ma_ratio(df, w=20):
    return df["volume"] / df["volume"].rolling(w).mean().replace(0, 1e-10)

def _log_turnover(df):
    return np.log1p(df["turnover"])

def _skewness(df, w=20):
    lr = np.log(df["price_close"] / df["price_close"].shift(1))
    return lr.rolling(w).apply(lambda x: pd.Series(x).skew(), raw=False)

def _kurtosis(df, w=20):
    lr = np.log(df["price_close"] / df["price_close"].shift(1))
    return lr.rolling(w).apply(lambda x: pd.Series(x).kurtosis(), raw=False)


FEATURE_CATALOGUE = [
    # d=2
    ("rvol_30d", lambda df: _rvol(df, 30)),
    # d=3
    ("log_volume", _log_volume),
    # d=4-5
    ("ret_7d", lambda df: _ret(df, 7)),
    ("ret_14d", lambda df: _ret(df, 14)),
    # d=6-10
    ("ret_1d", lambda df: _ret(df, 1)),
    ("ret_3d", lambda df: _ret(df, 3)),
    ("rvol_10d", lambda df: _rvol(df, 10)),
    ("rsi_14d", _rsi),
    ("hl_range", _hl_range),
    # d=11-15
    ("oc_range", _oc_range),
    ("ret_21d", lambda df: _ret(df, 21)),
    ("ret_30d", lambda df: _ret(df, 30)),
    ("ma_ratio_10_30", lambda df: _ma_ratio(df, 10, 30)),
    ("boll_width_20d", lambda df: _boll_width(df, 20)),
    # d=16-20
    ("rvol_5d", lambda df: _rvol(df, 5)),
    ("rvol_60d", lambda df: _rvol(df, 60)),
    ("vol_ma_ratio_20d", lambda df: _volume_ma_ratio(df, 20)),
    ("ma_ratio_20_60", lambda df: _ma_ratio(df, 20, 60)),
    ("vol_of_vol_20d", lambda df: _vol_of_vol(df, 20)),
    # d=21-25
    ("ret_60d", lambda df: _ret(df, 60)),
    ("ret_2d", lambda df: _ret(df, 2)),
    ("ret_5d", lambda df: _ret(df, 5)),
    ("ret_10d", lambda df: _ret(df, 10)),
    ("log_turnover", _log_turnover),
    # d=26-30
    ("skewness_20d", lambda df: _skewness(df, 20)),
    ("kurtosis_20d", lambda df: _kurtosis(df, 20)),
    ("rvol_20d", lambda df: _rvol(df, 20)),
    ("boll_width_10d", lambda df: _boll_width(df, 10)),
    ("ma_ratio_5_20", lambda df: _ma_ratio(df, 5, 20)),
]


def load_btc_daily(csv_path, start_date=None, end_date=None):
    # type: (str, str, str) -> pd.DataFrame
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
    if start_date:
        daily = daily[daily.index >= start_date]
    if end_date:
        daily = daily[daily.index <= end_date]
    return daily


def add_all_features(df):
    # type: (pd.DataFrame) -> pd.DataFrame
    df = df.copy()
    for name, func in FEATURE_CATALOGUE:
        df[name] = func(df)
    return df


def build_paths(df, window_size, step, dims, reference_spot=100.0):
    # type: (pd.DataFrame, int, int, int, float) -> np.ndarray
    """Build (nb_paths, dims, window_size+1) array.

    Feature 0 is always the normalised BTC close price.
    Features 1..dims-1 come from FEATURE_CATALOGUE and are scaled to
    reference_spot ± 10 % to keep all dimensions on a similar range.
    """
    feature_cols = ["price_close"]
    n_aux = min(dims - 1, len(FEATURE_CATALOGUE))
    for i in range(n_aux):
        feature_cols.append(FEATURE_CATALOGUE[i][0])
    actual_dims = len(feature_cols)

    data = df[feature_cols].values.astype(np.float64)
    valid = ~np.isnan(data).any(axis=1)
    first_ok = int(np.argmax(valid))
    data = data[first_ok:]
    valid = valid[first_ok:]

    n = len(data)
    paths = []
    for i in range(0, n - window_size, step):
        seg = data[i: i + window_size + 1].copy()
        if np.isnan(seg).any():
            continue
        s0 = seg[0, 0]
        if s0 <= 0:
            continue
        seg[:, 0] = seg[:, 0] / s0 * reference_spot
        for k in range(1, actual_dims):
            col = seg[:, k]
            mu = col.mean()
            sigma = col.std()
            if sigma > 1e-12:
                z = (col - mu) / sigma
            else:
                z = np.zeros_like(col)
            seg[:, k] = reference_spot + z * (reference_spot * 0.1)
        paths.append(seg.T)

    if not paths:
        return np.empty((0, actual_dims, window_size + 1))
    return np.array(paths, dtype=np.float64)


def augment_paths(paths, n_extra, sigma=0.008, seed=None):
    # type: (np.ndarray, int, float, int) -> np.ndarray
    if seed is not None:
        np.random.seed(seed)
    nb, nd, nt = paths.shape
    idx = np.random.randint(0, nb, n_extra)
    extra = paths[idx].copy()
    for k in range(nd):
        noise = np.random.normal(0, sigma, (n_extra, nt - 1))
        lr = np.diff(np.log(np.maximum(extra[:, k, :], 1e-8)), axis=1) + noise
        extra[:, k, 1:] = extra[:, k, 0:1] * np.exp(np.cumsum(lr, axis=1))
    return extra


if __name__ == "__main__":
    csv = os.path.join(_ROOT, "data", "bybit_BTCUSDT_ohlc_interval_1min.csv")
    df = load_btc_daily(csv, "2021-07-06", "2023-12-31")
    df = add_all_features(df)
    print("Daily shape:", df.shape)
    for d in [1, 5, 10, 15, 20, 30]:
        p = build_paths(df, 30, 1, d)
        print("  d={:<3} -> paths: {}".format(d, p.shape))
