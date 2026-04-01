# -*- coding: utf-8 -*-
"""
Load Bybit BTCUSDT 1-min OHLCV data and prepare daily time series
for calibration and hedge backtesting.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_DEFAULT_FILE = "bybit_BTCUSDT_ohlc_interval_1min.csv"


def load_raw(path=None):
    # type: (str | None) -> pd.DataFrame
    if path is None:
        path = os.path.join(_DATA_DIR, _DEFAULT_FILE)
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def daily_close(df=None, path=None):
    # type: (pd.DataFrame | None, str | None) -> pd.Series
    if df is None:
        df = load_raw(path)
    daily = df.set_index("datetime").resample("1D")["price_close"].last().dropna()
    daily.name = "btc_close"
    return daily


def daily_log_returns(daily=None, df=None):
    # type: (pd.Series | None, pd.DataFrame | None) -> pd.Series
    if daily is None:
        daily = daily_close(df)
    lr = np.log(daily / daily.shift(1)).dropna()
    lr.name = "log_return"
    return lr


def get_windows(daily, calib_days=60, hedge_days=30, step_days=30):
    # type: (pd.Series, int, int, int) -> list[dict]
    """
    Generate non-overlapping rolling windows for calibration + hedge test.

    Returns list of dicts with keys:
      calib_start, calib_end, hedge_start, hedge_end,
      calib_prices (np.array), hedge_prices (np.array)
    """
    idx = daily.index
    vals = daily.values.astype(np.float64)
    total_needed = calib_days + hedge_days
    windows = []
    start = 0
    while start + total_needed <= len(idx):
        c_start = start
        c_end = start + calib_days
        h_start = c_end
        h_end = c_end + hedge_days
        if h_end > len(idx):
            break
        windows.append({
            "calib_start": idx[c_start],
            "calib_end": idx[c_end - 1],
            "hedge_start": idx[h_start],
            "hedge_end": idx[h_end - 1],
            "calib_prices": vals[c_start:c_end].copy(),
            "hedge_prices": vals[h_start:h_end].copy(),
        })
        start += step_days
    return windows


if __name__ == "__main__":
    daily = daily_close()
    print("Daily prices loaded: {} points".format(len(daily)))
    print("Date range: {} to {}".format(daily.index[0].date(), daily.index[-1].date()))
    print("Price range: {:.0f} -- {:.0f}".format(daily.min(), daily.max()))
    lr = daily_log_returns(daily)
    ann_vol = lr.std() * np.sqrt(365)
    print("Annualized vol: {:.1f}%".format(ann_vol * 100))
    wins = get_windows(daily)
    print("Rolling windows (60 calib + 30 hedge, step 30): {}".format(len(wins)))
