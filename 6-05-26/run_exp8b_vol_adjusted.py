# -*- coding: utf-8 -*-
"""
Experiment 8b: Volatility-Adjusted Market Validation.

The base experiment showed all models underestimate market prices because
historical BTC paths have lower realized volatility than the implied vol
priced into options (the "volatility risk premium").

This variant:
  1. Estimates historical realized vol from the training data
  2. Extracts market implied vol (mid_IV from options)
  3. Scales augmented training paths by the IV/RV ratio to match market expectations
  4. Re-trains models and compares with market prices

This demonstrates the practical workflow: calibrate path generation to market IV.
"""
from __future__ import annotations

import json, os, sys, time, math
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_DIR = str(Path(__file__).resolve().parent)
_ROOT = str(Path(__file__).resolve().parent.parent)
for p in (_DIR, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from feature_builder import load_btc_daily, build_paths, augment_paths
from historical_model import HistoricalModel
from backward_snapshots import fit_backward_snapshots
from btc_payoff import BtcPut
from btc_hedge import btc_hedge_one_path
from risk_metrics import var_cvar

CSV_OHLC = os.path.join(_ROOT, "data", "bybit_BTCUSDT_ohlc_interval_1min.csv")
CSV_OPTS = os.path.join(_ROOT, "6-05-26", "output", "btc_puts_market.csv")
OUT = os.path.join(_ROOT, "6-05-26", "output")

ALGOS = ["LSM", "RLSM", "NLSM"]
HIDDEN = 128
NB_EPOCHS = 80
AUGMENT = 3000
SEED = 42
REFERENCE_SPOT = 100.0


def estimate_realized_vol(df, window=30):
    """Annualized realized volatility from daily close prices."""
    lr = np.log(df['price_close'] / df['price_close'].shift(1)).dropna()
    recent = lr.iloc[-window:]
    return float(recent.std()) * np.sqrt(252)


def scale_paths_to_iv(paths, rv, iv, reference_spot):
    """Scale path returns to match implied volatility level.
    
    If IV > RV, we need to stretch the returns by factor IV/RV.
    """
    if rv <= 0 or iv <= 0:
        return paths
    ratio = iv / rv
    if ratio < 0.5 or ratio > 5.0:
        ratio = np.clip(ratio, 0.5, 3.0)

    scaled = paths.copy()
    for i in range(scaled.shape[0]):
        for k in range(scaled.shape[1]):
            series = scaled[i, k, :]
            lr = np.diff(np.log(np.maximum(series, 1e-8)))
            lr_scaled = lr * ratio
            scaled[i, k, 1:] = series[0] * np.exp(np.cumsum(lr_scaled))
    return scaled


def select_options():
    """Select representative BTC puts with mid_IV available."""
    chunks = pd.read_csv(CSV_OPTS, chunksize=500000)
    frames = []
    for chunk in chunks:
        mask = (
            (chunk['moneyness'] >= 0.90) &
            (chunk['moneyness'] <= 1.05) &
            (chunk['bid_price_usd'].notna()) &
            (chunk['ask_price_usd'].notna()) &
            (chunk['bid_price_usd'] > 0) &
            (chunk['mid_IV'].notna()) &
            (chunk['mid_IV'] > 0.1) &
            (chunk['mid_IV'] < 3.0) &
            (chunk['open_interest'] >= 20) &
            (chunk['volume'] >= 1)
        )
        frames.append(chunk[mask])
        if sum(len(f) for f in frames) > 20000:
            break

    df = pd.concat(frames, ignore_index=True)
    print("  Liquid ATM puts with IV: {}".format(len(df)))

    buckets = [(5, 10), (12, 22), (28, 40)]
    selected = []
    for lo, hi in buckets:
        sub = df[(df['tte_days'] >= lo) & (df['tte_days'] <= hi)]
        if len(sub) == 0:
            continue
        sub = sub.copy()
        sub['date'] = pd.to_datetime(sub['current_time']).dt.date
        dates = sub['date'].unique()
        np.random.seed(SEED + 1)
        chosen = np.random.choice(dates, min(2, len(dates)), replace=False)
        for d in chosen:
            day_sub = sub[sub['date'] == d].copy()
            day_sub['atm_dist'] = abs(day_sub['moneyness'] - 1.0)
            best = day_sub.nsmallest(1, 'atm_dist').iloc[0]
            selected.append(best.to_dict())

    return selected


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 70)
    print("Experiment 8b: Vol-Adjusted Market Validation")
    print("=" * 70)
    sys.stdout.flush()

    options = select_options()
    print("  Selected {} options".format(len(options)))
    sys.stdout.flush()

    all_results = []
    for i, opt in enumerate(options):
        print("\n--- Option {}/{} ---".format(i + 1, len(options)))
        print("  {}, Strike={}, S={:.0f}, TTE={:.1f}d, IV={:.1%}".format(
            opt['full_name'], opt['strike'], opt['underlying_price'],
            opt['tte_days'], opt['mid_IV']))
        print("  Market: ${:.2f} (bid/ask: ${:.2f}/${:.2f})".format(
            opt['mark_price_usd'],
            opt.get('bid_price_usd', 0) or 0,
            opt.get('ask_price_usd', 0) or 0))
        sys.stdout.flush()

        maturity = opt['tte_days'] / 365.0
        nb_dates = max(int(opt['tte_days']), 7)
        norm_strike = (opt['strike'] / opt['underlying_price']) * REFERENCE_SPOT
        market_iv = opt['mid_IV']

        end_date = opt['current_time'][:10]
        start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=730)
        start_date = start_dt.strftime("%Y-%m-%d")

        df = load_btc_daily(CSV_OHLC, start_date, end_date)
        if len(df) < nb_dates + 30:
            print("  SKIP: insufficient data")
            continue

        rv = estimate_realized_vol(df, 60)
        print("  RV(60d)={:.1%}, Market IV={:.1%}, ratio={:.2f}".format(
            rv, market_iv, market_iv / rv if rv > 0 else 0))
        sys.stdout.flush()

        # Build paths
        paths_raw = build_paths(df, nb_dates, 1, 1, REFERENCE_SPOT)
        if paths_raw.shape[0] < 100:
            print("  SKIP: insufficient paths")
            continue

        # Scale paths to match IV
        paths_scaled = scale_paths_to_iv(paths_raw, rv, market_iv, REFERENCE_SPOT)

        # Augment scaled paths
        extra = augment_paths(paths_scaled, AUGMENT,
                              sigma=market_iv / np.sqrt(252) * 0.3, seed=SEED)
        train_paths = np.concatenate([paths_scaled, extra], axis=0)
        np.random.seed(SEED)
        train_paths = train_paths[np.random.permutation(train_paths.shape[0])]

        model = HistoricalModel(train_paths, 0.05, maturity, nb_dates)
        payoff_f = BtcPut(norm_strike)

        entry = {
            "full_name": opt['full_name'],
            "obs_date": end_date,
            "strike": int(opt['strike']),
            "underlying_price": round(opt['underlying_price'], 2),
            "tte_days": opt['tte_days'],
            "moneyness": opt['moneyness'],
            "market_mark_usd": opt['mark_price_usd'],
            "market_bid_usd": opt.get('bid_price_usd'),
            "market_ask_usd": opt.get('ask_price_usd'),
            "market_iv": round(market_iv, 4),
            "realized_vol": round(rv, 4),
            "iv_rv_ratio": round(market_iv / rv, 3) if rv > 0 else None,
        }

        for algo in ALGOS:
            t0 = time.time()
            bundle = fit_backward_snapshots(
                algo=algo, model=model, payoff_f=payoff_f,
                hidden_size=HIDDEN, nb_epochs_nlsm=NB_EPOCHS,
                rlsm_factors=(1.0,), train_itm_only=True,
                train_eval_split=2, seed=SEED)
            fit_t = time.time() - t0

            norm_price = float(bundle.price_terminal_discounted)
            usd_price = norm_price * (opt['underlying_price'] / REFERENCE_SPOT)

            err_pct = (usd_price - opt['mark_price_usd']) / max(opt['mark_price_usd'], 1) * 100

            entry["{}_usd".format(algo)] = round(usd_price, 2)
            entry["{}_error_pct".format(algo)] = round(err_pct, 1)
            entry["{}_fit_time".format(algo)] = round(fit_t, 2)

            print("  {} ${:.2f} (err={:+.1f}%, fit={:.2f}s)".format(
                algo, usd_price, err_pct, fit_t))
            sys.stdout.flush()

        # Check if within bid-ask spread
        bid = opt.get('bid_price_usd', 0) or 0
        ask = opt.get('ask_price_usd', 0) or 0
        for algo in ALGOS:
            p = entry["{}_usd".format(algo)]
            entry["{}_in_spread".format(algo)] = bool(bid <= p <= ask)

        all_results.append(entry)

    # Summary
    print("\n\n" + "=" * 80)
    print("VOL-ADJUSTED MARKET VALIDATION SUMMARY")
    print("=" * 80)
    print("{:<22} {:>5} {:>6} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8}".format(
        "Option", "TTE", "IV", "Market$", "LSM$", "RLSM$", "NLSM$", "LSMerr", "RLSMerr"))
    for r in all_results:
        print("{:<22} {:>4.0f}d {:>5.0f}% {:>9.0f} {:>9.0f} {:>9.0f} {:>9.0f} {:>+7.1f}% {:>+7.1f}%".format(
            r['full_name'][:22], r['tte_days'], r['market_iv']*100,
            r['market_mark_usd'],
            r.get('LSM_usd', 0), r.get('RLSM_usd', 0), r.get('NLSM_usd', 0),
            r.get('LSM_error_pct', 0), r.get('RLSM_error_pct', 0)))
    print("=" * 80)

    fout = os.path.join(OUT, "exp8b_vol_adjusted.json")
    with open(fout, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved:", fout)

    # Mean absolute errors
    for algo in ALGOS:
        errs = [abs(r["{}_error_pct".format(algo)]) for r in all_results
                if "{}_error_pct".format(algo) in r]
        if errs:
            print("  {} mean |error|: {:.1f}%".format(algo, np.mean(errs)))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
