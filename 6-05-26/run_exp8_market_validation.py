# -*- coding: utf-8 -*-
"""
Experiment 8: Market Validation.

Compare model-computed American put option prices with actual BTC option
market prices from Bybit/Deribit.

Strategy:
  1. Select representative put options from market data (near ATM, various TTE)
  2. For each option: train LSM/RLSM/NLSM on historical BTC paths
     ending at the observation date, with matched maturity
  3. Convert model prices to USD and compare with market mark_price

Usage:
    /home/kulikoval/.conda/envs/OptStopRandNN/bin/python -u 6-05-26/run_exp8_market_validation.py
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_DIR = str(Path(__file__).resolve().parent)
_ROOT = str(Path(__file__).resolve().parent.parent)
for p in (_DIR, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from feature_builder import load_btc_daily, add_all_features, build_paths, augment_paths
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
AUGMENT = 2000
NOISE_SIGMA = 0.008
SEED = 42
REFERENCE_SPOT = 100.0
NB_HEDGE = 40


def select_market_options(n_per_bucket=3):
    """Select representative put options: near-ATM, liquid, various TTE."""
    print("  Loading market options CSV (sampled) ...")
    sys.stdout.flush()

    # Read in chunks to handle large file
    chunks = pd.read_csv(CSV_OPTS, chunksize=500000)
    frames = []
    for chunk in chunks:
        # Filter: near ATM (moneyness 0.9-1.1), has bid and ask
        mask = (
            (chunk['moneyness'] >= 0.85) &
            (chunk['moneyness'] <= 1.10) &
            (chunk['bid_price_usd'].notna()) &
            (chunk['ask_price_usd'].notna()) &
            (chunk['bid_price_usd'] > 0) &
            (chunk['open_interest'] >= 10) &
            (chunk['volume'] >= 1)
        )
        frames.append(chunk[mask])
        if sum(len(f) for f in frames) > 50000:
            break

    df = pd.concat(frames, ignore_index=True)
    print("  Near-ATM liquid puts: {}".format(len(df)))
    sys.stdout.flush()

    # TTE buckets: 7d, 14d, 30d
    buckets = [(5, 10), (10, 20), (25, 40)]
    selected = []
    for lo, hi in buckets:
        sub = df[(df['tte_days'] >= lo) & (df['tte_days'] <= hi)]
        if len(sub) == 0:
            continue
        # Pick n_per_bucket distinct observation dates
        sub = sub.copy()
        sub['date'] = pd.to_datetime(sub['current_time']).dt.date
        dates = sub['date'].unique()
        np.random.seed(SEED)
        chosen_dates = np.random.choice(dates, min(n_per_bucket, len(dates)), replace=False)
        for d in chosen_dates:
            day_sub = sub[sub['date'] == d]
            # Pick the most liquid option closest to ATM
            day_sub = day_sub.copy()
            day_sub['atm_dist'] = abs(day_sub['moneyness'] - 1.0)
            best = day_sub.nsmallest(1, 'atm_dist').iloc[0]
            selected.append(best.to_dict())

    print("  Selected {} options for validation".format(len(selected)))
    sys.stdout.flush()
    return selected


def price_option_with_models(obs_date_str, strike, underlying_price,
                             tte_days, dims=1):
    """Train all models on historical paths and return model prices."""
    maturity = tte_days / 365.0
    nb_dates = max(int(tte_days), 7)

    # Normalised strike
    norm_strike = (strike / underlying_price) * REFERENCE_SPOT

    # Load training data ending at observation date
    end_date = obs_date_str
    start_dt = datetime.strptime(obs_date_str[:10], "%Y-%m-%d") - timedelta(days=730)
    start_date = start_dt.strftime("%Y-%m-%d")

    df = load_btc_daily(CSV_OHLC, start_date, end_date)
    if dims > 1:
        df = add_all_features(df)

    train_paths = build_paths(df, nb_dates, 1, dims, REFERENCE_SPOT)
    if train_paths.shape[0] < 100:
        return None

    extra = augment_paths(train_paths, AUGMENT, NOISE_SIGMA, seed=SEED)
    train_paths = np.concatenate([train_paths, extra], axis=0)
    np.random.seed(SEED)
    train_paths = train_paths[np.random.permutation(train_paths.shape[0])]

    model = HistoricalModel(train_paths, 0.05, maturity, nb_dates)
    payoff_f = BtcPut(norm_strike)

    results = {}
    for algo in ALGOS:
        t0 = time.time()
        bundle = fit_backward_snapshots(
            algo=algo, model=model, payoff_f=payoff_f,
            hidden_size=HIDDEN, nb_epochs_nlsm=NB_EPOCHS,
            rlsm_factors=(1.0,), train_itm_only=True,
            train_eval_split=2, seed=SEED)
        fit_t = time.time() - t0

        # Model price in normalised units
        norm_price = float(bundle.price_terminal_discounted)
        # Convert back to USD
        usd_price = norm_price * (underlying_price / REFERENCE_SPOT)

        results[algo] = {
            "norm_price": norm_price,
            "usd_price": round(usd_price, 2),
            "fit_time": round(fit_t, 2),
            "n_train": train_paths.shape[0],
        }

    return results


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 70)
    print("Experiment 8: Market Validation — Model vs Market BTC Option Prices")
    print("=" * 70)
    sys.stdout.flush()

    # Select options
    options = select_market_options(n_per_bucket=3)

    all_results = []
    for i, opt in enumerate(options):
        print("\n--- Option {}/{} ---".format(i + 1, len(options)))
        print("  Name: {}".format(opt.get('full_name', '?')))
        print("  Obs date: {}".format(opt['current_time'][:10]))
        print("  Strike: {}, Underlying: {:.0f}, TTE: {:.1f}d".format(
            opt['strike'], opt['underlying_price'], opt['tte_days']))
        print("  Market mark_price_usd: {:.2f}".format(opt['mark_price_usd']))
        print("  Market bid/ask USD: {:.2f} / {:.2f}".format(
            opt.get('bid_price_usd', 0) or 0,
            opt.get('ask_price_usd', 0) or 0))
        sys.stdout.flush()

        model_prices = price_option_with_models(
            obs_date_str=opt['current_time'][:10],
            strike=opt['strike'],
            underlying_price=opt['underlying_price'],
            tte_days=opt['tte_days'],
            dims=1
        )

        if model_prices is None:
            print("  SKIP: insufficient training data")
            continue

        entry = {
            "full_name": opt.get('full_name', ''),
            "obs_date": opt['current_time'][:10],
            "strike": int(opt['strike']),
            "underlying_price": round(opt['underlying_price'], 2),
            "tte_days": opt['tte_days'],
            "moneyness": opt['moneyness'],
            "market_mark_usd": opt['mark_price_usd'],
            "market_bid_usd": opt.get('bid_price_usd'),
            "market_ask_usd": opt.get('ask_price_usd'),
        }

        for algo in ALGOS:
            mp = model_prices[algo]
            entry["{}_usd".format(algo)] = mp["usd_price"]
            entry["{}_fit_time".format(algo)] = mp["fit_time"]
            err_pct = abs(mp["usd_price"] - opt['mark_price_usd']) / max(opt['mark_price_usd'], 1) * 100
            entry["{}_error_pct".format(algo)] = round(err_pct, 1)

            print("  {} price: ${:.2f} (err={:.1f}%, fit={:.2f}s)".format(
                algo, mp["usd_price"], err_pct, mp["fit_time"]))
            sys.stdout.flush()

        all_results.append(entry)

    # Summary
    print("\n\n" + "=" * 80)
    print("MARKET VALIDATION SUMMARY")
    print("=" * 80)
    hdr = "{:<22} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "Option", "TTE", "Market$", "LSM$", "RLSM$", "NLSM$", "LSMerr%", "RLSMerr%")
    print(hdr)
    print("-" * len(hdr))
    for r in all_results:
        print("{:<22} {:>5.1f}d {:>9.2f} {:>9.2f} {:>9.2f} {:>9.2f} {:>8.1f}% {:>8.1f}%".format(
            r['full_name'][:22], r['tte_days'],
            r['market_mark_usd'],
            r.get('LSM_usd', 0), r.get('RLSM_usd', 0), r.get('NLSM_usd', 0),
            r.get('LSM_error_pct', 0), r.get('RLSM_error_pct', 0)))
    print("=" * 80)

    # Save
    fout = os.path.join(OUT, "exp8_market_validation.json")
    with open(fout, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved:", fout)

    # Also save CSV
    df_out = pd.DataFrame(all_results)
    csv_out = os.path.join(OUT, "exp8_summary.csv")
    df_out.to_csv(csv_out, index=False)
    print("Saved:", csv_out)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
