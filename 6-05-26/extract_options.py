# -*- coding: utf-8 -*-
"""
Extract BTC put option market data from the SQL dump file.

Parses the COPY block for options_table and outputs a CSV with
BTC put options that have valid bid/ask and reasonable parameters.
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)

SQL_PATH = os.path.join(_ROOT, "data", "dump_2024-11-09_00_46_57.sql")
OUT_CSV = os.path.join(_ROOT, "6-05-26", "output", "btc_puts_market.csv")

COLUMNS = [
    "id", "full_name", "type", "open_interest", "expiry", "current_time",
    "strike", "interest_rate", "volume", "price_change", "mark_price",
    "bid_price", "bid_IV", "ask_price", "ask_IV", "mid_price", "mid_IV",
    "underlying_price"
]


def parse_value(val, col):
    if val == "\\N":
        return None
    if col in ("id", "strike"):
        return int(val)
    if col in ("open_interest", "interest_rate", "volume", "price_change",
               "mark_price", "bid_price", "bid_IV", "ask_price", "ask_IV",
               "mid_price", "mid_IV", "underlying_price"):
        return float(val)
    if col in ("expiry", "current_time"):
        return val.strip()
    return val.strip()


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    print("Extracting BTC put options from SQL dump...")
    print("  Source: {}".format(SQL_PATH))
    sys.stdout.flush()

    in_copy = False
    rows = []
    n_total = 0
    n_btc_put = 0

    with open(SQL_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("COPY public.options_table"):
                in_copy = True
                continue
            if in_copy:
                if line.strip() == "\\.":
                    in_copy = False
                    break
                n_total += 1
                parts = line.strip().split("\t")
                if len(parts) != len(COLUMNS):
                    continue

                full_name = parts[1]
                opt_type = parts[2]

                if "BTC" not in full_name:
                    continue
                if opt_type != "put":
                    continue

                n_btc_put += 1

                row = {}
                for i, col in enumerate(COLUMNS):
                    row[col] = parse_value(parts[i], col)

                # Filter: need mark_price, underlying_price, valid strike
                if row["mark_price"] is None or row["mark_price"] <= 0:
                    continue
                if row["underlying_price"] is None or row["underlying_price"] <= 0:
                    continue
                if row["strike"] is None or row["strike"] <= 0:
                    continue

                # Parse time-to-expiry
                try:
                    exp_str = row["expiry"]
                    cur_str = row["current_time"]
                    # Fix year: "0024" -> "2024"
                    if exp_str.startswith("0024"):
                        exp_str = "2024" + exp_str[4:]
                    exp_dt = datetime.strptime(exp_str[:19], "%Y-%m-%d %H:%M:%S")
                    cur_dt = datetime.strptime(cur_str[:19], "%Y-%m-%d %H:%M:%S")
                    tte_days = (exp_dt - cur_dt).total_seconds() / 86400.0
                except Exception:
                    continue

                if tte_days < 1 or tte_days > 90:
                    continue

                row["tte_days"] = round(tte_days, 2)
                row["moneyness"] = round(row["underlying_price"] / row["strike"], 4)

                # mark_price is in BTC terms, convert to USD
                row["mark_price_usd"] = round(
                    row["mark_price"] * row["underlying_price"], 2)

                if row["bid_price"] is not None and row["bid_price"] > 0:
                    row["bid_price_usd"] = round(
                        row["bid_price"] * row["underlying_price"], 2)
                else:
                    row["bid_price_usd"] = None

                if row["ask_price"] is not None and row["ask_price"] > 0:
                    row["ask_price_usd"] = round(
                        row["ask_price"] * row["underlying_price"], 2)
                else:
                    row["ask_price_usd"] = None

                rows.append(row)

                if n_btc_put % 500000 == 0:
                    print("  ... processed {} BTC puts, kept {}".format(
                        n_btc_put, len(rows)))
                    sys.stdout.flush()

    print("\n  Total options rows: {}".format(n_total))
    print("  BTC puts: {}".format(n_btc_put))
    print("  After filtering: {}".format(len(rows)))
    sys.stdout.flush()

    # Write CSV
    out_cols = ["id", "full_name", "current_time", "expiry", "tte_days",
                "strike", "underlying_price", "moneyness",
                "mark_price", "mark_price_usd",
                "bid_price", "bid_price_usd", "ask_price", "ask_price_usd",
                "bid_IV", "ask_IV", "mid_IV",
                "open_interest", "volume", "interest_rate"]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("  Saved: {}".format(OUT_CSV))
    print("  Rows written: {}".format(len(rows)))


if __name__ == "__main__":
    main()
