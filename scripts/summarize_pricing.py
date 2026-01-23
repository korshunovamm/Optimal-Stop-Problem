#!/usr/bin/env python3
import argparse
import os
import glob
import pandas as pd
import numpy as np

Z_95 = 1.96  # нормальная аппроксимация для 95% CI

def _pick_latest_csv(metrics_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(metrics_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {metrics_dir}")
    return files[-1]

def _to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def summarize_pricing(
    input_csv: str,
    out_csv: str,
    scenario_cols,
    algo_col: str = "algo",
):
    df = pd.read_csv(input_csv)

    # Ensure numeric for relevant columns
    df = _to_numeric(df, ["price", "duration", "time_path_gen", "comp_time"])

    # Drop rows without price (failed runs)
    df = df.dropna(subset=["price"])

    group_cols = list(scenario_cols) + [algo_col]

    def agg_block(g: pd.DataFrame) -> pd.Series:
        n = len(g)

        price_mean = g["price"].mean()
        price_std = g["price"].std(ddof=1) if n > 1 else 0.0
        price_se = price_std / np.sqrt(n) if n > 0 else np.nan
        ci_half = Z_95 * price_se if n > 0 else np.nan

        duration_mean = g["duration"].mean() if "duration" in g else np.nan
        duration_std = g["duration"].std(ddof=1) if ("duration" in g and n > 1) else 0.0

        time_path_gen_mean = g["time_path_gen"].mean() if "time_path_gen" in g else np.nan
        comp_time_mean = g["comp_time"].mean() if "comp_time" in g else np.nan

        return pd.Series({
            "n_runs": n,

            "price_mean": price_mean,
            "price_std": price_std,
            "price_se": price_se,
            "price_ci95_halfwidth": ci_half,
            "price_ci95_low": price_mean - ci_half,
            "price_ci95_high": price_mean + ci_half,

            "duration_mean": duration_mean,
            "duration_std": duration_std,
            "time_path_gen_mean": time_path_gen_mean,
            "comp_time_mean": comp_time_mean,
        })

    summary = df.groupby(group_cols, dropna=False).apply(agg_block).reset_index()

    # Optional: make it easier to compare algorithms by sorting
    sort_cols = list(scenario_cols) + [algo_col]
    summary = summary.sort_values(sort_cols).reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    summary.to_csv(out_csv, index=False)

    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="", help="Path to metrics_draft CSV. If empty, take latest from --metrics_dir.")
    parser.add_argument("--metrics_dir", default="output/metrics_draft", help="Directory with raw CSV files.")
    parser.add_argument("--out_csv", default="output/summary/pricing_summary.csv", help="Output CSV path.")
    parser.add_argument("--scenario_cols", default="", help="Comma-separated list of scenario columns. If empty, use recommended default.")
    args = parser.parse_args()

    input_csv = args.input_csv.strip() or _pick_latest_csv(args.metrics_dir)

    if args.scenario_cols.strip():
        scenario_cols = [c.strip() for c in args.scenario_cols.split(",") if c.strip()]
    else:
        scenario_cols = [
            "model", "payoff",
            "spot", "strike", "maturity",
            "volatility", "drift", "dividend",
            "nb_stocks", "nb_dates", "nb_paths",
        ]

    summarize_pricing(
        input_csv=input_csv,
        out_csv=args.out_csv,
        scenario_cols=scenario_cols,
        algo_col="algo",
    )

    print(f"Input:  {input_csv}")
    print(f"Output: {args.out_csv}")

if __name__ == "__main__":
    main()
