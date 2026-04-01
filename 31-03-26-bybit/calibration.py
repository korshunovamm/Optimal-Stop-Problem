# -*- coding: utf-8 -*-
"""
Calibrate GBM and Heston parameters from BTC daily prices.

GBM:  dS = r*S*dt + sigma*S*dW
      sigma = annualized realized vol over calibration window

Heston:  dS = r*S*dt + sqrt(V)*S*dW_S
         dV = kappa*(theta - V)*dt + xi*sqrt(V)*dW_V
         corr(dW_S, dW_V) = rho

Heston calibration uses method-of-moments on realized variance series.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def calibrate_gbm(prices, dt_years=1.0 / 365, r=0.05):
    # type: (np.ndarray, float, float) -> dict
    """
    Estimate GBM volatility from daily close prices.
    Returns dict with keys: sigma, r, spot, dt_years.
    """
    prices = np.asarray(prices, dtype=np.float64)
    log_ret = np.diff(np.log(prices))
    sigma = float(np.std(log_ret) / np.sqrt(dt_years))
    return {
        "sigma": sigma,
        "r": r,
        "spot": float(prices[-1]),
        "dt_years": dt_years,
        "n_obs": len(prices),
    }


def calibrate_heston(prices, dt_years=1.0 / 365, r=0.05):
    # type: (np.ndarray, float, float) -> dict
    """
    Estimate Heston parameters from daily close prices using
    method-of-moments on realized variance.

    Realized variance proxy: V_i = (log S_i - log S_{i-1})^2 / dt
    Then fit: E[V] = theta, Var(V) gives xi, autocorr(V) gives kappa,
    corr(dS/S, dV) gives rho.
    """
    prices = np.asarray(prices, dtype=np.float64)
    log_ret = np.diff(np.log(prices))

    rv = log_ret ** 2 / dt_years
    theta = float(np.mean(rv))
    var_rv = float(np.var(rv))

    # xi^2 * theta / (2*kappa) ≈ Var(V) in stationary regime
    # autocorr(V, lag=1) ≈ exp(-kappa * dt)
    n = len(rv)
    if n > 2:
        autocov = float(np.mean((rv[1:] - theta) * (rv[:-1] - theta)))
        autocorr = autocov / max(var_rv, 1e-12)
        autocorr = np.clip(autocorr, 0.01, 0.99)
        kappa = float(-np.log(autocorr) / dt_years)
    else:
        kappa = 2.0

    kappa = np.clip(kappa, 0.5, 50.0)

    xi_sq = 2.0 * kappa * var_rv / max(theta, 1e-12)
    xi = float(np.sqrt(max(xi_sq, 1e-6)))
    xi = np.clip(xi, 0.1, 10.0)

    rho = float(np.corrcoef(log_ret[1:], rv[1:] - rv[:-1])[0, 1])
    rho = np.clip(rho, -0.99, 0.99)
    if np.isnan(rho):
        rho = -0.5

    sigma0 = float(np.sqrt(max(theta, 1e-6)))

    return {
        "sigma": sigma0,
        "r": r,
        "spot": float(prices[-1]),
        "theta": theta,
        "kappa": kappa,
        "xi": xi,
        "rho": rho,
        "dt_years": dt_years,
        "n_obs": len(prices),
    }


def bs_put_price(S, K, T, r, sigma):
    # type: (float, float, float, float, float) -> float
    """European put price (Black-Scholes closed form)."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_put_delta(S, K, T, r, sigma):
    # type: (float, float, float, float, float) -> float
    """European put delta (Black-Scholes closed form)."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1) - 1.0)


if __name__ == "__main__":
    from btc_data_loader import daily_close, get_windows
    daily = daily_close()
    wins = get_windows(daily)
    w = wins[0]
    gbm = calibrate_gbm(w["calib_prices"])
    heston = calibrate_heston(w["calib_prices"])
    print("Window 0: {} -- {}".format(w["calib_start"].date(), w["hedge_end"].date()))
    print("GBM:    sigma={:.3f}".format(gbm["sigma"]))
    print("Heston: kappa={:.2f}  theta={:.4f}  xi={:.2f}  rho={:.2f}  sigma0={:.3f}".format(
        heston["kappa"], heston["theta"], heston["xi"], heston["rho"], heston["sigma"]))
