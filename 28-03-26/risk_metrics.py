# -*- coding: utf-8 -*-
"""Блок E: квантили, VaR, CVaR и «лямбда»-резерв по распределению ошибок хеджа."""

from __future__ import annotations

import numpy as np


def var_cvar(losses: np.ndarray, alpha: float = 0.95) -> dict:
    """losses — выборка максимальных дисконтированных потерь по траекториям (>=0)."""
    x = np.sort(np.asarray(losses, dtype=np.float64))
    n = len(x)
    if n == 0:
        return {"var": 0.0, "cvar": 0.0, "lambda_reserve_mean": 0.0}
    idx = int(np.floor(alpha * n))
    idx = min(max(idx, 0), n - 1)
    var = float(x[idx])
    tail = x[idx:]
    cvar = float(np.mean(tail))
    return {
        "var": var,
        "cvar": cvar,
        "lambda_reserve_mean": cvar,
        "quantile_level": alpha,
    }


def summary_stats(losses: np.ndarray) -> dict:
    x = np.asarray(losses, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
        "n": int(x.size),
    }
