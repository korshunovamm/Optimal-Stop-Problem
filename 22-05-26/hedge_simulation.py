# -*- coding: utf-8 -*-
"""
Discrete delta-hedge for multi-dimensional American options.
Portfolio: cash + stock positions in d assets, self-financing with discrete rebalancing.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_PKG = str(Path(__file__).resolve().parent)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from american_value import american_value, get_delta
from backward_snapshots import SnapshotBundle


def _intrinsic(payoff_f, s_vec):
    x = np.atleast_2d(s_vec.astype(np.float64))
    return float(np.asarray(payoff_f.eval(x)).ravel()[0])


def hedge_one_path(bundle, payoff_f, spot_path, hidden_size, bump):
    # type: (SnapshotBundle, Any, np.ndarray, int, float) -> dict
    """
    Discrete self-financing delta-hedge along one path.
    
    spot_path: shape (nb_stocks, nb_dates+1)  -- prices S_0, ..., S_n for d assets.
    
    Returns dict with per-timestep shortfalls.
    """
    r = bundle.rate
    dt = bundle.maturity / bundle.nb_dates
    grow = math.exp(r * dt)
    n = bundle.nb_dates
    d = spot_path.shape[0]

    v0 = float(bundle.price_terminal_discounted or 0.0)
    s0 = spot_path[:, 0]  # (d,)
    d0 = get_delta(bundle, payoff_f, 0, s0, hidden_size, bump)  # (d,)

    cash = v0 - np.dot(d0, s0)
    stock_pos = d0.copy()

    sf_mv = []
    sf_intr = []

    for i in range(n):
        s_next = spot_path[:, i + 1]  # (d,)
        cash = cash * grow
        phi_i = cash + np.dot(stock_pos, s_next)

        v_i = american_value(bundle, payoff_f, i + 1, s_next, hidden_size)
        intr_i = _intrinsic(payoff_f, s_next)

        t_loc = (i + 1) * dt
        dfac = math.exp(-r * t_loc)
        sf_mv.append(dfac * max(0.0, v_i - phi_i))
        sf_intr.append(dfac * max(0.0, intr_i - phi_i))

        if i < n - 1:
            d_next = get_delta(bundle, payoff_f, i + 1, s_next, hidden_size, bump)
            cash = phi_i - np.dot(d_next, s_next)
            stock_pos = d_next.copy()

    return {
        "disc_shortfall_vs_model": np.array(sf_mv),
        "disc_shortfall_vs_intrinsic": np.array(sf_intr),
    }


def max_hedge_losses(bundle, payoff_f, test_paths, hidden_size, bump):
    # type: (SnapshotBundle, Any, np.ndarray, int, float) -> Tuple[np.ndarray, np.ndarray]
    """
    test_paths: (nb_paths, nb_stocks, nb_dates+1)
    Returns per-path max discounted shortfall arrays.
    """
    n_p = test_paths.shape[0]
    m1 = np.zeros(n_p)
    m2 = np.zeros(n_p)
    for j in range(n_p):
        path_j = test_paths[j, :, :]  # (nb_stocks, nb_dates+1)
        out = hedge_one_path(bundle, payoff_f, path_j, hidden_size, bump)
        m1[j] = float(np.max(out["disc_shortfall_vs_model"]))
        m2[j] = float(np.max(out["disc_shortfall_vs_intrinsic"]))
    return m1, m2
