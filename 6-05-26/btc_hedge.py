# -*- coding: utf-8 -*-
"""
BTC-specific hedge simulation.

When the state is multi-dimensional (price + vol + volume …), the payoff
and the hedge position depend only on the first coordinate (BTC price).
Auxiliary features are *observed* but not *traded*.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
_PKG = str(Path(__file__).resolve().parent)
for p in (_PKG, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from backward_snapshots import SnapshotBundle
from american_value import american_value, get_delta
from btc_payoff import BtcPut


def _btc_delta(bundle, payoff_f, date_idx, s_vec, hidden_size, bump):
    # type: (...) -> float
    """Compute hedge delta w.r.t. BTC price only (first coordinate).

    For NN methods (RLSM/NLSM), uses the full gradient but extracts [0].
    Falls back to 1D finite-difference on the price coordinate.
    """
    try:
        full_delta = get_delta(bundle, payoff_f, date_idx, s_vec,
                               hidden_size, bump)
        d0 = float(full_delta[0])
        if abs(d0) < 10.0:
            return np.clip(d0, -1.0, 1.0)
    except Exception:
        pass

    s_up = s_vec.copy()
    s_down = s_vec.copy()
    s_up[0] += bump
    s_down[0] -= bump
    v_up = american_value(bundle, payoff_f, date_idx, s_up, hidden_size)
    v_down = american_value(bundle, payoff_f, date_idx, s_down, hidden_size)
    return np.clip((v_up - v_down) / (2.0 * bump), -1.0, 1.0)


def btc_hedge_one_path(bundle, payoff_f, spot_path, hidden_size, bump):
    # type: (SnapshotBundle, BtcPut, np.ndarray, int, float) -> dict
    """
    Discrete self-financing delta-hedge along one BTC path.

    spot_path : shape (nb_stocks, nb_dates+1)
        Full feature path. Only coordinate 0 is traded.

    Returns dict with per-timestep shortfall arrays.
    """
    r = bundle.rate
    dt = bundle.maturity / bundle.nb_dates
    grow = math.exp(r * dt)
    n = bundle.nb_dates

    v0 = float(bundle.price_terminal_discounted or 0.0)
    s0 = spot_path[:, 0]                    # (d,) full feature vector
    s0_price = float(s0[0])                 # BTC price only
    d0 = _btc_delta(bundle, payoff_f, 0, s0, hidden_size, bump)

    cash = v0 - d0 * s0_price
    stock_pos = d0

    sf_mv = []
    sf_intr = []

    for i in range(n):
        s_next = spot_path[:, i + 1]
        s_next_price = float(s_next[0])
        cash = cash * grow
        phi_i = cash + stock_pos * s_next_price

        v_i = american_value(bundle, payoff_f, i + 1, s_next, hidden_size)
        intr_i = float(payoff_f.eval(s_next.reshape(1, -1))[0])

        t_loc = (i + 1) * dt
        dfac = math.exp(-r * t_loc)
        sf_mv.append(dfac * max(0.0, v_i - phi_i))
        sf_intr.append(dfac * max(0.0, intr_i - phi_i))

        if i < n - 1:
            d_next = _btc_delta(bundle, payoff_f, i + 1, s_next,
                                hidden_size, bump)
            cash = phi_i - d_next * s_next_price
            stock_pos = d_next

    return {
        "disc_shortfall_vs_model": np.array(sf_mv),
        "disc_shortfall_vs_intrinsic": np.array(sf_intr),
    }


def btc_max_hedge_losses(bundle, payoff_f, test_paths, hidden_size, bump):
    # type: (...) -> tuple
    """Run hedge on multiple paths, return per-path max shortfalls."""
    n_p = test_paths.shape[0]
    m1 = np.zeros(n_p)
    m2 = np.zeros(n_p)
    for j in range(n_p):
        out = btc_hedge_one_path(bundle, payoff_f, test_paths[j],
                                 hidden_size, bump)
        m1[j] = float(np.max(out["disc_shortfall_vs_model"]))
        m2[j] = float(np.max(out["disc_shortfall_vs_intrinsic"]))
    return m1, m2
