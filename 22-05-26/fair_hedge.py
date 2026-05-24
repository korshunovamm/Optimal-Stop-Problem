# -*- coding: utf-8 -*-
"""
Fair-capital hedge simulation.

Key difference from btc_hedge.py: accepts an external V0 so that all
algorithms start with the SAME initial capital.  This isolates delta
quality from pricing accuracy.
"""
from __future__ import annotations

import math, sys
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
    try:
        full_delta = get_delta(bundle, payoff_f, date_idx, s_vec,
                               hidden_size, bump)
        d0 = float(full_delta[0])
        if abs(d0) < 10.0:
            return np.clip(d0, -1.0, 1.0)
    except Exception:
        pass
    s_up = s_vec.copy(); s_down = s_vec.copy()
    s_up[0] += bump;     s_down[0] -= bump
    v_up = american_value(bundle, payoff_f, date_idx, s_up, hidden_size)
    v_dn = american_value(bundle, payoff_f, date_idx, s_down, hidden_size)
    return np.clip((v_up - v_dn) / (2.0 * bump), -1.0, 1.0)


def fair_hedge_one_path(bundle, payoff_f, spot_path, hidden_size, bump,
                        external_v0=None):
    """
    Delta-hedge along one path.

    If external_v0 is given, use it as initial portfolio value instead of
    the model's own price.  This makes the comparison fair across algos.

    Returns dict with per-step shortfall arrays and the full portfolio
    value trajectory.
    """
    r = bundle.rate
    dt = bundle.maturity / bundle.nb_dates
    grow = math.exp(r * dt)
    n = bundle.nb_dates

    v0 = external_v0 if external_v0 is not None else float(
        bundle.price_terminal_discounted or 0.0)

    s0 = spot_path[:, 0]
    s0_price = float(s0[0])
    d0 = _btc_delta(bundle, payoff_f, 0, s0, hidden_size, bump)

    cash = v0 - d0 * s0_price
    stock_pos = d0

    shortfalls = []
    portfolio_vals = [v0]
    deltas = [d0]

    for i in range(n):
        s_next = spot_path[:, i + 1]
        s_next_price = float(s_next[0])
        cash = cash * grow
        phi_i = cash + stock_pos * s_next_price
        portfolio_vals.append(phi_i)

        intr_i = float(payoff_f.eval(s_next.reshape(1, -1))[0])
        t_loc = (i + 1) * dt
        dfac = math.exp(-r * t_loc)
        shortfalls.append(dfac * max(0.0, intr_i - phi_i))

        if i < n - 1:
            d_next = _btc_delta(bundle, payoff_f, i + 1, s_next,
                                hidden_size, bump)
            cash = phi_i - d_next * s_next_price
            stock_pos = d_next
            deltas.append(d_next)

    return {
        "shortfalls": np.array(shortfalls),
        "max_shortfall": float(np.max(shortfalls)),
        "mean_shortfall": float(np.mean(shortfalls)),
        "portfolio": np.array(portfolio_vals),
        "deltas": np.array(deltas),
        "v0_used": v0,
    }
