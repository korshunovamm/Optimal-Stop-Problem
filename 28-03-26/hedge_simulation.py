# -*- coding: utf-8 -*-
"""
Дискретный дельта-хедж по оценке V(S,t) из снимков (блок D).
Портфель: Φ_i = Δ_i S_i + B_i, самофинансирование с дисконтированием exp(r Δt) на шаг.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from american_value import american_value, delta_central
from backward_snapshots import SnapshotBundle


def _intrinsic_1d(payoff_f, s: float) -> float:
    return float(np.asarray(payoff_f.eval(np.array([[s]], dtype=np.float64))).ravel()[0])


def hedge_one_path(
    bundle: SnapshotBundle,
    payoff_f,
    spot_path: np.ndarray,
    hidden_size: int,
    bump: float,
) -> dict:
    """
    spot_path: shape (nb_dates+1,) цены S_0,...,S_n.
    """
    r = bundle.rate
    dt = bundle.maturity / bundle.nb_dates
    grow = math.exp(r * dt)
    n = bundle.nb_dates

    v0 = float(bundle.price_terminal_discounted or 0.0)
    s0 = float(spot_path[0])
    d0 = delta_central(bundle, payoff_f, 0, s0, hidden_size, bump)
    B = v0 - d0 * s0
    d_curr = d0

    disc_factors = []
    phi_w = []
    v_m = []
    intr_w = []
    sf_mv = []
    sf_intr = []

    for i in range(n):
        s_next = float(spot_path[i + 1])
        wealth_pre = d_curr * s_next + B * grow
        d_next = delta_central(bundle, payoff_f, i + 1, s_next, hidden_size, bump)
        B = wealth_pre - d_next * s_next
        d_curr = d_next
        phi_i = d_curr * s_next + B
        v_i = american_value(
            bundle,
            payoff_f,
            i + 1,
            np.array([[s_next]], dtype=np.float64),
            hidden_size,
        )
        intr = _intrinsic_1d(payoff_f, s_next)
        t_loc = (i + 1) * dt
        dfac = math.exp(-r * t_loc)
        disc_factors.append(dfac)
        phi_w.append(phi_i)
        v_m.append(v_i)
        intr_w.append(intr)
        sf_mv.append(dfac * max(0.0, v_i - phi_i))
        sf_intr.append(dfc_intr_helper(dfac, intr, phi_i))

    return {
        "discount": np.array(disc_factors),
        "phi": np.array(phi_w),
        "V_model": np.array(v_m),
        "intrinsic": np.array(intr_w),
        "disc_shortfall_vs_model": np.array(sf_mv),
        "disc_shortfall_vs_intrinsic": np.array(sf_intr),
    }


def dfc_intr_helper(dfac: float, intrinsic: float, phi: float) -> float:
    """
    Консервативная величина для short put: дисконтированная недостача покрытия intrinsic
    реплицирующим портфелем long-опциона (перестановка знака относительно V-Phi).
    Здесь: max(0, intrinsic - phi) если считать phi как хеджируемое богатство long-V.
    Для сопоставимости с сообщением руководителя — отдельная метрика; сама формула
    зависит от соглашения о знаке; см. отчёт.
    """
    return dfac * max(0.0, intrinsic - phi)


def max_hedge_losses(
    bundle: SnapshotBundle,
    payoff_f,
    paths: np.ndarray,
    hidden_size: int,
    bump: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    paths: (nb_paths, nb_dates+1)
    Возвращает для каждой траектории:
      max дисконтированного shortfall model vs phi,
      max дисконтированного shortfall intrinsic vs phi.
    """
    n_p = paths.shape[0]
    m1 = np.zeros(n_p)
    m2 = np.zeros(n_p)
    for j in range(n_p):
        out = hedge_one_path(
            bundle, payoff_f, paths[j, 0, :], hidden_size, bump
        )
        m1[j] = float(np.max(out["disc_shortfall_vs_model"]))
        m2[j] = float(np.max(out["disc_shortfall_vs_intrinsic"]))
    return m1, m2
