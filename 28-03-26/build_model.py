# -*- coding: utf-8 -*-
"""Блок B: фабрики моделей и пэйоффа под optimal_stopping."""

from __future__ import annotations

import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from optimal_stopping.data import stock_model as sm
from optimal_stopping.payoffs import payoff

from config import ExperimentParams


def build_payoff(params: ExperimentParams):
    return payoff.Put1Dim(params.strike)


def build_black_scholes(params: ExperimentParams) -> sm.BlackScholes:
    return sm.BlackScholes(
        drift=params.drift,
        volatility=params.volatility,
        nb_paths=params.nb_paths_train,
        nb_stocks=1,
        nb_dates=params.nb_dates,
        spot=params.spot,
        maturity=params.maturity,
        dividend=params.dividend,
    )


def build_heston(params: ExperimentParams) -> sm.Heston:
    return sm.Heston(
        drift=params.drift,
        volatility=params.volatility,
        mean=params.heston_mean,
        speed=params.heston_speed,
        correlation=params.heston_corr,
        nb_stocks=1,
        nb_paths=params.nb_paths_train,
        nb_dates=params.nb_dates,
        spot=params.spot,
        maturity=params.maturity,
        dividend=params.dividend,
    )


def build_market_model(params: ExperimentParams):
    if params.stock_model == "BlackScholes":
        return build_black_scholes(params)
    if params.stock_model == "Heston":
        return build_heston(params)
    raise ValueError(f"Unknown stock_model: {params.stock_model}")


def clone_model_same_params(model, nb_paths: int):
    """Та же модель, другое число траекторий (для теста хеджа)."""
    if model.name == "BlackScholes":
        return sm.BlackScholes(
            drift=model.rate,
            volatility=float(model.volatility),
            nb_paths=nb_paths,
            nb_stocks=model.nb_stocks,
            nb_dates=model.nb_dates,
            spot=model.spot,
            maturity=model.maturity,
            dividend=model.dividend,
        )
    if model.name == "Heston":
        return sm.Heston(
            drift=model.rate,
            volatility=float(model.volatility),
            mean=model.mean,
            speed=model.speed,
            correlation=model.correlation,
            nb_stocks=model.nb_stocks,
            nb_paths=nb_paths,
            nb_dates=model.nb_dates,
            spot=model.spot,
            maturity=model.maturity,
            dividend=model.dividend,
        )
    raise ValueError(f"Unsupported model clone: {model.name}")
