# -*- coding: utf-8 -*-
"""Factories for payoffs and stock models supporting multi-dimensional options."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_PKG = str(Path(__file__).resolve().parent)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from optimal_stopping.data import stock_model as sm
from optimal_stopping.payoffs import payoff

from config import ExperimentParams

_PAYOFFS = {
    "MaxCall": payoff.MaxCall,
    "MaxPut": payoff.MaxPut,
    "BasketCall": payoff.BasketCall,
    "GeometricPut": payoff.GeometricPut,
    "Put1Dim": payoff.Put1Dim,
    "Call1Dim": payoff.Call1Dim,
    "MinPut": payoff.MinPut,
}


def build_payoff(params):
    # type: (ExperimentParams) -> payoff.Payoff
    cls = _PAYOFFS.get(params.payoff_name)
    if cls is None:
        raise ValueError("Unknown payoff: {}".format(params.payoff_name))
    return cls(params.strike)


def build_market_model(params):
    # type: (ExperimentParams) -> sm.Model
    if params.stock_model == "BlackScholes":
        return sm.BlackScholes(
            drift=params.drift,
            volatility=params.volatility,
            nb_paths=params.nb_paths_train,
            nb_stocks=params.nb_stocks,
            nb_dates=params.nb_dates,
            spot=params.spot,
            maturity=params.maturity,
            dividend=params.dividend,
        )
    if params.stock_model == "Heston":
        return sm.Heston(
            drift=params.drift,
            volatility=params.volatility,
            mean=params.heston_mean,
            speed=params.heston_speed,
            correlation=params.heston_corr,
            nb_stocks=params.nb_stocks,
            nb_paths=params.nb_paths_train,
            nb_dates=params.nb_dates,
            spot=params.spot,
            maturity=params.maturity,
            dividend=params.dividend,
        )
    if params.stock_model == "HestonWithVar":
        return sm.HestonWithVar(
            drift=params.drift,
            volatility=params.volatility,
            mean=params.heston_mean,
            speed=params.heston_speed,
            correlation=params.heston_corr,
            nb_stocks=params.nb_stocks,
            nb_paths=params.nb_paths_train,
            nb_dates=params.nb_dates,
            spot=params.spot,
            maturity=params.maturity,
            dividend=params.dividend,
        )
    raise ValueError("Unknown stock_model: {}".format(params.stock_model))


def clone_model(model, nb_paths):
    # type: (sm.Model, int) -> sm.Model
    """Clone a model with a different number of paths."""
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
    if model.name == "HestonWithVar":
        return sm.HestonWithVar(
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
    raise ValueError("Unsupported model clone: {}".format(model.name))
