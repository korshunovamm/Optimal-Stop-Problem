# -*- coding: utf-8 -*-
"""Adapter that wraps a pre-built numpy array of paths into the Model
interface expected by ``fit_backward_snapshots``."""
from __future__ import annotations

import math
import numpy as np


class HistoricalModel(object):
    """Drop-in replacement for ``stock_model.Model`` when paths come from
    historical data rather than a stochastic generator.

    Parameters
    ----------
    paths : np.ndarray, shape (nb_paths, nb_stocks, nb_dates + 1)
        Pre-built price paths (already normalised if needed).
    rate : float
        Annualised risk-free rate.
    maturity : float
        Option maturity in years.
    nb_dates : int
        Number of exercise dates (steps in each path).
    """

    def __init__(self, paths, rate, maturity, nb_dates):
        # type: (np.ndarray, float, float, int) -> None
        assert paths.ndim == 3, "Expected shape (nb_paths, nb_stocks, nb_dates+1)"
        assert paths.shape[2] == nb_dates + 1, \
            "paths.shape[2]={} but expected nb_dates+1={}".format(paths.shape[2], nb_dates + 1)
        self._paths = paths.astype(np.float64)
        self.nb_paths = paths.shape[0]
        self.nb_stocks = paths.shape[1]
        self.nb_dates = nb_dates
        self.maturity = maturity
        self.rate = rate
        self.spot = float(paths[0, 0, 0])
        self.dividend = 0.0
        self.volatility = 0.0
        self.dt = maturity / nb_dates
        self.df = math.exp(-rate * self.dt)
        self.name = "Historical"
        self.return_var = False

    def generate_paths(self, nb_paths=None):
        # type: (int) -> tuple
        """Return stored paths (optionally sub-sampled)."""
        if nb_paths is not None and nb_paths < self.nb_paths:
            idx = np.random.choice(self.nb_paths, size=nb_paths, replace=False)
            return self._paths[idx].copy(), None
        return self._paths.copy(), None
