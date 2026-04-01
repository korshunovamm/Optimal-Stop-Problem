# -*- coding: utf-8 -*-
"""Custom payoff for multi-dimensional BTC paths.

The payoff is an American Put on the first coordinate (normalised BTC price).
Auxiliary features (realized vol, volume, etc.) are ignored by the payoff
but used by the regression model for continuation value estimation.
"""
from __future__ import annotations

import numpy as np

import sys
from pathlib import Path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from optimal_stopping.payoffs.payoff import Payoff


class BtcPut(Payoff):
    """American Put on the first asset coordinate: max(K - S1, 0)."""
    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        return self.eval(X)

    def eval(self, X):
        # X: (nb_paths, nb_stocks) or (nb_paths,)
        X = np.atleast_2d(X)
        return np.maximum(0.0, self.strike - X[:, 0])
