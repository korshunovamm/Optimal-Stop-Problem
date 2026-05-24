# -*- coding: utf-8 -*-
"""
Backward induction with per-timestep snapshot storage.
Supports multi-dimensional options (nb_stocks > 1).

The standard NLSM in the library only retains the last network;
this module stores a snapshot (coefficients / state_dict) for every date,
enabling V(S,t) evaluation and delta-hedging afterwards.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata

import sys
from pathlib import Path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from optimal_stopping.algorithms.backward_induction import regression as reg
from optimal_stopping.algorithms.utils import basis_functions as bf
from optimal_stopping.algorithms.utils import neural_networks

AlgoName = str   # "LSM", "NLSM", "RLSM"


def _training_masks(immediate, split, train_itm_only):
    if train_itm_only:
        in_the_money = np.where(immediate[:split] > 0)
        in_the_money_all = np.where(immediate > 0)
    else:
        in_the_money = np.where(immediate[:split] < np.inf)
        in_the_money_all = np.where(immediate < np.inf)
    return in_the_money, in_the_money_all


def _stop_rule(immediate, continuation, train_itm_only):
    rule = np.zeros(len(immediate), dtype=np.float64)
    if train_itm_only:
        which = (immediate > continuation) & (immediate > np.finfo(float).eps)
    else:
        which = immediate > continuation
    rule[which] = 1.0
    return rule


@dataclass
class LSMSnapshot:
    coefficients: np.ndarray
    nb_stocks: int


@dataclass
class RLSMSnapshot:
    output_coef: np.ndarray


@dataclass
class NLSMSnapshot:
    state_dict: Optional[dict] = None
    fallback_mean: Optional[float] = None


@dataclass
class RLSMBackbone:
    reservoir_state: Dict[str, Any]
    hidden_size: int
    factors: Tuple[float, ...]


@dataclass
class SnapshotBundle:
    algo: AlgoName
    disc_factor: float
    nb_dates: int
    nb_stocks: int
    maturity: float
    rate: float
    split: int
    stock_paths: np.ndarray
    snapshots: Dict[int, Any] = field(default_factory=dict)
    continuation_scalar_t0: float = 0.0
    price_terminal_discounted: Optional[float] = None
    path_gen_seconds: float = 0.0
    rlsm_backbone: Optional[RLSMBackbone] = None
    hidden_size: int = 128


def _nlsm_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def _fit_nlsm_batch(x_itm, y_itm, state_dim, hidden_size, nb_epochs,
                     batch_size=2000):
    net = neural_networks.NetworkNLSM(state_dim, hidden_size=hidden_size).double()
    net.apply(_nlsm_init_weights)
    opt = optim.Adam(net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss(reduction="mean")
    x_t = torch.from_numpy(x_itm).double()
    y_t = torch.from_numpy(y_itm).double().view(-1, 1)
    net.train(True)
    for _ in range(nb_epochs):
        for batch in tdata.BatchSampler(
            tdata.RandomSampler(range(len(x_t)), replacement=False),
            batch_size=batch_size, drop_last=False,
        ):
            opt.zero_grad()
            out = net(x_t[batch])
            loss = loss_fn(out, y_t[batch])
            loss.backward()
            opt.step()
    return copy.deepcopy(net.state_dict())


def fit_backward_snapshots(
    algo,           # type: AlgoName
    model,
    payoff_f,
    hidden_size=128,    # type: int
    nb_epochs_nlsm=100, # type: int
    rlsm_factors=(1.0,),  # type: Tuple[float, ...]
    train_itm_only=True,   # type: bool
    train_eval_split=2,    # type: int
    seed=42,               # type: Optional[int]
):
    # type: (...) -> SnapshotBundle
    import time

    t_path0 = time.time()
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    stock_paths, _var = model.generate_paths()
    path_gen_elapsed = time.time() - t_path0

    nb_stocks = model.nb_stocks
    split = max(1, int(len(stock_paths) / train_eval_split))
    disc_factor = math.exp((-model.rate) * model.maturity / model.nb_dates)
    nb_dates = model.nb_dates

    # stock_paths shape: (nb_paths, nb_stocks, nb_dates+1)
    values = np.asarray(payoff_f.eval(stock_paths[:, :, -1])).ravel()
    snapshots = {}  # type: Dict[int, Any]

    basis = bf.BasisFunctions(nb_stocks)
    fac = rlsm_factors if rlsm_factors else (1.0,)
    rlsm_regressor = None
    if algo == "RLSM":
        rlsm_regressor = reg.ReservoirLeastSquares2(
            nb_stocks, hidden_size,
            factors=fac,
            activation=torch.nn.LeakyReLU(fac[0] / 2),
        )

    for date in range(stock_paths.shape[2] - 2, 0, -1):
        immediate = np.asarray(payoff_f.eval(stock_paths[:, :, date])).ravel()
        paths = stock_paths[:, :, date]   # (nb_paths, nb_stocks)
        y_next = values * disc_factor

        in_m, in_m_all = _training_masks(immediate, split, train_itm_only)

        if algo == "LSM":
            nb_paths_loc = paths.shape[0]
            reg_mat = np.empty((nb_paths_loc, basis.nb_base_fcts))
            for j in range(basis.nb_base_fcts):
                reg_mat[:, j] = basis.base_fct(j, paths[:, :], d2=True)
            coef, *_ = np.linalg.lstsq(
                reg_mat[in_m[0]], y_next[in_m[0]], rcond=None)
            cont = np.zeros(nb_paths_loc, dtype=np.float64)
            cont[in_m_all[0]] = reg_mat[in_m_all[0]] @ coef
            snapshots[date] = LSMSnapshot(
                coefficients=coef.astype(np.float64), nb_stocks=nb_stocks)

        elif algo == "NLSM":
            x_itm = paths[in_m[0]]
            y_itm = y_next[in_m[0]]
            if len(x_itm) < 5:
                cont = y_next.astype(np.float64).copy()
                fb_mean = float(np.mean(y_next)) if len(y_next) else 0.0
                snapshots[date] = NLSMSnapshot(
                    state_dict=None, fallback_mean=fb_mean)
            else:
                state_dict = _fit_nlsm_batch(
                    x_itm, y_itm, nb_stocks, hidden_size, nb_epochs_nlsm)
                net = neural_networks.NetworkNLSM(nb_stocks, hidden_size).double()
                net.load_state_dict(state_dict)
                net.eval()
                with torch.no_grad():
                    px = torch.from_numpy(paths[in_m_all[0]]).double()
                    cont = np.zeros(len(paths), dtype=np.float64)
                    if px.shape[0] > 0:
                        cont[in_m_all[0]] = net(px).numpy().ravel()
                snapshots[date] = NLSMSnapshot(
                    state_dict=state_dict, fallback_mean=None)

        elif algo == "RLSM":
            assert rlsm_regressor is not None
            x_np = paths.astype(np.float32)
            x_t = torch.from_numpy(x_np)
            reg_input = np.concatenate(
                [rlsm_regressor.reservoir(x_t).detach().numpy(),
                 np.ones((len(paths), 1))], axis=1)
            coef, *_ = np.linalg.lstsq(
                reg_input[in_m[0]], y_next[in_m[0]], rcond=None)
            cont = reg_input @ coef
            snapshots[date] = RLSMSnapshot(output_coef=coef.astype(np.float64))
        else:
            raise ValueError(algo)

        rule = _stop_rule(immediate, cont, train_itm_only)
        ex = rule > 0.5
        values = np.where(ex, immediate, values * disc_factor)

    cont0 = float(np.mean(values[split:]) * disc_factor)
    p0 = float(np.asarray(payoff_f.eval(stock_paths[:, :, 0])).ravel().mean())
    est_price = max(p0, cont0)

    backbone = None
    if algo == "RLSM" and rlsm_regressor is not None:
        backbone = RLSMBackbone(
            reservoir_state=copy.deepcopy(rlsm_regressor.reservoir.state_dict()),
            hidden_size=hidden_size,
            factors=fac,
        )

    bundle = SnapshotBundle(
        algo=algo,
        disc_factor=disc_factor,
        nb_dates=nb_dates,
        nb_stocks=nb_stocks,
        maturity=model.maturity,
        rate=model.rate,
        split=split,
        stock_paths=stock_paths,
        snapshots=snapshots,
        continuation_scalar_t0=cont0,
        price_terminal_discounted=est_price,
        path_gen_seconds=path_gen_elapsed,
        rlsm_backbone=backbone,
        hidden_size=hidden_size,
    )
    return bundle
