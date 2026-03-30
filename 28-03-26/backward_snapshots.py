# -*- coding: utf-8 -*-
"""
Обратная индукция с сохранением аппроксимаций продолжения на каждом шаге времени.
Нужно для оценки V(S, t) и хеджа (блоки C–D); стандартный NLSM в библиотеке хранит только последнюю сеть.
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

from optimal_stopping.algorithms.backward_induction import regression as reg
from optimal_stopping.algorithms.utils import basis_functions as bf
from optimal_stopping.algorithms.utils import neural_networks

# Одно из: "LSM", "NLSM", "RLSM".
AlgoName = str


def _training_masks(
    immediate: np.ndarray,
    split: int,
    train_itm_only: bool,
):
    if train_itm_only:
        in_the_money = np.where(immediate[:split] > 0)
        in_the_money_all = np.where(immediate > 0)
    else:
        in_the_money = np.where(immediate[:split] < np.inf)
        in_the_money_all = np.where(immediate < np.inf)
    return in_the_money, in_the_money_all


def _stop_rule(
    immediate: np.ndarray,
    continuation: np.ndarray,
    train_itm_only: bool,
) -> np.ndarray:
    rule = np.zeros(len(immediate), dtype=np.float64)
    if train_itm_only:
        which = (immediate > continuation) & (immediate > np.finfo(float).eps)
    else:
        which = immediate > continuation
    rule[which] = 1.0
    return rule


@dataclass
class LSMSnapshot:
    coefficients: np.ndarray  # (nb_basis,)
    nb_stocks: int


@dataclass
class RLSMSnapshot:
    output_coef: np.ndarray  # (hidden+1,)


@dataclass
class NLSMSnapshot:
    """Веса сети или грубая константа продолжения, если ITM-выборка слишком мала."""

    state_dict: Optional[dict] = None
    fallback_mean: Optional[float] = None


@dataclass
class RLSMBackbone:
    """Общий резервуар для всех дат (как в оригинальном RLSM)."""

    reservoir_state: Dict[str, Any]
    hidden_size: int
    factors: Tuple[float, ...]


@dataclass
class SnapshotBundle:
    algo: AlgoName
    disc_factor: float
    nb_dates: int
    maturity: float
    rate: float
    split: int
    stock_paths: np.ndarray
    # snapshots[date] = объект для даты date in {1, ..., nb_dates-1}
    snapshots: Dict[int, Any] = field(default_factory=dict)
    # Как в LSM: среднее дисконтированное «продолжение» с t=0 (не функция S).
    continuation_scalar_t0: float = 0.0
    price_terminal_discounted: Optional[float] = None
    path_gen_seconds: float = 0.0
    rlsm_backbone: Optional[RLSMBackbone] = None


def _nlsm_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def _fit_nlsm_batch(
    x_itm: np.ndarray,
    y_itm: np.ndarray,
    state_dim: int,
    hidden_size: int,
    nb_epochs: int,
    batch_size: int = 2000,
) -> dict:
    net = neural_networks.NetworkNLSM(state_dim, hidden_size=hidden_size).double()
    net.apply(_nlsm_init_weights)
    opt = optim.Adam(net.parameters())
    loss_fn = nn.MSELoss(reduction="mean")
    x_t = torch.from_numpy(x_itm).double()
    y_t = torch.from_numpy(y_itm).double().view(-1, 1)
    net.train(True)
    for _ in range(nb_epochs):
        for batch in tdata.BatchSampler(
            tdata.RandomSampler(range(len(x_t)), replacement=False),
            batch_size=batch_size,
            drop_last=False,
        ):
            opt.zero_grad()
            out = net(x_t[batch])
            loss = loss_fn(out, y_t[batch])
            loss.backward()
            opt.step()
    return copy.deepcopy(net.state_dict())


def fit_backward_snapshots(
    algo: AlgoName,
    model,
    payoff_f,
    hidden_size: int = 32,
    nb_epochs_nlsm: int = 30,
    rlsm_factors: Tuple[float, ...] = (1.0,),
    train_itm_only: bool = True,
    train_eval_split: int = 2,
    seed: Optional[int] = 42,
) -> SnapshotBundle:
    import time

    t_path0 = time.time()
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    stock_paths, _var = model.generate_paths()
    path_gen_elapsed = time.time() - t_path0
    if _var is not None and getattr(model, "return_var", False):
        raise NotImplementedError(
            "Дипломный модуль рассчитан на марковский случай без v в регрессорах."
        )
    split = max(1, int(len(stock_paths) / train_eval_split))
    disc_factor = math.exp((-model.rate) * model.maturity / model.nb_dates)
    nb_dates = model.nb_dates

    values = np.asarray(payoff_f.eval(stock_paths[:, :, -1])).ravel()
    snapshots: Dict[int, Any] = {}

    basis = bf.BasisFunctions(model.nb_stocks)
    fac = rlsm_factors if rlsm_factors else (1.0,)
    rlsm_regressor = None
    if algo == "RLSM":
        # Полный tuple `fac` в Reservoir2 (см. RLSM в run_algo: иначе fac[1:] может быть пустым).
        rlsm_regressor = reg.ReservoirLeastSquares2(
            model.nb_stocks,
            hidden_size,
            factors=fac,
            activation=torch.nn.LeakyReLU(fac[0] / 2),
        )

    for date in range(stock_paths.shape[2] - 2, 0, -1):
        immediate = np.asarray(payoff_f.eval(stock_paths[:, :, date])).ravel()
        paths = stock_paths[:, :, date]
        y_next = values * disc_factor

        in_m, in_m_all = _training_masks(immediate, split, train_itm_only)

        if algo == "LSM":
            nb_paths, nb_s = paths.shape
            reg_mat = np.empty((nb_paths, basis.nb_base_fcts))
            for j in range(basis.nb_base_fcts):
                reg_mat[:, j] = basis.base_fct(j, paths[:, :], d2=True)
            coef, *_ = np.linalg.lstsq(
                reg_mat[in_m[0]], y_next[in_m[0]], rcond=None
            )
            cont = np.zeros(nb_paths, dtype=np.float64)
            cont[in_m_all[0]] = reg_mat[in_m_all[0]] @ coef
            snapshots[date] = LSMSnapshot(
                coefficients=coef.astype(np.float64),
                nb_stocks=model.nb_stocks,
            )
        elif algo == "NLSM":
            x_itm = paths[in_m[0]]
            y_itm = y_next[in_m[0]]
            if len(x_itm) < 5:
                cont = y_next.astype(np.float64).copy()
                fb_mean = float(np.mean(y_next)) if len(y_next) else 0.0
                snapshots[date] = NLSMSnapshot(
                    state_dict=None, fallback_mean=fb_mean
                )
            else:
                state_dict = _fit_nlsm_batch(
                    x_itm,
                    y_itm,
                    model.nb_stocks,
                    hidden_size,
                    nb_epochs_nlsm,
                )
                net = neural_networks.NetworkNLSM(model.nb_stocks, hidden_size).double()
                net.load_state_dict(state_dict)
                net.eval()
                with torch.no_grad():
                    px = torch.from_numpy(paths[in_m_all[0]]).double()
                    cont = np.zeros(len(paths), dtype=np.float64)
                    if px.shape[0] > 0:
                        cont[in_m_all[0]] = net(px).numpy().ravel()
                snapshots[date] = NLSMSnapshot(
                    state_dict=state_dict, fallback_mean=None
                )
        elif algo == "RLSM":
            assert rlsm_regressor is not None
            x_np = paths.astype(np.float32)
            x_t = torch.from_numpy(x_np)
            reg_input = np.concatenate(
                [rlsm_regressor.reservoir(x_t).detach().numpy(), np.ones((len(paths), 1))],
                axis=1,
            )
            coef, *_ = np.linalg.lstsq(
                reg_input[in_m[0]], y_next[in_m[0]], rcond=None
            )
            cont = reg_input @ coef
            snapshots[date] = RLSMSnapshot(output_coef=coef.astype(np.float64))
        else:
            raise ValueError(algo)

        rule = _stop_rule(immediate, cont, train_itm_only)
        ex = rule > 0.5
        values = np.where(ex, immediate, values * disc_factor)

    cont0 = float(np.mean(values[split:]) * disc_factor)
    p0 = float(np.asarray(payoff_f.eval(stock_paths[:, :, 0])).ravel()[0])
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
        maturity=model.maturity,
        rate=model.rate,
        split=split,
        stock_paths=stock_paths,
        snapshots=snapshots,
        continuation_scalar_t0=cont0,
        price_terminal_discounted=est_price,
        path_gen_seconds=path_gen_elapsed,
        rlsm_backbone=backbone,
    )
    return bundle
