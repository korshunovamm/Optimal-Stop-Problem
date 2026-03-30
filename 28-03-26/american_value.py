# -*- coding: utf-8 -*-
"""Оценка V(S, t) и дельты по снимкам обратной индукции (блок D)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from backward_snapshots import (
    LSMSnapshot,
    NLSMSnapshot,
    RLSMSnapshot,
    SnapshotBundle,
)
from optimal_stopping.algorithms.utils import basis_functions as bf
from optimal_stopping.algorithms.utils import neural_networks


def _intrinsic(payoff_f, s: np.ndarray) -> float:
    x = np.atleast_2d(s.astype(np.float64))
    return float(np.asarray(payoff_f.eval(x)).ravel()[0])


def continuation_value(
    bundle: SnapshotBundle,
    payoff_f,
    date_idx: int,
    s: np.ndarray,
    hidden_size: int,
) -> float:
    """
    date_idx: 0 .. nb_dates (совпадает с осью времени в stock_paths[:,:,date_idx]).
    Возвращает Q (оценка условного ожидания при продолжении), без max с intrinsic.
    """
    if date_idx <= 0:
        return float(bundle.continuation_scalar_t0)
    if date_idx >= bundle.nb_dates:
        return 0.0

    snap = bundle.snapshots.get(date_idx)
    paths = np.atleast_2d(s.astype(np.float64))

    if bundle.algo == "LSM":
        assert isinstance(snap, LSMSnapshot)
        basis = bf.BasisFunctions(snap.nb_stocks)
        reg_mat = np.empty((1, basis.nb_base_fcts))
        for j in range(basis.nb_base_fcts):
            reg_mat[0, j] = basis.base_fct(j, paths[:, :], d2=True)
        return float(reg_mat[0] @ snap.coefficients)

    if bundle.algo == "RLSM":
        assert isinstance(snap, RLSMSnapshot)
        from optimal_stopping.algorithms.backward_induction import regression as reg

        bb = bundle.rlsm_backbone
        if bb is None:
            raise RuntimeError("RLSM backbone missing in bundle")
        fac = bb.factors
        rlsm = reg.ReservoirLeastSquares2(
            paths.shape[1],
            bb.hidden_size,
            factors=fac,
            activation=torch.nn.LeakyReLU(fac[0] / 2),
        )
        rlsm.reservoir.load_state_dict(bb.reservoir_state)
        rlsm.reservoir.eval()
        x_t = torch.from_numpy(paths.astype(np.float32))
        reg_input = np.concatenate(
            [rlsm.reservoir(x_t).detach().numpy(), np.ones((1, 1))], axis=1
        )
        return float(reg_input @ snap.output_coef)

    if bundle.algo == "NLSM":
        if snap is None:
            return 0.0
        assert isinstance(snap, NLSMSnapshot)
        if snap.fallback_mean is not None:
            return float(snap.fallback_mean)
        if snap.state_dict is None:
            return 0.0
        net = neural_networks.NetworkNLSM(paths.shape[1], hidden_size=hidden_size).double()
        net.load_state_dict(snap.state_dict)
        net.eval()
        with torch.no_grad():
            p = torch.from_numpy(paths).double()
            return float(net(p).numpy().ravel()[0])

    raise ValueError(bundle.algo)


def american_value(
    bundle: SnapshotBundle,
    payoff_f,
    date_idx: int,
    s: np.ndarray,
    hidden_size: int,
) -> float:
    ex = _intrinsic(payoff_f, s)
    q = continuation_value(bundle, payoff_f, date_idx, s, hidden_size)
    if date_idx >= bundle.nb_dates:
        return ex
    return max(ex, q)


def delta_central(
    bundle: SnapshotBundle,
    payoff_f,
    date_idx: int,
    s_scalar: float,
    hidden_size: int,
    bump: float,
) -> float:
    """∂V/∂S (численно), единый протокол для LSM / NLSM / RLSM."""
    s0 = float(s_scalar)
    vp = american_value(
        bundle, payoff_f, date_idx, np.array([[s0 + bump]]), hidden_size
    )
    vm = american_value(
        bundle, payoff_f, date_idx, np.array([[s0 - bump]]), hidden_size
    )
    return (vp - vm) / (2.0 * bump)


def delta_nlsm_autograd(
    bundle: SnapshotBundle,
    payoff_f,
    date_idx: int,
    s_scalar: float,
    hidden_size: int,
) -> Optional[float]:
    """Дельта от сети продолжения (только NLSM и только если date_idx в 1..n-1)."""
    if bundle.algo != "NLSM":
        return None
    if date_idx <= 0 or date_idx >= bundle.nb_dates:
        return None
    snap = bundle.snapshots.get(date_idx)
    if snap is None or not isinstance(snap, NLSMSnapshot):
        return None
    if snap.state_dict is None:
        return None
    net = neural_networks.NetworkNLSM(1, hidden_size=hidden_size).double()
    net.load_state_dict(snap.state_dict)
    net.train(False)
    x = torch.tensor([[float(s_scalar)]], dtype=torch.double, requires_grad=True)
    q = net(x).squeeze()
    q.backward()
    d_q = float(x.grad[0, 0].item())
    ex = _intrinsic(payoff_f, np.array([[s_scalar]]))
    q0 = float(
        torch.detach(net(torch.tensor([[float(s_scalar)]], dtype=torch.double))).item()
    )
    if ex > q0:
        # exercise region — дельта по intrinsic
        return -1.0 if s_scalar < payoff_f.strike else 0.0
    return d_q
