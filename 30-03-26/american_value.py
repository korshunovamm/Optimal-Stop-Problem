# -*- coding: utf-8 -*-
"""
V(S, t) evaluation and delta computation for multi-dimensional options.
Supports vector S (nb_stocks >= 1) and provides:
  - continuation_value: Q(S, t) from stored snapshots
  - american_value: max(intrinsic, Q)
  - delta_central: finite-difference delta (vector, one per asset)
  - delta_nlsm_autograd: exact gradient via PyTorch autograd (NLSM only)
  - delta_rlsm_analytical: analytical gradient through reservoir (RLSM only)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_PKG = str(Path(__file__).resolve().parent)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from backward_snapshots import (
    LSMSnapshot, NLSMSnapshot, RLSMSnapshot, SnapshotBundle,
)
from optimal_stopping.algorithms.utils import basis_functions as bf
from optimal_stopping.algorithms.utils import neural_networks


def _intrinsic(payoff_f, s_vec):
    # type: (Any, np.ndarray) -> float
    """Intrinsic value for a single state vector s_vec of shape (nb_stocks,)."""
    x = np.atleast_2d(s_vec.astype(np.float64))  # (1, nb_stocks)
    return float(np.asarray(payoff_f.eval(x)).ravel()[0])


def continuation_value(bundle, payoff_f, date_idx, s_vec, hidden_size):
    # type: (SnapshotBundle, Any, int, np.ndarray, int) -> float
    """
    Q(S, t) -- continuation value from stored snapshots.
    s_vec: shape (nb_stocks,)
    """
    if date_idx <= 0:
        return float(bundle.continuation_scalar_t0)
    if date_idx >= bundle.nb_dates:
        return 0.0

    snap = bundle.snapshots.get(date_idx)
    paths = np.atleast_2d(s_vec.astype(np.float64))  # (1, nb_stocks)

    if bundle.algo == "LSM":
        assert isinstance(snap, LSMSnapshot)
        basis = bf.BasisFunctions(snap.nb_stocks)
        reg_mat = np.empty((1, basis.nb_base_fcts))
        for j in range(basis.nb_base_fcts):
            reg_mat[0, j] = basis.base_fct(j, paths[:, :], d2=True)
        return float(reg_mat[0] @ snap.coefficients)

    if bundle.algo == "RLSM":
        assert isinstance(snap, RLSMSnapshot)
        from optimal_stopping.algorithms.backward_induction import regression as reg_mod

        bb = bundle.rlsm_backbone
        if bb is None:
            raise RuntimeError("RLSM backbone missing")
        fac = bb.factors
        rlsm = reg_mod.ReservoirLeastSquares2(
            paths.shape[1], bb.hidden_size,
            factors=fac,
            activation=torch.nn.LeakyReLU(fac[0] / 2),
        )
        rlsm.reservoir.load_state_dict(bb.reservoir_state)
        rlsm.reservoir.eval()
        x_t = torch.from_numpy(paths.astype(np.float32))
        reg_input = np.concatenate(
            [rlsm.reservoir(x_t).detach().numpy(), np.ones((1, 1))], axis=1)
        return float(reg_input @ snap.output_coef)

    if bundle.algo == "NLSM":
        if snap is None:
            return 0.0
        assert isinstance(snap, NLSMSnapshot)
        if snap.fallback_mean is not None:
            return float(snap.fallback_mean)
        if snap.state_dict is None:
            return 0.0
        nb_s = paths.shape[1]
        net = neural_networks.NetworkNLSM(nb_s, hidden_size=hidden_size).double()
        net.load_state_dict(snap.state_dict)
        net.eval()
        with torch.no_grad():
            p = torch.from_numpy(paths).double()
            return float(net(p).numpy().ravel()[0])

    raise ValueError(bundle.algo)


def american_value(bundle, payoff_f, date_idx, s_vec, hidden_size):
    # type: (SnapshotBundle, Any, int, np.ndarray, int) -> float
    """max(intrinsic, continuation) for state s_vec at date_idx."""
    ex = _intrinsic(payoff_f, s_vec)
    q = continuation_value(bundle, payoff_f, date_idx, s_vec, hidden_size)
    if date_idx >= bundle.nb_dates:
        return ex
    return max(ex, q)


def delta_central(bundle, payoff_f, date_idx, s_vec, hidden_size, bump):
    # type: (SnapshotBundle, Any, int, np.ndarray, int, float) -> np.ndarray
    """
    Finite-difference delta (central) for each asset.
    Returns shape (nb_stocks,).
    """
    s = s_vec.astype(np.float64).ravel()
    d = len(s)
    delta = np.zeros(d, dtype=np.float64)
    for k in range(d):
        s_up = s.copy()
        s_down = s.copy()
        s_up[k] += bump
        s_down[k] -= bump
        vp = american_value(bundle, payoff_f, date_idx, s_up, hidden_size)
        vm = american_value(bundle, payoff_f, date_idx, s_down, hidden_size)
        delta[k] = (vp - vm) / (2.0 * bump)
    return delta


def delta_nlsm_autograd(bundle, payoff_f, date_idx, s_vec, hidden_size):
    # type: (SnapshotBundle, Any, int, np.ndarray, int) -> Optional[np.ndarray]
    """
    Exact gradient via autograd for NLSM continuation network.
    Returns shape (nb_stocks,) or None if not applicable.
    In exercise region, returns intrinsic gradient instead.
    """
    if bundle.algo != "NLSM":
        return None
    if date_idx <= 0 or date_idx >= bundle.nb_dates:
        return None
    snap = bundle.snapshots.get(date_idx)
    if snap is None or not isinstance(snap, NLSMSnapshot):
        return None
    if snap.state_dict is None:
        return None

    s = s_vec.astype(np.float64).ravel()
    nb_s = len(s)
    net = neural_networks.NetworkNLSM(nb_s, hidden_size=hidden_size).double()
    net.load_state_dict(snap.state_dict)
    net.eval()

    x = torch.tensor(s.reshape(1, -1), dtype=torch.double, requires_grad=True)
    q = net(x).squeeze()
    grad = torch.autograd.grad(q, x, create_graph=False)[0]
    d_q = grad[0].detach().numpy().copy()  # (nb_stocks,)

    q_val = float(q.item())
    ex_val = _intrinsic(payoff_f, s)

    if ex_val > q_val:
        return _intrinsic_gradient(payoff_f, s)
    return d_q


def delta_rlsm_analytical(bundle, payoff_f, date_idx, s_vec, hidden_size):
    # type: (SnapshotBundle, Any, int, np.ndarray, int) -> Optional[np.ndarray]
    """
    Analytical gradient through reservoir for RLSM.
    Uses autograd through the fixed reservoir network.
    Returns shape (nb_stocks,) or None.
    """
    if bundle.algo != "RLSM":
        return None
    if date_idx <= 0 or date_idx >= bundle.nb_dates:
        return None
    snap = bundle.snapshots.get(date_idx)
    if snap is None or not isinstance(snap, RLSMSnapshot):
        return None

    bb = bundle.rlsm_backbone
    if bb is None:
        return None

    from optimal_stopping.algorithms.backward_induction import regression as reg_mod

    s = s_vec.astype(np.float64).ravel()
    fac = bb.factors
    rlsm = reg_mod.ReservoirLeastSquares2(
        len(s), bb.hidden_size,
        factors=fac, activation=torch.nn.LeakyReLU(fac[0] / 2),
    )
    rlsm.reservoir.load_state_dict(bb.reservoir_state)
    rlsm.reservoir.eval()

    x = torch.tensor(s.reshape(1, -1).astype(np.float32), requires_grad=True)
    features = rlsm.reservoir(x)  # (1, hidden_size)
    features_with_bias = torch.cat(
        [features, torch.ones(1, 1)], dim=1)
    coef_t = torch.tensor(snap.output_coef, dtype=torch.float32)
    q = (features_with_bias @ coef_t.unsqueeze(1)).squeeze()
    grad = torch.autograd.grad(q, x, create_graph=False)[0]
    d_q = grad[0].detach().numpy().copy().astype(np.float64)

    q_val = float(q.item())
    ex_val = _intrinsic(payoff_f, s)
    if ex_val > q_val:
        return _intrinsic_gradient(payoff_f, s)
    return d_q


def _intrinsic_gradient(payoff_f, s_vec, eps=0.01):
    # type: (Any, np.ndarray, float) -> np.ndarray
    """Numerical gradient of intrinsic value w.r.t. each asset."""
    s = s_vec.ravel().astype(np.float64)
    d = len(s)
    grad = np.zeros(d, dtype=np.float64)
    for k in range(d):
        s_up = s.copy()
        s_down = s.copy()
        s_up[k] += eps
        s_down[k] -= eps
        grad[k] = (_intrinsic(payoff_f, s_up) - _intrinsic(payoff_f, s_down)) / (2.0 * eps)
    return grad


def get_delta(bundle, payoff_f, date_idx, s_vec, hidden_size, bump=0.5):
    # type: (SnapshotBundle, Any, int, np.ndarray, int, float) -> np.ndarray
    """
    Unified delta function: uses autograd for NLSM, analytical for RLSM,
    falls back to finite-difference for LSM or when autograd is unavailable.
    Returns shape (nb_stocks,).
    """
    if bundle.algo == "NLSM":
        d = delta_nlsm_autograd(bundle, payoff_f, date_idx, s_vec, hidden_size)
        if d is not None:
            return d
    if bundle.algo == "RLSM":
        d = delta_rlsm_analytical(bundle, payoff_f, date_idx, s_vec, hidden_size)
        if d is not None:
            return d
    return delta_central(bundle, payoff_f, date_idx, s_vec, hidden_size, bump)
