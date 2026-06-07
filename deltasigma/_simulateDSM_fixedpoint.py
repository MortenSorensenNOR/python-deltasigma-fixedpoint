# -*- coding: utf-8 -*-
# _simulateDSM_fixedpoint.py
# Fixed-point simulation backend for delta-sigma modulators.
# This file is part of python-deltasigma.
#
# Copyright (c) 2026, Morten Sørensen
# SPDX-License-Identifier: BSD-2-Clause
#
# python-deltasigma is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# LICENSE file for the licensing terms.

"""A pure-Python simulation backend that performs the modulator inner-loop
arithmetic in user-configured fixed-point Q-format.

The implementation mirrors :mod:`deltasigma._simulateDSM_python` but replaces
the numpy ``float64`` state and coefficient values with
``fixedpoint.FixedPoint`` scalars. The quantizer itself is reused from
:func:`deltasigma._ds_quantize.ds_quantize` on the float view of the
accumulator output: the quantized result is an integer in
``[-(nlev-1), nlev-1]`` which is then re-wrapped into the coefficient Q-format
for the feedback path.

Because ``fixedpoint`` is a pure-Python scalar library, this backend is
typically 10x-100x slower than the float Cython backends. It is intended for
*validating* an implementation choice (which wordlength still gives N dB of
SNR?), not for high-throughput Monte-Carlo sweeps.
"""

from __future__ import annotations

import collections
from warnings import warn

import numpy as np
from scipy.linalg import inv, norm, orth
from scipy.signal import zpk2ss

from ._ds_quantize import ds_quantize as _ds_quantize_float
from ._fixedpoint_config import FixedPointConfig, to_fp
from ._fixedpoint_constraints import _snap_to_constraint
from ._utils import _get_zpk, carray


def _coeff_matrix_to_fp(M, qfmt, matrix_name=None, cfg=None):
    """Convert a 2D numpy coefficient matrix to a nested list of FixedPoints,
    applying per-entry hardware-form constraints from ``cfg`` if given."""
    rows = []
    for r in range(M.shape[0]):
        row = []
        for c in range(M.shape[1]):
            value = float(M[r, c])
            if cfg is not None and matrix_name is not None:
                constraint = cfg.constraint_for(matrix_name, r, c)
                if constraint is not None:
                    value = _snap_to_constraint(value, constraint, qfmt)
            row.append(to_fp(value, qfmt))
        rows.append(row)
    return rows


def _matvec_fp(M_fp, x_fp, target_qfmt):
    """Compute ``M @ x`` with FixedPoint scalars; resize each result row to
    ``target_qfmt``. ``M_fp`` is a list-of-lists, ``x_fp`` a list."""
    nrows = len(M_fp)
    ncols = len(M_fp[0]) if nrows else 0
    out = []
    for r in range(nrows):
        # Start the accumulator at zero in the target Q-format. We use the
        # target format so that the very first product, when added to zero, is
        # rounded/clamped to that format right away.
        acc = to_fp(0.0, target_qfmt)
        for c in range(ncols):
            term = M_fp[r][c] * x_fp[c]
            acc = acc + term
            acc.resize(target_qfmt.m, target_qfmt.n)
        out.append(acc)
    return out


def simulateDSM(u, arg2, nlev=2, x0=0., fixedpoint=None):
    """Simulate a delta-sigma modulator with a fixed-point datapath.

    Parameters
    ----------
    u : ndarray or sequence
        Input signal. Same shape conventions as the float :func:`simulateDSM`.
    arg2 : ndarray or LTI description
        ABCD matrix or NTF (zpk). Identical interpretation to the float
        backend.
    nlev : int or sequence
        Quantizer level count.
    x0 : float or sequence
        Initial state.
    fixedpoint : FixedPointConfig
        Required. Describes the Q-format used for state, coefficients, input
        and accumulator.

    Returns
    -------
    v, xn, xmax, y : tuple of ndarray
        Same shapes and meanings as the float backend. All returned arrays are
        ``float64`` for downstream compatibility (FFTs, SNR calculation, etc.).
    """
    if fixedpoint is None:
        raise ValueError(
            "simulateDSM_fixedpoint requires a FixedPointConfig in "
            "`fixedpoint=`; pass one or use the float backend instead."
        )
    cfg: FixedPointConfig = fixedpoint
    out_qfmt = cfg.y_or_state

    # --- argument normalisation: identical to _simulateDSM_python ---
    nlev = carray(nlev)
    u = np.array(u) if not hasattr(u, "ndim") else u
    if not max(u.shape) == np.prod(u.shape):
        warn("Multiple input delta sigma structures have had little testing.")
    if u.ndim == 1:
        u = u.reshape((1, -1))
    nu = u.shape[0]
    nq = 1 if np.isscalar(nlev) else nlev.shape[0]

    if (hasattr(arg2, "inputs") and not arg2.inputs == 1) or \
       (hasattr(arg2, "outputs") and not arg2.outputs == 1):
        raise TypeError("The supplied TF isn't a SISO transfer function.")

    if isinstance(arg2, np.ndarray):
        ABCD = np.asarray(arg2, dtype=np.float64)
        if ABCD.shape[1] != ABCD.shape[0] + nu:
            raise ValueError("The ABCD argument does not have proper dimensions.")
        form = 1
    else:
        zeros, poles, _k = _get_zpk(arg2)
        form = 2

    order = carray(zeros).shape[0] if form == 2 else ABCD.shape[0] - nq

    if not isinstance(x0, collections.abc.Iterable):
        x0 = x0 * np.ones((order,), dtype=np.float64)
    else:
        x0 = np.array(x0).reshape((-1,))

    if form == 1:
        A = ABCD[:order, :order]
        B = ABCD[:order, order:order + nu + nq]
        C = ABCD[order:order + nq, :order]
        D1 = ABCD[order:order + nq, order:order + nu]
    else:
        A, B2, C, D2 = zpk2ss(poles, zeros, -1)
        C, D2 = np.real_if_close(C), np.real_if_close(D2)
        Sinv = orth(np.hstack((np.transpose(C), np.eye(order)))) / norm(C)
        S = inv(Sinv)
        C = np.dot(C, Sinv)
        if C[0, 0] < 0:
            S = -S
            Sinv = -Sinv
        A = np.dot(np.dot(S, A), Sinv)
        B2 = np.dot(S, B2)
        C = np.hstack((np.ones((1, 1)), np.zeros((1, order - 1))))
        D2 = np.zeros((0,))
        B1 = -B2
        D1 = np.ones((1, 1))
        B = np.hstack((B1, B2))

    N = np.max(np.shape(u))

    # --- fixed-point conversion of coefficients and initial state ---
    A_fp = _coeff_matrix_to_fp(np.real(A), cfg.coeff, "A", cfg)
    B_fp = _coeff_matrix_to_fp(np.real(B), cfg.coeff, "B", cfg)
    C_fp = _coeff_matrix_to_fp(np.real(C), cfg.coeff, "C", cfg)
    D1_fp = _coeff_matrix_to_fp(np.atleast_2d(np.real(D1)),
                                cfg.coeff, "D", cfg)

    x_fp = [to_fp(v, cfg.state) for v in x0]

    # --- output buffers (returned as float so downstream code is unchanged) ---
    v = np.zeros((nq, N), dtype=np.float64)
    y = np.zeros((nq, N), dtype=np.float64)
    xn = np.zeros((order, N), dtype=np.float64)
    xmax = np.abs(x0).reshape((-1, 1))

    nlev_arr = nlev if isinstance(nlev, np.ndarray) else np.array([nlev])

    for i in range(N):
        # --- input column to FP ---
        u_fp = [to_fp(u[r, i], cfg.input) for r in range(nu)]

        # --- y = C*x + D1*u, in output Q-format ---
        Cx_fp = _matvec_fp(C_fp, x_fp, out_qfmt)
        D1u_fp = _matvec_fp(D1_fp, u_fp, out_qfmt)
        y_fp = []
        for r in range(nq):
            s = Cx_fp[r] + D1u_fp[r]
            s.resize(out_qfmt.m, out_qfmt.n)
            y_fp.append(s)

        y0_float = np.array([float(s) for s in y_fp])
        y[:, i] = y0_float

        # --- quantizer: on float view, result is small integer ---
        v_float = _ds_quantize_float(y0_float.reshape(-1, 1), nlev_arr).reshape(-1)
        v[:, i] = v_float

        # --- feedback v in coeff Q-format, concatenate with u ---
        v_fp = [to_fp(val, cfg.coeff) for val in v_float]
        uv_fp = u_fp + v_fp  # shape (nu + nq,)

        # --- x_next = A*x + B*[u;v], in state Q-format ---
        Ax_fp = _matvec_fp(A_fp, x_fp, cfg.state)
        Buv_fp = _matvec_fp(B_fp, uv_fp, cfg.state)
        x_next = []
        for r in range(order):
            s = Ax_fp[r] + Buv_fp[r]
            s.resize(cfg.state.m, cfg.state.n)
            x_next.append(s)
        x_fp = x_next

        x_float = np.array([float(s) for s in x_fp])
        xn[:, i] = x_float
        xmax = np.max(
            np.hstack((np.abs(x_float).reshape((-1, 1)), xmax)),
            axis=1, keepdims=True,
        )

    return v.squeeze(), xn.squeeze(), xmax, y.squeeze()
