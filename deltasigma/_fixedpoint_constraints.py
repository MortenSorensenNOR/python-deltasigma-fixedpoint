# -*- coding: utf-8 -*-
# _fixedpoint_constraints.py
# Coefficient form constraints (power-of-2, CSD) for the fixed-point
# simulation backend.
# This file is part of python-deltasigma.

"""Snap coefficient values onto hardware-implementable forms.

A real FPGA / ASIC implementation often cannot freely choose every coefficient
value: an interstage gain that is a power of two costs only a hard-wired
shift, an arbitrary value costs a full multiplier. The functions in this
module map a desired float coefficient onto the nearest value reachable by:

* a single signed power of two -- ``"po2"`` -- or
* a sum of at most N signed powers of two with distinct exponents --
  ``"csd:N"`` (greedy approximation to a canonical signed digit form).

The snapping is done with respect to a target :class:`QFormat`, so the result
is always representable in the user's chosen wordlength.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ._fixedpoint_config import FixedPointConfig, QFormat


def _qfmt_bounds(qfmt: QFormat) -> tuple[float, float]:
    """Min and max representable values of ``qfmt``."""
    if qfmt.signed:
        max_val = 2.0 ** (qfmt.m - 1) - 2.0 ** (-qfmt.n)
        min_val = -(2.0 ** (qfmt.m - 1))
    else:
        max_val = 2.0 ** qfmt.m - 2.0 ** (-qfmt.n)
        min_val = 0.0
    return min_val, max_val


def _k_max_for_sign(qfmt: QFormat, sign: int) -> int:
    """Largest exponent ``k`` such that ``sign * 2**k`` fits in ``qfmt``."""
    if qfmt.signed:
        # Signed range is [-2**(m-1), 2**(m-1) - 2**-n]: the negative endpoint
        # is achievable as -2**(m-1), but the positive max +2**(m-1) is not.
        return qfmt.m - 1 if sign < 0 else qfmt.m - 2
    # Unsigned [0, 2**m - 2**-n]; max po2 is 2**(m-1).
    return qfmt.m - 1


def _snap_po2(value: float, qfmt: QFormat) -> float:
    """Nearest signed power of two representable in ``qfmt``."""
    if value == 0.0:
        return 0.0
    if not qfmt.signed and value < 0.0:
        return 0.0
    sign = 1 if value > 0 else -1
    k = int(round(math.log2(abs(value))))
    k_min = -qfmt.n
    k_max = _k_max_for_sign(qfmt, sign)
    if k_max < k_min:
        return 0.0
    k = max(k_min, min(k_max, k))
    return float(sign) * 2.0 ** k


def _snap_csd(value: float, qfmt: QFormat, n_terms: int) -> float:
    """Greedy approximation to the closest sum of <= ``n_terms`` signed
    powers of two (distinct exponents) representable in ``qfmt``."""
    if value == 0.0 or n_terms <= 0:
        return 0.0
    min_val, max_val = _qfmt_bounds(qfmt)
    used: set[int] = set()
    accumulated = 0.0
    residual = float(value)
    eps = 2.0 ** (-qfmt.n - 1)  # half-LSB: stop when residual is smaller
    for _ in range(n_terms):
        if abs(residual) < eps:
            break
        sign = 1 if residual > 0 else -1
        if not qfmt.signed and sign < 0:
            break
        k_target = int(round(math.log2(abs(residual))))
        k_min = -qfmt.n
        k_max = _k_max_for_sign(qfmt, sign)
        # Find the largest unused k <= min(k_target, k_max) that's >= k_min.
        k = min(k_target, k_max)
        while k >= k_min and k in used:
            k -= 1
        if k < k_min:
            break
        term = float(sign) * 2.0 ** k
        new_acc = accumulated + term
        if new_acc > max_val or new_acc < min_val:
            break
        accumulated = new_acc
        used.add(k)
        residual = value - accumulated
    return accumulated


def _snap_to_constraint(value: float, constraint: Optional[str],
                        qfmt: QFormat) -> float:
    """Apply a ``po2`` / ``csd:N`` constraint to ``value`` for ``qfmt``."""
    if constraint is None:
        return float(value)
    if constraint == "po2":
        return _snap_po2(float(value), qfmt)
    if constraint.startswith("csd:"):
        n_terms = int(constraint.split(":", 1)[1])
        return _snap_csd(float(value), qfmt, n_terms)
    raise ValueError(f"unknown coefficient constraint: {constraint!r}")


def constrain_coefficients(ABCD: np.ndarray, fixedpoint: FixedPointConfig,
                           nlev: int = 2):
    """Return the post-constraint, post-quantization ABCD matrices.

    Run the coefficient pipeline that the FP simulation will apply (per-entry
    constraint snap + Q-format rounding) and hand back the actual float
    values the simulator would use. Useful for sanity-checking what your HDL
    has to encode.

    Parameters
    ----------
    ABCD : ndarray
        The float ABCD matrix as passed to ``simulateDSM``.
    fixedpoint : FixedPointConfig
        The configuration whose ``coeff`` Q-format and constraint hooks
        should be applied.
    nlev : int, optional
        Quantizer level count. Determines how ABCD splits into A/B/C/D.

    Returns
    -------
    A, B, C, D : ndarray
        Float64 sub-matrices reflecting (a) per-entry constraint snapping and
        (b) Q-format rounding, exactly as the simulation would see them.
    """
    from fixedpoint import FixedPoint  # local: optional dep

    ABCD = np.asarray(ABCD, dtype=np.float64)
    nq = 1 if np.isscalar(nlev) else int(np.asarray(nlev).shape[0])
    nu = ABCD.shape[1] - ABCD.shape[0]
    order = ABCD.shape[0] - nq
    A = ABCD[:order, :order]
    B = ABCD[:order, order:order + nu + nq]
    C = ABCD[order:order + nq, :order]
    D = ABCD[order:order + nq, order:order + nu]

    qfmt = fixedpoint.coeff

    def _do(M, name):
        out = np.zeros_like(M, dtype=np.float64)
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                constraint = fixedpoint.constraint_for(name, r, c)
                snapped = _snap_to_constraint(float(M[r, c]), constraint, qfmt)
                fp = FixedPoint(
                    snapped, signed=qfmt.signed, m=qfmt.m, n=qfmt.n,
                    overflow=qfmt.overflow, rounding=qfmt.rounding,
                    overflow_alert=qfmt.overflow_alert,
                    mismatch_alert="ignore", implicit_cast_alert="ignore",
                )
                out[r, c] = float(fp)
        return out

    return _do(A, "A"), _do(B, "B"), _do(C, "C"), _do(D, "D")
