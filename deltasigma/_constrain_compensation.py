# -*- coding: utf-8 -*-
# _constrain_compensation.py
# Coefficient compensation: snap topology c_i values to a hardware-friendly
# form (po2 / CSD) while absorbing the change into a_i, b_i, c_(i+1), and
# resonator g_j so the NTF is preserved.
# This file is part of python-deltasigma.

"""Compensate for hardware-form constraints on integrator interstage gains.

When a CRFB DSM is mapped to silicon, the interstage gains ``c_i`` are
attractive candidates for power-of-two / CSD restriction because each one
costs only a shifter (and maybe a tiny adder). A *naive* snap, however, just
moves the NTF zeros and destroys the noise-shaping the user designed for.

The trick is that the snap is exactly a state-scaling of the loop filter:
forcing ``c_i -> c_i_hat`` is equivalent to multiplying state ``x_(i+1)`` by
``alpha = c_i_hat / c_i``. State-scaling is a similarity transform on
``(A, B, C, D)``, so it preserves the input-output transfer functions (NTF
and STF) **exactly**. The cost shows up as scaled values of ``a_(i+1)``,
``b_(i+1)``, ``c_(i+1)``, and any resonator ``g_j`` touching the scaled
state -- those become whatever they need to be (general fixed-point, not
hardware-cheap), which is the usual trade in real DSM design.

CRFB and CRFF are both supported. The two differ in *which* ``c_i`` are
interstage gains:

* CRFB -- ``c[0..n-2]`` are interstage; ``c[n-1]`` is the output gain into
  ``y``. Compensation snaps the interstage ones.
* CRFF -- ``c[1..n-1]`` are interstage; ``c[0]`` is the feedback DAC gain
  (it plays the role ``a[0]`` does in CRFB). Compensation snaps the
  interstage ones.

Both cases derive a vector of state-scaling factors that yields the desired
``c_i`` snaps and then propagate the inverse scalings to the remaining
topology coefficients. Other topologies (CIFB, CIFF, Stratos, ...) need
their own derivation; calling this function with one raises
``NotImplementedError``.
"""

from __future__ import annotations

from typing import Tuple
from warnings import warn

import numpy as np

from ._fixedpoint_config import QFormat
from ._fixedpoint_constraints import _snap_to_constraint


def constrain_with_compensation(
    a, g, b, c, form: str, qfmt: QFormat,
    constraint: str = "po2",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Snap interstage gains to ``constraint`` and compensate via state scaling.

    Parameters
    ----------
    a, g, b, c : array_like
        Topology coefficients as returned by :func:`realizeNTF`. Sizes
        follow the package convention: ``a`` length ``n``, ``g`` length
        ``floor(n/2)``, ``b`` length ``n + 1``, ``c`` length ``n``, where
        ``n`` is the loop order.
    form : str
        Modulator topology. ``"CRFB"`` or ``"CRFF"``.
    qfmt : QFormat
        Coefficient Q-format -- determines the representable range used by
        the po2 / CSD snap. Typically pass ``FixedPointConfig.coeff`` here.
    constraint : str
        ``"po2"`` (default) or ``"csd:N"``. Applied to the interstage
        gains only -- ``c[0..n-2]`` for CRFB, ``c[1..n-1]`` for CRFF. The
        remaining ``c`` (output gain for CRFB, feedback DAC for CRFF) is
        left unconstrained: forcing it has different physical meaning that
        state-scaling cannot compensate for in a single similarity
        transform.

    Returns
    -------
    a_new, g_new, b_new, c_new : ndarray
        Float64 topology coefficients. ``c_new[0..n-2]`` lie exactly on the
        constraint grid; the other coefficients have absorbed the inverse
        scalings. Feed these to :func:`stuffABCD` to get an ABCD matrix that,
        once simulated in fixed-point, leaves the NTF zeros where you
        designed them.

    Notes
    -----
    The compensation is exact in real arithmetic. Once the resulting
    coefficients are themselves Q-format-rounded by the simulation, sub-LSB
    drift on ``a_i`` / ``b_i`` / ``g_j`` causes a residual NTF zero
    perturbation -- that residual is what the fixed-point simulation will
    legitimately reveal, in contrast to the gross NTF damage you would see
    from naive (uncompensated) c-snapping.
    """
    if form not in ("CRFB", "CRFF"):
        raise NotImplementedError(
            f"constrain_with_compensation supports form='CRFB' and 'CRFF'; "
            f"got form={form!r}."
        )

    a = np.asarray(a, dtype=np.float64).reshape(-1).copy()
    g = np.asarray(g, dtype=np.float64).reshape(-1).copy()
    b = np.asarray(b, dtype=np.float64).reshape(-1).copy()
    c = np.asarray(c, dtype=np.float64).reshape(-1).copy()

    n = a.shape[0]
    if c.shape[0] != n:
        raise ValueError(f"len(c)={c.shape[0]} must equal len(a)={n}")
    if b.shape[0] != n + 1:
        raise ValueError(f"len(b)={b.shape[0]} must equal len(a)+1={n+1}")

    if form == "CRFB":
        return _compensate_CRFB(a, g, b, c, qfmt, constraint, n)
    return _compensate_CRFF(a, g, b, c, qfmt, constraint, n)


def _compensate_CRFB(a, g, b, c, qfmt, constraint, n):
    # Derive state-scaling factors s[0..n-1] that achieve the desired c snap.
    # In CRFB, c[i] is the gain from x_i to x_(i+1); state scaling x' = s*x
    # gives c'_i = (s_(i+1)/s_i) * c_i. For c'_i = snap(c_i):
    #     s_(i+1) = s_i * (snap(c_i) / c_i).
    # We constrain c[0..n-2] (the interstage gains); c[n-1] is the output gain.
    s = np.ones(n, dtype=np.float64)
    for i in range(n - 1):
        if c[i] == 0.0:
            continue
        c_target = _snap_to_constraint(float(c[i]), constraint, qfmt)
        if c_target == 0.0:
            warn(f"constrain_with_compensation: snap of c[{i}]={c[i]} gave 0; "
                 f"leaving this c unconstrained.")
            continue
        s[i + 1] = s[i] * (c_target / c[i])

    # Apply the similarity transform x' = diag(s) * x:
    #   c'_(i)     = (s_(i+1)/s_i) * c_i   for i = 0..n-2  (interstage)
    #   c'_(n-1)   = c_(n-1) / s_(n-1)     (output gain)
    #   a'_i       = s_i * a_i             (feedback DAC into integrator i)
    #   b'_i       = s_i * b_i             for i = 0..n-1
    #   b'_n       = b_n                   (direct feed to y)
    #   g'_j       = (s_row/s_col) * g_j   resonator pair (row, col)
    c_new = c.copy()
    for i in range(n - 1):
        c_new[i] = (s[i + 1] / s[i]) * c[i]
    c_new[n - 1] = c[n - 1] / s[n - 1]

    a_new = s * a
    b_new = b.copy()
    b_new[:n] = s * b[:n]

    g_new = _scale_g_pairs(g, s, n)
    return a_new, g_new, b_new, c_new


def _compensate_CRFF(a, g, b, c, qfmt, constraint, n):
    # Derive state-scaling factors. In CRFF, c[0] is the feedback DAC gain
    # (B2 = -c[0], feedback only enters the first integrator). c[1..n-1] are
    # interstage gains -- c[i] gains from x_(i-1) to x_i -- so state scaling
    # x' = s*x gives c'_i = (s_i / s_(i-1)) * c_i. For c'_i = snap(c_i):
    #     s_i = s_(i-1) * (snap(c_i) / c_i).
    # We constrain c[1..n-1]; c[0] is left alone (it sets the quantizer-side
    # loop gain, which is not what state-scaling compensates for).
    s = np.ones(n, dtype=np.float64)
    for i in range(1, n):
        if c[i] == 0.0:
            continue
        c_target = _snap_to_constraint(float(c[i]), constraint, qfmt)
        if c_target == 0.0:
            warn(f"constrain_with_compensation: snap of c[{i}]={c[i]} gave 0; "
                 f"leaving this c unconstrained.")
            continue
        s[i] = s[i - 1] * (c_target / c[i])

    # Apply the similarity transform x' = diag(s) * x. In CRFF:
    #   - y = sum(a_i * x_i)  =>  a'_i = a_i / s_i  (forward-summing weights)
    #   - x_i[n+1] += c_i * x_(i-1)  =>  c'_i = (s_i / s_(i-1)) * c_i
    #   - x_0[n+1] += -c[0] * v       =>  c'[0] = s_0 * c[0] = c[0] (s_0 == 1)
    #   - b_i feed into integrator i  =>  b'_i = s_i * b_i for i = 0..n-1
    #   - b_n goes straight to y      =>  b'_n unchanged
    #   - g_j as in CRFB
    c_new = c.copy()
    c_new[0] = s[0] * c[0]
    for i in range(1, n):
        c_new[i] = (s[i] / s[i - 1]) * c[i]

    a_new = a / s
    b_new = b.copy()
    b_new[:n] = s * b[:n]

    g_new = _scale_g_pairs(g, s, n)
    return a_new, g_new, b_new, c_new


def _scale_g_pairs(g, s, n):
    # Resonator g_j sits on the superdiagonal of A at row (odd + 2j),
    # col (odd + 2j + 1), where odd = n % 2. Verified by inspection of
    # stuffABCD's CRFB and CRFF branches (the CRFF even-order multg path
    # produces the same A[row, col] = -g_j after row reduction).
    odd = n % 2
    g_new = g.copy()
    for j in range(g.shape[0]):
        row = odd + 2 * j
        col = row + 1
        if col < n:
            g_new[j] = (s[row] / s[col]) * g[j]
    return g_new
