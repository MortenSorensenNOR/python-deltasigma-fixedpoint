# -*- coding: utf-8 -*-
# _constrain_compensation.py
# Coefficient compensation: snap topology c_i values to a hardware-friendly
# form (po2 / CSD) while absorbing the change into a_i, b_i, c_(i+1), and
# resonator g_j so the NTF is preserved.
# This file is part of python-deltasigma.
#
# Copyright (c) 2026, Morten Sørensen
# SPDX-License-Identifier: BSD-2-Clause
#
# python-deltasigma is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# LICENSE file for the licensing terms.

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

from typing import Optional, Tuple
from warnings import warn

import numpy as np

from ._fixedpoint_config import QFormat, _validate_constraint
from ._fixedpoint_constraints import _snap_to_constraint


def constrain_with_compensation(
    a, g, b, c, form: str, qfmt: QFormat,
    c_constraint: str = "po2",
    a_constraint: Optional[str] = None,
    b_constraint: Optional[str] = None,
    g_constraint: Optional[str] = None,
    zero_circle_margin: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Snap interstage gains and optionally other coefficients to hardware forms.

    The interstage c-gains (``c[0..n-2]`` for CRFB, ``c[1..n-1]`` for CRFF)
    are snapped to ``c_constraint`` and the resulting perturbation is absorbed
    into the other coefficients via a state-scaling similarity transform, so
    the NTF is preserved exactly.  Optionally, the remaining coefficient groups
    (``a``, ``b[0..n-1]``, ``g``) can each be snapped to independent
    constraints *after* compensation; these direct snaps do not preserve the
    NTF but allow every coefficient group to target its own hardware form.

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
        all po2 / CSD snaps.  Typically pass ``FixedPointConfig.coeff`` here.
    c_constraint : str
        Constraint applied to the interstage gains via NTF-preserving state
        scaling.  ``"po2"`` (default) or ``"csd:N"``.  Applied to
        ``c[0..n-2]`` for CRFB, ``c[1..n-1]`` for CRFF; the remaining
        ``c`` entry (output gain / feedback DAC) is left unconstrained.
    a_constraint : str or None
        Constraint applied directly to every ``a_i`` after compensation.
        ``None`` (default) leaves ``a`` values as the exact compensation
        result; ``"po2"`` or ``"csd:N"`` snaps them to the hardware grid.
        This snap does **not** preserve the NTF.
    b_constraint : str or None
        Same as ``a_constraint`` but applied to ``b[0..n-1]`` (the
        integrator input feed-in gains).  ``b[n]`` (direct feedthrough to
        the quantizer input ``y``) is always left unconstrained.
    g_constraint : str or None
        Same as ``a_constraint`` but applied to every resonator coefficient
        ``g_j``.  Runs before the ``zero_circle_margin`` check so that
        quantised ``g`` values are subject to the unit-circle guard.
    zero_circle_margin : float
        Maximum allowed magnitude of any NTF zero beyond the unit circle.
        After all snaps, every NTF zero must satisfy
        ``|z| <= 1 + zero_circle_margin``.

        * ``0.0`` (default) -- strict unit-circle containment.  Float64
          arithmetic is accurate to ~1e-15, so false positives are
          extremely unlikely for well-conditioned designs.
        * Positive value (e.g. ``1e-6``) -- explicit floating-point
          tolerance when zero magnitudes land just above 1 due to numerical
          noise.
        * Negative value (e.g. ``-0.05``) -- stricter than the unit circle;
          requires zeros to sit at ``|z| <= 1 - 0.05``.  Note that for
          ideal resonator coefficients ``g_j ∈ (0, 4)`` the zeros are
          exactly on the unit circle (``|z| = 1``), so a negative margin
          will always produce a warning for resonator-based designs.

        When a violation is detected the function attempts to bring zeros
        back inside the boundary by clamping any ``g_j`` that is outside
        ``(0, 4)`` -- the range for which the resonator zeros lie exactly on
        the unit circle.  If all ``g_j`` are already in ``(0, 4)`` and
        zeros still violate the margin (caused by ``a``/``b``/``c``
        coefficient interactions), a warning is issued but the coefficients
        are returned unchanged.

    Returns
    -------
    a_new, g_new, b_new, c_new : ndarray
        Float64 topology coefficients after compensation and any requested
        direct snaps.  Feed these to :func:`stuffABCD` to get the hardware
        ABCD matrix.

    Notes
    -----
    Coefficient groups and their hardware roles (CRFB example):

    * ``c_i`` -- interstage shift gains; snapping to po2/CSD costs only a
      hard-wired shift per stage and preserves the NTF exactly (via
      ``c_constraint``).
    * ``a_i`` -- feedback DAC weights; may differ in wordlength from ``c``
      but are general multipliers; use ``a_constraint`` to target a specific
      grid.
    * ``b_i`` -- input feed-in gains; often set to ``[b_0, 0, ..., 0, 1]``
      for a single-feed design; use ``b_constraint`` if the nonzero entries
      need to be hardware-friendly.
    * ``g_j`` -- resonator coupling coefficients; small values that set zero
      frequencies; use ``g_constraint`` with care -- the
      ``zero_circle_margin`` guard will clamp any result that pushes zeros
      outside the unit circle.

    The ``c``-compensation is exact in real arithmetic.  The ``a``/``b``/
    ``g`` direct snaps introduce additional NTF perturbation on top of any
    Q-format rounding applied later by the simulator.
    """
    if form not in ("CRFB", "CRFF"):
        raise NotImplementedError(
            f"constrain_with_compensation supports form='CRFB' and 'CRFF'; "
            f"got form={form!r}."
        )
    _validate_constraint(c_constraint, "c_constraint")
    _validate_constraint(a_constraint, "a_constraint")
    _validate_constraint(b_constraint, "b_constraint")
    _validate_constraint(g_constraint, "g_constraint")

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
        a_new, g_new, b_new, c_new = _compensate_CRFB(a, g, b, c, qfmt, c_constraint, n)
    else:
        a_new, g_new, b_new, c_new = _compensate_CRFF(a, g, b, c, qfmt, c_constraint, n)

    # Direct snaps for the remaining coefficient groups.  These run after
    # c-compensation so they operate on the already-scaled values.
    if a_constraint is not None:
        a_new = np.array([_snap_to_constraint(float(v), a_constraint, qfmt)
                          for v in a_new])

    if b_constraint is not None:
        b_new = b_new.copy()
        b_new[:n] = np.array([_snap_to_constraint(float(v), b_constraint, qfmt)
                               for v in b_new[:n]])

    # g_constraint runs before zero_circle_margin so the guard sees the
    # quantised g values.
    if g_constraint is not None:
        g_new = np.array([_snap_to_constraint(float(v), g_constraint, qfmt)
                          for v in g_new])

    a_new, g_new, b_new, c_new = _enforce_zero_circle_margin(
        a_new, g_new, b_new, c_new, form, qfmt, zero_circle_margin
    )
    return a_new, g_new, b_new, c_new


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


def _enforce_zero_circle_margin(
    a: np.ndarray, g: np.ndarray, b: np.ndarray, c: np.ndarray,
    form: str, qfmt: QFormat, margin: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Check NTF zeros and clamp g coefficients if any |z| > 1 + margin.

    Resonator coefficients ``g_j ∈ (0, 4)`` place NTF zeros exactly on the
    unit circle.  A large state-scaling factor can push ``g_j`` outside this
    range after compensation; this helper detects that and clamps ``g_j``
    back, warning the caller that exact NTF preservation no longer holds.

    If all ``g_j`` are already in ``(0, 4)`` but zeros still exceed the
    margin (due to ``a``/``b``/``c`` interactions), a warning is issued and
    coefficients are returned unchanged.
    """
    from ._stuffABCD import stuffABCD

    # NTF zeros = eigenvalues of the loop-filter state-transition matrix A.
    # This holds because the NTF zeros are the poles of H_loop, and
    # H_loop(z) = C(zI-A)^{-1}B + D has poles at the eigenvalues of A.
    # Using eigvals avoids the polynomial round-trip in calculateTF which
    # triggers scipy BadCoefficients warnings for clustered zeros (e.g. all
    # NTF zeros at z=1 for LP designs) and is more numerically stable.
    #
    # A small float64 round-off floor (~4 ULPs at 1.0) prevents false
    # positives when float arithmetic leaves eigenvalues at |z| = 1 ± 1e-15.
    _fp_eps = 4e-15
    threshold = 1.0 + margin + _fp_eps

    n = len(a)
    ABCD = stuffABCD(a, g, b, c, form=form)
    A = ABCD[:n, :n]
    ntf_zeros = np.linalg.eigvals(A)
    mags = np.abs(ntf_zeros)

    # Clamp g_j to the open interval (0, 4).  Within this range the isolated
    # resonator zeros lie exactly on the unit circle; outside it they move to
    # the real axis past |z|=1.  Cross-coupling between resonator sections in
    # higher-order filters can mask the eigenvalue effect in the full A matrix,
    # so we always check g values directly rather than relying on eigvals alone.
    lsb = 2.0 ** (-qfmt.n)
    g_new = g.copy()
    clamped = []

    for j in range(g_new.shape[0]):
        old = float(g_new[j])
        if g_new[j] <= 0.0:
            g_new[j] = lsb
        elif g_new[j] >= 4.0:
            g_new[j] = 4.0 - lsb
        if g_new[j] != old:
            clamped.append(j)

    if clamped:
        warn(
            f"constrain_with_compensation: g{clamped} outside (0, 4) after "
            f"state-scaling compensation; clamped to keep NTF zeros within "
            f"unit circle (zero_circle_margin={margin}). "
            f"Exact NTF preservation no longer holds for these resonators.",
            stacklevel=3,
        )
        ABCD2 = stuffABCD(a, g_new, b, c, form=form)
        ntf_zeros2 = np.linalg.eigvals(ABCD2[:n, :n])
        worst = float(np.max(np.abs(ntf_zeros2)))
        if worst > threshold:
            warn(
                f"constrain_with_compensation: after g clamping, "
                f"max |NTF zero| = {worst:.8g} still exceeds "
                f"1 + margin = {1.0 + margin:.8g}. "
                f"Residual violation is caused by a/b/c coefficient interactions.",
                stacklevel=3,
            )
        return a, g_new, b, c

    # All g in (0, 4).  Eigenvalue check for violations caused by a/b/c drift.
    if not np.all(mags <= threshold):
        worst = float(np.max(mags))
        warn(
            f"constrain_with_compensation: max |NTF zero| = {worst:.8g} > "
            f"1 + zero_circle_margin = {1.0 + margin:.8g}. "
            f"All g ∈ (0, 4), so the violation is caused by compensated "
            f"a/b/c coefficient drift and cannot be corrected by g clamping. "
            f"Consider a finer constraint or wider Q-format.",
            stacklevel=3,
        )

    return a, g, b, c
