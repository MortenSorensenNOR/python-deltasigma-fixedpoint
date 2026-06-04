# -*- coding: utf-8 -*-
# _fixedpoint_config.py
# Fixed-point configuration dataclasses for simulateDSM.
# This file is part of python-deltasigma.
#
# python-deltasigma is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# LICENSE file for the licensing terms.

"""Configuration objects for the fixed-point simulation backend.

A :class:`QFormat` describes one signal's word format (signed-ness, integer
bits, fractional bits, rounding and overflow handling). A
:class:`FixedPointConfig` bundles one ``QFormat`` per datapath signal class
(state, coefficients, input, and the quantizer-input accumulator ``y``) and is
the object that the user passes to ``simulateDSM`` to request a fixed-point
simulation.

The fields map directly onto the ``fixedpoint.FixedPoint`` constructor
arguments, so any value accepted by the upstream library is accepted here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class QFormat:
    """Q-format / arithmetic policy for one class of signals.

    Parameters
    ----------
    signed : bool
        Two's-complement signed representation if True, unsigned otherwise.
    m : int
        Integer-part bits. **Includes the sign bit when ``signed=True``**
        (this matches the upstream ``fixedpoint`` library convention). For
        example, a signed ``QFormat(m=4, n=12)`` covers ``[-8, 8)`` with
        fractional resolution ``2**-12``.
    n : int
        Fractional-part bits. Must be >= 0.
    overflow : {"clamp", "wrap"}
        Behaviour when a value cannot be represented in (m, n).
    rounding : str
        Rounding mode, passed to ``fixedpoint.FixedPoint``. Common choices:
        ``"convergent"``, ``"nearest"``, ``"down"``, ``"up"``.
    overflow_alert : {"error", "warning", "ignore"}
        How the underlying library should react when an overflow occurs.
        Use ``"error"`` to make overflow raise, which is useful for sweeping
        wordlengths until the modulator becomes unstable.
    """

    signed: bool
    m: int
    n: int
    overflow: str = "clamp"
    rounding: str = "convergent"
    overflow_alert: str = "error"

    def __post_init__(self) -> None:
        if self.m < 0 or self.n < 0:
            raise ValueError("QFormat: m and n must be non-negative")
        if self.overflow not in ("clamp", "wrap"):
            raise ValueError("QFormat: overflow must be 'clamp' or 'wrap'")
        if self.overflow_alert not in ("error", "warning", "ignore"):
            raise ValueError(
                "QFormat: overflow_alert must be 'error', 'warning' or 'ignore'"
            )


_CONSTRAINT_RE = re.compile(r"^(po2|csd:[1-9]\d*)$")


def _validate_constraint(c: Optional[str], where: str) -> None:
    if c is None:
        return
    if not isinstance(c, str) or not _CONSTRAINT_RE.match(c):
        raise ValueError(
            f"{where}: constraint must be None, 'po2', or 'csd:N' "
            f"with N a positive integer; got {c!r}"
        )


@dataclass(frozen=True)
class FixedPointConfig:
    """Bundle of Q-formats describing the modulator datapath.

    Parameters
    ----------
    state : QFormat
        Format used for the modulator state vector ``x``.
    coeff : QFormat
        Format used for the ABCD coefficients and for the (integer-valued)
        quantizer output ``v`` when it is fed back into the state update.
    input : QFormat
        Format used for the input sequence ``u``.
    y : QFormat, optional
        Format used for the quantizer input ``y`` (i.e. the accumulator that
        computes ``C*x + D*u`` before quantization). Named ``y`` to match
        the DSM convention -- the modulator output ``v`` is the quantizer's
        output and is determined by ``nlev``, not by a Q-format. If not
        given, the ``state`` format is reused.
    coeff_constraint : str, optional
        Hardware-form constraint applied to every coefficient by default.
        Recognised values:

        * ``None`` (default) -- no constraint, just round to the ``coeff``
          grid.
        * ``"po2"`` -- snap to the nearest signed power of two. Models a
          coefficient implemented as a single hard-wired shift.
        * ``"csd:N"`` -- snap to the closest sum of at most ``N`` signed
          powers of two (greedy approximation, distinct exponents).
          Models a small shift-and-add tree of depth ``N``.

        Overridden per-entry by ``coeff_constraint_for``.
    coeff_constraint_for : callable, optional
        ``(matrix_name, row, col) -> str | None`` returning the constraint
        for one specific coefficient. ``matrix_name`` is one of
        ``"A"``, ``"B"``, ``"C"``, ``"D"``. Returning ``None`` falls back
        to ``coeff_constraint``.

        Use this when only some coefficients have to be hardware-friendly
        (e.g. constraining the CRFB interstage gains ``c_i`` while leaving
        the feedback ``a_i`` and resonator ``g_i`` general).
    """

    state: QFormat
    coeff: QFormat
    input: QFormat
    y: Optional[QFormat] = None
    coeff_constraint: Optional[str] = None
    coeff_constraint_for: Optional[
        Callable[[str, int, int], Optional[str]]
    ] = None

    def __post_init__(self) -> None:
        _validate_constraint(self.coeff_constraint, "FixedPointConfig")

    @property
    def y_or_state(self) -> QFormat:
        """The ``y`` Q-format, falling back to ``state`` if unset."""
        return self.y if self.y is not None else self.state

    def constraint_for(self, matrix_name: str, row: int, col: int) -> Optional[str]:
        """Return the constraint string in force for one coefficient."""
        if self.coeff_constraint_for is not None:
            specific = self.coeff_constraint_for(matrix_name, row, col)
            _validate_constraint(specific,
                f"coeff_constraint_for({matrix_name!r},{row},{col})")
            if specific is not None:
                return specific
        return self.coeff_constraint


def to_fp(value, qfmt: QFormat):
    """Construct a ``fixedpoint.FixedPoint`` value in the given Q-format.

    Imported lazily so that ``fixedpoint`` is only required when a fixed-point
    simulation is actually requested.
    """
    from fixedpoint import FixedPoint  # local import: optional dep

    return FixedPoint(
        float(value),
        signed=qfmt.signed,
        m=qfmt.m,
        n=qfmt.n,
        overflow=qfmt.overflow,
        rounding=qfmt.rounding,
        overflow_alert=qfmt.overflow_alert,
        mismatch_alert="ignore",
        implicit_cast_alert="ignore",
    )
