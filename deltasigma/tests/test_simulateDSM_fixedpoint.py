# -*- coding: utf-8 -*-
# test_simulateDSM_fixedpoint.py
# Tests for the fixed-point simulation backend.
# This file is part of python-deltasigma.

"""Tests for ``simulateDSM`` with a ``FixedPointConfig`` argument."""

import unittest

import numpy as np
from numpy.fft import fft

from deltasigma import (
    FixedPointConfig,
    QFormat,
    calculateSNR,
    ds_hann,
    realizeNTF,
    scaleABCD,
    simulateDSM,
    stuffABCD,
    synthesizeNTF,
)


def _scaled_modulator(order=5, OSR=32):
    """Build a 5th-order CRFB modulator with its state vector scaled to ~1."""
    H = synthesizeNTF(order, OSR, 1)
    a, g, b, c = realizeNTF(H, "CRFB")
    b = np.concatenate(([b[0]], np.zeros(b.shape[0] - 1)))
    ABCD = stuffABCD(a, g, b, c, form="CRFB")
    ABCDs, umax, _ = scaleABCD(ABCD, nlev=2, f=1.0 / (OSR * 4))
    return ABCDs, umax


def _snr(v, N, OSR, f_bin):
    spec = fft(v * ds_hann(N)) / (N / 4)
    fB = int(np.ceil(N / (2 * OSR)))
    return calculateSNR(spec[:fB], f_bin)


class TestSimulateDSMFixedPoint(unittest.TestCase):
    """Tests for the FixedPointConfig path through simulateDSM."""

    @classmethod
    def setUpClass(cls):
        cls.OSR = 32
        cls.ABCD, cls.umax = _scaled_modulator(order=5, OSR=cls.OSR)
        cls.N = 1024
        cls.f_bin = 11
        cls.u = 0.5 * cls.umax * np.sin(
            2 * np.pi * cls.f_bin / cls.N * np.arange(cls.N)
        )
        cls.snr_float = _snr(
            simulateDSM(cls.u, cls.ABCD)[0], cls.N, cls.OSR, cls.f_bin
        )

    def _wide_cfg(self):
        # m=4 covers [-8, 8) (signed convention includes sign bit) which holds
        # both the scaled state (~ [-1, 1]) and the largest coefficient (~2.7)
        # with headroom. n=20 fractional bits make quantization essentially
        # negligible compared to the quantizer's noise floor.
        qf = QFormat(signed=True, m=4, n=20)
        return FixedPointConfig(state=qf, coeff=qf, input=qf, output=qf)

    def test_returns_match_float_shapes(self):
        """FP backend returns the same shapes as the float backend."""
        v, xn, xmax, y = simulateDSM(self.u, self.ABCD, fixedpoint=self._wide_cfg())
        self.assertEqual(v.shape, (self.N,))
        self.assertEqual(xn.shape, (5, self.N))
        self.assertEqual(xmax.shape, (5, 1))
        self.assertEqual(y.shape, (self.N,))
        self.assertTrue(np.all(np.isfinite(v)))
        self.assertTrue(np.all(np.isfinite(xn)))

    def test_wide_wordlength_matches_float_snr(self):
        """With wide Q-format the FP SNR should be within ~1.5 dB of float."""
        v_fp, _, _, _ = simulateDSM(self.u, self.ABCD, fixedpoint=self._wide_cfg())
        snr_fp = _snr(v_fp, self.N, self.OSR, self.f_bin)
        self.assertGreater(snr_fp, self.snr_float - 1.5,
            msg=f"FP SNR {snr_fp:.2f} dB << float {self.snr_float:.2f} dB")

    def test_snr_degrades_with_narrowing_wordlength(self):
        """Reducing fractional bits monotonically reduces SNR."""
        snrs = []
        for n_bits in (20, 12, 8):
            qf = QFormat(signed=True, m=4, n=n_bits)
            cfg = FixedPointConfig(state=qf, coeff=qf, input=qf, output=qf)
            v, _, _, _ = simulateDSM(self.u, self.ABCD, fixedpoint=cfg)
            snrs.append(_snr(v, self.N, self.OSR, self.f_bin))
        # Each step down in fractional resolution must give strictly less SNR.
        # We use a tolerance because random alignment of the quantizer with the
        # signal can give small reversals at the dB level.
        self.assertGreater(snrs[0], snrs[1] + 3.0,
            msg=f"SNR did not drop going 20->12 frac bits: {snrs}")
        self.assertGreater(snrs[1], snrs[2] + 3.0,
            msg=f"SNR did not drop going 12->8 frac bits: {snrs}")

    def test_overflow_alert_error_propagates(self):
        """A wordlength too small for the state must raise."""
        from fixedpoint import FixedPointOverflowError

        # Coefficients reach ~2.7, so m=1 (range [-1, 1)) cannot hold them.
        qf = QFormat(signed=True, m=1, n=20, overflow_alert="error")
        cfg = FixedPointConfig(state=qf, coeff=qf, input=qf, output=qf)
        with self.assertRaises(FixedPointOverflowError):
            simulateDSM(self.u, self.ABCD, fixedpoint=cfg)

    def test_mid_tread_quantizer_supported(self):
        """nlev=3 (mid-tread, zero is representable) works under FP."""
        ABCD, umax = _scaled_modulator(order=5, OSR=self.OSR)
        # nlev=3 needs its own scaling, so rebuild around it
        from deltasigma import (realizeNTF, scaleABCD, stuffABCD,
                                synthesizeNTF)
        H = synthesizeNTF(5, self.OSR, 1)
        a, g, b, c = realizeNTF(H, "CRFB")
        b = np.concatenate(([b[0]], np.zeros(b.shape[0] - 1)))
        ABCD3 = stuffABCD(a, g, b, c, form="CRFB")
        ABCD3s, umax3, _ = scaleABCD(ABCD3, nlev=3, f=1.0 / (self.OSR * 4))
        u3 = 0.5 * umax3 * np.sin(
            2 * np.pi * self.f_bin / self.N * np.arange(self.N)
        )
        v_fp, _, _, _ = simulateDSM(
            u3, ABCD3s, nlev=3, fixedpoint=self._wide_cfg()
        )
        levels = set(np.unique(v_fp))
        # mid-tread quantizer: outputs are even integers in [-(n-1), n-1].
        # nlev=3 -> {-2, 0, +2}; zero MUST be representable.
        self.assertIn(0.0, levels,
            msg=f"mid-tread quantizer should yield 0 in output; got {levels}")
        self.assertTrue(levels.issubset({-2.0, 0.0, 2.0}),
            msg=f"unexpected levels for nlev=3 mid-tread: {levels}")

    def test_missing_config_raises(self):
        """Calling the backend directly without a config is an error."""
        from deltasigma._simulateDSM_fixedpoint import simulateDSM as _fp
        with self.assertRaises(ValueError):
            _fp(self.u, self.ABCD, fixedpoint=None)


if __name__ == "__main__":
    unittest.main()
