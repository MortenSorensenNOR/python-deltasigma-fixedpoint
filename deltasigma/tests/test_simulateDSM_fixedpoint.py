# -*- coding: utf-8 -*-
# test_simulateDSM_fixedpoint.py
# Tests for the fixed-point simulation backend.
# This file is part of python-deltasigma.
#
# Copyright (c) 2026, Morten Sørensen
# SPDX-License-Identifier: BSD-2-Clause
#
# python-deltasigma is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# LICENSE file for the licensing terms.

"""Tests for ``simulateDSM`` with a ``FixedPointConfig`` argument."""

import unittest

import numpy as np
from numpy.fft import fft

from deltasigma import (
    FixedPointConfig,
    QFormat,
    calculateSNR,
    calculateTF,
    constrain_coefficients,
    constrain_with_compensation,
    cplxpair,
    ds_hann,
    mapABCD,
    realizeNTF,
    scaleABCD,
    simulateDSM,
    stuffABCD,
    synthesizeNTF,
)
from deltasigma._fixedpoint_constraints import (
    _snap_csd,
    _snap_po2,
    _snap_to_constraint,
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
        return FixedPointConfig(state=qf, coeff=qf, input=qf, y=qf)

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
            cfg = FixedPointConfig(state=qf, coeff=qf, input=qf, y=qf)
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
        cfg = FixedPointConfig(state=qf, coeff=qf, input=qf, y=qf)
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


class TestCompensation(unittest.TestCase):
    """State-scaling compensation when constraining c_i to a hardware form."""

    @classmethod
    def setUpClass(cls):
        # 5th-order CRFB with scaling so c[0..n-2] are non-trivial values.
        cls.OSR = 32
        H = synthesizeNTF(5, cls.OSR, 1)
        a, g, b, c = realizeNTF(H, "CRFB")
        b = np.concatenate(([b[0]], np.zeros(b.shape[0] - 1)))
        ABCD = stuffABCD(a, g, b, c, form="CRFB")
        ABCDs, _umax, _S = scaleABCD(ABCD, nlev=2, f=1.0 / (cls.OSR * 4))
        cls.a, cls.g, cls.b, cls.c = mapABCD(ABCDs, form="CRFB")
        cls.qf = QFormat(signed=True, m=4, n=20)

    def test_compensation_makes_interstage_c_po2(self):
        """After compensation, c[0..n-2] must be exact powers of 2."""
        _, _, _, c_new = constrain_with_compensation(
            self.a, self.g, self.b, self.c, "CRFB", self.qf, "po2",
        )
        for i, value in enumerate(c_new[:-1]):
            log2 = np.log2(abs(value))
            self.assertAlmostEqual(log2, round(log2), places=10,
                msg=f"c[{i}] = {value} is not a power of 2 after compensation")

    def test_compensation_preserves_NTF(self):
        """Compensated stuffABCD has the same NTF zeros and poles."""
        a2, g2, b2, c2 = constrain_with_compensation(
            self.a, self.g, self.b, self.c, "CRFB", self.qf, "po2",
        )
        ntf_orig = calculateTF(stuffABCD(self.a, self.g, self.b, self.c,
                                         form="CRFB"))[0]
        ntf_new = calculateTF(stuffABCD(a2, g2, b2, c2, form="CRFB"))[0]
        zo, po, ko = ntf_orig
        zn, pn, kn = ntf_new
        zero_diff = np.max(np.abs(cplxpair(zo) - cplxpair(zn)))
        pole_diff = np.max(np.abs(cplxpair(po) - cplxpair(pn)))
        self.assertLess(zero_diff, 1e-8,
            msg=f"NTF zeros drifted by {zero_diff} after compensation")
        self.assertLess(pole_diff, 1e-8,
            msg=f"NTF poles drifted by {pole_diff} after compensation")
        self.assertAlmostEqual(kn / ko, 1.0, places=8)

    def test_uncompensated_snap_shifts_zeros_significantly(self):
        """Sanity check: naive snap really does move the zeros (so the
        compensation is doing meaningful work)."""
        from deltasigma._fixedpoint_constraints import _snap_to_constraint
        c_naive = self.c.copy()
        for i in range(len(self.c) - 1):
            c_naive[i] = _snap_to_constraint(self.c[i], "po2", self.qf)
        ntf_orig = calculateTF(stuffABCD(self.a, self.g, self.b, self.c,
                                         form="CRFB"))[0]
        ntf_naive = calculateTF(stuffABCD(self.a, self.g, self.b, c_naive,
                                          form="CRFB"))[0]
        zo = cplxpair(ntf_orig[0])
        zn = cplxpair(ntf_naive[0])
        zero_diff = np.max(np.abs(zo - zn))
        # Should drift visibly; specific number depends on coefficients.
        self.assertGreater(zero_diff, 1e-4,
            msg="naive po2 snap didn't move the zeros -- this test is "
                "ill-set-up or constraints are trivial")

    def test_unsupported_form_raises(self):
        with self.assertRaises(NotImplementedError):
            constrain_with_compensation(
                self.a, self.g, self.b, self.c, "CIFB", self.qf, "po2",
            )

    def test_crff_compensation_makes_interstage_c_po2(self):
        """CRFF: c[1..n-1] must be exact powers of 2 after compensation.
        c[0] (the feedback DAC gain) is left untouched."""
        H = synthesizeNTF(5, self.OSR, 1)
        a, g, b, c = realizeNTF(H, "CRFF")
        b = np.concatenate(([b[0]], np.zeros(b.shape[0] - 1)))
        ABCD = stuffABCD(a, g, b, c, form="CRFF")
        ABCDs, _, _ = scaleABCD(ABCD, nlev=2, f=1.0 / (self.OSR * 4))
        a, g, b, c = mapABCD(ABCDs, form="CRFF")
        a2, g2, b2, c2 = constrain_with_compensation(
            a, g, b, c, "CRFF", self.qf, "po2",
        )
        # c[0] is feedback DAC, should be unchanged.
        self.assertEqual(c2[0], c[0])
        # c[1..n-1] should all be exact po2.
        for i, value in enumerate(c2[1:], start=1):
            log2 = np.log2(abs(value))
            self.assertAlmostEqual(log2, round(log2), places=10,
                msg=f"c[{i}] = {value} is not a power of 2 (CRFF)")

    def test_crff_compensation_preserves_NTF(self):
        """CRFF: NTF zeros and poles preserved to machine precision."""
        H = synthesizeNTF(5, self.OSR, 1)
        a, g, b, c = realizeNTF(H, "CRFF")
        b = np.concatenate(([b[0]], np.zeros(b.shape[0] - 1)))
        ABCD = stuffABCD(a, g, b, c, form="CRFF")
        ABCDs, _, _ = scaleABCD(ABCD, nlev=2, f=1.0 / (self.OSR * 4))
        a, g, b, c = mapABCD(ABCDs, form="CRFF")
        a2, g2, b2, c2 = constrain_with_compensation(
            a, g, b, c, "CRFF", self.qf, "po2",
        )
        ntf_orig = calculateTF(stuffABCD(a, g, b, c, form="CRFF"))[0]
        ntf_new = calculateTF(stuffABCD(a2, g2, b2, c2, form="CRFF"))[0]
        zero_diff = np.max(np.abs(cplxpair(ntf_orig[0]) -
                                  cplxpair(ntf_new[0])))
        pole_diff = np.max(np.abs(cplxpair(ntf_orig[1]) -
                                  cplxpair(ntf_new[1])))
        self.assertLess(zero_diff, 1e-8,
            msg=f"CRFF NTF zeros drifted by {zero_diff}")
        self.assertLess(pole_diff, 1e-8,
            msg=f"CRFF NTF poles drifted by {pole_diff}")

    def test_crff_end_to_end_fp_simulation(self):
        """A compensated CRFF design can be simulated in fixed-point."""
        H = synthesizeNTF(5, self.OSR, 1)
        a, g, b, c = realizeNTF(H, "CRFF")
        b = np.concatenate(([b[0]], np.zeros(b.shape[0] - 1)))
        ABCD = stuffABCD(a, g, b, c, form="CRFF")
        ABCDs, umax, _ = scaleABCD(ABCD, nlev=2, f=1.0 / (self.OSR * 4))
        a, g, b, c = mapABCD(ABCDs, form="CRFF")
        a2, g2, b2, c2 = constrain_with_compensation(
            a, g, b, c, "CRFF", self.qf, "po2",
        )
        ABCD_hw = stuffABCD(a2, g2, b2, c2, form="CRFF")
        # Use a state Q-format with enough integer bits for CRFF (a values
        # grow under compensation, so the state can swing wider).
        qf_state = QFormat(signed=True, m=5, n=18)
        cfg = FixedPointConfig(state=qf_state, coeff=qf_state,
                               input=qf_state, y=qf_state)
        N = 512
        f = 5
        u = 0.5 * umax * np.sin(2 * np.pi * f / N * np.arange(N))
        v, _, _, _ = simulateDSM(u, ABCD_hw, fixedpoint=cfg)
        self.assertEqual(v.shape, (N,))
        self.assertTrue(set(np.unique(v)).issubset({-1.0, 1.0}),
            msg=f"unexpected v levels for nlev=2: {set(np.unique(v))}")

    def test_csd_constraint_also_works(self):
        """csd:2 should give c values that are sums of two po2 -- the
        NTF-preservation property still holds."""
        a2, g2, b2, c2 = constrain_with_compensation(
            self.a, self.g, self.b, self.c, "CRFB", self.qf, "csd:2",
        )
        ntf_orig = calculateTF(stuffABCD(self.a, self.g, self.b, self.c,
                                         form="CRFB"))[0]
        ntf_new = calculateTF(stuffABCD(a2, g2, b2, c2, form="CRFB"))[0]
        zero_diff = np.max(np.abs(cplxpair(ntf_orig[0]) -
                                  cplxpair(ntf_new[0])))
        self.assertLess(zero_diff, 1e-8)


class TestCoeffConstraints(unittest.TestCase):
    """Tests for the po2 / CSD coefficient form constraints."""

    def setUp(self):
        # Wide signed format: range [-8, 8), 20 fractional bits.
        self.qf = QFormat(signed=True, m=4, n=20)

    # --- snap primitives -----------------------------------------------------

    def test_po2_snaps_to_nearest_power_of_two(self):
        cases = {
            0.0: 0.0,
            0.5: 0.5,
            0.7: 0.5,   # log2(0.7) ~ -0.51, round -> -1
            1.0: 1.0,
            1.5: 2.0,   # log2(1.5) ~ 0.58, round -> 1
            2.7: 2.0,
            -1.5: -2.0,
            -8.0: -8.0,  # signed Q4 lower bound is exactly representable
        }
        for value, expected in cases.items():
            self.assertEqual(_snap_po2(value, self.qf), expected,
                msg=f"po2({value}) -> got {_snap_po2(value, self.qf)}, "
                    f"expected {expected}")

    def test_po2_clips_out_of_range_value_to_in_range_po2(self):
        # Positive 8.0 is NOT representable in signed Q4.20 (max ~ 7.99...),
        # so the next po2 down (4.0) must be picked.
        self.assertEqual(_snap_po2(8.0, self.qf), 4.0)

    def test_csd_2_decomposes_to_two_signed_powers(self):
        # 0.7 -> 0.5 + 0.25 = 0.75 (nearest sum of two po2)
        self.assertAlmostEqual(_snap_csd(0.7, self.qf, 2), 0.75)
        # 0.75 already exact
        self.assertAlmostEqual(_snap_csd(0.75, self.qf, 2), 0.75)
        # Negative side
        self.assertAlmostEqual(_snap_csd(-0.7, self.qf, 2), -0.75)

    def test_csd_more_terms_gives_finer_approximation(self):
        # Increasing N must monotonically reduce |error| -- never make it
        # worse -- for any target value.
        value = 1.234
        errs = [
            abs(value - _snap_csd(value, self.qf, n)) for n in (1, 2, 3, 4)
        ]
        for a, b in zip(errs, errs[1:]):
            self.assertGreaterEqual(a, b,
                msg=f"CSD error did not decrease: {errs}")

    def test_snap_to_constraint_dispatch(self):
        self.assertEqual(_snap_to_constraint(0.7, None, self.qf), 0.7)
        self.assertEqual(_snap_to_constraint(0.7, "po2", self.qf), 0.5)
        self.assertAlmostEqual(_snap_to_constraint(0.7, "csd:2", self.qf),
                               0.75)
        with self.assertRaises(ValueError):
            _snap_to_constraint(0.7, "garbage", self.qf)

    # --- config validation ---------------------------------------------------

    def test_config_rejects_malformed_constraint(self):
        with self.assertRaises(ValueError):
            FixedPointConfig(state=self.qf, coeff=self.qf, input=self.qf,
                             coeff_constraint="not-a-mode")
        with self.assertRaises(ValueError):
            FixedPointConfig(state=self.qf, coeff=self.qf, input=self.qf,
                             coeff_constraint="csd:0")  # N must be positive

    def test_per_entry_constraint_overrides_default(self):
        # Default is csd:3; row 0 col 0 of A is overridden to po2.
        def only_A00(name, r, c):
            return "po2" if (name, r, c) == ("A", 0, 0) else None
        cfg = FixedPointConfig(
            state=self.qf, coeff=self.qf, input=self.qf,
            coeff_constraint="csd:3", coeff_constraint_for=only_A00,
        )
        self.assertEqual(cfg.constraint_for("A", 0, 0), "po2")
        self.assertEqual(cfg.constraint_for("A", 1, 0), "csd:3")
        self.assertEqual(cfg.constraint_for("B", 0, 0), "csd:3")

    # --- constrain_coefficients helper ---------------------------------------

    def test_constrain_coefficients_returns_po2_matrix(self):
        """All entries of the returned matrices must be signed po2 values."""
        ABCD, _ = _scaled_modulator(order=5, OSR=32)
        cfg = FixedPointConfig(
            state=self.qf, coeff=self.qf, input=self.qf,
            coeff_constraint="po2",
        )
        A, B, C, D = constrain_coefficients(ABCD, cfg)
        for name, M in (("A", A), ("B", B), ("C", C), ("D", D)):
            flat = M.flatten()
            for v in flat:
                if v == 0.0:
                    continue
                # log2(|v|) must be very close to an integer
                k = np.log2(abs(v))
                self.assertAlmostEqual(k, round(k), places=10,
                    msg=f"{name} value {v} is not a power of 2")

    def test_constrain_coefficients_passes_through_when_no_constraint(self):
        ABCD, _ = _scaled_modulator(order=5, OSR=32)
        cfg = FixedPointConfig(state=self.qf, coeff=self.qf, input=self.qf)
        A, B, C, D = constrain_coefficients(ABCD, cfg)
        # Without a constraint, the only delta vs. the input is Q-format
        # rounding -- which is sub-LSB for Q4.20 on these scaled values.
        # Verify A is within 2**-19 of the original A entries.
        order = ABCD.shape[0] - 1
        nu = ABCD.shape[1] - ABCD.shape[0]
        A_in = ABCD[:order, :order]
        self.assertTrue(np.allclose(A, A_in, atol=2.0 ** -19))

    # --- end-to-end integration ----------------------------------------------

    def test_po2_constraint_changes_simulation_output(self):
        """A po2-constrained run must differ from the unconstrained run
        (otherwise the constraint isn't being applied)."""
        ABCD, umax = _scaled_modulator(order=5, OSR=32)
        N = 512
        f = 7
        u = 0.5 * umax * np.sin(2 * np.pi * f / N * np.arange(N))
        qf = QFormat(signed=True, m=4, n=16)
        unconstrained = FixedPointConfig(state=qf, coeff=qf, input=qf)
        po2_all = FixedPointConfig(
            state=qf, coeff=qf, input=qf, coeff_constraint="po2",
        )
        v_unc = simulateDSM(u, ABCD, fixedpoint=unconstrained)[0]
        v_po2 = simulateDSM(u, ABCD, fixedpoint=po2_all)[0]
        # The two runs should not be identical: po2 forces coefficient drift.
        self.assertFalse(np.array_equal(v_unc, v_po2),
            msg="po2 constraint had no observable effect on the output")


class TestZeroCircleMargin(unittest.TestCase):
    """Tests for the zero_circle_margin enforcement in constrain_with_compensation."""

    OSR = 32

    def _build_crfb(self):
        H = synthesizeNTF(5, self.OSR, 1)
        a, g, b, c = realizeNTF(H, "CRFB")
        b = np.concatenate(([b[0]], np.zeros(b.shape[0] - 1)))
        ABCD = stuffABCD(a, g, b, c, form="CRFB")
        ABCDs, _, _ = scaleABCD(ABCD, nlev=2, f=1.0 / (self.OSR * 4))
        return mapABCD(ABCDs, form="CRFB")

    def _build_crff(self):
        H = synthesizeNTF(5, self.OSR, 1)
        a, g, b, c = realizeNTF(H, "CRFF")
        b = np.concatenate(([b[0]], np.zeros(b.shape[0] - 1)))
        ABCD = stuffABCD(a, g, b, c, form="CRFF")
        ABCDs, _, _ = scaleABCD(ABCD, nlev=2, f=1.0 / (self.OSR * 4))
        return mapABCD(ABCDs, form="CRFF")

    def _ntf_zero_mags(self, a, g, b, c, form):
        ntf = calculateTF(stuffABCD(a, g, b, c, form=form))[0]
        return np.abs(np.asarray(ntf[0]))

    # -- normal operation: no warnings, zeros stay on unit circle ---------------

    def test_crfb_normal_no_warning(self):
        """Normal CRFB compensation with default margin produces no warnings."""
        qf = QFormat(signed=True, m=4, n=20)
        a, g, b, c = self._build_crfb()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            a2, g2, b2, c2 = constrain_with_compensation(
                a, g, b, c, "CRFB", qf, "po2", zero_circle_margin=0.0
            )
        mags = self._ntf_zero_mags(a2, g2, b2, c2, "CRFB")
        self.assertTrue(np.all(mags <= 1.0 + 1e-10),
            msg=f"Zeros outside unit circle after CRFB compensation: max|z|={np.max(mags):.10f}")

    def test_crff_normal_no_warning(self):
        """Normal CRFF compensation with default margin produces no warnings."""
        qf = QFormat(signed=True, m=4, n=20)
        a, g, b, c = self._build_crff()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            a2, g2, b2, c2 = constrain_with_compensation(
                a, g, b, c, "CRFF", qf, "po2", zero_circle_margin=0.0
            )
        mags = self._ntf_zero_mags(a2, g2, b2, c2, "CRFF")
        self.assertTrue(np.all(mags <= 1.0 + 1e-10),
            msg=f"Zeros outside unit circle after CRFF compensation: max|z|={np.max(mags):.10f}")

    # -- g outside (0, 4): clamping kicks in -----------------------------------
    # These tests call _enforce_zero_circle_margin directly so that compensation
    # state-scaling (which can re-scale g back into range) does not obscure the
    # enforcement logic.  The function is module-level and can be imported.

    def _enforce(self, a, g, b, c, form, margin=0.0):
        from deltasigma._constrain_compensation import _enforce_zero_circle_margin
        qf = QFormat(signed=True, m=4, n=20)
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _enforce_zero_circle_margin(a, g, b, c, form, qf, margin)
        return result, caught

    def test_crfb_negative_g_clamped_and_warns(self):
        """CRFB: g < 0 detected directly in _enforce_zero_circle_margin → clamped."""
        a, g, b, c = self._build_crfb()
        g_bad = g.copy()
        g_bad[0] = -0.5  # negative → zeros go to real axis outside |z|=1
        (a2, g2, b2, c2), caught = self._enforce(a, g_bad, b, c, "CRFB")
        self.assertTrue(any("clamped" in str(w.message).lower() for w in caught),
            msg="Expected a 'clamped' warning for g < 0")
        self.assertGreater(g2[0], 0.0,
            msg=f"g2[0] = {g2[0]} should be positive after clamping")

    def test_crff_negative_g_clamped_and_warns(self):
        """CRFF: g < 0 detected directly in _enforce_zero_circle_margin → clamped."""
        a, g, b, c = self._build_crff()
        g_bad = g.copy()
        g_bad[0] = -0.3
        (a2, g2, b2, c2), caught = self._enforce(a, g_bad, b, c, "CRFF")
        self.assertTrue(any("clamped" in str(w.message).lower() for w in caught),
            msg="Expected a 'clamped' warning for g < 0 (CRFF)")
        self.assertGreater(g2[0], 0.0,
            msg=f"g2[0] = {g2[0]} should be positive after clamping (CRFF)")

    def test_crfb_g_above_4_clamped(self):
        """CRFB: g >= 4 detected directly → clamped to < 4."""
        a, g, b, c = self._build_crfb()
        g_bad = g.copy()
        g_bad[0] = 5.0  # above 4 → zeros go to real axis outside |z|=1
        (a2, g2, b2, c2), caught = self._enforce(a, g_bad, b, c, "CRFB")
        self.assertTrue(any("clamped" in str(w.message).lower() for w in caught),
            msg="Expected a 'clamped' warning for g >= 4")
        self.assertLess(g2[0], 4.0,
            msg=f"g2[0] = {g2[0]} should be < 4 after clamping")

    def test_crff_g_above_4_clamped(self):
        """CRFF: g >= 4 detected directly → clamped to < 4."""
        a, g, b, c = self._build_crff()
        g_bad = g.copy()
        g_bad[0] = 6.0
        (a2, g2, b2, c2), caught = self._enforce(a, g_bad, b, c, "CRFF")
        self.assertTrue(any("clamped" in str(w.message).lower() for w in caught),
            msg="Expected a 'clamped' warning for g >= 4 (CRFF)")
        self.assertLess(g2[0], 4.0,
            msg=f"g2[0] = {g2[0]} should be < 4 after clamping (CRFF)")

    # -- margin semantics ------------------------------------------------------

    def test_positive_margin_tolerates_tiny_exceedance(self):
        """A positive zero_circle_margin allows zeros just above |z|=1."""
        qf = QFormat(signed=True, m=4, n=20)
        a, g, b, c = self._build_crfb()
        # With a generous margin, a mildly out-of-range g should not warn.
        g_slightly_bad = g.copy()
        g_slightly_bad[0] = 4.001  # barely over 4; zeros go very slightly outside
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            constrain_with_compensation(
                a, g_slightly_bad, b, c, "CRFB", qf, "po2",
                zero_circle_margin=0.1,  # allow up to |z| = 1.1
            )
        clamped_warnings = [w for w in caught if "clamped" in str(w.message).lower()]
        self.assertEqual(len(clamped_warnings), 0,
            msg="margin=0.1 should tolerate zeros barely above |z|=1")

    def test_zero_circle_margin_backward_compatible(self):
        """Omitting zero_circle_margin (default 0.0) works for normal designs."""
        qf = QFormat(signed=True, m=4, n=20)
        a, g, b, c = self._build_crfb()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Must not raise -- the default 0.0 should be fine for a well-formed design.
            constrain_with_compensation(a, g, b, c, "CRFB", qf, "po2")


class TestPerGroupConstraints(unittest.TestCase):
    """Tests for per-group coefficient constraints in constrain_with_compensation.

    The function accepts independent ``a_constraint``, ``b_constraint``, and
    ``g_constraint`` parameters in addition to the primary ``c_constraint``.
    Each group is snapped after the c-compensation state-scaling step.
    """

    OSR = 32

    @classmethod
    def setUpClass(cls):
        H = synthesizeNTF(5, cls.OSR, 1)
        cls.a_crfb, cls.g_crfb, cls.b_crfb, cls.c_crfb = realizeNTF(H, "CRFB")
        cls.a_crff, cls.g_crff, cls.b_crff, cls.c_crff = realizeNTF(H, "CRFF")

    def setUp(self):
        self.qf = QFormat(signed=True, m=4, n=20)

    def _is_po2(self, v):
        if v == 0.0:
            return True
        k = np.log2(abs(v))
        return abs(k - round(k)) < 1e-10

    # -- a_constraint ----------------------------------------------------------

    def test_a_constraint_snaps_a_to_po2(self):
        """a_constraint='po2' forces every a_i to a power of two."""
        a2, g2, b2, c2 = constrain_with_compensation(
            self.a_crfb, self.g_crfb, self.b_crfb, self.c_crfb,
            "CRFB", self.qf, a_constraint="po2"
        )
        for i, v in enumerate(a2):
            self.assertTrue(self._is_po2(v),
                msg=f"a2[{i}]={v} is not a power of 2")

    def test_a_constraint_none_does_not_snap_a(self):
        """Without a_constraint the compensated a values are general floats."""
        a_base, _, _, _ = constrain_with_compensation(
            self.a_crfb, self.g_crfb, self.b_crfb, self.c_crfb,
            "CRFB", self.qf
        )
        a_po2, _, _, _ = constrain_with_compensation(
            self.a_crfb, self.g_crfb, self.b_crfb, self.c_crfb,
            "CRFB", self.qf, a_constraint="po2"
        )
        self.assertFalse(np.allclose(a_base, a_po2, atol=1e-10),
            msg="a_constraint='po2' had no effect; compensated a may already be po2")

    def test_a_constraint_csd2_works(self):
        """a_constraint='csd:2' snaps each a_i to a sum of two powers of 2."""
        a2, g2, b2, c2 = constrain_with_compensation(
            self.a_crfb, self.g_crfb, self.b_crfb, self.c_crfb,
            "CRFB", self.qf, a_constraint="csd:2"
        )
        self.assertTrue(np.all(np.isfinite(a2)))
        a_base, _, _, _ = constrain_with_compensation(
            self.a_crfb, self.g_crfb, self.b_crfb, self.c_crfb,
            "CRFB", self.qf
        )
        self.assertFalse(np.allclose(a2, a_base, atol=1e-10),
            msg="csd:2 a_constraint produced no change vs unconstrained")

    # -- b_constraint ----------------------------------------------------------

    def test_b_constraint_snaps_b_interstage_only(self):
        """b_constraint='po2' snaps b[0..n-1] but leaves b[n] (direct feed) unchanged."""
        a, g, b, c = self.a_crfb, self.g_crfb, self.b_crfb, self.c_crfb
        b_n_original = float(b[-1])
        a2, g2, b2, c2 = constrain_with_compensation(
            a, g, b, c, "CRFB", self.qf, b_constraint="po2"
        )
        n = len(a)
        for i in range(n):
            self.assertTrue(self._is_po2(b2[i]),
                msg=f"b2[{i}]={b2[i]} is not a power of 2 with b_constraint='po2'")
        self.assertAlmostEqual(float(b2[n]), b_n_original, places=15,
            msg=f"b[n] must not be snapped: before={b_n_original}, after={b2[n]}")

    # -- g_constraint ----------------------------------------------------------

    def test_g_constraint_snaps_g_to_po2(self):
        """g_constraint='po2' forces every g_j to a power of two."""
        a2, g2, b2, c2 = constrain_with_compensation(
            self.a_crfb, self.g_crfb, self.b_crfb, self.c_crfb,
            "CRFB", self.qf, g_constraint="po2"
        )
        for j, v in enumerate(g2):
            self.assertTrue(self._is_po2(v),
                msg=f"g2[{j}]={v} is not a power of 2")

    # -- CRFF ------------------------------------------------------------------

    def test_crff_a_and_b_constraints(self):
        """CRFF: a_constraint and b_constraint snap their groups independently."""
        a, g, b, c = self.a_crff, self.g_crff, self.b_crff, self.c_crff
        a2, g2, b2, c2 = constrain_with_compensation(
            a, g, b, c, "CRFF", self.qf,
            a_constraint="po2", b_constraint="po2"
        )
        for i, v in enumerate(a2):
            self.assertTrue(self._is_po2(v),
                msg=f"CRFF a2[{i}]={v} is not a power of 2")
        n = len(a)
        for i in range(n):
            self.assertTrue(self._is_po2(b2[i]),
                msg=f"CRFF b2[{i}]={b2[i]} is not a power of 2")

    # -- all groups simultaneously ---------------------------------------------

    def test_all_constraints_simultaneously(self):
        """Setting all four constraint groups at once: each group is snapped."""
        a, g, b, c = self.a_crfb, self.g_crfb, self.b_crfb, self.c_crfb
        a2, g2, b2, c2 = constrain_with_compensation(
            a, g, b, c, "CRFB", self.qf,
            c_constraint="po2",
            a_constraint="po2",
            b_constraint="po2",
            g_constraint="po2",
        )
        n = len(a)
        for i in range(n - 1):  # c[0..n-2] are interstage
            self.assertTrue(self._is_po2(c2[i]),
                msg=f"c2[{i}]={c2[i]} is not a power of 2 after c-compensation")
        for i, v in enumerate(a2):
            self.assertTrue(self._is_po2(v),
                msg=f"a2[{i}]={v} is not a power of 2")
        for j, v in enumerate(g2):
            self.assertTrue(self._is_po2(v),
                msg=f"g2[{j}]={v} is not a power of 2")
        for i in range(n):
            self.assertTrue(self._is_po2(b2[i]),
                msg=f"b2[{i}]={b2[i]} is not a power of 2")

    # -- validation ------------------------------------------------------------

    def test_invalid_a_constraint_raises(self):
        """An invalid a_constraint string raises ValueError."""
        with self.assertRaises(ValueError):
            constrain_with_compensation(
                self.a_crfb, self.g_crfb, self.b_crfb, self.c_crfb,
                "CRFB", self.qf, a_constraint="nonsense"
            )

    def test_invalid_g_constraint_raises(self):
        """An invalid g_constraint string (csd:0 is illegal) raises ValueError."""
        with self.assertRaises(ValueError):
            constrain_with_compensation(
                self.a_crfb, self.g_crfb, self.b_crfb, self.c_crfb,
                "CRFB", self.qf, g_constraint="csd:0"
            )


if __name__ == "__main__":
    unittest.main()
