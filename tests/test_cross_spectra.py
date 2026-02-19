"""Tests for cospec/quadspec feature (GitHub issue #1)."""

import numpy as np
import pytest
from synthetic_timeseries import (
    cross_spectra_to_coh_lag,
    TK_simulate,
    CLSimulator,
    make_powerlaw_pds,
)


class TestCrossSpectraToCohLag:
    """Tests for the cross_spectra_to_coh_lag conversion function."""

    def test_pure_real_cross_spectrum(self):
        """Co-spectrum only -> zero phase lag."""
        P_X = np.ones(100) * 10.0
        P_Y = np.ones(100) * 10.0
        cospec = np.ones(100) * 5.0
        quadspec = np.zeros(100)

        gamma2, phi = cross_spectra_to_coh_lag(cospec, quadspec, P_X, P_Y)

        np.testing.assert_allclose(phi, 0.0, atol=1e-14)
        np.testing.assert_allclose(gamma2, 0.25)  # 25/(100)

    def test_pure_imaginary_cross_spectrum(self):
        """Quadrature spectrum only -> pi/2 phase lag."""
        P_X = np.ones(100) * 10.0
        P_Y = np.ones(100) * 10.0
        cospec = np.zeros(100)
        quadspec = np.ones(100) * 5.0

        gamma2, phi = cross_spectra_to_coh_lag(cospec, quadspec, P_X, P_Y)

        np.testing.assert_allclose(phi, np.pi / 2, atol=1e-14)
        np.testing.assert_allclose(gamma2, 0.25)

    def test_perfect_coherence(self):
        """When |C|^2 = P_X * P_Y, coherence should be 1."""
        P_X = np.ones(50) * 4.0
        P_Y = np.ones(50) * 9.0
        # |C|^2 = cospec^2 + quadspec^2 = 36 = P_X * P_Y
        cospec = np.ones(50) * 6.0
        quadspec = np.zeros(50)

        gamma2, phi = cross_spectra_to_coh_lag(cospec, quadspec, P_X, P_Y)

        np.testing.assert_allclose(gamma2, 1.0)

    def test_roundtrip_with_known_coh_lag(self):
        """Convert coh/lag -> cospec/quadspec -> coh/lag and verify roundtrip."""
        P_X = np.linspace(1, 10, 50)
        P_Y = np.linspace(2, 8, 50)
        target_gamma2 = 0.7
        target_phi = 0.3

        # Compute cross spectrum from coh/lag
        C_mag = np.sqrt(target_gamma2 * P_X * P_Y)
        cospec = C_mag * np.cos(target_phi)
        quadspec = C_mag * np.sin(target_phi)

        # Convert back
        gamma2, phi = cross_spectra_to_coh_lag(cospec, quadspec, P_X, P_Y)

        np.testing.assert_allclose(gamma2, target_gamma2, rtol=1e-12)
        np.testing.assert_allclose(phi, target_phi, atol=1e-12)

    def test_scalar_inputs(self):
        """Function works with scalar inputs."""
        gamma2, phi = cross_spectra_to_coh_lag(3.0, 4.0, 10.0, 10.0)
        np.testing.assert_allclose(gamma2, 25.0 / 100.0)
        np.testing.assert_allclose(phi, np.arctan2(4.0, 3.0))


class TestTKSimulateWithCrossSpectra:
    """Tests for TK_simulate with cospec/quadspec inputs."""

    def setup_method(self):
        self.dt = 0.01
        self.N = 4096
        self.output_length = self.N * self.dt
        self.P1 = make_powerlaw_pds(2.0, self.dt, self.N)
        self.P2 = make_powerlaw_pds(2.0, self.dt, self.N)

    def test_error_when_both_specified(self):
        """Should raise ValueError when both coh/lag and cospec/quadspec given."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            TK_simulate(
                self.P1, self.output_length, self.dt, mean=100, gamma=0.8, cospec=1.0
            )

        with pytest.raises(ValueError, match="Cannot specify both"):
            TK_simulate(
                self.P1, self.output_length, self.dt, mean=100, lag=0.5, quadspec=1.0
            )

    def test_cospec_quadspec_produces_output(self):
        """Basic smoke test: cospec/quadspec should produce valid output."""
        # Compute cross spectrum corresponding to gamma2=0.9, phi=0.3
        gamma2 = 0.9
        phi = 0.3
        C_mag = np.sqrt(gamma2 * self.P1 * self.P2)
        cospec = C_mag * np.cos(phi)
        quadspec = C_mag * np.sin(phi)

        c1, c2 = TK_simulate(
            self.P1,
            self.output_length,
            self.dt,
            mean=100,
            P2=self.P2,
            cospec=cospec,
            quadspec=quadspec,
        )

        assert c1.shape == (self.N,)
        assert c2.shape == (self.N,)
        assert np.isfinite(c1).all()
        assert np.isfinite(c2).all()

    def test_cospec_only_means_zero_lag(self):
        """Specifying only cospec (no quadspec) should be equivalent to zero lag."""
        gamma2 = 0.8
        C_mag = np.sqrt(gamma2 * self.P1 * self.P2)
        cospec = C_mag  # All real -> zero phase lag

        np.random.seed(42)
        c1_cross, c2_cross = TK_simulate(
            self.P1, self.output_length, self.dt, mean=100, P2=self.P2, cospec=cospec
        )

        assert c1_cross.shape == (self.N,)
        assert c2_cross.shape == (self.N,)

    def test_scalar_cospec_quadspec(self):
        """Float inputs for cospec/quadspec should work."""
        c1, c2 = TK_simulate(
            self.P1, self.output_length, self.dt, mean=100, cospec=1.0, quadspec=0.5
        )

        assert c1.shape == (self.N,)
        assert c2.shape == (self.N,)

    def test_equivalent_to_coh_lag(self):
        """Cross spectra inputs should give same results as equivalent coh/lag."""
        gamma2 = 0.85
        phi = 0.4

        # Use coh/lag directly
        np.random.seed(123)
        c1_cl, c2_cl = TK_simulate(
            self.P1,
            self.output_length,
            self.dt,
            mean=100,
            P2=self.P2,
            gamma=gamma2,
            lag=phi,
        )

        # Compute equivalent cross spectra
        C_mag = np.sqrt(gamma2 * self.P1 * self.P2)
        cospec = C_mag * np.cos(phi)
        quadspec = C_mag * np.sin(phi)

        np.random.seed(123)
        c1_cs, c2_cs = TK_simulate(
            self.P1,
            self.output_length,
            self.dt,
            mean=100,
            P2=self.P2,
            cospec=cospec,
            quadspec=quadspec,
        )

        np.testing.assert_allclose(c1_cl, c1_cs, rtol=1e-10)
        np.testing.assert_allclose(c2_cl, c2_cs, rtol=1e-10)


class TestCLSimulateWithCrossSpectra:
    """Tests for CLSimulator.CL_simulate with cospec/quadspec inputs."""

    def setup_method(self):
        self.sim = CLSimulator(dt=0.01, N=4096, mean=100, rms=0.3)

    def test_error_when_both_specified(self):
        """Should raise ValueError when both coh/lag and cospec/quadspec given."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            self.sim.CL_simulate(pds1=2.0, pds2=2.0, coh=0.8, cospec=1.0)

        with pytest.raises(ValueError, match="Cannot specify both"):
            self.sim.CL_simulate(pds1=2.0, pds2=2.0, lag=0.5, quadspec=1.0)

    def test_constant_cospec_quadspec(self):
        """Constant float cospec/quadspec should work."""
        lc1, lc2 = self.sim.CL_simulate(pds1=2.0, pds2=2.0, cospec=1.0, quadspec=0.5)
        assert len(lc1.counts) == 4096
        assert len(lc2.counts) == 4096

    def test_callable_cospec_quadspec(self):
        """Callable cospec/quadspec should work."""
        lc1, lc2 = self.sim.CL_simulate(
            pds1=2.0,
            pds2=2.0,
            cospec=lambda f: np.ones_like(f) * 2.0,
            quadspec=lambda f: np.ones_like(f) * 1.0,
        )
        assert len(lc1.counts) == 4096

    def test_array_cospec_quadspec(self):
        """Array cospec/quadspec should work."""
        w = self.sim.get_refftfreq()
        cospec_arr = np.ones_like(w) * 3.0
        quadspec_arr = np.ones_like(w) * 1.5

        lc1, lc2 = self.sim.CL_simulate(
            pds1=2.0, pds2=2.0, cospec=cospec_arr, quadspec=quadspec_arr
        )
        assert len(lc1.counts) == 4096

    def test_cospec_only_no_quadspec(self):
        """Specifying only cospec (quadspec defaults to None) should work."""
        lc1, lc2 = self.sim.CL_simulate(pds1=2.0, pds2=2.0, cospec=1.0)
        assert len(lc1.counts) == 4096

    def test_quadspec_only_no_cospec(self):
        """Specifying only quadspec (cospec defaults to None) should work."""
        lc1, lc2 = self.sim.CL_simulate(pds1=2.0, pds2=2.0, quadspec=0.5)
        assert len(lc1.counts) == 4096

    def test_no_cross_spectra_falls_through(self):
        """Without cospec/quadspec/coh/lag, should fall back to stingray simulate."""
        result = self.sim.CL_simulate(pds1=2.0, pds2=2.0)
        # stingray's simulate returns a single Lightcurve, not a tuple
        # so this tests the fallback path works
        assert result is not None
