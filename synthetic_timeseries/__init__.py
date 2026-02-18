"""
Synthetic Time Series with Coherence and Phase Lag

A Python library for simulating synthetic X-ray time series (light curves) with
specified coherence and phase lag properties, implementing the method from
Larner, Nowak, & Wilms (2026).
"""

__version__ = "0.1.0"
__author__ = "Seth R. Larner"
__email__ = "larner@wustl.edu"

from .simulations import (
    CLSimulator,
    TK_simulate,
    compute_transfer_function,
    compute_normalization_constant,
    compute_theoretical_std,
    cross_spectra_to_coh_lag,
    make_powerlaw_pds,
    invert_fft,
    extract_and_scale,
)

__all__ = [
    # Main simulator class
    "CLSimulator",
    # High-level simulation functions
    "TK_simulate",
    # Mathematical utilities
    "compute_transfer_function",
    "compute_normalization_constant",
    "compute_theoretical_std",
    "cross_spectra_to_coh_lag",
    "make_powerlaw_pds",
    # Low-level utilities
    "invert_fft",
    "extract_and_scale",
]
