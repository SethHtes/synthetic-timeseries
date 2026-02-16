# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for simulating synthetic X-ray time series (light curves) with specified coherence and phase lag properties. The code implements the method described in Larner, Nowak, & Wilms (2026) for generating correlated time series using the Timmer-Koenig algorithm.

## Dependencies

The codebase relies on:
- **stingray**: Astronomy time series analysis library (provides `Simulator`, `Lightcurve`, `AveragedCrossspectrum`, `AveragedPowerspectrum`)
- **numpy**: Numerical computing
- **astropy**: Astronomy modeling tools
- **typing_extensions**: Type hints
- **tqdm**: Progress bars

## Architecture

### Core Module

**simulations.py**
- Mathematical core implementing Equations 9, 10, 12, and 15 from Larner, Nowak, & Wilms (2026)
- Key functions:
  - `compute_transfer_function()`: Implements Eq. 15 - computes complex transfer function T from coherence (γ²), phase lag (φ), and power spectra
  - `compute_normalization_constant()`: Implements Eq. 12 - computes K for the incoherent component
  - `TK_simulate()`: Unified simulation function implementing the full method with arbitrary coherence and phase lag. Defaults to perfect coherence (γ²=1) and zero phase lag (φ=0). Replaces the former `TK_coh_and_lag`, `TK_coherence`, and `TK_phaselag` functions (which are kept as deprecated wrappers).
  - `compute_theoretical_std()`: Uses Parseval's theorem to compute deterministic standard deviation from PSD, avoiding empirical normalization that breaks cross-spectral phase relationships
- `CLSimulator` class: High-level interface extending `stingray.simulator.Simulator` with flexible input parsing

## Key Implementation Details

### Coherence Simulation Method

The method generates two correlated Fourier transforms X and Y:
1. Reference transform: X = sqrt(P_X/2) * (A_r + i*B_r)  [Eq. 9]
2. Transfer function: T = sqrt(P_Y*γ²/P_X) * exp(i*φ)  [Eq. 15]
3. Normalization: K = sqrt((P_Y - P_X*|T|²)/2)  [Eq. 12]
4. Dependent transform: Y = K*(H_r + i*J_r) + T*X  [Eq. 10]

where A_r, B_r, H_r, J_r are independent standard normal random variables.

### Important Normalization Note

The code uses **theoretical standard deviation** (from `compute_theoretical_std()`) rather than empirical std when normalizing time series in coherence/lag simulations. This is critical: using empirical std introduces data-dependent normalization that breaks cross-spectral phase relationships and causes coherence to drop below target when phase lag is nonzero.

### Deprecated Functions

- `gamma_to_T()`: Old transfer function formula, kept for backward compatibility only. Use `compute_transfer_function()` instead.
- `TK_coh_and_lag()`, `TK_coherence()`, `TK_phaselag()`: Replaced by the unified `TK_simulate()`. Kept as thin wrappers for backward compatibility.

## Running Code

This is a research library meant to be imported and used interactively (likely in Jupyter notebooks or IPython):

```python
from synthetic_timeseries import CLSimulator, TK_simulate
import numpy as np

# Example: Create simulator
sim = CLSimulator(dt=0.01, N=10000, mean=100, rms=0.3)

# Simulate with constant coherence and lag
lc1, lc2 = sim.CL_simulate(pds1=2.0, pds2=2.0, lag=0.5, coh=0.7)

# Analyze with Stingray
from stingray import AveragedCrossspectrum
cs = AveragedCrossspectrum.from_lightcurve(lc1, lc2, segment_size=10.0)
lag, lag_err = cs.phase_lag()
```

## Known Limitations

- Red noise > 1 not fully implemented for coherence/lag methods (raises `NotImplementedError`)
- Segment size handling is experimental and has known issues
- The code contains debug statements and TODOs (see line 1 of simulations.py)
