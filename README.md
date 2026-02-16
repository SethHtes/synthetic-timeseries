# Synthetic Time Series with Coherence and Phase Lag

A Python library for simulating synthetic X-ray time series (light curves) with specified coherence and phase lag properties, implementing the method from Larner, Nowak, & Wilms (2026).

## Installation

```bash
# Clone from GitHub
git clone https://github.com/sethlarner/synthetic-timeseries.git
cd synthetic-timeseries

# Install in editable mode
pip install -e .
```

## Quick Start

```python
from synthetic_timeseries import CLSimulator
import numpy as np

# Create a simulator
sim = CLSimulator(dt=0.01, N=10000, mean=100, rms=0.3)

# Simulate two correlated light curves with coherence and phase lag
lc1, lc2 = sim.CL_simulate(pds1=2.0, pds2=2.0, lag=0.5, coh=0.7)

# lc1 and lc2 are stingray.Lightcurve objects
print(f"Light curve 1: {lc1.time}, {lc1.counts}")
print(f"Light curve 2: {lc2.time}, {lc2.counts}")
```

## Core Components

### 1. CLSimulator Class

The `CLSimulator` class is the high-level interface for simulating correlated time series.

#### Initialization Parameters

```python
CLSimulator(dt=1, N=1024, mean=0, rms=1, red_noise=1, random_state=None, poisson=False)
```

- **dt** (float): Time resolution in seconds. Default: 1
- **N** (int): Number of time bins in the output light curve. Default: 1024
- **mean** (float): Mean count rate in counts/sec. Default: 0
- **rms** (float or tuple): Fractional RMS variability. Can be:
  - float: same RMS for both light curves
  - tuple (rms1, rms2): different RMS for each light curve
- **red_noise** (int): Red noise multiplier (must be 1 for coherence/lag simulations). Default: 1
- **random_state** (int): Random seed for reproducibility. Default: None
- **poisson** (bool): Apply Poisson noise to final light curves. Default: False

#### Main Method: CL_simulate()

```python
lc1, lc2 = sim.CL_simulate(pds1, pds2, lag=None, coh=None, segment_size=None)
```

**Parameters:**

- **pds1**: Power spectrum for the reference time series. Can be:
  - `float`: Power law index (e.g., 2.0 for P ∝ f⁻²)
  - `np.ndarray`: Pre-computed power spectrum array
  - `callable`: Function f(frequency) → power
  - `astropy.modeling.Model`: Astropy model object

- **pds2**: Power spectrum for the dependent time series (same format as pds1). If None, uses pds1.

- **lag**: Phase lag spectrum in radians. Can be:
  - `None`: No phase lag (default)
  - `float`: Constant phase lag (range: -π to π)
  - `np.ndarray`: Frequency-dependent phase lag
  - `callable`: Function f(frequency) → lag

- **coh**: Coherence spectrum (γ²). Can be:
  - `None`: No coherence constraint (default)
  - `float`: Constant coherence (range: 0 to 1)
  - `np.ndarray`: Frequency-dependent coherence
  - `callable`: Function f(frequency) → coherence

- **segment_size** (int): Segment size for experimental segmented simulation. Default: None

**Returns:**
- Tuple of two `stingray.Lightcurve` objects (lc1, lc2)

## Usage Examples

### Example 1: Simple Power Law with Constant Coherence

```python
from synthetic_timeseries import CLSimulator
import numpy as np

# Initialize simulator
# 0.01 second time bins, 100 seconds total, mean=100 cts/s, 30% fractional RMS
sim = CLSimulator(dt=0.01, N=10000, mean=100, rms=0.3)

# Simulate with power law index 2.0, coherence 0.8, no phase lag
lc1, lc2 = sim.CL_simulate(pds1=2.0, pds2=2.0, coh=0.8)

# Access the data (lc1 and lc2 are stingray.Lightcurve objects)
print(f"Time: {lc1.time}")
print(f"Counts: {lc1.counts}")
print(f"Count rate: {lc1.countrate}")
```

### Example 2: Frequency-Dependent Coherence

```python
# Create a coherence spectrum that varies with frequency
def coherence_function(freq):
    # High coherence at low frequencies, decreasing at high frequencies
    return 0.9 * np.exp(-freq / 10.0) + 0.1

sim = CLSimulator(dt=0.01, N=10000, mean=100, rms=0.3)
lc1, lc2 = sim.CL_simulate(pds1=2.0, pds2=2.0, coh=coherence_function)
```

### Example 3: Phase Lag Only (Perfect Coherence)

```python
# Simulate with constant phase lag but perfect coherence
sim = CLSimulator(dt=0.01, N=10000, mean=100, rms=0.3)

# 0.5 radians constant phase lag
lc1, lc2 = sim.CL_simulate(pds1=2.0, lag=0.5)
```

### Example 4: Frequency-Dependent Phase Lag

```python
# Phase lag that increases with frequency (time delay)
def lag_function(freq):
    time_delay = 0.1  # seconds
    return 2 * np.pi * freq * time_delay

sim = CLSimulator(dt=0.01, N=10000, mean=100, rms=0.3)
lc1, lc2 = sim.CL_simulate(pds1=2.0, lag=lag_function, coh=0.8)
```

### Example 5: Both Coherence and Phase Lag

```python
# Combine frequency-dependent coherence and phase lag
def coherence_func(freq):
    return 0.9 * np.exp(-freq / 5.0) + 0.1

def lag_func(freq):
    return np.where(freq < 1, 0.5, -0.3)  # Different lags at different frequencies

sim = CLSimulator(dt=0.01, N=10000, mean=100, rms=0.3)
lc1, lc2 = sim.CL_simulate(pds1=2.0, pds2=2.0, lag=lag_func, coh=coherence_func)
```

### Example 6: Different Power Spectra for Each Band

```python
# Simulate two energy bands with different power spectral shapes
sim = CLSimulator(dt=0.01, N=10000, mean=100, rms=0.3)

# Soft band: steeper power law (index 2.5)
# Hard band: flatter power law (index 1.5)
lc1, lc2 = sim.CL_simulate(pds1=2.5, pds2=1.5, lag=0.3, coh=0.7)
```

### Example 7: Different RMS for Each Band

```python
# Different fractional RMS in each energy band
sim = CLSimulator(dt=0.01, N=10000, mean=100, rms=(0.3, 0.2))

# Soft band has 30% RMS, hard band has 20% RMS
lc1, lc2 = sim.CL_simulate(pds1=2.0, pds2=2.0, lag=0.5, coh=0.8)
```

### Example 8: Using Pre-computed Power Spectrum

```python
# Get the frequency array
sim = CLSimulator(dt=0.01, N=10000, mean=100, rms=0.3)
freq = sim.get_refftfreq()

# Create custom power spectrum
# Broken power law: flat below 1 Hz, steep above
power_spectrum = np.where(freq < 1, 1.0, (freq)**(-2.5))

lc1, lc2 = sim.CL_simulate(pds1=power_spectrum, pds2=power_spectrum,
                           lag=0.5, coh=0.8)
```

### Example 9: Adding Poisson Noise

```python
# Simulate with Poisson noise for realistic photon counting statistics
sim = CLSimulator(dt=0.01, N=10000, mean=100, rms=0.3, poisson=True)
lc1, lc2 = sim.CL_simulate(pds1=2.0, pds2=2.0, lag=0.5, coh=0.8)
```

## Low-Level Functions

For more control, you can use the underlying functions directly:

### TK_coh_and_lag()

Most general function supporting arbitrary coherence and phase lag.

```python
from simulations import TK_coh_and_lag, make_powerlaw_pds
import numpy as np

# Create power spectrum
dt = 0.01
N = 10000
P1 = make_powerlaw_pds(index=2.0, dt=dt, bins=N)

# Create coherence and lag spectra
freq = np.fft.rfftfreq(N, d=dt)[1:]
gamma = np.ones_like(P1) * 0.8  # Constant coherence
lag = np.ones_like(P1) * 0.5    # Constant phase lag

# Simulate
counts1, counts2 = TK_coh_and_lag(
    P1=P1,
    P2=P1,
    output_length=N*dt,
    dt=dt,
    mean=100,
    lag=lag,
    gamma=gamma,
    rms=0.3
)
```

### TK_coherence()

Special case with coherence but no phase lag (φ=0).

```python
from simulations import TK_coherence, make_powerlaw_pds

P1 = make_powerlaw_pds(index=2.0, dt=0.01, bins=10000)

counts1, counts2 = TK_coherence(
    P1=P1,
    output_length=100,
    dt=0.01,
    mean=100,
    gamma=0.8,  # Can be float or array
    rms=0.3
)
```

### TK_phaselag()

Special case with phase lag but perfect coherence (γ²=1).

```python
from simulations import TK_phaselag, make_powerlaw_pds

pds = make_powerlaw_pds(index=2.0, dt=0.01, bins=10000)

counts1, counts2 = TK_phaselag(
    pds=pds,
    output_length=100,
    dt=0.01,
    mean=100,
    lag=0.5,  # Can be float or array
    rms=0.3
)
```

## Analyzing Results

The simulated light curves are `stingray.Lightcurve` objects. You can analyze them using Stingray's tools:

```python
from stingray import AveragedCrossspectrum

# Create cross-spectrum
cs = AveragedCrossspectrum.from_lightcurve(lc1, lc2, segment_size=10.0)

# Calculate phase lag and coherence
lag, lag_err = cs.phase_lag()
coherence = cs.coherence()

print(f"Phase lag: {lag}")
print(f"Coherence: {coherence}")
```

## Understanding the Output

### Coherence (γ²)
- Range: 0 to 1
- γ² = 0: No correlation between time series
- γ² = 1: Perfect correlation
- Measures the fraction of power in lc2 that is linearly correlated with lc1

### Phase Lag (φ)
- Range: -π to π radians
- φ > 0: lc2 lags behind lc1 (lc1 leads)
- φ < 0: lc2 leads lc1
- φ = 0: No time delay (in phase)
- Related to time delay: Δt = φ / (2πf) at frequency f

### Power Spectrum
- Describes the power distribution across frequencies
- Units depend on normalization (typically rms²/Hz)
- Power law index α: P(f) ∝ f⁻ᵅ
  - α = 0: White noise
  - α = 1: Pink/flicker noise
  - α = 2: Red/Brownian noise

## Technical Notes

### Important Implementation Details

1. **Coherence is γ² (squared)**: The `coh` and `gamma` parameters expect coherence squared, not coherence amplitude.

2. **Theoretical vs Empirical Normalization**: The code uses theoretical standard deviation (computed from the power spectrum via Parseval's theorem) rather than empirical std. This is critical for preserving cross-spectral phase relationships when coherence < 1.

3. **Red Noise Limitation**: The `red_noise` parameter must equal 1 for coherence/lag simulations (not yet implemented for red_noise > 1).

4. **Frequency Array**: Use `sim.get_refftfreq()` to get the frequency array corresponding to the power spectrum.

### Mathematical Background

The method implements Equations 9, 10, 12, and 15 from Larner, Nowak, & Wilms (2026):

1. Reference transform: X = √(P_X/2) × (A_r + iB_r)
2. Transfer function: T = √(P_Y×γ²/P_X) × exp(iφ)
3. Normalization: K = √((P_Y - P_X×|T|²)/2)
4. Dependent transform: Y = K×(H_r + iJ_r) + T×X

where A_r, B_r, H_r, J_r are independent standard normal random variables.

## Common Issues

### Issue: Coherence drops below target when phase lag is present
**Solution**: This is fixed in the current implementation using theoretical normalization. If you see this, ensure you're using the latest version of the code.

### Issue: Red noise simulation fails
**Error**: `NotImplementedError: Red noise > 1 is not implemented`
**Solution**: Set `red_noise=1` in CLSimulator initialization.

### Issue: Irregular results with segment_size
**Note**: The `segment_size` parameter is experimental and has known issues. Use with caution.

## Citation

If you use this package in published research, please cite:

```bibtex
@article{larner2026,
  author = {Larner, Seth R. and Nowak, Michael A. and Wilms, J\"orn},
  title = {Simulating Synthetic Time Series with Coherence and Phase Lag},
  journal = {In preparation},
  year = {2026}
}
```

## References

Larner, S. R., Nowak, M. A., & Wilms, J. 2026 (in prep)

Timmer, J., & Koenig, M. 1995, A&A, 300, 707

Uttley, P., et al. 2014, A&ARv, 22, 72

## Example: X-ray Astronomy Soft vs Hard Band

```python
from synthetic_timeseries import CLSimulator
import numpy as np

# Simulate soft and hard X-ray bands
sim = CLSimulator(dt=0.01, N=20000, mean=100, rms=0.3)

def lag_spec(freq):
    # Soft leads hard: negative lag at low freq
    return -0.1 * np.exp(-freq / 2.0)

def coh_spec(freq):
    # High coherence at low freq
    return 0.95 * np.exp(-freq / 10.0) + 0.05

lc_soft, lc_hard = sim.CL_simulate(pds1=2.0, pds2=1.8,
                                   lag=lag_spec, coh=coh_spec)

# Analyze with Stingray
from stingray import AveragedCrossspectrum
cs = AveragedCrossspectrum.from_lightcurve(lc_soft, lc_hard, segment_size=10.0)
lag, lag_err = cs.phase_lag()
print(f"Measured phase lag: {lag}")
```

## License

This code is research software. Please cite Larner, Nowak, & Wilms (2026) if you use it in published work.
