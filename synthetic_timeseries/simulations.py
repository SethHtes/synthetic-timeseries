import numpy as np
from typing import Iterable, Tuple, Optional, Union, Callable, Literal
from stingray import utils
from stingray.simulator import Simulator
from astropy.modeling import Model
from stingray import Lightcurve, AveragedCrossspectrum


def invert_fft(x: Iterable[complex], mean: float, nbins: int) -> np.ndarray:

    f = np.hstack([mean * nbins, x])

    return np.fft.irfft(f, n=nbins)


def compute_transfer_function(
    gamma2: Union[float, np.ndarray[float]],
    phi: Union[float, np.ndarray[float]],
    P_X: Union[float, np.ndarray[float]],
    P_Y: Union[float, np.ndarray[float]],
) -> Union[complex, np.ndarray[complex]]:
    """Compute the transfer function T from coherence and phase lag.

    Implements Equation 15 from Larner, Nowak, & Wilms (2026):
        T = sqrt(P_Y * γ² / P_X) * exp(i*φ)

    Parameters:
    -----------
    gamma2 : float | np.ndarray
        Coherence (squared) value or array. Range: [0, 1]
    phi : float | np.ndarray
        Phase lag in radians. Range: [-π, π]
    P_X : float | np.ndarray
        Power spectrum of the reference time series
    P_Y : float | np.ndarray
        Power spectrum of the dependent time series

    Returns:
    --------
    T : complex | np.ndarray[complex]
        Complex transfer function relating the two time series in Fourier space

    Notes:
    ------
    The transfer function encodes both the coherence (in its magnitude) and
    phase lag (in its argument). This formulation ensures proper normalization
    of the power spectra.

    References:
    -----------
    Larner, S. R., Nowak, M. A., & Wilms, J. 2026 (in prep)
    """
    magnitude = np.sqrt(P_Y * gamma2 / P_X)
    return magnitude * np.exp(1j * phi)


def compute_normalization_constant(
    P_X: Union[float, np.ndarray[float]],
    P_Y: Union[float, np.ndarray[float]],
    T: Union[complex, np.ndarray[complex]],
) -> Union[float, np.ndarray[float]]:
    """Compute the normalization constant K for the incoherent component.

    Implements Equation 12 from Larner, Nowak, & Wilms (2026):
        K = sqrt((P_Y - P_X*|T|²) / 2)

    Parameters:
    -----------
    P_X : float | np.ndarray
        Power spectrum of the reference time series
    P_Y : float | np.ndarray
        Power spectrum of the dependent time series
    T : complex | np.ndarray[complex]
        Transfer function (from compute_transfer_function)

    Returns:
    --------
    K : float | np.ndarray
        Normalization constant for the incoherent component

    Notes:
    ------
    This constant ensures that the dependent time series Y has the correct
    power spectrum P_Y. The factor of 2 accounts for the variance of complex
    Gaussian random variables.

    References:
    -----------
    Larner, S. R., Nowak, M. A., & Wilms, J. 2026 (in prep)
    """
    T_mag_squared = np.abs(T) ** 2
    # Clamp to zero to avoid NaN from floating-point rounding when gamma=1
    return np.sqrt(np.maximum(P_Y - P_X * T_mag_squared, 0.0) / 2)


def cross_spectra_to_coh_lag(
    cospec: Union[float, np.ndarray],
    quadspec: Union[float, np.ndarray],
    P_X: Union[float, np.ndarray],
    P_Y: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert co-spectrum and quadrature spectrum to coherence and phase lag.

    Parameters:
    -----------
    cospec : float | np.ndarray
        Real part of the cross spectrum, Re[C]
    quadspec : float | np.ndarray
        Imaginary part of the cross spectrum, Im[C]
    P_X : float | np.ndarray
        Power spectrum of the reference time series
    P_Y : float | np.ndarray
        Power spectrum of the dependent time series

    Returns:
    --------
    gamma2 : np.ndarray
        Coherence squared, γ² = (Re[C]² + Im[C]²) / (P_X * P_Y)
    phi : np.ndarray
        Phase lag in radians, φ = arctan2(Im[C], Re[C])
    """
    gamma2 = (cospec**2 + quadspec**2) / (P_X * P_Y)
    phi = np.arctan2(quadspec, cospec)
    return gamma2, phi


def make_powerlaw_pds(index: float, dt: float, bins: int) -> np.ndarray[float]:
    w = np.fft.rfftfreq(bins, d=dt)[1:]

    p = np.power((1 / w), index)

    return p


def compute_theoretical_std(P: np.ndarray, N: int) -> float:
    """Compute the theoretical standard deviation of a time series from its PSD.

    Given a one-sided power spectrum P (at positive frequencies, excluding DC),
    compute the expected standard deviation of the corresponding real-valued
    time series produced by np.fft.irfft with n=N.

    This uses Parseval's theorem. For N even, the Nyquist bin (P[-1]) gets
    weight 1 while all other positive-frequency bins get weight 2:
        E[Var] = (2*sum(P) - P[-1]) / N^2

    For N odd, there is no Nyquist bin, so all bins get weight 2:
        E[Var] = 2*sum(P) / N^2

    Parameters
    ----------
    P : np.ndarray
        One-sided power spectrum at positive frequencies (rfftfreq[1:]).
        Length should be N//2 for even N, or (N-1)//2 for odd N.
    N : int
        Length of the time series (number of time-domain samples).
        This is the `n` parameter passed to np.fft.irfft.

    Returns
    -------
    std : float
        Theoretical (deterministic) standard deviation.

    Notes
    -----
    Using this instead of the empirical std avoids a nonlinear,
    data-dependent normalization that breaks cross-spectral phase
    relationships and causes coherence to drop below target when
    the phase lag is nonzero.

    References
    ----------
    Larner, S. R., Nowak, M. A., & Wilms, J. 2026 (in prep)
    """
    if N % 2 == 0:
        # Even N: Nyquist bin exists at index N//2, gets weight 1
        theoretical_var = (2.0 * np.sum(P) - P[-1]) / (N * N)
    else:
        # Odd N: no Nyquist bin, all positive-frequency bins get weight 2
        theoretical_var = 2.0 * np.sum(P) / (N * N)

    return np.sqrt(theoretical_var)


def extract_and_scale(
    long_lc: np.ndarray,
    N: int,
    mean: float,
    red_noise: Optional[int] = 1,
    random_state: Optional[int] = None,
    rms: Optional[float] = 1.0,
    theoretical_std: Optional[float] = None,
):
    """
    i) Make a random cut and extract a light curve of required
    length.

    ii) Rescale light curve i) with zero mean and unit standard
    deviation, and ii) user provided mean and rms (fractional
    rms * mean)

    Parameters
    ----------
    long_lc : numpy.ndarray
        Simulated lightcurve of length 'N' times 'red_noise'
    N : int, default 1024
        bins count of simulated light curve
    red_noise : int, default 1
        multiple of real length of light curve, by
        which to simulate, to avoid red noise leakage
    random_state : int, default None
        seed value for random processes
    rms : float, default 1
        fractional rms of the simulated light curve,
        actual rms is calculated by mean*rms
    theoretical_std : float, optional
        If provided, use this theoretical standard deviation instead of
        computing empirical std from the light curve. This preserves
        cross-spectral phase relationships in coherence simulations.
    Returns
    -------
    lc : numpy.ndarray
        Normalized and extracted lightcurve of length 'N'
    """

    random_state = utils.get_random_state(random_state)

    empirical_std = long_lc.std()
    if theoretical_std is not None:
        std = theoretical_std
    else:
        std = empirical_std

    if red_noise == 1:
        lc = long_lc
    else:
        extract = random_state.randint(0, red_noise * N - N + 1)
        lc = np.take(long_lc, range(extract, extract + N))

    mean_lc = np.mean(lc)

    use_target_mean = theoretical_std is not None

    if use_target_mean and mean != 0:
        # Use target mean for centering (deterministic)
        result = (lc - mean) / std * mean * rms + mean
        return result
    else:
        # Original behavior: use empirical mean
        if mean == 0:
            return (lc - mean_lc) / std * rms
        else:
            return (lc - mean_lc) / std * mean * rms + mean


def TK_simulate(
    P1: np.ndarray,
    output_length: float,
    dt: float,
    mean: Union[float, Tuple[float, float]],
    lag: Optional[Union[float, np.ndarray[float]]] = None,
    gamma: Optional[Union[float, np.ndarray[float]]] = None,
    red_noise: Optional[int] = 1,
    rms: Optional[Union[float, Tuple[float, float]]] = 0.1,
    P2: Optional[np.ndarray] = None,
    poisson: bool = False,
    cospec: Optional[Union[float, np.ndarray[float]]] = None,
    quadspec: Optional[Union[float, np.ndarray[float]]] = None,
) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    """Simulate two correlated time series with arbitrary coherence and phase lag.

    Implements the method from Larner, Nowak, & Wilms (2026) using Equations 9, 10,
    12, and 15. The reference time series is generated using the Timmer-Koenig method,
    and the dependent time series is constructed as a weighted sum of a coherent
    component (derived from the reference via a transfer function) and an incoherent
    component.

    This is the unified simulation function that handles all cases:
    - Coherence only (lag omitted or 0): equivalent to the former TK_coherence
    - Phase lag only (gamma omitted or 1): equivalent to the former TK_phaselag
    - Both coherence and phase lag: equivalent to the former TK_coh_and_lag

    Parameters:
    -----------
    P1 : np.ndarray
        Power spectrum array of the reference time series (P_X in paper notation)
    output_length : float
        Length of the output lightcurve in seconds
    dt : float
        Time resolution of the output lightcurve in seconds
    mean : float or tuple of float
        Mean count rate of the output lightcurves in counts/sec. If tuple (mean1, mean2),
        different means for each band.
    lag : float or np.ndarray, optional
        Phase lag spectrum in radians (φ in paper notation). Range: [-π, π]
        - float: constant phase lag across all frequencies
        - np.ndarray: frequency-dependent phase lag spectrum (same length as P1)
        Default: 0 (no phase lag)
    gamma : float or np.ndarray, optional
        Coherence spectrum (γ² in paper notation). Range: [0, 1]
        Note: This is coherence SQUARED (γ²), not the coherence amplitude γ
        - float: constant coherence across all frequencies
        - np.ndarray: frequency-dependent coherence spectrum (same length as P1)
        Default: 1.0 (perfect coherence)
    red_noise : int, optional
        Red noise factor (default: 1). Values > 1 not fully implemented
    rms : float or tuple of float, optional
        Fractional RMS variability. If float, same RMS for both bands.
        If tuple (rms1, rms2), different RMS for each band. Default: 0.1
    P2 : np.ndarray, optional
        Power spectrum of dependent time series (P_Y in paper notation).
        If None, P2 = P1. Must have same shape as P1 if provided.
    poisson : bool, optional
        If True, draw final count arrays from Poisson distribution. Default: False
    cospec : float or np.ndarray, optional
        Real part of the cross spectrum (co-spectrum), Re[C].
        - float: constant across all frequencies
        - np.ndarray: frequency-dependent (same length as P1)
        Cannot be specified together with gamma/lag.
    quadspec : float or np.ndarray, optional
        Imaginary part of the cross spectrum (quadrature spectrum), Im[C].
        - float: constant across all frequencies
        - np.ndarray: frequency-dependent (same length as P1)
        Cannot be specified together with gamma/lag.

    Returns:
    --------
    counts1 : np.ndarray
        First simulated time series (reference)
    counts2 : np.ndarray
        Second simulated time series (dependent, correlated with first)

    Raises:
    -------
    ValueError
        If P1 and P2 have different shapes

    Notes:
    ------
    The method follows these steps for each Fourier frequency:
    1. Generate reference Fourier transform: X = sqrt(P_X/2) * (A_r + i*B_r)  [Eq. 9]
    2. Compute transfer function: T = sqrt(P_Y*γ²/P_X) * exp(i*φ)  [Eq. 15]
    3. Compute normalization: K = sqrt((P_Y - P_X*|T|²)/2)  [Eq. 12]
    4. Generate dependent transform: Y = K*(H_r + i*J_r) + T*X  [Eq. 10]

    where A_r, B_r, H_r, J_r are independent standard normal random variables.

    This formulation ensures:
    - Both time series have the correct power spectra (P_X and P_Y)
    - The coherence between them is γ² at each frequency
    - The phase lag between them is φ at each frequency

    When γ² = 1, the incoherent component vanishes (K = 0) and Y = T*X, giving
    perfect coherence with the specified phase lag.

    References:
    -----------
    Larner, S. R., Nowak, M. A., & Wilms, J. 2026 (in prep)
    Timmer, J., & Koenig, M. 1995, A&A, 300, 707
    """

    if P2 is None:
        P2 = P1
    elif P1.shape != P2.shape:
        raise ValueError("Both power spectra must have the same shape!")

    # Validate that cospec/quadspec and gamma/lag are not both specified
    use_cross_spectra = cospec is not None or quadspec is not None
    use_coh_lag = gamma is not None or lag is not None
    if use_cross_spectra and use_coh_lag:
        raise ValueError(
            "Cannot specify both cospec/quadspec and gamma/lag. "
            "Use one pair or the other."
        )

    # Convert cospec/quadspec to gamma and lag
    if use_cross_spectra:
        pds_size = P1.size
        if cospec is None:
            cospec = np.zeros(pds_size)
        elif isinstance(cospec, (float, int)):
            cospec = np.ones(pds_size) * cospec
        if quadspec is None:
            quadspec = np.zeros(pds_size)
        elif isinstance(quadspec, (float, int)):
            quadspec = np.ones(pds_size) * quadspec
        gamma, lag = cross_spectra_to_coh_lag(cospec, quadspec, P1, P2)

    pds_size = P1.size

    # Default: no phase lag
    if lag is None:
        lag = np.zeros(pds_size)
    elif isinstance(lag, (float, int)):
        lag = np.ones(pds_size) * lag

    # Default: perfect coherence
    if gamma is None:
        gamma = np.ones(pds_size)
    elif isinstance(gamma, (float, int)):
        gamma = np.ones(pds_size) * gamma

    randint = np.random.randint(low=0, high=10000)

    N = int(output_length / dt)
    long_N = int(output_length * red_noise / dt)

    # Generate random variables (one set per frequency)
    Ar = np.random.normal(size=pds_size)
    Br = np.random.normal(size=pds_size)
    Hr = np.random.normal(size=pds_size)
    Jr = np.random.normal(size=pds_size)

    # Equation 9: Reference Fourier transform
    # X = sqrt(P_X/2) * (A_r + i*B_r)
    X = np.sqrt(P1 / 2) * (Ar + 1j * Br)

    # Equation 15: Transfer function (includes phase lag)
    # T = sqrt(P_Y * γ² / P_X) * exp(i*φ)
    T = compute_transfer_function(gamma2=gamma, phi=lag, P_X=P1, P_Y=P2)

    # Equation 12: Normalization constant for incoherent component
    # K = sqrt((P_Y - P_X*|T|²) / 2)
    K = compute_normalization_constant(P_X=P1, P_Y=P2, T=T)

    # Equation 10: Dependent Fourier transform
    # Y = K*(H_r + i*J_r) + T*X
    Y = K * (Hr + 1j * Jr) + T * X

    # Handle separate RMS for each band if provided as tuple
    if isinstance(rms, tuple):
        rms1, rms2 = rms
    else:
        rms1 = rms2 = rms

    # Compute theoretical std from PSDs (deterministic, preserves coherence)
    theoretical_std_X = compute_theoretical_std(P1, long_N)
    theoretical_std_Y = compute_theoretical_std(P2, long_N)

    # Inverse FFT to get time series
    counts1 = invert_fft(X, mean=mean, nbins=long_N)
    counts1 = extract_and_scale(
        long_lc=counts1,
        N=N,
        red_noise=red_noise,
        rms=rms1,
        random_state=randint,
        mean=mean,
        theoretical_std=theoretical_std_X,
    )

    counts2 = invert_fft(Y, mean=mean, nbins=long_N)
    counts2 = extract_and_scale(
        long_lc=counts2,
        N=N,
        red_noise=red_noise,
        rms=rms2,
        random_state=randint,
        mean=mean,
        theoretical_std=theoretical_std_Y,
    )

    if poisson:
        counts1 = np.random.poisson(counts1)
        counts2 = np.random.poisson(counts2)

    return (counts1, counts2)


class CLSimulator(Simulator):
    """
    Methods to simulate and visualize light curves with arbitrary coherence
    and phase lag distributions.

    Built on top of the `stingray.simulator.Simulator` implementation.

    Parameters
    ----------
    dt : int, default 1
        time resolution of simulated light curve
    N : int, default 1024
        bins count of simulated light curve
    mean : float, default 0
        mean value of the simulated light curve
    rms : float, default 1
        Fractional root mean square of the output lightcurves. If float, same
        RMS is used for both bands. If tuple (rms1, rms2), different RMS for
        each band. Default is 1.
    err : float, default 0
        the errorbars on the final light curve
    red_noise : int, default 1
        multiple of real length of light curve, by
        which to simulate, to avoid red noise leakage
    random_state : int, default None
        seed value for random processes
    poisson : bool, default False
        return Poisson-distributed light curves.
    """

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except TypeError as e:
            if "'<=' not supported between instances of 'tuple' and 'int'" not in str(
                e
            ):
                raise e

    def get_refftfreq(self) -> np.ndarray:
        """
        Calculate the positive frequencies for the real-valued FFT of a signal.

        This method computes the frequencies corresponding to the positive
        half of the discrete Fourier Transform (DFT) of a signal, based on
        the parameters of the red noise and the number of samples.

        Returns:
            np.ndarray: An array of positive frequency values corresponding
            to the real FFT of the signal.
        """

        return np.fft.rfftfreq(self.red_noise * self.N, d=self.dt)[1:]

    def CL_simulate(
        self,
        pds1: Union[str, float, Model, Callable[[Iterable], Iterable]],
        pds2: Optional[Union[str, float, Model, Callable[[Iterable], Iterable]]],
        params: Optional[Union[list, dict]] = None,
        lag: Optional[
            Union[str, float, Model, Callable[[Iterable], Iterable], Iterable]
        ] = None,
        coh: Optional[
            Union[str, float, Model, Callable[[Iterable], Iterable], Iterable]
        ] = None,
        cospec: Optional[Union[float, Callable[[Iterable], Iterable], Iterable]] = None,
        quadspec: Optional[
            Union[float, Callable[[Iterable], Iterable], Iterable]
        ] = None,
    ) -> Tuple[Lightcurve, Lightcurve]:
        """Simulate two LightCurves from a power spectrum and with a specified
        phase lag and/or coherence distribution.

        Parameters:
        -----------
        pds1 (str | float| Model | Callable | Iterable)
            - Defines the shape of the power spectrum used to simulate the
            reference time series:
            - If string, model defined in `stingray.simulator.models`.
            - If float, defines the index of a power law power spectrum.
            - If `astropy.modeling.Model`, power spectrum takes shape of model.
            - If other callable, has signature f(frequency)->pds.
            - If iterable, defines the power spectrum at each frequency in self.get_rfftfreq()

        pds2 (str | float| Model | Callable | Iterable, optional)
            - Defines the shape of the power spectrum used to simulate the
            dependent time series:
            - If string, model defined in `stingray.simulator.models`.
            - If float, defines the index of a power law power spectrum.
            - If `astropy.modeling.Model`, power spectrum takes shape of model.
            - If other callable, has signature f(frequency)->pds.
            - If iterable, defines the power spectrum at each frequency in self.get_rfftfreq().
            - If not given, is set to pds1.
            - Cannot be given if simulating time series only with phase lag,
            only pds1 will be used.

        params (list | dict, optional)
            - Parameters for use in predefined model string.

        lag (str | float | Model | Callable | Iterable, optional)
            - Defines the shape of the phase lag spectrum used.
            - If ommited, no phase lag spectrum is simulated.
            - If string, model defined in `stingray.simulator.models`
            - If float, is a constant value in [-π,π]
            - If `astropy.modeling.Model`, lag spectrum takes shape of model.
            - If other callable, has signature f(frequency)->lag.
            - If iterable, must be lag at each frequency in self.get_refftfreq()

        coh (str | float | Model | Callable | Iterable, optional)
            - Defines the shape of the coherence spectrum used.
            - If ommited, no coherence spectrum is simulated.
            - If string, model defined in `stingray.simulator.models`
            - If float, is a constant value in [0,1]
            - If `astropy.modeling.Model`, coherence spectrum takes shape of model.
            - If other callable, has signature f(frequency)->coherence.
            - If iterable, must be coherence at each frequency in self.get_refftfreq()

        cospec (float | Callable | Iterable, optional)
            - Real part of the cross spectrum (co-spectrum).
            - Cannot be specified together with coh/lag.
            - If float, constant value across all frequencies.
            - If callable, has signature f(frequency)->cospec.
            - If iterable, must be value at each frequency in self.get_refftfreq()

        quadspec (float | Callable | Iterable, optional)
            - Imaginary part of the cross spectrum (quadrature spectrum).
            - Cannot be specified together with coh/lag.
            - If float, constant value across all frequencies.
            - If callable, has signature f(frequency)->quadspec.
            - If iterable, must be value at each frequency in self.get_refftfreq()

        Returns:
        --------
        (lc1, lc2):
            - Tuple of `Lightcurves` representing the input and response light
            curves, respectively.

        Raises:
        -------
        ValueError:
            - If using a model string and model or parameters cannot be parsed.

        """
        if pds2 is None:
            pds2 = pds1

        use_coh = coh is not None
        use_lag = lag is not None
        use_cross_spectra = cospec is not None or quadspec is not None

        if use_cross_spectra and (use_coh or use_lag):
            raise ValueError(
                "Cannot specify both cospec/quadspec and coh/lag. "
                "Use one pair or the other."
            )

        w = self.get_refftfreq()

        # Inspect the input and generate...
        # pds_distribution 1
        if isinstance(pds1, (float, int)):
            pds_shape1 = make_powerlaw_pds(pds1, self.dt, self.red_noise * self.N)

        elif isinstance(pds1, str):
            from stingray.simulator import models

            if not hasattr(models, pds1):
                raise ValueError("Model string not defined")

            if isinstance(params, dict):
                model = getattr(models, pds1)(**params)
                pds_shape1 = model(w)
            elif isinstance(params, list):
                model_func = getattr(models, pds1)
                pds_shape1 = model_func(w, params)
            else:
                raise ValueError("Params should be list or dictionary!")

        else:
            pds_shape1 = pds1(w)

        # pds distribution 2
        if isinstance(pds2, (float, int)):
            pds_shape2 = make_powerlaw_pds(pds2, self.dt, self.red_noise * self.N)

        elif isinstance(pds2, str):
            from stingray.simulator import models

            if not hasattr(models, pds2):
                raise ValueError("Model string not defined")

            if isinstance(params, dict):
                model = getattr(models, pds2)(**params)
                pds_shape2 = model(w)
            elif isinstance(params, list):
                model_func = getattr(models, pds2)
                pds_shape2 = model_func(w, params)
            else:
                raise ValueError("Params should be list or dictionary!")

        else:
            pds_shape2 = pds2(w)

        # Parse cospec distribution
        cospec_shape = None
        if cospec is not None:
            if isinstance(cospec, (float, int)):
                cospec_shape = np.ones_like(w) * cospec
            elif isinstance(cospec, Iterable):
                cospec_shape = cospec
            else:
                cospec_shape = cospec(w)

        # Parse quadspec distribution
        quadspec_shape = None
        if quadspec is not None:
            if isinstance(quadspec, (float, int)):
                quadspec_shape = np.ones_like(w) * quadspec
            elif isinstance(quadspec, Iterable):
                quadspec_shape = quadspec
            else:
                quadspec_shape = quadspec(w)

        # If neither lag nor coh nor cross spectra specified, fall back to stingray's simulate
        if not use_lag and not use_coh and not use_cross_spectra:
            if params is not None:
                return self.simulate(pds1, params)
            else:
                return self.simulate(pds1)

        # Parse lag distribution
        lag_shape = None
        if use_lag:
            if isinstance(lag, (float, int)):
                lag_shape = np.ones_like(w) * lag
            elif isinstance(lag, str):
                raise NotImplementedError("Model strings not implemented")
            elif isinstance(lag, Iterable):
                lag_shape = lag
            else:
                lag_shape = lag(w)

        # Parse coh distribution
        coh_shape = None
        if use_coh:
            if isinstance(coh, (float, int)):
                coh_shape = np.ones_like(w) * coh
            elif isinstance(coh, str):
                raise NotImplementedError("Model strings not implemented")
            elif isinstance(coh, Iterable):
                coh_shape = coh
            else:
                coh_shape = coh(w)

        c1, c2 = TK_simulate(
            P1=pds_shape1,
            P2=pds_shape2,
            output_length=self.N * self.dt,
            dt=self.dt,
            mean=self.mean,
            lag=lag_shape,
            gamma=coh_shape,
            red_noise=self.red_noise,
            rms=self.rms,
            poisson=self.poisson,
            cospec=cospec_shape,
            quadspec=quadspec_shape,
        )

        t = np.arange(len(c1)) * self.dt

        lc1 = Lightcurve(
            time=t, counts=c1, err=np.zeros_like(c1), dt=self.dt, skip_checks=True
        )
        lc2 = Lightcurve(
            time=t, counts=c2, err=np.zeros_like(c2), dt=self.dt, skip_checks=True
        )

        return (lc1, lc2)

    @staticmethod
    def crossspectrum(
        lc1: Lightcurve,
        lc2: Lightcurve,
        seg_size: Optional[float] = None,
        norm: Optional[Literal["frac", "abs", "leahy", "none"]] = "frac",
    ) -> np.ndarray:
        """
        Make a cross spectrum of the simulated light curves.

        Parameters
        ----------
        lc1 (lightcurve.Lightcurve object | iterable of the same):
            The reference light curve data to be Fourier-transformed.

        lc2 (lightcurve.Lightcurve | iterable of the same):
            The dependent light curve data to be Fourier-transformed.

        seg_size (float, optional):
            Segment size (in seconds).

        Returns
        -------
        power : numpy.ndarray[complex]
            The array of complex cross powers.

        Notes:
        -----
        lc1 and lc2 must be the same length.
        """

        # Following stingray convention by including this method

        if seg_size is None:
            seg_size = lc1.tseg

        return AveragedCrossspectrum(lc1, lc2, seg_size, silent=True, norm=norm).power
