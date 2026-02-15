# TODO: remove debug statements
from typing_extensions import deprecated
import numpy as np
from typing import Iterable, Tuple, Optional, Union, Callable, Literal
from stingray import utils 
from stingray.simulator import Simulator
from astropy.modeling import Model 
from stingray import Lightcurve, AveragedCrossspectrum
from tqdm import tqdm

def invert_fft(x:Iterable[complex],
               mean: float,
               nbins: int)->np.ndarray:
    
    f = np.hstack([mean * nbins, x])

    return np.fft.irfft(f,n=nbins)

def gamma_to_T(gamma:Union[float, np.ndarray[float]],
               P1: Union[float, np.ndarray[float]],
               P2: Union[float, np.ndarray[float]])-> Union[float, np.ndarray[float]]:
    '''DEPRECATED: Use compute_transfer_function instead.

    This function uses an old formula that does not match the published method.
    Kept for backward compatibility only.

    Parameters:
    -----------
    gamma (float | np.ndarray)
        Coherence value or array of coherence values.
    P1 (float | np.ndarray)
        Power spectrum of the first (reference) time series.
    P2 (float | np.ndarray)
        Power spectrum of the second (dependent) time series.

    Returns:
    --------
    |T| (float | np.ndarray)
        Coherence fraction or array of coherence fractions.

    Note:
    -----
    If a phase is not explicitly set by simulating a phase lag, T is assumed
    to be real.
    '''

    # TODO: Fix limitation where this function breaks if gamma = 1
    # We should be able to handle that gracefully!

    return np.sqrt(gamma * P2/(P1 - gamma * P1))


def compute_transfer_function(gamma2: Union[float, np.ndarray[float]],
                              phi: Union[float, np.ndarray[float]],
                              P_X: Union[float, np.ndarray[float]],
                              P_Y: Union[float, np.ndarray[float]]) -> Union[complex, np.ndarray[complex]]:
    '''Compute the transfer function T from coherence and phase lag.

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
    '''
    magnitude = np.sqrt(P_Y * gamma2 / P_X)
    return magnitude * np.exp(1j * phi)


def compute_normalization_constant(P_X: Union[float, np.ndarray[float]],
                                   P_Y: Union[float, np.ndarray[float]],
                                   T: Union[complex, np.ndarray[complex]]) -> Union[float, np.ndarray[float]]:
    '''Compute the normalization constant K for the incoherent component.

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
    '''
    T_mag_squared = np.abs(T)**2
    return np.sqrt((P_Y - P_X * T_mag_squared) / 2)

def make_powerlaw_pds(index:float,dt:float,bins:int,segment_size:Optional[float]=None)->np.ndarray[float]:
    w = np.fft.rfftfreq(bins, d=dt)[1:]

    p = np.power((1/w),index)

    if segment_size is not None:
        # p[np.where(w < 1/segment_size)[0]] = 0. # Makes the problem better, but does not solve it 
        pass
    
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

def extract_and_scale(long_lc:np.ndarray,
                       N:int,
                       mean: float,
                       red_noise:Optional[int] = 1,
                       random_state:Optional[int] = None,
                       rms:Optional[float] = 1.,
                       segment_size:Optional[int]=None,
                       theoretical_std:Optional[float]=None):
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
    segment_size:  float,
        optional, given in bins
    theoretical_std : float, optional
        If provided, use this theoretical standard deviation instead of
        computing empirical std from the light curve. This preserves
        cross-spectral phase relationships in coherence simulations.
    Returns
    -------
    lc : numpy.ndarray
        Normalized and extracted lightcurve of length 'N'
    """

    # if red_noise > 1 and segment_size is not None:
    #     raise NotImplementedError()

    random_state = utils.get_random_state(random_state)
    segment_size = None

    # std = long_lc.std() # maybe we need to do this segment-wise also
    empirical_std = long_lc.std()
    if theoretical_std is not None:
        std = theoretical_std
    else:
        std = empirical_std

    if segment_size is None:
        
        if red_noise == 1:
            lc = long_lc
        else:
            # Make random cut and extract light curve of length 'N'
            extract = random_state.randint(0, red_noise * N - N + 1) #this doesnt make sense? 
            lc = np.take(long_lc, range(extract, extract + N))
    
    else:
        if red_noise == 1:
            lc = long_lc
        else:
            lc = [] #this is going to make bad light curves, but I think it will be nice to know if it solves the pi/2 problem
            # This does not solve the problem and it makes bad light curves  
            Nsegments = int(np.ceil(N/segment_size))
            print('Drawing segments (this may take a minute)...')
            for i in tqdm(range(Nsegments),total=Nsegments):
                extract = random_state.randint(0,red_noise*N-N+1)
                lc_seg = np.take(long_lc,range(extract,extract + segment_size))
                
                t = np.arange(len(lc_seg)) - segment_size/2
                kernel_func = np.sqrt(2/np.pi) * np.sin(segment_size*t/2)/t
                kernel_func[np.where(np.isnan(kernel_func))[0][0]] = 1.
                lc_seg = np.convolve(lc_seg,kernel_func,mode='same')
                lc.extend(lc_seg)

            lc = np.array(lc)


    mean_lc = np.mean(lc)

    # EXPERIMENTAL: Try using target mean instead of empirical mean
    # This avoids subtracting different random variables from each time series
    use_target_mean = (theoretical_std is not None)

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

def TK_coherence(P1:np.ndarray,
                 output_length:float,
                 dt:float,
                 mean:float,
                 gamma:Union[float,np.ndarray[float]],
                 red_noise:int = 1,
                 rms:float = 0.1,
                 segment_size:Optional[float]=None,
                 P2:Optional[np.ndarray] = None)->Tuple[np.ndarray[float],np.ndarray[float]]:

    '''Simulate two correlated time series with specified coherence (no phase lag).

    This is a special case of TK_coh_and_lag with phase lag φ = 0.
    Implements Equations 9, 10, 12, and 15 from Larner, Nowak, & Wilms (2026).

    Parameters
    ----------
    P1 : np.ndarray
        Power spectrum array of the reference time series (P_X in paper notation)
    output_length : float
        Length of the output lightcurve in seconds
    dt : float
        Time resolution of the output lightcurve in seconds
    mean : float
        Mean count rate of the output lightcurves in counts/sec
    gamma : float or np.ndarray
        Coherence (γ² in paper notation) between the two time series. Range: [0, 1]
        Note: This is coherence SQUARED, not coherence amplitude.
        - float: constant coherence across all frequencies
        - np.ndarray: frequency-dependent coherence spectrum (same length as P1)
    red_noise : int, optional, default 1
        Red noise multiplier (not fully implemented for values > 1)
    rms : float, optional, default 0.1
        Fractional RMS variability amplitude
    segment_size : float, optional
        Size of segments if using segmented simulation (experimental)
    P2 : np.ndarray, optional
        Power spectrum of dependent time series (P_Y in paper notation).
        If None, P2 = P1. Must have same shape as P1 if provided.

    Returns
    -------
    counts1 : np.ndarray
        First simulated time series (reference)
    counts2 : np.ndarray
        Second simulated time series (correlated with first)

    Raises
    ------
    NotImplementedError
        If red_noise > 1 (not yet implemented for coherence simulation)
    ValueError
        If P1 and P2 have different shapes

    Notes
    -----
    This function is equivalent to TK_coh_and_lag with lag=0. The method:
    1. Generates reference transform: X = sqrt(P_X/2) * (A_r + i*B_r)  [Eq. 9]
    2. Computes transfer function: T = sqrt(P_Y*γ²/P_X)  [Eq. 15 with φ=0]
    3. Computes normalization: K = sqrt((P_Y - P_X*|T|²)/2)  [Eq. 12]
    4. Generates dependent transform: Y = K*(H_r + i*J_r) + T*X  [Eq. 10]

    See Also
    --------
    TK_coh_and_lag : Simulate with both coherence and phase lag
    compute_transfer_function : Compute transfer function from coherence and phase

    References
    ----------
    Larner, S. R., Nowak, M. A., & Wilms, J. 2026 (in prep)
    Timmer, J., & Koenig, M. 1995, A&A, 300, 707
    '''

    randint = np.random.randint(low=0,high=10000)

    if red_noise > 1:
        raise NotImplementedError('Red noise > 1 is not implemented for coherence/lag simulation methods')

    if P2 is None:
        P2 = P1
    elif P1.shape != P2.shape:
        raise ValueError('Both power spectra must have the same shape!')

    N = int(output_length/dt)
    long_N = int(output_length * red_noise / dt)

    pds_size = P1.size

    # Generate random variables (one set per frequency)
    Ar = np.random.normal(size=pds_size)
    Br = np.random.normal(size=pds_size)
    Hr = np.random.normal(size=pds_size)
    Jr = np.random.normal(size=pds_size)

    # Equation 9: Reference Fourier transform
    # X = sqrt(P_X/2) * (A_r + i*B_r)
    X = np.sqrt(P1 / 2) * (Ar + 1j * Br)

    # Equation 15: Transfer function with φ = 0 (no phase lag)
    # T = sqrt(P_Y * γ² / P_X) * exp(i*0) = sqrt(P_Y * γ² / P_X)
    lag = np.zeros(pds_size)  # Zero phase lag
    T = compute_transfer_function(gamma2=gamma, phi=lag, P_X=P1, P_Y=P2)

    # Equation 12: Normalization constant for incoherent component
    # K = sqrt((P_Y - P_X*|T|²) / 2)
    K = compute_normalization_constant(P_X=P1, P_Y=P2, T=T)

    # Equation 10: Dependent Fourier transform
    # Y = K*(H_r + i*J_r) + T*X
    Y = K * (Hr + 1j * Jr) + T * X

    # Compute theoretical std from PSDs (deterministic, preserves coherence)
    theoretical_std_X = compute_theoretical_std(P1, long_N)
    theoretical_std_Y = compute_theoretical_std(P2, long_N)

    # Inverse FFT to get time series
    counts1 = invert_fft(X, mean=mean, nbins=long_N)
    counts1 = extract_and_scale(long_lc=counts1, N=N, red_noise=red_noise,
                                rms=rms, random_state=randint, mean=mean,
                                segment_size=segment_size,
                                theoretical_std=theoretical_std_X)

    counts2 = invert_fft(Y, mean=mean, nbins=long_N)
    counts2 = extract_and_scale(long_lc=counts2, N=N, red_noise=red_noise,
                                rms=rms, random_state=randint, mean=mean,
                                segment_size=segment_size,
                                theoretical_std=theoretical_std_Y)

    return (counts1, counts2)

def TK_phaselag(pds:np.ndarray,
                output_length:float,
                dt:float,
                mean:float,
                lag:Union[float,np.ndarray[float]],
                red_noise:int = 1,
                segment_size:Optional[float] = None,
                rms:float = 0.1)->Tuple[np.ndarray[float],np.ndarray[float]]:
    '''
    Simulate two time series with specified power spectrum and phase lag using Timmer-Koenig.

    This function generates two correlated time series that share the same power spectrum
    but have a frequency-dependent phase lag relationship. Based on the SITAR implementation.

    Parameters
    ----------
    pds : np.ndarray
        Power spectrum array for both time series
    output_length : float
        Length of the output lightcurves in seconds
    dt : float
        Time resolution of the output lightcurves in seconds
    mean : float
        Mean count rate of the output lightcurves in counts/sec
    lag : float or np.ndarray
        Phase lag between the two time series in radians. Can be:
        - float: constant phase lag across all frequencies
        - np.ndarray: frequency-dependent phase lag spectrum (same length as pds)
    red_noise : int, optional, default 1
        Red noise multiplier (not fully implemented for values > 1)
    segment_size : float, optional
        Size of segments if using segmented simulation (experimental)
    rms : float, optional, default 0.1
        Fractional RMS variability amplitude

    Returns
    -------
    lc1 : np.ndarray
        First simulated time series (reference)
    lc2 : np.ndarray
        Second simulated time series (phase-lagged relative to first)

    Raises
    ------
    NotImplementedError
        If red_noise != 1 (not yet implemented for phase lag simulation)

    Notes
    -----
    The phase lag is imposed in Fourier space by multiplying the Fourier transform
    of the first light curve by exp(i*lag) to create the second light curve.
    Both light curves have identical power spectra but differ in phase.

    Unlike TK_coherence, this method maintains perfect coherence (gamma=1) while
    introducing a frequency-dependent time delay.

    See Also
    --------
    TK_coherence : Simulate with coherence relationship
    TK_lightcurve : Simulate single time series

    References
    ----------
    Timmer, J., & Koenig, M. 1995, A&A, 300, 707
    Uttley, P., et al. 2014, A&ARv, 22, 72 (SITAR method)
    '''

    randint = np.random.randint(low=0,high=10000)

    long_N = int(output_length * red_noise / dt)
    N = int(output_length/dt)

    if red_noise != 1:
        # TODO: Implement
        raise NotImplementedError("Red noise simulation for values other than 1 is not implemented.")
    
    pds_size = pds.size

    Ar = np.random.normal(size=pds_size) * np.sqrt(pds * 0.5)
    Br = np.random.normal(size = pds_size) * np.sqrt(pds * 0.5)

    z1 = np.array([complex(r,i) for r,i in zip(Ar,Br)])

    # Add in the phase lag for the second lc
    z2 = z1 * np.exp(1j * lag)

    # Compute theoretical std (same for both since same PSD)
    theoretical_std_val = compute_theoretical_std(pds, long_N)

    counts1 = invert_fft(z1,mean=mean,nbins=long_N)
    counts1 = extract_and_scale(long_lc=counts1,N=N,red_noise=red_noise,rms=rms,random_state=randint,mean=mean,segment_size=segment_size,theoretical_std=theoretical_std_val)
    counts2 = invert_fft(z2,mean=mean,nbins=long_N)
    counts2 = extract_and_scale(long_lc=counts2,N=N,red_noise=red_noise,rms=rms,random_state=randint,mean=mean,segment_size=segment_size,theoretical_std=theoretical_std_val)

    return (counts1,counts2)

def TK_coh_and_lag(P1:np.ndarray,
                   output_length:float,
                   dt:float,
                   mean:Union[float,Tuple[float,float]],
                   lag:np.ndarray[float],
                   gamma:np.ndarray[float],
                   red_noise:Optional[int] = 1,
                   rms:Optional[Union[float,Tuple[float,float]]] = 0.1,
                   segment_size:Optional[int]=None,
                   P2:Optional[np.ndarray]=None,
                   poisson: bool = False)->Tuple[np.ndarray[float],np.ndarray[float]]:

    '''Simulate two correlated time series with arbitrary coherence and phase lag.

    Implements the method from Larner, Nowak, & Wilms (2026) using Equations 9, 10,
    12, and 15. The reference time series is generated using the Timmer-Koenig method,
    and the dependent time series is constructed as a weighted sum of a coherent
    component (derived from the reference via a transfer function) and an incoherent
    component.

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
    lag : np.ndarray
        Phase lag spectrum in radians (φ in paper notation). Range: [-π, π]
    gamma : np.ndarray
        Coherence spectrum (γ² in paper notation). Range: [0, 1]
        Note: This is coherence SQUARED (γ²), not the coherence amplitude γ
    red_noise : int, optional
        Red noise factor (default: 1). Values > 1 not fully implemented
    rms : float or tuple of float, optional
        Fractional RMS variability. If float, same RMS for both bands.
        If tuple (rms1, rms2), different RMS for each band. Default: 0.1
    segment_size : int, optional
        Segment size for segmented simulation (experimental)
    P2 : np.ndarray, optional
        Power spectrum of dependent time series (P_Y in paper notation).
        If None, P2 = P1. Must have same shape as P1 if provided.
    poisson : bool, optional
        If True, draw final count arrays from Poisson distribution. Default: False

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

    References:
    -----------
    Larner, S. R., Nowak, M. A., & Wilms, J. 2026 (in prep)
    Timmer, J., & Koenig, M. 1995, A&A, 300, 707
    '''

    if P2 is None:
        P2 = P1
    elif P1.shape != P2.shape:
        raise ValueError('Both power spectra must have the same shape!')

    randint = np.random.randint(low=0,high=10000)

    N = int(output_length/dt)
    long_N = int(output_length * red_noise / dt)

    pds_size = P1.size

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
    counts1 = extract_and_scale(long_lc=counts1, N=N, red_noise=red_noise,
                                rms=rms1, random_state=randint, mean=mean,
                                segment_size=segment_size,
                                theoretical_std=theoretical_std_X)

    counts2 = invert_fft(Y, mean=mean, nbins=long_N)
    counts2 = extract_and_scale(long_lc=counts2, N=N, red_noise=red_noise,
                                rms=rms2, random_state=randint, mean=mean,
                                segment_size=segment_size,
                                theoretical_std=theoretical_std_Y)

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
            if "'<=' not supported between instances of 'tuple' and 'int'" not in str(e):
                raise e

        # if self.poisson or self.err != 0:
        #     raise NotImplementedError('Simulating coherence and phase lag light curves with nonzero errors is not implemented.')
        # if self.red_noise != 1:
        #     raise NotImplementedError('Simulating coherence and phase lag light curves with red noise > 1 is not implemented.')

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
        
        return np.fft.rfftfreq(self.red_noise * self.N, d = self.dt)[1:]

    def CL_simulate(self, 
                    pds1:Union[str,float,Model,Callable[[Iterable], Iterable]], 
                    pds2:Optional[Union[str,float,Model,Callable[[Iterable], Iterable]]], 
                    params: Optional[Union[list,dict]] = None,
                    lag:Optional[Union[str,float,Model,Callable[[Iterable], Iterable],Iterable]]=None,
                    coh:Optional[Union[str,float,Model,Callable[[Iterable], Iterable],Iterable]]=None,
                    segment_size:Optional[int]=None)->Tuple[Lightcurve, Lightcurve]:
        
        '''Simulate two LightCurves from a power spectrum and with a specified 
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
            - If float, is a constant value in [-\pi,\pi]
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

        Returns:
        --------
        (lc1, lc2):
            - Tuple of `Lightcurves` representing the input and response light 
            curves, respectively.

        Raises:
        -------
        ValueError:
            - If using a model string and model or parameters cannot be parsed.

        '''
        if pds2 is None:
            pds2 = pds1
        
        use_coh = coh is not None
        use_lag = lag is not None

        w = self.get_refftfreq()

        # Inspect the input and generate...
        # pds_distribution 1
        if isinstance(pds1, (float,int)):
            pds_shape1 = make_powerlaw_pds(pds1,self.dt,self.red_noise*self.N,segment_size=segment_size*self.dt)

        elif isinstance(pds1, str):
            from stingray.simulator import models

            if pds1 not in dir(models):
                raise ValueError('Model string not defined')
            
            if isinstance(params, dict):
                model = eval("models." + pds1 + "(**params)") # This is how Stingray does it
                pds_shape1 = model(w)
            elif isinstance(params,list):
                pds_shape1 = eval("models." + pds1 + "(w, params)")
            else:
                raise ValueError("Params should be list or dictionary!")

        else:
            pds_shape1 = pds1(w)

        # pds distribution 2
        if isinstance(pds2, (float,int)):
            pds_shape2 = make_powerlaw_pds(pds2,self.dt,self.red_noise*self.N,segment_size=segment_size*self.dt)

        elif isinstance(pds2, str):
            from stingray.simulator import models

            if pds2 not in dir(models):
                raise ValueError('Model string not defined')
            
            if isinstance(params, dict):
                model = eval("models." + pds2 + "(**params)")
                pds_shape2 = model(w)
            elif isinstance(params,list):
                pds_shape2 = eval("models." + pds2 + "(w, params)")
            else:
                raise ValueError("Params should be list or dictionary!")

        else:
            pds_shape2 = pds2(w)

        # lag_distribution
        if use_lag:
            
            if isinstance(lag, (float,int)):
                lag_shape = np.ones_like(w) * lag

            elif isinstance(lag, str):
                raise NotImplementedError('Model strings not implemented')
            
            elif isinstance(lag,Iterable):
                lag_shape = lag
            
            else:
                lag_shape = lag(w)

        # coh_distribution 
        if use_coh:
            
            if isinstance(coh, (float,int)):
                coh_shape = np.ones_like(w) * coh

            elif isinstance(coh, str):
                raise NotImplementedError('Model strings not implemented')
            
            elif isinstance(coh,Iterable):
                coh_shape = coh
            
            else:
                coh_shape = coh(w)


        if use_lag and use_coh:
            c1, c2 = TK_coh_and_lag(P1 = pds_shape1,
                                    P2 = pds_shape2,
                                    output_length=self.N * self.dt,
                                    dt = self.dt,
                                    mean= self.mean,
                                    lag = lag_shape,
                                    gamma = coh_shape,
                                    red_noise = self.red_noise,
                                    rms = self.rms,
                                    segment_size=segment_size,
                                    poisson=self.poisson)
        elif use_coh:
            c1,c2 = TK_coherence(P1 = pds_shape1,
                                 P2 = pds_shape2,
                                 output_length=self.N * self.dt,
                                 dt = self.dt,
                                 mean = self.mean,
                                 gamma = coh_shape,
                                 red_noise = self.red_noise,
                                 rms = self.rms)
            
        elif use_lag:
            if not np.allclose(pds_shape1,pds_shape2):
                raise ValueError('Lag only simulation cannot use different power spectral distributions')
            c1,c2 = TK_phaselag(pds = pds_shape1,
                                output_length=self.N * self.dt,
                                mean = self.mean, 
                                lag = lag_shape,
                                red_noise = self.red_noise,
                                rms = self.rms)
        else:
            if params is not None:
                return self.simulate(pds1,params)
            else:
                return self.simulate(pds1)
            
        t = np.arange(len(c1)) * self.dt
        
        lc1 = Lightcurve(time=t, counts = c1, err = np.zeros_like(c1), dt = self.dt,skip_checks=True)
        lc2 = Lightcurve(time=t, counts = c2, err = np.zeros_like(c2), dt = self.dt,skip_checks=True)
        
        return (lc1,lc2)
    
    @staticmethod
    def crossspectrum(lc1:Lightcurve, lc2:Lightcurve, seg_size:Optional[float] = None, norm:Optional[Literal['frac', 'abs', 'leahy', 'none']] = 'frac')->np.ndarray:
        '''
        Make a cross spectrum oif the simulated light curves. 

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
        '''

        # Following stingray convention by including this method 

        if seg_size is None:
            seg_size = lc1.tseg

        return AveragedCrossspectrum(lc1,lc2,seg_size,silent=True,norm=norm).power


