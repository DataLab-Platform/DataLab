# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Signal Processing Algorithms (see parent package :mod:`cdl.algorithms`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.interpolate
from scipy.optimize import leastsq

from cdl.algorithms import fit


# ----- Filtering functions ----------------------------------------------------
def moving_average(y: np.ndarray, n: int) -> np.ndarray:
    """Compute moving average.

    Args:
        y: Input array
        n: Window size

    Returns:
        Moving average
    """
    y_padded = np.pad(y, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(y_padded, np.ones((n,)) / n, mode="valid")


# ----- Misc. functions --------------------------------------------------------
def derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute numerical derivative.

    Args:
        x: X data
        y: Y data

    Returns:
        Numerical derivative
    """
    dy = np.zeros_like(y)
    dy[0:-1] = np.diff(y) / np.diff(x)
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return dy


def normalize(
    yin: np.ndarray,
    parameter: Literal["maximum", "amplitude", "area", "energy", "rms"] = "maximum",
) -> np.ndarray:
    """Normalize input array to a given parameter.

    Args:
        yin: Input array
        parameter: Normalization parameter. Defaults to "maximum"

    Returns:
        Normalized array
    """
    axis = len(yin.shape) - 1
    if parameter == "maximum":
        maximum = np.max(yin, axis)
        if axis == 1:
            maximum = maximum.reshape((len(maximum), 1))
        maxarray = np.tile(maximum, yin.shape[axis]).reshape(yin.shape)
        return yin / maxarray
    if parameter == "amplitude":
        ytemp = np.array(yin, copy=True)
        minimum = np.min(yin, axis)
        if axis == 1:
            minimum = minimum.reshape((len(minimum), 1))
        ytemp -= minimum
        return normalize(ytemp, parameter="maximum")
    if parameter == "area":
        return yin / yin.sum()
    if parameter == "energy":
        return yin / np.sqrt(np.sum(yin * yin.conjugate()))
    if parameter == "rms":
        return yin / np.sqrt(np.mean(yin * yin.conjugate()))
    raise RuntimeError(f"Unsupported parameter {parameter}")


def xy_fft(
    x: np.ndarray, y: np.ndarray, shift: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FFT on X,Y data.

    Args:
        x: X data
        y: Y data
        shift: Shift the zero frequency to the center of the spectrum. Defaults to True.

    Returns:
        X data, Y data (tuple)
    """
    y1 = np.fft.fft(y)
    x1 = np.fft.fftfreq(x.shape[-1], d=x[1] - x[0])
    if shift:
        x1 = np.fft.fftshift(x1)
        y1 = np.fft.fftshift(y1)
    return x1, y1


def xy_ifft(
    x: np.ndarray, y: np.ndarray, shift: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Compute iFFT on X,Y data.

    Args:
        x: X data
        y: Y data
        shift: Shift the zero frequency to the center of the spectrum. Defaults to True.

    Returns:
        X data, Y data (tuple)
    """
    x1 = np.fft.fftfreq(x.shape[-1], d=x[1] - x[0])
    if shift:
        x1 = np.fft.ifftshift(x1)
        y = np.fft.ifftshift(y)
    y1 = np.fft.ifft(y)
    return x1, y1.real


def sort_frequencies(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sort from X,Y data by computing FFT(y).

    Args:
        x: X data
        y: Y data

    Returns:
        Sorted frequencies in ascending order
    """
    freqs, fourier = xy_fft(x, y, shift=False)
    return freqs[np.argsort(fourier)]


# ----- Peak detection functions -----------------------------------------------
def peak_indexes(
    y, thres: float = 0.3, min_dist: int = 1, thres_abs: bool = False
) -> np.ndarray:
    #  Copyright (c) 2014 Lucas Hermann Negri
    #  Unmodified code snippet from PeakUtils 1.3.0
    """Peak detection routine.

    Finds the numeric index of the peaks in *y* by taking its first order
    difference. By using *thres* and *min_dist* parameters, it is possible
    to reduce the number of detected peaks. *y* must be signed.

    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.
    thres_abs: boolean
        If True, the thres value will be interpreted as an absolute value,
        instead of a normalized threshold.

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    if not thres_abs:
        thres = thres * (np.max(y) - np.min(y)) + np.min(y)

    # compute first order difference
    dy = np.diff(y)

    # propagate left and right values successively to fill all plateau pixels
    # (0-value)
    (zeros,) = np.where(dy == 0)

    # check if the signal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([])

    if len(zeros):
        # compute first order difference of zero indexes
        zeros_diff = np.diff(zeros)
        # check when zeros are not chained together
        (zeros_diff_not_one,) = np.add(np.where(zeros_diff != 1), 1)
        # make an array of the chained zero indexes
        zero_plateaus = np.split(zeros, zeros_diff_not_one)

        # fix if leftmost value in dy is zero
        if zero_plateaus[0][0] == 0:
            dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
            zero_plateaus.pop(0)

        # fix if rightmost value of dy is zero
        if len(zero_plateaus) > 0 and zero_plateaus[-1][-1] == len(dy) - 1:
            dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
            zero_plateaus.pop(-1)

        # for each chain of zero indexes
        for plateau in zero_plateaus:
            median = np.median(plateau)
            # set leftmost values to leftmost non zero values
            dy[plateau[plateau < median]] = dy[plateau[0] - 1]
            # set rightmost and middle values to rightmost non zero values
            dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]

    # find the peaks by using the first order difference
    peaks = np.where(
        (np.hstack([dy, 0.0]) < 0.0)
        & (np.hstack([0.0, dy]) > 0.0)
        & (np.greater(y, thres))
    )[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks


def xpeak(x: np.ndarray, y: np.ndarray) -> float:
    """Return default peak X-position (assuming a single peak).

    Args:
        x: X data
        y: Y data

    Returns:
        Peak X-position
    """
    peaks = peak_indexes(y)
    if peaks.size == 1:
        return x[peaks[0]]
    return np.average(x, weights=y)


def interpolate(
    x: np.ndarray,
    y: np.ndarray,
    xnew: np.ndarray,
    method: Literal["linear", "spline", "quadratic", "cubic", "barycentric", "pchip"],
    fill_value: float | None = None,
) -> np.ndarray:
    """Interpolate data.

    Args:
        x: X data
        y: Y data
        xnew: New X data
        method: Interpolation method
        fill_value: Fill value. Defaults to None.
         This value is used to fill in for requested points outside of the
         X data range. It is only used if the method argument is 'linear',
         'cubic' or 'pchip'.

    Returns:
        Interpolated Y data
    """
    interpolator_extrap = None
    if method == "linear":
        # Linear interpolation using NumPy's interp function:
        ynew = np.interp(xnew, x, y, left=fill_value, right=fill_value)
    elif method == "spline":
        # Spline using 1-D interpolation with SciPy's interpolate package:
        # pylint: disable=unbalanced-tuple-unpacking
        knots, coeffs, degree = scipy.interpolate.splrep(x, y, s=0)
        ynew = scipy.interpolate.splev(xnew, (knots, coeffs, degree), der=0)
    elif method == "quadratic":
        # Quadratic interpolation using NumPy's polyval function:
        coeffs = np.polyfit(x, y, 2)
        ynew = np.polyval(coeffs, xnew)
    elif method == "cubic":
        # Cubic interpolation using SciPy's Akima1DInterpolator class:
        interpolator_extrap = scipy.interpolate.Akima1DInterpolator(x, y)
    elif method == "barycentric":
        # Barycentric interpolation using SciPy's BarycentricInterpolator class:
        interpolator = scipy.interpolate.BarycentricInterpolator(x, y)
        ynew = interpolator(xnew)
    elif method == "pchip":
        # PCHIP interpolation using SciPy's PchipInterpolator class:
        interpolator_extrap = scipy.interpolate.PchipInterpolator(x, y)
    else:
        raise ValueError(f"Invalid interpolation method {method}")
    if interpolator_extrap is not None:
        ynew = interpolator_extrap(xnew, extrapolate=fill_value is None)
        if fill_value is not None:
            ynew[xnew < x[0]] = fill_value
            ynew[xnew > x[-1]] = fill_value
    return ynew


def windowing(
    y: np.ndarray,
    method: Literal[
        "barthann",
        "bartlett",
        "blackman",
        "blackman-harris",
        "bohman",
        "boxcar",
        "cosine",
        "exponential",
        "flat-top",
        "hamming",
        "hanning",
        "lanczos",
        "nuttall",
        "parzen",
        "rectangular",
        "taylor",
        "tukey",
        "kaiser",
        "gaussian",
    ] = "hamming",
    alpha: float = 0.5,
    beta: float = 14.0,
    sigma: float = 7.0,
) -> np.ndarray:
    """Apply windowing to the input data.

    Args:
        x: X data
        y: Y data
        method: Windowing function. Defaults to "hamming".
        alpha: Tukey window parameter. Defaults to 0.5.
        beta: Kaiser window parameter. Defaults to 14.0.
        sigma: Gaussian window parameter. Defaults to 7.0.

    Returns:
        Windowed Y data
    """
    # Cases without parameters:
    win_func = {
        "barthann": scipy.signal.windows.barthann,
        "bartlett": np.bartlett,
        "blackman": np.blackman,
        "blackman-harris": scipy.signal.windows.blackmanharris,
        "bohman": scipy.signal.windows.bohman,
        "boxcar": scipy.signal.windows.boxcar,
        "cosine": scipy.signal.windows.cosine,
        "exponential": scipy.signal.windows.exponential,
        "flat-top": scipy.signal.windows.flattop,
        "hamming": np.hamming,
        "hanning": np.hanning,
        "lanczos": scipy.signal.windows.lanczos,
        "nuttall": scipy.signal.windows.nuttall,
        "parzen": scipy.signal.windows.parzen,
        "rectangular": np.ones,
        "taylor": scipy.signal.windows.taylor,
    }.get(method)
    if win_func is not None:
        return y * win_func(len(y))
    # Cases with parameters:
    if method == "tukey":
        return y * scipy.signal.windows.tukey(len(y), alpha)
    if method == "kaiser":
        return y * np.kaiser(len(y), beta)
    if method == "gaussian":
        return y * scipy.signal.windows.gaussian(len(y), sigma)
    raise ValueError(f"Invalid window type {method}")


def find_nearest_zero_point_idx(y: np.ndarray) -> np.ndarray:
    """Find the x indexes where the corresponding y is the closest to zero

    Args:
        y: Y data

    Returns:
        Indexes of the points right before or at zero crossing
    """
    xi = np.where((y[:-1] >= 0) & (y[1:] <= 0) | (y[:-1] <= 0) & (y[1:] >= 0))[0]
    return xi


def find_x_at_value(x: np.ndarray, y: np.ndarray, value: float) -> np.ndarray:
    """Find the x value where the y value is the closest to the given value using
    linear interpolation to deduce the precise x value.

    Args:
        x: X data
        y: Y data
        value: Value to find

    Returns:
        X value where the Y value is the closest to the given value
    """
    leveled_y = y - value
    xi_before = find_nearest_zero_point_idx(leveled_y)
    xi_after = xi_before + 1

    if len(xi_before) == 0:
        return np.array([0.0])

    # linear interpolation
    p = (leveled_y[xi_after] - leveled_y[xi_before]) / (x[xi_after] - x[xi_before])
    ori = leveled_y[xi_after] - p * x[xi_after]
    x0 = -ori / p  # where the curve cut the absissa
    return x0


def bandwidth(x: np.ndarray, y: np.ndarray, level: float = 3.0) -> float:
    """Compute the bandwidth of the signal at a given level.

    Args:
        x: x signal data
        y: y signal data
        level: Level in dB at which the bandwidth is computed. Defaults to 3.0.

    Returns:
        Bandwidth of the signal at the given level
    """
    half_max: float = np.max(y) - level
    bw = find_x_at_value(x, y, half_max)
    return bw[0]


# MARK: ENOB, SINAD, THD, SFDR, SNR
# ======================================================================================


def sinusoidal_model(
    x: np.ndarray, a: float, f: float, phi: float, offset: float
) -> np.ndarray:
    """Sinusoidal model function."""
    return a * np.sin(2 * np.pi * f * x + phi) + offset


def sinusoidal_fit(
    x: np.ndarray, y: np.ndarray
) -> tuple[tuple[float, float, float, float], float]:
    """Fit a sinusoidal model to the input data.

    Args:
        x: X data
        y: Y data

    Returns:
        A tuple containing the fit parameters (amplitude, frequency, phase, offset)
        and the residuals
    """
    # Initial guess for the parameters
    # ==================================================================================
    offset = np.mean(y)
    amp = (np.max(y) - np.min(y)) / 2
    phase_origin = 0
    # Search for the maximum of the FFT
    i_maxfft = np.argmax(np.abs(np.fft.fft(y - offset)))
    if i_maxfft > len(x) / 2:
        # If the index is greater than N/2, we are in the mirrored half spectrum
        # (negative frequencies)
        i_maxfft = len(x) - i_maxfft
    freq = i_maxfft / (x[-1] - x[0])
    # ==================================================================================

    def optim_func(fitparams: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Optimization function."""
        return y - sinusoidal_model(x, *fitparams)

    # Fit the model to the data
    fitparams = leastsq(optim_func, [amp, freq, phase_origin, offset], args=(x, y))[0]
    y_th = sinusoidal_model(x, *fitparams)
    residuals = np.std(y - y_th)
    return fitparams, residuals


def sinus_frequency(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the frequency of a sinusoidal signal.

    Args:
        x: x signal data
        y: y signal data

    Returns:
        Frequency of the sinusoidal signal
    """
    fitparams, residuals = sinusoidal_fit(x, y)
    return fitparams[1]


def enob(x: np.ndarray, y: np.ndarray, full_scale: float = 1.0) -> float:
    """Compute Effective Number of Bits (ENOB).

    Args:
        x: x signal data
        y: y signal data
        full_scale: Full scale(V). Defaults to 1.0.

    Returns:
        Effective Number of Bits (ENOB)
    """
    _fitparams, residuals = sinusoidal_fit(x, y)
    enob = -np.log2(residuals * np.sqrt(12) / full_scale)
    return enob


def sinad(
    x: np.ndarray,
    y: np.ndarray,
    full_scale: float = 1.0,
    unit: Literal["dBc", "dBFS"] = "dBc",
) -> float:
    """Compute Signal-to-Noise and Distortion Ratio (SINAD).

    Args:
        x: x signal data
        y: y signal data
        full_scale: Full scale(V). Defaults to 1.0.
        unit: Unit of the input data. Valid values are 'dBc' and 'dBFS'.
         Defaults to 'dBc'.

    Returns:
        Signal-to-Noise and Distortion Ratio (SINAD)
    """
    fitparams, residuals = sinusoidal_fit(x, y)
    amp = fitparams[0]

    # Compute the power of the fundamental
    powf = np.abs(amp / np.sqrt(2)) if unit == "dBc" else full_scale / (2 * np.sqrt(2))

    sinad = 20 * np.log10(powf / residuals)
    return sinad


def thd(
    x: np.ndarray,
    y: np.ndarray,
    full_scale: float = 1.0,
    unit: Literal["dBc", "dBFS"] = "dBc",
    nb_harm: int = 5,
) -> float:
    """Compute Total Harmonic Distortion (THD).

    Args:
        x: x signal data
        y: y signal data
        full_scale: Full scale(V). Defaults to 1.0.
        unit: Unit of the input data. Valid values are 'dBc' and 'dBFS'.
         Defaults to 'dBc'.
        nb_harm: Number of harmonics to consider. Defaults to 5.

    Returns:
        Total Harmonic Distortion (THD)
    """
    fitparams, residuals = sinusoidal_fit(x, y)
    offset = np.mean(y)
    amp, freq = fitparams[:2]
    ampfft = np.abs(np.fft.fft(y - offset))

    # Compute the power of the fundamental
    if unit == "dBc":
        powfund = np.max(ampfft[: len(ampfft) // 2])
    else:
        powfund = (full_scale / (2 * np.sqrt(2))) * (len(x) / np.sqrt(2))

    sumharm = 0
    for i in np.arange(nb_harm + 2)[2:]:
        a = i * np.ceil(freq * (x[-1] - x[0]))
        amp = ampfft[int(a - 5) : int(a + 5)]
        if len(amp) > 0:
            sumharm += np.max(amp)
    thd = 20 * np.log10(sumharm / powfund)
    return thd


def sfdr(
    x: np.ndarray,
    y: np.ndarray,
    full_scale: float = 1.0,
    unit: Literal["dBc", "dBFS"] = "dBc",
) -> float:
    """Compute Spurious-Free Dynamic Range (SFDR).

    Args:
        x: x signal data
        y: y signal data
        full_scale: Full scale(V). Defaults to 1.0.
        unit: Unit of the input data. Valid values are 'dBc' and 'dBFS'.
         Defaults to 'dBc'.

    Returns:
        Spurious-Free Dynamic Range (SFDR)
    """
    fitparams, _residuals = sinusoidal_fit(x, y)

    # Compute the power of the fundamental
    if unit == "dBc":
        powfund = np.max(np.abs(np.fft.fft(y)))
    else:
        powfund = (full_scale / (2 * np.sqrt(2))) * (len(x) / np.sqrt(2))

    maxspike = np.max(np.abs(np.fft.fft(y - sinusoidal_model(x, *fitparams))))
    sfdr = 20 * np.log10(powfund / maxspike)
    return sfdr


def snr(
    x: np.ndarray,
    y: np.ndarray,
    full_scale: float = 1.0,
    unit: Literal["dBc", "dBFS"] = "dBc",
) -> float:
    """Compute Signal-to-Noise Ratio (SNR).

    Args:
        x: x signal data
        y: y signal data
        full_scale: Full scale(V). Defaults to 1.0.
        unit: Unit of the input data. Valid values are 'dBc' and 'dBFS'.
         Defaults to 'dBc'.

    Returns:
        Signal-to-Noise Ratio (SNR)
    """
    fitparams, _residuals = sinusoidal_fit(x, y)

    # Compute the power of the fundamental
    if unit == "dBc":
        powfund = np.max(np.abs(np.fft.fft(y)))
    else:
        powfund = (full_scale / (2 * np.sqrt(2))) * (len(x) / np.sqrt(2))

    noise = np.sqrt(np.mean((y - sinusoidal_model(x, *fitparams)) ** 2))
    snr = 20 * np.log10(powfund / noise)
    return snr


def fwhm(
    data: np.ndarray,
    method: Literal["zero-crossing", "gauss", "lorentz", "voigt"] = "zero-crossing",
    xmin: float | None = None,
    xmax: float | None = None,
) -> tuple[float, float, float, float]:
    """Compute Full Width at Half Maximum (FWHM) of the input data

    Args:
        data: X,Y data
        method: Calculation method. Two types of methods are supported: a zero-crossing
         method and fitting methods (based on various models: Gauss, Lorentz, Voigt).
         Defaults to "zero-crossing".
        xmin: Lower X bound for the fitting. Defaults to None (no lower bound,
         i.e. the fitting starts from the first point).
        xmax: Upper X bound for the fitting. Defaults to None (no upper bound,
         i.e. the fitting ends at the last point)

    Returns:
        FWHM segment coordinates
    """
    x, y = data
    dx, dy, base = np.max(x) - np.min(x), np.max(y) - np.min(y), np.min(y)
    sigma, mu = dx * 0.1, xpeak(x, y)
    if isinstance(xmin, float):
        indexes = np.where(x >= xmin)[0]
        x = x[indexes]
        y = y[indexes]
    if isinstance(xmax, float):
        indexes = np.where(x <= xmax)[0]
        x = x[indexes]
        y = y[indexes]

    if method == "zero-crossing":
        hmax = dy * 0.5 + np.min(y)
        fx = find_x_at_value(x, y, hmax)
        assert fx.size == 2, f"Number of half-max points must be 2, not {fx.size}"
        return fx[0], hmax, fx[-1], hmax

    try:
        FitModelClass: type[fit.FitModel] = {
            "gauss": fit.GaussianModel,
            "lorentz": fit.LorentzianModel,
            "voigt": fit.VoigtModel,
        }[method]
    except KeyError as exc:
        raise ValueError(f"Invalid method {method}") from exc

    def func(params):
        """Fitting model function"""
        # pylint: disable=cell-var-from-loop
        return y - FitModelClass.func(x, *params)

    amp = FitModelClass.get_amp_from_amplitude(dy, sigma)
    (amp, sigma, mu, base), _ier = leastsq(func, np.array([amp, sigma, mu, base]))
    return FitModelClass.half_max_segment(amp, sigma, mu, base)


def fw1e2(data: np.ndarray) -> tuple[float, float, float, float]:
    """Compute Full Width at 1/e² of the input data (using a Gaussian model fitting).

    Args:
        data: X,Y data

    Returns:
        FW at 1/e² segment coordinates
    """
    x, y = data
    dx, dy, base = np.max(x) - np.min(x), np.max(y) - np.min(y), np.min(y)
    sigma, mu = dx * 0.1, xpeak(x, y)
    amp = fit.GaussianModel.get_amp_from_amplitude(dy, sigma)
    p_in = np.array([amp, sigma, mu, base])

    def func(params):
        """Fitting model function"""
        # pylint: disable=cell-var-from-loop
        return y - fit.GaussianModel.func(x, *params)

    p_out, _ier = leastsq(func, p_in)
    amp, sigma, mu, base = p_out
    hw = 2 * sigma
    yhm = fit.GaussianModel.amplitude(amp, sigma) / np.e**2 + base
    return mu - hw, yhm, mu + hw, yhm
