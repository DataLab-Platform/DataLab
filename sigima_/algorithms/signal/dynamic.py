# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Dynamic Parameters (see parent package :mod:`sigima_.algorithms.signal`)

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import scipy.optimize


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

    def optfunc(fitparams: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Optimization function."""
        return y - sinusoidal_model(x, *fitparams)

    # Fit the model to the data
    fitparams = scipy.optimize.leastsq(
        optfunc, [amp, freq, phase_origin, offset], args=(x, y)
    )[0]
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
    fitparams, _residuals = sinusoidal_fit(x, y)
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
    return -np.log2(residuals * np.sqrt(12) / full_scale)


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

    return 20 * np.log10(powf / residuals)


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
    fitparams, _residuals = sinusoidal_fit(x, y)
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
    return 20 * np.log10(sumharm / powfund)


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
    return 20 * np.log10(powfund / maxspike)


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
    return 20 * np.log10(powfund / noise)


def sampling_period(x: np.ndarray) -> float:
    """Compute sampling period

    Args:
        x: X data

    Returns:
        Sampling period
    """
    steps = np.diff(x)
    if not np.isclose(np.diff(steps).max(), 0, atol=1e-10):
        warnings.warn("Non-constant sampling signal")
    return steps[0]


def sampling_rate(x: np.ndarray) -> float:
    """Compute mean sampling rate

    Args:
        x: X data

    Returns:
        Sampling rate
    """
    return 1.0 / sampling_period(x)
