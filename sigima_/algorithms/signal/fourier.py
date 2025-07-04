# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Fourier Analysis (see parent package :mod:`sigima_.algorithms.signal`)

"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal

from sigima_.algorithms.signal.dynamic import sampling_rate


def zero_padding(x: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Append n zeros at the end of the signal.

    Args:
        x: X data
        y: Y data
        n: Number of zeros to append

    Returns:
        X data, Y data (tuple)
    """
    if n < 1:
        raise ValueError("Number of zeros to append must be greater than 0")
    x1 = np.linspace(x[0], x[-1] + n * (x[1] - x[0]), len(y) + n)
    y1 = np.append(y, np.zeros(n))
    return x1, y1


def fft1d(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    shifted: bool = True,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.complexfloating]]:
    """
    Compute the Fast Fourier Transform (FFT) of a 1D real signal.

    Args:
        x: Time domain axis (evenly spaced).
        y: Signal values.
        shifted: If True, shift zero frequency to the center of the spectrum.

    Returns:
        Tuple (f, sp): Frequency axis and corresponding FFT values.
    """
    dt = x[1] - x[0]
    f = np.fft.fftfreq(x.size, d=dt)  # Frequency axis
    sp = np.fft.fft(y)  # Spectrum values
    if shifted:
        f = np.fft.fftshift(f)
        sp = np.fft.fftshift(sp)
    return f, sp


def ifft1d(
    f: npt.NDArray[np.floating],
    sp: npt.NDArray[np.complexfloating],
    shifted: bool = True,
    initial: float = 0.0,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Compute the inverse FFT of a spectrum.

    Args:
        f: Frequency axis (evenly spaced).
        spectrum: FFT values.
        shifted: If True, the input spectrum is centered (zero frequency in the middle).
        initial: Starting value for the time axis.

    Returns:
        Tuple (x, y): Time axis and reconstructed real signal.

    Raises:
        ValueError: If frequency array is not evenly spaced or has fewer than 2 points.
    """
    if f.size < 2:
        raise ValueError("Frequency array must have at least two elements.")
    if shifted:
        sp = np.fft.ifftshift(sp)
    else:
        f = np.fft.fftshift(f)

    df = np.mean(np.diff(f))
    if not np.allclose(np.diff(f), df):
        raise ValueError("Frequency array must be evenly spaced.")

    y = np.fft.ifft(sp)
    dt = 1.0 / (f.size * df)
    x = np.linspace(initial, initial + (len(y) - 1) * dt, y.size)
    return x, y.real


def magnitude_spectrum(
    x: np.ndarray, y: np.ndarray, log_scale: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Compute magnitude spectrum.

    Args:
        x: X data
        y: Y data
        log_scale: Use log scale. Defaults to False.

    Returns:
        Magnitude spectrum (X data, Y data)
    """
    x1, y1 = fft1d(x, y)
    if log_scale:
        y_mag = 20 * np.log10(np.abs(y1))
    else:
        y_mag = np.abs(y1)
    return x1, y_mag


def phase_spectrum(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute phase spectrum.

    Args:
        x: X data
        y: Y data

    Returns:
        Phase spectrum in degrees (X data, Y data)
    """
    x1, y1 = fft1d(x, y)
    y_phase = np.rad2deg(np.angle(y1))
    return x1, y_phase


def psd(
    x: np.ndarray, y: np.ndarray, log_scale: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Power Spectral Density (PSD), using the Welch method.

    Args:
        x: X data
        y: Y data
        log_scale: Use log scale. Defaults to False.

    Returns:
        Power Spectral Density (PSD): X data, Y data (tuple)
    """
    x1, y1 = scipy.signal.welch(y, fs=sampling_rate(x))
    if log_scale:
        y1 = 10 * np.log10(y1)
    return x1, y1


def sort_frequencies(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sort from X,Y data by computing FFT(y).

    Args:
        x: X data
        y: Y data

    Returns:
        Sorted frequencies in ascending order
    """
    freqs, fourier = fft1d(x, y, shifted=False)
    return freqs[np.argsort(fourier)]
