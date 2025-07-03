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
    x1 = np.fft.fftfreq(x.size, d=x[1] - x[0])
    if shift:
        x1 = np.fft.fftshift(x1)
        y1 = np.fft.fftshift(y1)
    return x1, y1


def ifft1d(
    f: npt.NDArray[np.floating],
    sp: npt.NDArray[np.complexfloating],
    shifted: bool = True,
    initial: float = 0.0,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute inverse Fast Fourier Transform (iFFT).

    Args:
        f: 1-D evenly-spaced values in frequency domain.
        sp: 1-D spectrum values.
        shifted: If `True`, input has zero-frequency component shifted to the center.
        initial: Starting value for the reconstructed time axis.

    Returns:
        Tuple containing the signal x and y data in time domain.

    Raises:
        ValueError: If the frequency array has less than two elements.
    """
    if f.size < 2:
        raise ValueError("Frequency array must have at least two elements.")
    if shifted:
        # Move zero-frequency component to the beginning.
        sp = np.fft.ifftshift(sp)
    else:
        # Move zero-frequency to the center.
        f = np.fft.fftshift(f)
    if not np.allclose(np.diff(f), np.mean(np.diff(f))):
        raise ValueError("Frequency array must be evenly spaced.")
    y = np.fft.ifft(sp)
    df = np.mean(np.diff(f))
    dt = 1.0 / (len(f) * df)
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
    freqs, fourier = fft1d(x, y, shift=False)
    return freqs[np.argsort(fourier)]
