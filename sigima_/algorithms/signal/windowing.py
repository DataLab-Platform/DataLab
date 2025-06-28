# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Windowing (see parent package :mod:`sigima_.algorithms.signal`)

"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import scipy.signal.windows


def get_window(
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
    ],
) -> Callable[[int], np.ndarray]:
    """Get the window function.

    .. note::

        The window functions are from `scipy.signal.windows` and `numpy`.
        All functions take an integer argument that specifies the length of the window,
        and return a numpy array of the same length.

    Args:
        method: Windowing function name.

    Returns:
        Window function

    Raises:
        ValueError: If the method is not recognized.
    """
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
        return win_func
    raise ValueError(f"Invalid window type {method}")


def apply_window(
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

    Raises:
        ValueError: If the method is not recognized.
    """
    # Cases with parameters:
    if method == "tukey":
        return y * scipy.signal.windows.tukey(len(y), alpha)
    if method == "kaiser":
        return y * np.kaiser(len(y), beta)
    if method == "gaussian":
        return y * scipy.signal.windows.gaussian(len(y), sigma)
    # Cases without parameters:
    win_func = get_window(method)
    return y * win_func(len(y))
