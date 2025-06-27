# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Scaling (see parent package :mod:`sigima_.algorithms.signal`)

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import Literal

import numpy as np


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
        maximum = np.nanmax(yin, axis)
        if axis == 1:
            maximum = maximum.reshape((len(maximum), 1))
        maxarray = np.tile(maximum, yin.shape[axis]).reshape(yin.shape)
        return yin / maxarray
    if parameter == "amplitude":
        ytemp = np.array(yin, copy=True)
        minimum = np.nanmin(yin, axis)
        if axis == 1:
            minimum = minimum.reshape((len(minimum), 1))
        ytemp -= minimum
        return normalize(ytemp, parameter="maximum")
    if parameter == "area":
        return yin / np.nansum(yin)
    if parameter == "energy":
        return yin / np.sqrt(np.nansum(yin * yin.conjugate()))
    if parameter == "rms":
        return yin / np.sqrt(np.nanmean(yin * yin.conjugate()))
    raise RuntimeError(f"Unsupported parameter {parameter}")


def zscore(yin: np.ndarray) -> np.ndarray:
    """Standardize signal to zero mean and unit variance.

    Args:
        yin: Input array

    Returns:
        Standardized array with mean 0 and std 1
    """
    mean = np.nanmean(yin)
    std = np.nanstd(yin)
    if std == 0:
        raise ValueError("Standard deviation is zero; z-score is undefined.")
    return (yin - mean) / std


def robust_scale(yin: np.ndarray) -> np.ndarray:
    """Scale signal by median and interquartile range (IQR).

    Args:
        yin: Input array

    Returns:
        Scaled array with median 0 and IQR ~1
    """
    median = np.nanmedian(yin)
    q75, q25 = np.nanpercentile(yin, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        raise ValueError("Interquartile range is zero; robust scaling is undefined.")
    return (yin - median) / iqr
