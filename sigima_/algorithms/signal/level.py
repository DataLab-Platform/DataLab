# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Level Adjustment (see parent package :mod:`sigima_.algorithms.signal`)

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
