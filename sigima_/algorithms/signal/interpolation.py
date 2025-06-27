# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Interpolation (see parent package :mod:`sigima_.algorithms.signal`)

"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.interpolate


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
