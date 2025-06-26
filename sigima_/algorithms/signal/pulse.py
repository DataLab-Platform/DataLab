# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Pulse analysis (see parent package :mod:`sigima_.algorithms.signal`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import scipy.optimize

from sigima_.algorithms.signal import fitmodels, misc, peakdetection

# MARK: Pulse analysis -----------------------------------------------------------------


def full_width_at_y(
    data: np.ndarray, level: float
) -> tuple[float, float, float, float]:
    """Compute the full width at a given y level using zero-crossing method.

    Args:
        data: X,Y data
        level: The Y level at which to compute the width

    Returns:
        Full width segment coordinates
    """
    x, y = data
    crossings = misc.find_x_at_value(x, y, level)
    if crossings.size < 2:
        raise ValueError("Not enough zero-crossing points found")
    return crossings[0], level, crossings[-1], level


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
    sigma, mu = dx * 0.1, peakdetection.xpeak(x, y)
    if isinstance(xmin, float):
        indices = np.where(x >= xmin)[0]
        x = x[indices]
        y = y[indices]
    if isinstance(xmax, float):
        indices = np.where(x <= xmax)[0]
        x = x[indices]
        y = y[indices]

    if method == "zero-crossing":
        hmax = dy * 0.5 + np.min(y)
        fx = misc.find_x_at_value(x, y, hmax)
        if fx.size > 2:
            warnings.warn(f"Ambiguous zero-crossing points (found {fx.size} points)")
        elif fx.size < 2:
            raise ValueError("No zero-crossing points found")
        return fx[0], hmax, fx[-1], hmax

    try:
        fit_model_class: type[fitmodels.FitModel] = {
            "gauss": fitmodels.GaussianModel,
            "lorentz": fitmodels.LorentzianModel,
            "voigt": fitmodels.VoigtModel,
        }[method]
    except KeyError as exc:
        raise ValueError(f"Invalid method {method}") from exc

    def func(params):
        """Fitting model function"""
        # pylint: disable=cell-var-from-loop
        return y - fit_model_class.func(x, *params)

    amp = fit_model_class.get_amp_from_amplitude(dy, sigma)
    (amp, sigma, mu, base), _ier = scipy.optimize.leastsq(
        func, np.array([amp, sigma, mu, base])
    )
    return fit_model_class.half_max_segment(amp, sigma, mu, base)


def fw1e2(data: np.ndarray) -> tuple[float, float, float, float]:
    """Compute Full Width at 1/e� of the input data (using a Gaussian model fitting).

    Args:
        data: X,Y data

    Returns:
        FW at 1/e² segment coordinates
    """
    x, y = data
    dx, dy, base = np.max(x) - np.min(x), np.max(y) - np.min(y), np.min(y)
    sigma, mu = dx * 0.1, peakdetection.xpeak(x, y)
    amp = fitmodels.GaussianModel.get_amp_from_amplitude(dy, sigma)
    p_in = np.array([amp, sigma, mu, base])

    def func(params):
        """Fitting model function"""
        # pylint: disable=cell-var-from-loop
        return y - fitmodels.GaussianModel.func(x, *params)

    p_out, _ier = scipy.optimize.leastsq(func, p_in)
    amp, sigma, mu, base = p_out
    hw = 2 * sigma
    yhm = fitmodels.GaussianModel.amplitude(amp, sigma) / np.e**2 + base
    return mu - hw, yhm, mu + hw, yhm
