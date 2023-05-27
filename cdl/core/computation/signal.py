# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Signal Computation module
-------------------------------

This module defines the signal parameters and functions used by the
:mod:`cdl.core.gui.processor` module.

It is based on the :mod:`cdl.algorithms` module, which defines the algorithms
that are applied to the data, and on the :mod:`cdl.core.model` module, which
defines the data model.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import guidata.dataset.dataitems as gdi
import guidata.dataset.datatypes as gdt
import numpy as np
import scipy.integrate as spt
import scipy.ndimage as spi
import scipy.optimize as spo
import scipy.signal as sps

from cdl.algorithms import fit
from cdl.algorithms.signal import (
    derivative,
    moving_average,
    normalize,
    peak_indexes,
    xpeak,
    xy_fft,
    xy_ifft,
)
from cdl.config import Conf, _
from cdl.core.computation.base import (
    ClipParam,
    GaussianParam,
    MovingAverageParam,
    MovingMedianParam,
    ThresholdParam,
)
from cdl.core.model.signal import SignalObj


def extract_multiple_roi(
    x: np.ndarray, y: np.ndarray, group: gdt.DataSetGroup
) -> tuple[np.ndarray, np.ndarray]:
    """Extract multiple regions of interest from data
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
        group (gdt.DataSetGroup): group of parameters
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    xout, yout = np.ones_like(x) * np.nan, np.ones_like(y) * np.nan
    for p in group.datasets:
        slice0 = slice(p.col1, p.col2 + 1)
        xout[slice0], yout[slice0] = x[slice0], y[slice0]
    nans = np.isnan(xout) | np.isnan(yout)
    return xout[~nans], yout[~nans]


def extract_single_roi(
    x: np.ndarray, y: np.ndarray, p: gdt.DataSet
) -> tuple[np.ndarray, np.ndarray]:
    """Extract single region of interest from data
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
        p (gdt.DataSet): parameters
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    return x[p.col1 : p.col2 + 1], y[p.col1 : p.col2 + 1]


def compute_swap_axes(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Swap axes
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    return y, x


def compute_abs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute absolute value
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    return (x, np.abs(y))


def compute_log10(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Log10
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    return (x, np.log10(y))


class PeakDetectionParam(gdt.DataSet):
    """Peak detection parameters"""

    threshold = gdi.IntItem(
        _("Threshold"), default=30, min=0, max=100, slider=True, unit="%"
    )
    min_dist = gdi.IntItem(_("Minimum distance"), default=1, min=1, unit="points")


def compute_peak_detection(
    x: np.ndarray, y: np.ndarray, p: PeakDetectionParam
) -> tuple[np.ndarray, np.ndarray]:
    """Peak detection
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
        p (PeakDetectionParam): parameters
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    indexes = peak_indexes(y, thres=p.threshold * 0.01, min_dist=p.min_dist)
    return x[indexes], y[indexes]


class NormalizeParam(gdt.DataSet):
    """Normalize parameters"""

    methods = (
        (_("maximum"), "maximum"),
        (_("amplitude"), "amplitude"),
        (_("sum"), "sum"),
        (_("energy"), "energy"),
    )
    method = gdi.ChoiceItem(_("Normalize with respect to"), methods)


def compute_normalize(
    x: np.ndarray, y: np.ndarray, p: NormalizeParam
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize data
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
        p (NormalizeParam): parameters
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    return (x, normalize(y, p.method))


def compute_derivative(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute derivative
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    return (x, derivative(x, y))


def compute_integral(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute integral
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    return (x, spt.cumtrapz(y, x, initial=0.0))


def compute_threshold(
    x: np.ndarray, y: np.ndarray, p: ThresholdParam
) -> tuple[np.ndarray, np.ndarray]:
    """Compute threshold clipping
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
        p (ThresholdParam): parameters
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    return (x, np.clip(y, p.value, y.max()))


def compute_clip(
    x: np.ndarray, y: np.ndarray, p: ClipParam
) -> tuple[np.ndarray, np.ndarray]:
    """Compute maximum data clipping
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
        p (ClipParam): parameters
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    return (x, np.clip(y, y.min(), p.value))


def compute_gaussian_filter(
    x: np.ndarray, y: np.ndarray, p: GaussianParam
) -> tuple[np.ndarray, np.ndarray]:
    """Compute gaussian filter
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
        p (GaussianParam): parameters
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    return (x, spi.gaussian_filter1d(y, p.sigma))


def compute_moving_average(
    x: np.ndarray, y: np.ndarray, p: MovingAverageParam
) -> tuple[np.ndarray, np.ndarray]:
    """Compute moving average
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
        p (MovingAverageParam): parameters
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    return (x, moving_average(y, p.n))


def compute_moving_median(
    x: np.ndarray, y: np.ndarray, p: MovingMedianParam
) -> tuple[np.ndarray, np.ndarray]:
    """Compute moving median
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
        p (MovingMedianParam): parameters
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    return (x, sps.medfilt(y, kernel_size=p.n))


def compute_wiener(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Wiener filter
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    return (x, sps.wiener(y))


class XYCalibrateParam(gdt.DataSet):
    """Signal calibration parameters"""

    axes = (("x", _("X-axis")), ("y", _("Y-axis")))
    axis = gdi.ChoiceItem(_("Calibrate"), axes, default="y")
    a = gdi.FloatItem("a", default=1.0)
    b = gdi.FloatItem("b", default=0.0)


def compute_calibration(
    x: np.ndarray, y: np.ndarray, p: XYCalibrateParam
) -> tuple[np.ndarray, np.ndarray]:
    """Compute linear calibration
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
        p (XYCalibrateParam): parameters
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    if p.axis == "x":
        return p.a * x + p.b, y
    return x, p.a * y + p.b


class FFTParam(gdt.DataSet):
    """FFT parameters"""

    shift = gdi.BoolItem(
        _("Shift"),
        default=Conf.proc.fft_shift_enabled.get(),
        help=_("Shift zero frequency to center"),
    )


def compute_fft(
    x: np.ndarray, y: np.ndarray, p: FFTParam
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FFT
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
        p (FFTParam): parameters
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    return xy_fft(x, y, shift=p.shift)


def compute_ifft(
    x: np.ndarray, y: np.ndarray, p: FFTParam
) -> tuple[np.ndarray, np.ndarray]:
    """Compute iFFT
    Args:
        x (np.ndarray): X-axis data
        y (np.ndarray): Y-axis data
        p (FFTParam): parameters
    Returns:
        tuple[np.ndarray, np.ndarray]: X-axis data, Y-axis data
    """
    return xy_ifft(x, y, shift=p.shift)


class PolynomialFitParam(gdt.DataSet):
    """Polynomial fitting parameters"""

    degree = gdi.IntItem(_("Degree"), 3, min=1, max=10, slider=True)


class FWHMParam(gdt.DataSet):
    """FWHM parameters"""

    fittypes = (
        ("GaussianModel", _("Gaussian")),
        ("LorentzianModel", _("Lorentzian")),
        ("VoigtModel", "Voigt"),
    )

    fittype = gdi.ChoiceItem(_("Fit type"), fittypes, default="GaussianModel")


def compute_fwhm(signal: SignalObj, param: FWHMParam):
    """Compute FWHM"""
    res = []
    for i_roi in signal.iterate_roi_indexes():
        x, y = signal.get_data(i_roi)
        dx = np.max(x) - np.min(x)
        dy = np.max(y) - np.min(y)
        base = np.min(y)
        sigma, mu = dx * 0.1, xpeak(x, y)
        FitModel = getattr(fit, param.fittype)
        amp = FitModel.get_amp_from_amplitude(dy, sigma)

        def func(params):
            """Fitting model function"""
            # pylint: disable=cell-var-from-loop
            return y - FitModel.func(x, *params)

        (amp, sigma, mu, base), _ier = spo.leastsq(
            func, np.array([amp, sigma, mu, base])
        )
        x0, y0, x1, y1 = FitModel.half_max_segment(amp, sigma, mu, base)
        res.append([i_roi, x0, y0, x1, y1])
    return np.array(res)


def compute_fw1e2(signal: SignalObj):
    """Compute FW at 1/e²"""
    res = []
    for i_roi in signal.iterate_roi_indexes():
        x, y = signal.get_data(i_roi)
        dx = np.max(x) - np.min(x)
        dy = np.max(y) - np.min(y)
        base = np.min(y)
        sigma, mu = dx * 0.1, xpeak(x, y)
        amp = fit.GaussianModel.get_amp_from_amplitude(dy, sigma)
        p_in = np.array([amp, sigma, mu, base])

        def func(params):
            """Fitting model function"""
            # pylint: disable=cell-var-from-loop
            return y - fit.GaussianModel.func(x, *params)

        p_out, _ier = spo.leastsq(func, p_in)
        amp, sigma, mu, base = p_out
        hw = 2 * sigma
        amplitude = fit.GaussianModel.amplitude(amp, sigma)
        yhm = amplitude / np.e**2 + base
        res.append([i_roi, mu - hw, yhm, mu + hw, yhm])
    return np.array(res)
