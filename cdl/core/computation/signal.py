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

# Note:
# ----
# All dataset classes must also be imported in the cdl.core.computation.param module.

from __future__ import annotations

import guidata.dataset as gds
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
from cdl.config import _
from cdl.core.computation.base import (
    ClipParam,
    FFTParam,
    GaussianParam,
    MovingAverageParam,
    MovingMedianParam,
    ThresholdParam,
)
from cdl.core.model.signal import SignalObj

VALID_DTYPES_STRLIST = SignalObj.get_valid_dtypenames()


def dst_11(src: SignalObj, name: str, suffix: str | None = None) -> SignalObj:
    """Create result signal object for compute_11 function

    Args:
        src (SignalObj): source signal
        name (str): name of the function

    Returns:
        SignalObj: result signal object
    """
    dst = src.copy(title=f"{name}({src.short_id})")
    if suffix is not None:
        dst.title += "|" + suffix
    return dst


def dst_n1n(src1: SignalObj, src2: SignalObj, name: str, suffix: str | None = None):
    """Create result signal object for compute_n1n function

    Args:
        src1 (SignalObj): source signal 1
        src2 (SignalObj): source signal 2
        name (str): name of the function

    Returns:
        SignalObj: result signal object
    """
    dst = src1.copy(title=f"{name}({src1.short_id}, {src2.short_id})")
    if suffix is not None:
        dst.title += "|" + suffix
    return dst


# -------- compute_n1 functions --------------------------------------------------------
# Functions with N input signals and 1 output signal
# --------------------------------------------------------------------------------------
# Those functions are perfoming a computation on N input signals and return a single
# output signal. If we were only executing these functions locally, we would not need
# to define them here, but since we are using the multiprocessing module, we need to
# define them here so that they can be pickled and sent to the worker processes.
# Also, we need to systematically return the output signal object, even if it is already
# modified in place, because the multiprocessing module will not be able to retrieve
# the modified object from the worker processes.


def compute_add(dst: SignalObj, src: SignalObj) -> SignalObj:
    """Add signal to result signal
    Args:
        dst (SignalObj): destination signal
        src (SignalObj): source signal
    """
    dst.y += np.array(src.y, dtype=dst.y.dtype)
    if dst.dy is not None:
        dst.dy = np.sqrt(dst.dy**2 + src.dy**2)
    return dst


def compute_product(dst: SignalObj, src: SignalObj) -> SignalObj:
    """Multiply signal to result signal
    Args:
        dst (SignalObj): destination signal
        src (SignalObj): source signal
    """
    dst.y *= np.array(src.y, dtype=dst.y.dtype)
    if dst.dy is not None:
        dst.dy = dst.y * np.sqrt((dst.dy / dst.y) ** 2 + (src.dy / src.y) ** 2)
    return dst


# -------- compute_n1n functions -------------------------------------------------------
# Functions with N input images + 1 input image and N output images
# --------------------------------------------------------------------------------------


def compute_difference(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute difference between two signals
    Args:
        src1 (SignalObj): source signal 1
        src2 (SignalObj): source signal 2
    Returns:
        SignalObj: result signal object
    """
    dst = dst_n1n(src1, src2, "difference")
    dst.y = src1.y - src2.y
    if dst.dy is not None:
        dst.dy = np.sqrt(src1.dy**2 + src2.dy**2)
    return dst


def compute_quadratic_difference(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute quadratic difference between two signals
    Args:
        src1 (SignalObj): source signal 1
        src2 (SignalObj): source signal 2
    Returns:
        SignalObj: result signal object
    """
    dst = dst_n1n(src1, src2, "quadratic_difference")
    x1, y1 = src1.get_data()
    _x2, y2 = src2.get_data()
    dst.set_xydata(x1, (y1 - np.array(y2, dtype=y1.dtype)) / np.sqrt(2.0))
    if np.issubdtype(dst.data.dtype, np.unsignedinteger):
        dst.data[src1.data < src2.data] = 0
    if dst.dy is not None:
        dst.dy = np.sqrt(src1.dy**2 + src2.dy**2)
    return dst


def compute_division(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute division between two signals
    Args:
        src1 (SignalObj): source signal 1
        src2 (SignalObj): source signal 2
    Returns:
        SignalObj: result signal object
    """
    dst = dst_n1n(src1, src2, "division")
    x1, y1 = src1.get_data()
    _x2, y2 = src2.get_data()
    dst.set_xydata(x1, y1 / np.array(y2, dtype=y1.dtype))
    return dst


# -------- compute_11 functions --------------------------------------------------------
# Functions with 1 input image and 1 output image
# --------------------------------------------------------------------------------------


def extract_multiple_roi(src: SignalObj, group: gds.DataSetGroup) -> SignalObj:
    """Extract multiple regions of interest from data
    Args:
        src (SignalObj): source signal
        group (gds.DataSetGroup): group of parameters
    Returns:
        SignalObj: signal with multiple regions of interest
    """
    suffix = None
    if len(group.datasets) == 1:
        p = group.datasets[0]
        suffix = f"indexes={p.col1:d}:{p.col2:d}"
    dst = dst_11(src, "extract_multiple_roi", suffix)
    x, y = src.get_data()
    xout, yout = np.ones_like(x) * np.nan, np.ones_like(y) * np.nan
    for p in group.datasets:
        slice0 = slice(p.col1, p.col2 + 1)
        xout[slice0], yout[slice0] = x[slice0], y[slice0]
    nans = np.isnan(xout) | np.isnan(yout)
    dst.set_xydata(xout[~nans], yout[~nans])
    # TODO: [P2] Instead of removing geometric shapes, apply roi extract
    dst.remove_all_shapes()
    return dst


def extract_single_roi(src: SignalObj, p: gds.DataSet) -> SignalObj:
    """Extract single region of interest from data
    Args:
        src (SignalObj): source signal
        p (gds.DataSet): parameters
    Returns:
        SignalObj: signal with single region of interest
    """
    dst = dst_11(src, "extract_single_roi", f"indexes={p.col1:d}:{p.col2:d}")
    x, y = src.get_data()
    dst.set_xydata(x[p.col1 : p.col2 + 1], y[p.col1 : p.col2 + 1])
    # TODO: [P2] Instead of removing geometric shapes, apply roi extract
    dst.remove_all_shapes()
    return dst


def compute_swap_axes(src: SignalObj) -> SignalObj:
    """Swap axes
    Args:
        src (SignalObj): source signal
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(src, "swap_axes")
    x, y = src.get_data()
    dst.set_xydata(y, x)
    return dst


def compute_abs(src: SignalObj) -> SignalObj:
    """Compute absolute value
    Args:
        src (SignalObj): source signal
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(src, "abs")
    x, y = src.get_data()
    dst.set_xydata(x, np.abs(y))
    return dst


def compute_re(src: SignalObj) -> SignalObj:
    """Compute real part
    Args:
        src (SignalObj): source signal
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(src, "re")
    x, y = src.get_data()
    dst.set_xydata(x, np.real(y))
    return dst


def compute_im(src: SignalObj) -> SignalObj:
    """Compute imaginary part
    Args:
        src (SignalObj): source signal
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(src, "im")
    x, y = src.get_data()
    dst.set_xydata(x, np.imag(y))
    return dst


class DataTypeSParam(gds.DataSet):
    """Convert signal data type parameters"""

    dtype_str = gds.ChoiceItem(
        _("Destination data type"),
        list(zip(VALID_DTYPES_STRLIST, VALID_DTYPES_STRLIST)),
        help=_("Output image data type."),
    )


def compute_astype(src: SignalObj, p: DataTypeSParam) -> SignalObj:
    """Convert data type
    Args:
        src: source signal
        p: parameters
    Returns:
        Result signal object
    """
    dst = dst_11(src, "astype", f"dtype={p.dtype_str}")
    dst.xydata = src.xydata.astype(p.dtype_str)
    return dst


def compute_log10(src: SignalObj) -> SignalObj:
    """Compute Log10
    Args:
        src (SignalObj): source signal
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(src, "log10")
    x, y = src.get_data()
    dst.set_xydata(x, np.log10(y))
    return dst


class PeakDetectionParam(gds.DataSet):
    """Peak detection parameters"""

    threshold = gds.IntItem(
        _("Threshold"), default=30, min=0, max=100, slider=True, unit="%"
    )
    min_dist = gds.IntItem(_("Minimum distance"), default=1, min=1, unit="points")


def compute_peak_detection(src: SignalObj, p: PeakDetectionParam) -> SignalObj:
    """Peak detection
    Args:
        src (SignalObj): source signal
        p (PeakDetectionParam): parameters
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(
        src, "peak_detection", f"threshold={p.threshold}%, min_dist={p.min_dist}pts"
    )
    x, y = src.get_data()
    indexes = peak_indexes(y, thres=p.threshold * 0.01, min_dist=p.min_dist)
    dst.set_xydata(x[indexes], y[indexes])
    dst.metadata["curvestyle"] = "Sticks"
    return dst


class NormalizeYParam(gds.DataSet):
    """Normalize parameters"""

    methods = (
        (_("maximum"), "maximum"),
        (_("amplitude"), "amplitude"),
        (_("sum"), "sum"),
        (_("energy"), "energy"),
    )
    method = gds.ChoiceItem(_("Normalize with respect to"), methods)


def compute_normalize(src: SignalObj, p: NormalizeYParam) -> SignalObj:
    """Normalize data
    Args:
        src (SignalObj): source signal
        p (NormalizeYParam): parameters
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(src, "normalize", f"ref={p.method}")
    x, y = src.get_data()
    dst.set_xydata(x, normalize(y, p.method))
    return dst


def compute_derivative(src: SignalObj) -> SignalObj:
    """Compute derivative
    Args:
        src (SignalObj): source signal
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(src, "derivative")
    x, y = src.get_data()
    dst.set_xydata(x, derivative(x, y))
    return dst


def compute_integral(src: SignalObj) -> SignalObj:
    """Compute integral
    Args:
        src (SignalObj): source signal
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(src, "integral")
    x, y = src.get_data()
    dst.set_xydata(x, spt.cumtrapz(y, x, initial=0.0))
    return dst


class XYCalibrateParam(gds.DataSet):
    """Signal calibration parameters"""

    axes = (("x", _("X-axis")), ("y", _("Y-axis")))
    axis = gds.ChoiceItem(_("Calibrate"), axes, default="y")
    a = gds.FloatItem("a", default=1.0)
    b = gds.FloatItem("b", default=0.0)


def compute_calibration(src: SignalObj, p: XYCalibrateParam) -> SignalObj:
    """Compute linear calibration
    Args:
        src (SignalObj): source signal
        p (XYCalibrateParam): parameters
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(src, "calibration", f"{p.axis}={p.a}*{p.axis}+{p.b}")
    x, y = src.get_data()
    if p.axis == "x":
        dst.set_xydata(p.a * x + p.b, y)
    else:
        dst.set_xydata(x, p.a * y + p.b)
    return dst


def compute_threshold(src: SignalObj, p: ThresholdParam) -> SignalObj:
    """Compute threshold clipping
    Args:
        src (SignalObj): source signal
        p (ThresholdParam): parameters
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(src, "threshold", f"min={p.value}")
    x, y = src.get_data()
    dst.set_xydata(x, np.clip(y, p.value, y.max()))
    return dst


def compute_clip(src: SignalObj, p: ClipParam) -> SignalObj:
    """Compute maximum data clipping
    Args:
        src (SignalObj): source signal
        p (ClipParam): parameters
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(src, "clip", f"max={p.value}")
    x, y = src.get_data()
    dst.set_xydata(x, np.clip(y, y.min(), p.value))
    return dst


def compute_gaussian_filter(src: SignalObj, p: GaussianParam) -> SignalObj:
    """Compute gaussian filter
    Args:
        src (SignalObj): source signal
        p (GaussianParam): parameters
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(src, "gaussian_filter", f"σ={p.sigma:.3f}")
    x, y = src.get_data()
    dst.set_xydata(x, spi.gaussian_filter1d(y, p.sigma))
    return dst


def compute_moving_average(src: SignalObj, p: MovingAverageParam) -> SignalObj:
    """Compute moving average
    Args:
        src (SignalObj): source signal
        p (MovingAverageParam): parameters
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(src, "moving_average", f"n={p.n:d}")
    x, y = src.get_data()
    dst.set_xydata(x, moving_average(y, p.n))
    return dst


def compute_moving_median(src: SignalObj, p: MovingMedianParam) -> SignalObj:
    """Compute moving median
    Args:
        src (SignalObj): source signal
        p (MovingMedianParam): parameters
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(src, "moving_median", f"n={p.n:d}")
    x, y = src.get_data()
    dst.set_xydata(x, sps.medfilt(y, kernel_size=p.n))
    return dst


def compute_wiener(src: SignalObj) -> SignalObj:
    """Compute Wiener filter
    Args:
        src (SignalObj): source signal
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(src, "wiener")
    x, y = src.get_data()
    dst.set_xydata(x, sps.wiener(y))
    return dst


def compute_fft(src: SignalObj, p: FFTParam) -> SignalObj:
    """Compute FFT
    Args:
        src (SignalObj): source signal
        p (FFTParam): parameters
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(src, "fft")
    x, y = src.get_data()
    dst.set_xydata(*xy_fft(x, y, shift=p.shift))
    return dst


def compute_ifft(src: SignalObj, p: FFTParam) -> SignalObj:
    """Compute iFFT
    Args:
        src (SignalObj): source signal
        p (FFTParam): parameters
    Returns:
        SignalObj: result signal object
    """
    dst = dst_11(src, "ifft")
    x, y = src.get_data()
    dst.set_xydata(*xy_ifft(x, y, shift=p.shift))
    return dst


class PolynomialFitParam(gds.DataSet):
    """Polynomial fitting parameters"""

    degree = gds.IntItem(_("Degree"), 3, min=1, max=10, slider=True)


class FWHMParam(gds.DataSet):
    """FWHM parameters"""

    fittypes = (
        ("GaussianModel", _("Gaussian")),
        ("LorentzianModel", _("Lorentzian")),
        ("VoigtModel", "Voigt"),
    )

    fittype = gds.ChoiceItem(_("Fit type"), fittypes, default="GaussianModel")


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
