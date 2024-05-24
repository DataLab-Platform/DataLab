# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Signal computation objects (see parent package :mod:`cdl.core.computation`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# MARK: Important note
# --------------------
# All `guidata.dataset.DataSet` classes must also be imported in the `cdl.param` module.

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

import guidata.dataset as gds
import numpy as np
import scipy.integrate as spt
import scipy.ndimage as spi
import scipy.signal as sps

import cdl.algorithms.signal as alg
from cdl.config import _
from cdl.core.computation.base import (
    ClipParam,
    ConstantOperationParam,
    FFTParam,
    GaussianParam,
    HistogramParam,
    MovingAverageParam,
    MovingMedianParam,
    NormalizeParam,
    ThresholdParam,
    calc_resultproperties,
    dst_11,
    dst_n1n,
    new_signal_result,
)
from cdl.obj import ResultProperties, ResultShape, ShapeTypes, SignalObj

VALID_DTYPES_STRLIST = SignalObj.get_valid_dtypenames()


class Wrap11Func:
    """Wrap a 1 array → 1 array function (the simple case of y1 = f(y0)) to produce
    a 1 signal → 1 signal function, which can be used inside DataLab's infrastructure
    to perform computations with :class:`cdl.core.gui.processor.signal.SignalProcessor`.

    This wrapping mechanism using a class is necessary for the resulted function to be
    pickable by the ``multiprocessing`` module.

    The instance of this wrapper is callable and returns a :class:`cdl.obj.SignalObj`
    object.

    Example:

        >>> import numpy as np
        >>> from cdl.core.computation.signal import Wrap11Func
        >>> import cdl.obj
        >>> def square(y):
        ...     return y**2
        >>> compute_square = Wrap11Func(square)
        >>> x = np.linspace(0, 10, 100)
        >>> y = np.sin(x)
        >>> sig0 = cdl.obj.create_signal("Example", x, y)
        >>> sig1 = compute_square(sig0)

    Args:
        func: 1 array → 1 array function
    """

    def __init__(self, func: Callable) -> None:
        self.func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__call__.__func__.__doc__ = self.func.__doc__

    def __call__(self, src: SignalObj) -> SignalObj:
        dst = dst_11(src, self.func.__name__)
        x, y = src.get_data()
        dst.set_xydata(x, self.func(y))
        return dst


# MARK: compute_n1 functions -----------------------------------------------------------
# Functions with N input signals and 1 output signal
# --------------------------------------------------------------------------------------
# Those functions are perfoming a computation on N input signals and return a single
# output signal. If we were only executing these functions locally, we would not need
# to define them here, but since we are using the multiprocessing module, we need to
# define them here so that they can be pickled and sent to the worker processes.
# Also, we need to systematically return the output signal object, even if it is already
# modified in place, because the multiprocessing module will not be able to retrieve
# the modified object from the worker processes.


def compute_addition(dst: SignalObj, src: SignalObj) -> SignalObj:
    """Add **dst** and **src** signals and return **dst** signal modified in place

    Args:
        dst: destination signal
        src: source signal

    Returns:
        Modified **dst** signal (modified in place)
    """
    dst.y += np.array(src.y, dtype=dst.y.dtype)
    if dst.dy is not None:
        dst.dy = np.sqrt(dst.dy**2 + src.dy**2)
    return dst


def compute_product(dst: SignalObj, src: SignalObj) -> SignalObj:
    """Multiply **dst** and **src** signals and return **dst** signal modified in place

    Args:
        dst: destination signal
        src: source signal

    Returns:
        Modified **dst** signal (modified in place)
    """
    dst.y *= np.array(src.y, dtype=dst.y.dtype)
    if dst.dy is not None:
        dst.dy = dst.y * np.sqrt((dst.dy / dst.y) ** 2 + (src.dy / src.y) ** 2)
    return dst


def compute_addition_constant(src: SignalObj, p: ConstantOperationParam) -> SignalObj:
    """Add **dst** and a constant value and return a the new result signal object

    Args:
        src: input signal object
        p: constant value

    Returns:
        Result signal object **src** + **p.value** (new object)
    """
    dst = dst_11(src, "+", str(p.value))
    dst.y += p.value
    return dst


def compute_difference_constant(src: SignalObj, p: ConstantOperationParam) -> SignalObj:
    """Subtract a constant value from a signal

    Args:
        src: input signal object
        p: constant value

    Returns:
        Result signal object **src** - **p.value** (new object)
    """
    dst = dst_11(src, "-", str(p.value))
    dst.y -= p.value
    return dst


def compute_product_constant(src: SignalObj, p: ConstantOperationParam) -> SignalObj:
    """Multiply **dst** by a constant value and return the new result signal object

    Args:
        src: input signal object
        p: constant value

    Returns:
        Result signal object **src** * **p.value** (new object)
    """
    dst = dst_11(src, "×", str(p.value))
    dst.y *= p.value
    return dst


def compute_division_constant(src: SignalObj, p: ConstantOperationParam) -> SignalObj:
    """Divide a signal by a constant value

    Args:
        src: input signal object
        p: constant value

    Returns:
        Result signal object **src** / **p.value** (new object)
    """
    dst = dst_11(src, "/", str(p.value))
    dst.y /= p.value
    return dst


# MARK: compute_n1n functions ----------------------------------------------------------
# Functions with N input images + 1 input image and N output images
# --------------------------------------------------------------------------------------


def compute_difference(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute difference between two signals

    .. note::

        If uncertainty is available, it is propagated.

    Args:
        src1: source signal 1
        src2: source signal 2

    Returns:
        Result signal object **src1** - **src2**
    """
    dst = dst_n1n(src1, src2, "-")
    dst.y = src1.y - src2.y
    if dst.dy is not None:
        dst.dy = np.sqrt(src1.dy**2 + src2.dy**2)
    return dst


def compute_quadratic_difference(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute quadratic difference between two signals

    .. note::

        If uncertainty is available, it is propagated.

    Args:
        src1: source signal 1
        src2: source signal 2

    Returns:
        Result signal object (**src1** - **src2**) / sqrt(2.0)
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
        src1: source signal 1
        src2: source signal 2

    Returns:
        Result signal object **src1** / **src2**
    """
    dst = dst_n1n(src1, src2, "/")
    x1, y1 = src1.get_data()
    _x2, y2 = src2.get_data()
    dst.set_xydata(x1, y1 / np.array(y2, dtype=y1.dtype))
    return dst


# MARK: compute_11 functions -----------------------------------------------------------
# Functions with 1 input image and 1 output image
# --------------------------------------------------------------------------------------


def extract_multiple_roi(src: SignalObj, group: gds.DataSetGroup) -> SignalObj:
    """Extract multiple regions of interest from data

    Args:
        src: source signal
        group: group of parameters

    Returns:
        Signal with multiple regions of interest
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
        src: source signal
        p: parameters

    Returns:
        Signal with single region of interest
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
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_11(src, "swap_axes")
    x, y = src.get_data()
    dst.set_xydata(y, x)
    return dst


def compute_abs(src: SignalObj) -> SignalObj:
    """Compute absolute value

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap11Func(np.abs)(src)


def compute_re(src: SignalObj) -> SignalObj:
    """Compute real part

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap11Func(np.real)(src)


def compute_im(src: SignalObj) -> SignalObj:
    """Compute imaginary part

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap11Func(np.imag)(src)


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
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap11Func(np.log10)(src)


def compute_exp(src: SignalObj) -> SignalObj:
    """Compute exponential

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap11Func(np.exp)(src)


def compute_sqrt(src: SignalObj) -> SignalObj:
    """Compute square root

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap11Func(np.sqrt)(src)


class PowParam(gds.DataSet):
    """Power parameters"""

    power = gds.FloatItem(_("Power"), default=2.0)


def compute_pow(src: SignalObj, p: PowParam) -> SignalObj:
    """Compute power

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_11(src, "pow", f"n={p.power}")
    x, y = src.get_data()
    dst.set_xydata(x, y**p.power)
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
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_11(
        src, "peak_detection", f"threshold={p.threshold}%, min_dist={p.min_dist}pts"
    )
    x, y = src.get_data()
    indexes = alg.peak_indexes(y, thres=p.threshold * 0.01, min_dist=p.min_dist)
    dst.set_xydata(x[indexes], y[indexes])
    dst.metadata["curvestyle"] = "Sticks"
    return dst


def compute_normalize(src: SignalObj, p: NormalizeParam) -> SignalObj:
    """Normalize data

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_11(src, "normalize", f"ref={p.method}")
    x, y = src.get_data()
    dst.set_xydata(x, alg.normalize(y, p.method))
    return dst


def compute_derivative(src: SignalObj) -> SignalObj:
    """Compute derivative

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_11(src, "derivative")
    x, y = src.get_data()
    dst.set_xydata(x, alg.derivative(x, y))
    return dst


def compute_integral(src: SignalObj) -> SignalObj:
    """Compute integral

    Args:
        src: source signal

    Returns:
        Result signal object
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
        src: source signal
        p: parameters

    Returns:
        Result signal object
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
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_11(src, "threshold", f"min={p.value}")
    x, y = src.get_data()
    dst.set_xydata(x, np.clip(y, p.value, y.max()))
    return dst


def compute_clip(src: SignalObj, p: ClipParam) -> SignalObj:
    """Compute maximum data clipping

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_11(src, "clip", f"max={p.value}")
    x, y = src.get_data()
    dst.set_xydata(x, np.clip(y, y.min(), p.value))
    return dst


def compute_gaussian_filter(src: SignalObj, p: GaussianParam) -> SignalObj:
    """Compute gaussian filter

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_11(src, "gaussian_filter", f"σ={p.sigma:.3f}")
    x, y = src.get_data()
    dst.set_xydata(x, spi.gaussian_filter1d(y, p.sigma))
    return dst


def compute_moving_average(src: SignalObj, p: MovingAverageParam) -> SignalObj:
    """Compute moving average

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_11(src, "moving_average", f"n={p.n:d}")
    x, y = src.get_data()
    dst.set_xydata(x, alg.moving_average(y, p.n))
    return dst


def compute_moving_median(src: SignalObj, p: MovingMedianParam) -> SignalObj:
    """Compute moving median

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_11(src, "moving_median", f"n={p.n:d}")
    x, y = src.get_data()
    dst.set_xydata(x, sps.medfilt(y, kernel_size=p.n))
    return dst


def compute_wiener(src: SignalObj) -> SignalObj:
    """Compute Wiener filter

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap11Func(sps.wiener)(src)


class FilterType(Enum):
    """Filter types"""

    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"


class BaseHighLowBandParam(gds.DataSet):
    """Base class for high-pass, low-pass, band-pass and band-stop filters"""

    methods = (
        ("bessel", _("Bessel")),
        ("butter", _("Butterworth")),
        ("cheby1", _("Chebyshev type 1")),
        ("cheby2", _("Chebyshev type 2")),
        ("ellip", _("Elliptic")),
    )

    TYPE: FilterType = FilterType.LOWPASS
    _type_prop = gds.GetAttrProp("TYPE")

    # Must be overwriten by the child class
    _method_prop = gds.GetAttrProp("method")
    method = gds.ChoiceItem(_("Filter method"), methods).set_prop(
        "display", store=_method_prop
    )

    order = gds.IntItem(_("Filter order"), default=3, min=1)
    f_cut0 = gds.FloatItem(
        _("Low cutoff frequency"), min=0, nonzero=True, unit="Hz"
    ).set_prop(
        "display", hide=gds.FuncProp(_type_prop, lambda x: x is FilterType.HIGHPASS)
    )
    f_cut1 = gds.FloatItem(
        _("High cutoff frequency"), min=0, nonzero=True, unit="Hz"
    ).set_prop(
        "display", hide=gds.FuncProp(_type_prop, lambda x: x is FilterType.LOWPASS)
    )
    rp = gds.FloatItem(
        _("Passband ripple"), min=0, default=1, nonzero=True, unit="dB"
    ).set_prop(
        "display",
        active=gds.FuncProp(_method_prop, lambda x: x in ("cheby1", "ellip")),
    )
    rs = gds.FloatItem(
        _("Stopband attenuation"), min=0, default=1, nonzero=True, unit="dB"
    ).set_prop(
        "display",
        active=gds.FuncProp(_method_prop, lambda x: x in ("cheby2", "ellip")),
    )

    @staticmethod
    def get_nyquist_frequency(obj: SignalObj) -> float:
        """Return the Nyquist frequency of a signal object

        Args:
            obj: signal object
        """
        fs = float(obj.x.size - 1) / (obj.x[-1] - obj.x[0])
        return fs / 2.0

    def update_from_signal(self, obj: SignalObj) -> None:
        """Update the filter parameters from a signal object

        Args:
            obj: signal object
        """
        f_nyquist = self.get_nyquist_frequency(obj)
        if self.f_cut0 is None:
            self.f_cut0 = 0.1 * f_nyquist
        if self.f_cut1 is None:
            self.f_cut1 = 0.9 * f_nyquist

    def get_filter_params(self, obj: SignalObj) -> tuple[float | str, float | str]:
        """Return the filter parameters (a and b) as a tuple. These parameters are used
        in the scipy.signal filter functions (eg. `scipy.signal.filtfilt`).

        Args:
            obj: signal object

        Returns:
            tuple: filter parameters
        """
        f_nyquist = self.get_nyquist_frequency(obj)
        func = getattr(sps, self.method)
        args: list[float | str | tuple[float, ...]] = [self.order]  # type: ignore
        if self.method == "cheby1":
            args += [self.rp]
        elif self.method == "cheby2":
            args += [self.rs]
        elif self.method == "ellip":
            args += [self.rp, self.rs]
        if self.TYPE is FilterType.HIGHPASS:
            args += [self.f_cut1 / f_nyquist]
        elif self.TYPE is FilterType.LOWPASS:
            args += [self.f_cut0 / f_nyquist]
        else:
            args += [[self.f_cut0 / f_nyquist, self.f_cut1 / f_nyquist]]
        args += [self.TYPE.value]
        return func(*args)


class LowPassFilterParam(BaseHighLowBandParam):
    """Low-pass filter parameters"""

    TYPE = FilterType.LOWPASS


class HighPassFilterParam(BaseHighLowBandParam):
    """High-pass filter parameters"""

    TYPE = FilterType.HIGHPASS


class BandPassFilterParam(BaseHighLowBandParam):
    """Band-pass filter parameters"""

    TYPE = FilterType.BANDPASS


class BandStopFilterParam(BaseHighLowBandParam):
    """Band-stop filter parameters"""

    TYPE = FilterType.BANDSTOP


def compute_filter(src: SignalObj, p: BaseHighLowBandParam) -> SignalObj:
    """Compute frequency filter (low-pass, high-pass, band-pass, band-stop)

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    name = f"{p.TYPE.value}"
    suffix = f"order={p.order:d}"
    if p.TYPE is FilterType.LOWPASS:
        suffix += f", cutoff={p.f_cut0:.2f}"
    elif p.TYPE is FilterType.HIGHPASS:
        suffix += f", cutoff={p.f_cut1:.2f}"
    else:
        suffix += f", cutoff={p.f_cut0:.2f}:{p.f_cut1:.2f}"
    dst = dst_11(src, name, suffix)
    b, a = p.get_filter_params(dst)
    dst.y = sps.filtfilt(b, a, dst.y)
    return dst


def compute_fft(src: SignalObj, p: FFTParam | None = None) -> SignalObj:
    """Compute FFT

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_11(src, "fft")
    x, y = src.get_data()
    dst.set_xydata(*alg.xy_fft(x, y, shift=True if p is None else p.shift))
    return dst


def compute_ifft(src: SignalObj, p: FFTParam | None = None) -> SignalObj:
    """Compute iFFT

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_11(src, "ifft")
    x, y = src.get_data()
    dst.set_xydata(*alg.xy_ifft(x, y, shift=True if p is None else p.shift))
    return dst


class PolynomialFitParam(gds.DataSet):
    """Polynomial fitting parameters"""

    degree = gds.IntItem(_("Degree"), 3, min=1, max=10, slider=True)


def compute_histogram(src: SignalObj, p: HistogramParam) -> SignalObj:
    """Compute histogram

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    # Extract data from ROIs:
    datalist = []
    for i_roi in src.iterate_roi_indexes():
        datalist.append(src.get_data(i_roi)[1])
    data = np.concatenate(datalist)

    suffix = p.get_suffix(data)  # Also updates p.lower and p.upper

    # Compute histogram:
    y, bin_edges = np.histogram(data, bins=p.bins, range=(p.lower, p.upper))
    x = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Note: we use the `new_signal_result` function to create the result signal object
    # because the `dst_11` would copy the source signal, which is not what we want here
    # (we want a brand new signal object).
    dst = new_signal_result(
        src,
        "histogram",
        suffix=suffix,
        units=(src.yunit, ""),
        labels=(src.ylabel, _("Counts")),
    )
    dst.set_xydata(x, y)
    dst.metadata["shade"] = 0.5
    return dst


class InterpolationParam(gds.DataSet):
    """Interpolation parameters"""

    methods = (
        ("linear", _("Linear")),
        ("spline", _("Spline")),
        ("quadratic", _("Quadratic")),
        ("cubic", _("Cubic")),
        ("barycentric", _("Barycentric")),
        ("pchip", _("PCHIP")),
    )
    method = gds.ChoiceItem(_("Interpolation method"), methods, default="linear")
    fill_value = gds.FloatItem(
        _("Fill value"),
        default=None,
        help=_(
            "Value to use for points outside the interpolation domain (used only "
            "with linear, cubic and pchip methods)."
        ),
        check=False,
    )


def compute_interpolation(
    src1: SignalObj, src2: SignalObj, p: InterpolationParam
) -> SignalObj:
    """Interpolate data

    Args:
        src1: source signal 1
        src2: source signal 2
        p: parameters

    Returns:
        Result signal object
    """
    suffix = f"method={p.method}"
    if p.fill_value is not None and p.method in ("linear", "cubic", "pchip"):
        suffix += f", fill_value={p.fill_value}"
    dst = dst_n1n(src1, src2, "interpolation", suffix)
    x1, y1 = src1.get_data()
    xnew, _y2 = src2.get_data()
    ynew = alg.interpolate(x1, y1, xnew, p.method, p.fill_value)
    dst.set_xydata(xnew, ynew)
    return dst


class ResamplingParam(InterpolationParam):
    """Resample parameters"""

    xmin = gds.FloatItem(_("X<sub>min</sub>"))
    xmax = gds.FloatItem(_("X<sub>max</sub>"))
    _prop = gds.GetAttrProp("dx_or_nbpts")
    _modes = (("dx", "ΔX"), ("nbpts", _("Number of points")))
    mode = gds.ChoiceItem(_("Mode"), _modes, default="nbpts", radio=True).set_prop(
        "display", store=_prop
    )
    dx = gds.FloatItem("ΔX").set_prop(
        "display", active=gds.FuncProp(_prop, lambda x: x == "dx")
    )
    nbpts = gds.IntItem(_("Number of points")).set_prop(
        "display", active=gds.FuncProp(_prop, lambda x: x == "nbpts")
    )


def compute_resampling(src: SignalObj, p: ResamplingParam) -> SignalObj:
    """Resample data

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    suffix = f"method={p.method}"
    if p.fill_value is not None and p.method in ("linear", "cubic", "pchip"):
        suffix += f", fill_value={p.fill_value}"
    if p.mode == "dx":
        suffix += f", dx={p.dx:.3f}"
    else:
        suffix += f", nbpts={p.nbpts:d}"
    dst = dst_11(src, "resample", suffix)
    x, y = src.get_data()
    if p.mode == "dx":
        xnew = np.arange(p.xmin, p.xmax, p.dx)
    else:
        xnew = np.linspace(p.xmin, p.xmax, p.nbpts)
    ynew = alg.interpolate(x, y, xnew, p.method, p.fill_value)
    dst.set_xydata(xnew, ynew)
    return dst


class DetrendingParam(gds.DataSet):
    """Detrending parameters"""

    methods = (("linear", _("Linear")), ("constant", _("Constant")))
    method = gds.ChoiceItem(_("Detrending method"), methods, default="linear")


def compute_detrending(src: SignalObj, p: DetrendingParam) -> SignalObj:
    """Detrend data

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_11(src, "detrending", f"method={p.method}")
    x, y = src.get_data()
    dst.set_xydata(x, sps.detrend(y, type=p.method))
    return dst


def compute_convolution(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute convolution of two signals

    Args:
        src1: source signal 1
        src2: source signal 2

    Returns:
        Result signal object
    """
    dst = dst_n1n(src1, src2, "⊛")
    x1, y1 = src1.get_data()
    _x2, y2 = src2.get_data()
    ynew = np.real(sps.convolve(y1, y2, mode="same"))
    dst.set_xydata(x1, ynew)
    return dst


class WindowingParam(gds.DataSet):
    """Windowing parameters"""

    methods = (
        ("barthann", "Barthann"),
        ("bartlett", "Bartlett"),
        ("blackman", "Blackman"),
        ("blackman-harris", "Blackman-Harris"),
        ("bohman", "Bohman"),
        ("boxcar", "Boxcar"),
        ("cosine", _("Cosine")),
        ("exponential", _("Exponential")),
        ("flat-top", _("Flat top")),
        ("gaussian", _("Gaussian")),
        ("hamming", "Hamming"),
        ("hanning", "Hanning"),
        ("kaiser", "Kaiser"),
        ("lanczos", "Lanczos"),
        ("nuttall", "Nuttall"),
        ("parzen", "Parzen"),
        ("rectangular", _("Rectangular")),
        ("taylor", "Taylor"),
        ("tukey", "Tukey"),
    )
    _meth_prop = gds.GetAttrProp("method")
    method = gds.ChoiceItem(_("Method"), methods, default="hamming").set_prop(
        "display", store=_meth_prop
    )
    alpha = gds.FloatItem(
        "Alpha",
        default=0.5,
        help=_("Shape parameter of the Tukey windowing function"),
    ).set_prop("display", active=gds.FuncProp(_meth_prop, lambda x: x == "tukey"))
    beta = gds.FloatItem(
        "Beta",
        default=14.0,
        help=_("Shape parameter of the Kaiser windowing function"),
    ).set_prop("display", active=gds.FuncProp(_meth_prop, lambda x: x == "kaiser"))
    sigma = gds.FloatItem(
        "Sigma",
        default=0.5,
        help=_("Shape parameter of the Gaussian windowing function"),
    ).set_prop("display", active=gds.FuncProp(_meth_prop, lambda x: x == "gaussian"))


def compute_windowing(src: SignalObj, p: WindowingParam) -> SignalObj:
    """Compute windowing (available methods: hamming, hanning, bartlett, blackman,
    tukey, rectangular)

    Args:
        dst: destination signal
        src: source signal

    Returns:
        Result signal object
    """
    suffix = f"method={p.method}"
    if p.method == "tukey":
        suffix += f", alpha={p.alpha:.3f}"
    elif p.method == "kaiser":
        suffix += f", beta={p.beta:.3f}"
    elif p.method == "gaussian":
        suffix += f", sigma={p.sigma:.3f}"
    dst = dst_11(src, "windowing", suffix)  # type: ignore
    dst.y = alg.windowing(dst.y, p.method, p.alpha)  # type: ignore
    return dst


def compute_reverse_x(src: SignalObj) -> SignalObj:
    """Reverse x-axis

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_11(src, "reverse_x")
    dst.y = dst.y[::-1]
    return dst


# MARK: compute_10 functions -----------------------------------------------------------
# Functions with 1 input signal and 0 output signals (ResultShape or ResultProperties)
# --------------------------------------------------------------------------------------


def calc_resultshape(
    label: str, shapetype: ShapeTypes, obj: SignalObj, func: Callable, *args: Any
) -> ResultShape | None:
    """Calculate result shape by executing a computation function on a signal object,
    taking into account the signal ROIs.

    Args:
        label: result shape label
        shapetype: result shape type
        obj: input image object
        func: computation function
        *args: computation function arguments

    Returns:
        Result shape object or None if no result is found

    .. warning::

        The computation function must take either a single argument (the data) or
        multiple arguments (the data followed by the computation parameters).

        Moreover, the computation function must return a 1D NumPy array (or a list,
        or a tuple) containing the result of the computation.
    """
    res = []
    for i_roi in obj.iterate_roi_indexes():
        data_roi = obj.get_data(i_roi)
        if args is None:
            results: np.ndarray = func(data_roi)
        else:
            results: np.ndarray = func(data_roi, *args)
        if not isinstance(results, (np.ndarray, list, tuple)):
            raise ValueError(
                "The computation function must return a NumPy array, a list or a tuple"
            )
        results = np.array(results)
        if results.size:
            if results.ndim != 1:
                raise ValueError(
                    "The computation function must return a 1D NumPy array"
                )
            results = np.array([i_roi] + results.tolist())
            res.append(results)
    if res:
        return ResultShape(label, np.vstack(res), shapetype)
    return None


class FWHMParam(gds.DataSet):
    """FWHM parameters"""

    methods = (
        ("zero-crossing", _("Zero-crossing")),
        ("gauss", _("Gaussian fit")),
        ("lorentz", _("Lorentzian fit")),
        ("voigt", _("Voigt fit")),
    )
    method = gds.ChoiceItem(_("Method"), methods, default="zero-crossing")
    xmin = gds.FloatItem(
        "X<sub>MIN</sub>",
        default=None,
        check=False,
        help=_("Lower X boundary (empty for no limit, i.e. start of the signal)"),
    )
    xmax = gds.FloatItem(
        "X<sub>MAX</sub>",
        default=None,
        check=False,
        help=_("Upper X boundary (empty for no limit, i.e. end of the signal)"),
    )


def compute_fwhm(obj: SignalObj, param: FWHMParam) -> ResultShape | None:
    """Compute FWHM

    Args:
        obj: source signal
        param: parameters

    Returns:
        Segment coordinates
    """
    return calc_resultshape(
        "fwhm",
        ShapeTypes.SEGMENT,
        obj,
        alg.fwhm,
        param.method,
        param.xmin,
        param.xmax,
    )


def compute_fw1e2(obj: SignalObj) -> ResultShape | None:
    """Compute FW at 1/e²

    Args:
        obj: source signal

    Returns:
        Segment coordinates
    """
    return calc_resultshape("fw1e2", ShapeTypes.SEGMENT, obj, alg.fw1e2)


def compute_stats(obj: SignalObj) -> ResultProperties:
    """Compute statistics on a signal

    Args:
        obj: source signal

    Returns:
        Result properties object
    """
    statfuncs = {
        "min(y)": lambda xy: xy[1].min(),
        "max(y)": lambda xy: xy[1].max(),
        "<y>": lambda xy: xy[1].mean(),
        "median(y)": lambda xy: np.median(xy[1]),
        "σ(y)": lambda xy: xy[1].std(),
        "<y>/σ(y)": lambda xy: xy[1].mean() / xy[1].std(),
        "peak-to-peak(y)": lambda xy: xy[1].ptp(),
        "Σ(y)": lambda xy: xy[1].sum(),
        "∫ydx": lambda xy: np.trapz(xy[1], xy[0]),
    }
    return calc_resultproperties("stats", obj, statfuncs)


def compute_bandwidth_3db(obj: SignalObj) -> ResultProperties:
    """Compute bandwidth

    Args:
        obj: source signal

    Returns:
        Result properties with bandwidth
    """
    return calc_resultshape("bandwidth", ShapeTypes.SEGMENT, obj, alg.bandwidth, 3.0)


class DynamicParam(gds.DataSet):
    """Parameters for dynamic range computation (ENOB, SNR, SINAD, THD, SFDR)"""

    full_scale = gds.FloatItem(_("Full scale (V)"), default=0.16, min=0.0)
    _units = ("dBc", "dBFS")
    unit = gds.ChoiceItem(
        _("Unit"), zip(_units, _units), default="dBc", help=_("Unit for SINAD")
    )
    nb_harm = gds.IntItem(
        _("Number of harmonics"),
        default=5,
        min=1,
        help=_("Number of harmonics to consider for THD"),
    )


def compute_dynamic_parameters(src: SignalObj, p: DynamicParam) -> ResultProperties:
    """Compute Dynamic parameters (ENOB, SNR, SINAD, THD, SFDR)

    Args:
        src: source signal
        p: parameters

    Returns:
        Result properties with ENOB, SNR, SINAD, THD, SFDR
    """
    funcs = {
        "f": lambda xy: alg.sinus_frequency(xy[0], xy[1]),
        "ENOB": lambda xy: alg.enob(xy[0], xy[1], p.full_scale),
        "SNR": lambda xy: alg.snr(xy[0], xy[1], p.unit),
        "SINAD": lambda xy: alg.sinad(xy[0], xy[1], p.unit),
        "THD": lambda xy: alg.thd(xy[0], xy[1], p.full_scale, p.unit, p.nb_harm),
        "SFDR": lambda xy: alg.sfdr(xy[0], xy[1], p.full_scale, p.unit),
    }
    return calc_resultproperties("ADC", src, funcs)


def compute_sampling_rate_period(obj: SignalObj) -> ResultProperties:
    """Compute sampling rate and period

    Args:
        obj: source signal

    Returns:
        Result properties with sampling rate and period
    """
    return calc_resultproperties(
        "sampling_rate_period",
        obj,
        {
            "fs": lambda xy: alg.sampling_rate(xy[0]),
            "T": lambda xy: alg.sampling_period(xy[0]),
        },
    )


def compute_contrast(obj: SignalObj) -> ResultProperties:
    """Compute contrast"""
    return calc_resultproperties(
        "contrast",
        obj,
        {
            "contrast": lambda xy: alg.contrast(xy[1]),
        },
    )


def compute_x_at_minmax(obj: SignalObj) -> ResultProperties:
    """Compute x at min/max"""
    return calc_resultproperties(
        "x@min,max",
        obj,
        {
            "X@Ymin": lambda xy: xy[0][np.argmin(xy[1])],
            "X@Ymax": lambda xy: xy[0][np.argmax(xy[1])],
        },
    )
