# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Signal computation objects (see parent package :mod:`sigima_.computation`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# MARK: Important note
# --------------------
# All `guidata.dataset.DataSet` classes must also be imported
# in the `sigima_.param` module.

from __future__ import annotations

import warnings
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

import guidata.dataset as gds
import numpy as np
import scipy.integrate as spt
import scipy.ndimage as spi
import scipy.signal as sps

from cdl.config import Conf, _
from sigima_.algorithms import coordinates
from sigima_.algorithms.signal import (
    dynamic,
    features,
    fourier,
    interpolation,
    level,
    peakdetection,
    pulse,
    stability,
    windowing,
)
from sigima_.computation import computation_function
from sigima_.computation.base import (
    ArithmeticParam,
    ClipParam,
    ConstantParam,
    FFTParam,
    GaussianParam,
    HistogramParam,
    MovingAverageParam,
    MovingMedianParam,
    NormalizeParam,
    SpectrumParam,
    calc_resultproperties,
    dst_1_to_1,
    dst_2_to_1,
    dst_n_to_1,
    new_signal_result,
)
from sigima_.obj.base import ResultProperties, ResultShape
from sigima_.obj.signal import ROI1DParam, SignalObj


def restore_data_outside_roi(dst: SignalObj, src: SignalObj) -> None:
    """Restore data outside the Region Of Interest (ROI) of the input signal
    after a computation, only if the input signal has a ROI,
    and if the output signal has the same ROI as the input signal,
    and if the data types are the same,
    and if the shapes are the same.
    Otherwise, do nothing.

    Args:
        dst: destination signal object
        src: source signal object
    """
    if src.maskdata is not None and dst.maskdata is not None:
        if (
            np.array_equal(src.maskdata, dst.maskdata)
            and dst.xydata.dtype == src.xydata.dtype
            and dst.xydata.shape == src.xydata.shape
        ):
            dst.xydata[src.maskdata] = src.xydata[src.maskdata]


class Wrap1to1Func:
    """Wrap a 1 array → 1 array function (the simple case of y1 = f(y0)) to produce
    a 1 signal → 1 signal function, which can be used inside DataLab's infrastructure
    to perform computations with :class:`cdl.gui.processor.signal.SignalProcessor`.

    This wrapping mechanism using a class is necessary for the resulted function to be
    pickable by the ``multiprocessing`` module.

    The instance of this wrapper is callable and returns
    a :class:`sigima_.obj.SignalObj` object.

    Example:

        >>> import numpy as np
        >>> from sigima_.computation.signal import Wrap1to1Func
        >>> import sigima_.obj
        >>> def square(y):
        ...     return y**2
        >>> compute_square = Wrap1to1Func(square)
        >>> x = np.linspace(0, 10, 100)
        >>> y = np.sin(x)
        >>> sig0 = sigima_.obj.create_signal("Example", x, y)
        >>> sig1 = compute_square(sig0)

    Args:
        func: 1 array → 1 array function
        *args: Additional positional arguments to pass to the function
        **kwargs: Additional keyword arguments to pass to the function
    """

    def __init__(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__call__.__func__.__doc__ = self.func.__doc__

    def __call__(self, src: SignalObj) -> SignalObj:
        """Compute the function on the input signal and return the result signal

        Args:
            src: input signal object

        Returns:
            Result signal object
        """
        suffix = ", ".join(
            [str(arg) for arg in self.args]
            + [f"{k}={v}" for k, v in self.kwargs.items() if v is not None]
        )
        dst = dst_1_to_1(src, self.func.__name__, suffix)
        x, y = src.get_data()
        dst.set_xydata(x, self.func(y, *self.args, **self.kwargs))
        restore_data_outside_roi(dst, src)
        return dst


# MARK: n_to_1 functions -------------------------------------------------------
# Functions with N input signals and 1 output signal
# --------------------------------------------------------------------------------------
# Those functions are perfoming a computation on N input signals and return a single
# output signal. If we were only executing these functions locally, we would not need
# to define them here, but since we are using the multiprocessing module, we need to
# define them here so that they can be pickled and sent to the worker processes.
# Also, we need to systematically return the output signal object, even if it is already
# modified in place, because the multiprocessing module will not be able to retrieve
# the modified object from the worker processes.


@computation_function()
def addition(src_list: list[SignalObj]) -> SignalObj:
    """Add **src** signals and return a new result signal object

    Args:
        src_list: list of source signals

    Returns:
        Modified **dst** signal (modified in place)
    """
    dst = dst_n_to_1(src_list, "Σ")  # `dst` data is initialized to `src_list[0]` data
    for src in src_list[1:]:
        dst.y += np.array(src.y, dtype=dst.y.dtype)
        if dst.dy is not None:
            dst.dy = np.sqrt(dst.dy**2 + src.dy**2)
    restore_data_outside_roi(dst, src_list[0])
    return dst


@computation_function()
def average(src_list: list[SignalObj]) -> SignalObj:
    """Average **src** signals and return a new result signal object

    Args:
        src_list: list of source signals

    Returns:
        Modified **dst** signal (modified in place)
    """
    dst = dst_n_to_1(src_list, "µ")  # `dst` data is initialized to `src_list[0]` data
    for src in src_list[1:]:
        dst.y += np.array(src.y, dtype=dst.y.dtype)
        if dst.dy is not None:
            dst.dy = np.sqrt(dst.dy**2 + src.dy**2)
    dst.y /= len(src_list)
    restore_data_outside_roi(dst, src_list[0])
    return dst


@computation_function()
def product(src_list: list[SignalObj]) -> SignalObj:
    """Multiply **dst** by **src** signals and return a new result signal object

    Args:
        src_list: list of source signals

    Returns:
        Modified **dst** signal (modified in place)
    """
    dst = dst_n_to_1(src_list, "Π")  # `dst` data is initialized to `src_list[0]` data
    for src in src_list[1:]:
        dst.y *= np.array(src.y, dtype=dst.y.dtype)
        if dst.dy is not None:
            dst.dy = dst.y * np.sqrt((dst.dy / dst.y) ** 2 + (src.dy / src.y) ** 2)
    restore_data_outside_roi(dst, src_list[0])
    return dst


@computation_function()
def addition_constant(src: SignalObj, p: ConstantParam) -> SignalObj:
    """Add **dst** and a constant value and return a the new result signal object

    Args:
        src: input signal object
        p: constant value

    Returns:
        Result signal object **src** + **p.value** (new object)
    """
    dst = dst_1_to_1(src, "+", str(p.value))
    dst.y += p.value
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def difference_constant(src: SignalObj, p: ConstantParam) -> SignalObj:
    """Subtract a constant value from a signal

    Args:
        src: input signal object
        p: constant value

    Returns:
        Result signal object **src** - **p.value** (new object)
    """
    dst = dst_1_to_1(src, "-", str(p.value))
    dst.y -= p.value
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def product_constant(src: SignalObj, p: ConstantParam) -> SignalObj:
    """Multiply **dst** by a constant value and return the new result signal object

    Args:
        src: input signal object
        p: constant value

    Returns:
        Result signal object **src** * **p.value** (new object)
    """
    dst = dst_1_to_1(src, "×", str(p.value))
    dst.y *= p.value
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def division_constant(src: SignalObj, p: ConstantParam) -> SignalObj:
    """Divide a signal by a constant value

    Args:
        src: input signal object
        p: constant value

    Returns:
        Result signal object **src** / **p.value** (new object)
    """
    dst = dst_1_to_1(src, "/", str(p.value))
    dst.y /= p.value
    restore_data_outside_roi(dst, src)
    return dst


# MARK: 2_to_1 functions -------------------------------------------------------
# Functions with N input images + 1 input image and N output images
# --------------------------------------------------------------------------------------


@computation_function()
def arithmetic(src1: SignalObj, src2: SignalObj, p: ArithmeticParam) -> SignalObj:
    """Perform arithmetic operation on two signals

    Args:
        src1: source signal 1
        src2: source signal 2
        p: parameters

    Returns:
        Result signal object
    """
    initial_dtype = src1.xydata.dtype
    title = p.operation.replace("obj1", "{0}").replace("obj2", "{1}")
    dst = src1.copy(title=title)
    if not Conf.proc.keep_results.get():
        dst.delete_results()  # Remove any previous results
    o, a, b = p.operator, p.factor, p.constant
    if o in ("×", "/") and a == 0.0:
        dst.y = np.ones_like(src1.y) * b
    elif p.operator == "+":
        dst.y = (src1.y + src2.y) * a + b
    elif p.operator == "-":
        dst.y = (src1.y - src2.y) * a + b
    elif p.operator == "×":
        dst.y = (src1.y * src2.y) * a + b
    elif p.operator == "/":
        dst.y = (src1.y / src2.y) * a + b
    if dst.dy is not None and p.operator in ("+", "-"):
        dst.dy = np.sqrt(src1.dy**2 + src2.dy**2)
    if dst.dy is not None:
        dst.dy *= p.factor
    # Eventually convert to initial data type
    if p.restore_dtype:
        dst.xydata = dst.xydata.astype(initial_dtype)
    restore_data_outside_roi(dst, src1)
    return dst


@computation_function()
def difference(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute difference between two signals

    .. note::

        If uncertainty is available, it is propagated.

    Args:
        src1: source signal 1
        src2: source signal 2

    Returns:
        Result signal object **src1** - **src2**
    """
    dst = dst_2_to_1(src1, src2, "-")
    dst.y = src1.y - src2.y
    if dst.dy is not None:
        dst.dy = np.sqrt(src1.dy**2 + src2.dy**2)
    restore_data_outside_roi(dst, src1)
    return dst


@computation_function()
def quadratic_difference(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute quadratic difference between two signals

    .. note::

        If uncertainty is available, it is propagated.

    Args:
        src1: source signal 1
        src2: source signal 2

    Returns:
        Result signal object (**src1** - **src2**) / sqrt(2.0)
    """
    dst = dst_2_to_1(src1, src2, "quadratic_difference")
    x1, y1 = src1.get_data()
    _x2, y2 = src2.get_data()
    dst.set_xydata(x1, (y1 - np.array(y2, dtype=y1.dtype)) / np.sqrt(2.0))
    if np.issubdtype(dst.data.dtype, np.unsignedinteger):
        dst.data[src1.data < src2.data] = 0
    if dst.dy is not None:
        dst.dy = np.sqrt(src1.dy**2 + src2.dy**2)
    restore_data_outside_roi(dst, src1)
    return dst


@computation_function()
def division(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute division between two signals

    Args:
        src1: source signal 1
        src2: source signal 2

    Returns:
        Result signal object **src1** / **src2**
    """
    dst = dst_2_to_1(src1, src2, "/")
    x1, y1 = src1.get_data()
    _x2, y2 = src2.get_data()
    dst.set_xydata(x1, y1 / np.array(y2, dtype=y1.dtype))
    restore_data_outside_roi(dst, src1)
    return dst


# MARK: 1_to_1 functions -------------------------------------------------------
# Functions with 1 input image and 1 output image
# --------------------------------------------------------------------------------------


@computation_function()
def extract_rois(src: SignalObj, params: list[ROI1DParam]) -> SignalObj:
    """Extract multiple regions of interest from data

    Args:
        src: source signal
        params: list of ROI parameters

    Returns:
        Signal with multiple regions of interest
    """
    suffix = None
    if len(params) == 1:
        p: ROI1DParam = params[0]
        suffix = f"{p.xmin:.3g}≤x≤{p.xmax:.3g}"
    dst = dst_1_to_1(src, "extract_rois", suffix)
    x, y = src.get_data()
    xout, yout = np.ones_like(x) * np.nan, np.ones_like(y) * np.nan
    for p in params:
        idx1, idx2 = np.searchsorted(x, p.xmin), np.searchsorted(x, p.xmax)
        slice0 = slice(idx1, idx2)
        xout[slice0], yout[slice0] = x[slice0], y[slice0]
    nans = np.isnan(xout) | np.isnan(yout)
    dst.set_xydata(xout[~nans], yout[~nans])
    # TODO: [P2] Instead of removing geometric shapes, apply roi extract
    dst.remove_all_shapes()
    return dst


@computation_function()
def extract_roi(src: SignalObj, p: ROI1DParam) -> SignalObj:
    """Extract single region of interest from data

    Args:
        src: source signal
        p: ROI parameters

    Returns:
        Signal with single region of interest
    """
    dst = dst_1_to_1(src, "extract_roi", f"{p.xmin:.3g}≤x≤{p.xmax:.3g}")
    x, y = p.get_data(src).copy()
    dst.set_xydata(x, y)
    # TODO: [P2] Instead of removing geometric shapes, apply roi extract
    dst.remove_all_shapes()
    return dst


@computation_function()
def swap_axes(src: SignalObj) -> SignalObj:
    """Swap axes

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "swap_axes")
    x, y = src.get_data()
    dst.set_xydata(y, x)
    return dst


@computation_function()
def inverse(src: SignalObj) -> SignalObj:
    """Compute inverse with :py:data:`numpy.invert`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "invert")
    x, y = src.get_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dst.set_xydata(x, np.reciprocal(y))
        dst.y[np.isinf(dst.y)] = np.nan
    if dst.dy is not None:
        dst.dy = dst.y * src.dy / (src.y**2)
        dst.dy[np.isinf(dst.dy)] = np.nan
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def absolute(src: SignalObj) -> SignalObj:
    """Compute absolute value with :py:data:`numpy.absolute`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(np.absolute)(src)


@computation_function()
def real(src: SignalObj) -> SignalObj:
    """Compute real part with :py:func:`numpy.real`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(np.real)(src)


@computation_function()
def imag(src: SignalObj) -> SignalObj:
    """Compute imaginary part with :py:func:`numpy.imag`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(np.imag)(src)


class DataTypeSParam(gds.DataSet):
    """Convert signal data type parameters"""

    dtype_str = gds.ChoiceItem(
        _("Destination data type"),
        list(zip(SignalObj.get_valid_dtypenames(), SignalObj.get_valid_dtypenames())),
        help=_("Output image data type."),
    )


@computation_function()
def astype(src: SignalObj, p: DataTypeSParam) -> SignalObj:
    """Convert data type with :py:func:`numpy.astype`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "astype", f"dtype={p.dtype_str}")
    dst.xydata = src.xydata.astype(p.dtype_str)
    return dst


@computation_function()
def log10(src: SignalObj) -> SignalObj:
    """Compute Log10 with :py:data:`numpy.log10`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(np.log10)(src)


@computation_function()
def exp(src: SignalObj) -> SignalObj:
    """Compute exponential with :py:data:`numpy.exp`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(np.exp)(src)


@computation_function()
def sqrt(src: SignalObj) -> SignalObj:
    """Compute square root with :py:data:`numpy.sqrt`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(np.sqrt)(src)


class PowerParam(gds.DataSet):
    """Power parameters"""

    power = gds.FloatItem(_("Power"), default=2.0)


@computation_function()
def power(src: SignalObj, p: PowerParam) -> SignalObj:
    """Compute power with :py:data:`numpy.power`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "^", str(p.power))
    dst.y = np.power(src.y, p.power)
    restore_data_outside_roi(dst, src)
    return dst


class PeakDetectionParam(gds.DataSet):
    """Peak detection parameters"""

    threshold = gds.IntItem(
        _("Threshold"), default=30, min=0, max=100, slider=True, unit="%"
    )
    min_dist = gds.IntItem(_("Minimum distance"), default=1, min=1, unit="points")


@computation_function()
def peak_detection(src: SignalObj, p: PeakDetectionParam) -> SignalObj:
    """Peak detection with
    :py:func:`sigima_.algorithms.signal.peakdetection.peak_indices`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(
        src, "peak_detection", f"threshold={p.threshold}%, min_dist={p.min_dist}pts"
    )
    x, y = src.get_data()
    indices = peakdetection.peak_indices(
        y, thres=p.threshold * 0.01, min_dist=p.min_dist
    )
    dst.set_xydata(x[indices], y[indices])
    dst.set_metadata_option("curvestyle", "Sticks")
    return dst


@computation_function()
def normalize(src: SignalObj, p: NormalizeParam) -> SignalObj:
    """Normalize data with :py:func:`sigima_.algorithms.signal.level.normalize`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "normalize", f"ref={p.method}")
    x, y = src.get_data()
    dst.set_xydata(x, level.normalize(y, p.method))
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def derivative(src: SignalObj) -> SignalObj:
    """Compute derivative with :py:func:`numpy.gradient`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "derivative")
    x, y = src.get_data()
    dst.set_xydata(x, np.gradient(y, x))
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def integral(src: SignalObj) -> SignalObj:
    """Compute integral with :py:func:`scipy.integrate.cumulative_trapezoid`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "integral")
    x, y = src.get_data()
    dst.set_xydata(x, spt.cumulative_trapezoid(y, x, initial=0.0))
    restore_data_outside_roi(dst, src)
    return dst


class XYCalibrateParam(gds.DataSet):
    """Signal calibration parameters"""

    axes = (("x", _("X-axis")), ("y", _("Y-axis")))
    axis = gds.ChoiceItem(_("Calibrate"), axes, default="y")
    a = gds.FloatItem("a", default=1.0)
    b = gds.FloatItem("b", default=0.0)


@computation_function()
def calibration(src: SignalObj, p: XYCalibrateParam) -> SignalObj:
    """Compute linear calibration

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "calibration", f"{p.axis}={p.a}*{p.axis}+{p.b}")
    x, y = src.get_data()
    if p.axis == "x":
        dst.set_xydata(p.a * x + p.b, y)
    else:
        dst.set_xydata(x, p.a * y + p.b)
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def clip(src: SignalObj, p: ClipParam) -> SignalObj:
    """Compute maximum data clipping with :py:func:`numpy.clip`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return Wrap1to1Func(np.clip, a_min=p.lower, a_max=p.upper)(src)


@computation_function()
def offset_correction(src: SignalObj, p: ROI1DParam) -> SignalObj:
    """Correct offset: subtract the mean value of the signal in the specified range
    (baseline correction)

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "offset_correction", f"{p.xmin:.3g}≤x≤{p.xmax:.3g}")
    _roi_x, roi_y = p.get_data(src)
    dst.y -= np.mean(roi_y)
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def gaussian_filter(src: SignalObj, p: GaussianParam) -> SignalObj:
    """Compute gaussian filter with :py:func:`scipy.ndimage.gaussian_filter`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return Wrap1to1Func(spi.gaussian_filter, sigma=p.sigma)(src)


@computation_function()
def moving_average(src: SignalObj, p: MovingAverageParam) -> SignalObj:
    """Compute moving average with :py:func:`scipy.ndimage.uniform_filter`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return Wrap1to1Func(spi.uniform_filter, size=p.n, mode=p.mode)(src)


@computation_function()
def moving_median(src: SignalObj, p: MovingMedianParam) -> SignalObj:
    """Compute moving median with :py:func:`scipy.ndimage.median_filter`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return Wrap1to1Func(spi.median_filter, size=p.n, mode=p.mode)(src)


@computation_function()
def wiener(src: SignalObj) -> SignalObj:
    """Compute Wiener filter with :py:func:`scipy.signal.wiener`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    return Wrap1to1Func(sps.wiener)(src)


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

    def update_from_obj(self, obj: SignalObj) -> None:
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


def frequency_filter(src: SignalObj, p: BaseHighLowBandParam) -> SignalObj:
    """Compute frequency filter (low-pass, high-pass, band-pass, band-stop),
    with :py:func:`scipy.signal.filtfilt`

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
    dst = dst_1_to_1(src, name, suffix)
    b, a = p.get_filter_params(dst)
    dst.y = sps.filtfilt(b, a, dst.y)
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def lowpass(src: SignalObj, p: LowPassFilterParam) -> SignalObj:
    """Compute low-pass filter with :py:func:`scipy.signal.filtfilt`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return frequency_filter(src, p)


@computation_function()
def highpass(src: SignalObj, p: HighPassFilterParam) -> SignalObj:
    """Compute high-pass filter with :py:func:`scipy.signal.filtfilt`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return frequency_filter(src, p)


@computation_function()
def bandpass(src: SignalObj, p: BandPassFilterParam) -> SignalObj:
    """Compute band-pass filter with :py:func:`scipy.signal.filtfilt`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return frequency_filter(src, p)


@computation_function()
def bandstop(src: SignalObj, p: BandStopFilterParam) -> SignalObj:
    """Compute band-stop filter with :py:func:`scipy.signal.filtfilt`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    return frequency_filter(src, p)


class ZeroPadding1DParam(gds.DataSet):
    """Zero padding parameters"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__obj: SignalObj | None = None

    def update_from_obj(self, obj: SignalObj) -> None:
        """Update parameters from signal"""
        self.__obj = obj
        self.choice_callback(None, self.strategy)

    def choice_callback(self, item, value):  # pylint: disable=unused-argument
        """Callback for choice item"""
        size = self.__obj.x.size
        if value == "next_pow2":
            self.n = 2 ** np.ceil(np.log2(size)).astype(int) - size
        elif value == "double":
            self.n = size
        elif value == "triple":
            self.n = 2 * size

    strategies = ("next_pow2", "double", "triple", "custom")
    _prop = gds.GetAttrProp("strategy")
    strategy = gds.ChoiceItem(
        _("Strategy"), zip(strategies, strategies), default=strategies[-1]
    ).set_prop("display", store=_prop, callback=choice_callback)
    _func_prop = gds.FuncProp(_prop, lambda x: x == "custom")
    n = gds.IntItem(
        _("Number of points"), min=1, help=_("Number of points to add")
    ).set_prop("display", active=_func_prop)


@computation_function()
def zero_padding(src: SignalObj, p: ZeroPadding1DParam) -> SignalObj:
    """Compute zero padding with
    :py:func:`sigima_.algorithms.signal.fourier.zero_padding`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    if p.strategy == "custom":
        suffix = f"n={p.n}"
    else:
        suffix = f"strategy={p.strategy}"
    dst = dst_1_to_1(src, "zero_padding", suffix)
    x, y = src.get_data()
    dst.set_xydata(*fourier.zero_padding(x, y, p.n))
    return dst


@computation_function()
def fft(src: SignalObj, p: FFTParam | None = None) -> SignalObj:
    """Compute FFT with :py:func:`sigima_.algorithms.signal.fourier.fft1d`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "fft")
    x, y = src.get_data()
    dst.set_xydata(*fourier.fft1d(x, y, shift=True if p is None else p.shift))
    dst.save_attr_to_metadata("xunit", "Hz" if dst.xunit == "s" else "")
    dst.save_attr_to_metadata("yunit", "")
    dst.save_attr_to_metadata("xlabel", _("Frequency"))
    return dst


@computation_function()
def ifft(src: SignalObj, p: FFTParam | None = None) -> SignalObj:
    """Compute iFFT with :py:func:`sigima_.algorithms.signal.fourier.ifft1d`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "ifft")
    x, y = src.get_data()
    dst.set_xydata(*fourier.ifft1d(x, y, shift=True if p is None else p.shift))
    dst.restore_attr_from_metadata("xunit", "s" if src.xunit == "Hz" else "")
    dst.restore_attr_from_metadata("yunit", "")
    dst.restore_attr_from_metadata("xlabel", "")
    return dst


@computation_function()
def magnitude_spectrum(src: SignalObj, p: SpectrumParam | None = None) -> SignalObj:
    """Compute magnitude spectrum
    with :py:func:`sigima_.algorithms.signal.fourier.magnitude_spectrum`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "magnitude_spectrum")
    x, y = src.get_data()
    log_scale = p is not None and p.log
    dst.set_xydata(*fourier.magnitude_spectrum(x, y, log_scale=log_scale))
    dst.xlabel = _("Frequency")
    dst.xunit = "Hz" if dst.xunit == "s" else ""
    dst.yunit = "dB" if log_scale else ""
    return dst


@computation_function()
def phase_spectrum(src: SignalObj) -> SignalObj:
    """Compute phase spectrum
    with :py:func:`sigima_.algorithms.signal.fourier.phase_spectrum`

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "phase_spectrum")
    x, y = src.get_data()
    dst.set_xydata(*fourier.phase_spectrum(x, y))
    dst.xlabel = _("Frequency")
    dst.xunit = "Hz" if dst.xunit == "s" else ""
    dst.yunit = ""
    return dst


@computation_function()
def psd(src: SignalObj, p: SpectrumParam | None = None) -> SignalObj:
    """Compute power spectral density
    with :py:func:`sigima_.algorithms.signal.fourier.psd`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "psd")
    x, y = src.get_data()
    log_scale = p is not None and p.log
    psd_x, psd_y = fourier.psd(x, y, log_scale=log_scale)
    dst.xydata = np.vstack((psd_x, psd_y))
    dst.xlabel = _("Frequency")
    dst.xunit = "Hz" if dst.xunit == "s" else ""
    dst.yunit = "dB/Hz" if log_scale else ""
    return dst


class PolynomialFitParam(gds.DataSet):
    """Polynomial fitting parameters"""

    degree = gds.IntItem(_("Degree"), 3, min=1, max=10, slider=True)


@computation_function()
def histogram(src: SignalObj, p: HistogramParam) -> SignalObj:
    """Compute histogram with :py:func:`numpy.histogram`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    data = src.get_masked_view().compressed()
    suffix = p.get_suffix(data)  # Also updates p.lower and p.upper

    # Compute histogram:
    y, bin_edges = np.histogram(data, bins=p.bins, range=(p.lower, p.upper))
    x = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Note: we use the `new_signal_result` function to create the result signal object
    # because the `dst_1_to_1` would copy the source signal, which is not what we want
    # here (we want a brand new signal object).
    dst = new_signal_result(
        src,
        "histogram",
        suffix=suffix,
        units=(src.yunit, ""),
        labels=(src.ylabel, _("Counts")),
    )
    dst.set_xydata(x, y)
    dst.set_metadata_option("shade", 0.5)
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


@computation_function()
def interpolate(src1: SignalObj, src2: SignalObj, p: InterpolationParam) -> SignalObj:
    """Interpolate data with
    :py:func:`sigima_.algorithms.signal.interpolation.interpolate`

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
    dst = dst_2_to_1(src1, src2, "interpolate", suffix)
    x1, y1 = src1.get_data()
    xnew, _y2 = src2.get_data()
    ynew = interpolation.interpolate(x1, y1, xnew, p.method, p.fill_value)
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

    def update_from_obj(self, obj: SignalObj) -> None:
        """Update parameters from a signal object."""
        if self.xmin is None:
            self.xmin = obj.x[0]
        if self.xmax is None:
            self.xmax = obj.x[-1]
        if self.dx is None:
            self.dx = obj.x[1] - obj.x[0]
        if self.nbpts is None:
            self.nbpts = len(obj.x)


@computation_function()
def resampling(src: SignalObj, p: ResamplingParam) -> SignalObj:
    """Resample data with :py:func:`sigima_.algorithms.signal.interpolation.interpolate`

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
    dst = dst_1_to_1(src, "resample", suffix)
    x, y = src.get_data()
    if p.mode == "dx":
        xnew = np.arange(p.xmin, p.xmax, p.dx)
    else:
        xnew = np.linspace(p.xmin, p.xmax, p.nbpts)
    ynew = interpolation.interpolate(x, y, xnew, p.method, p.fill_value)
    dst.set_xydata(xnew, ynew)
    return dst


class DetrendingParam(gds.DataSet):
    """Detrending parameters"""

    methods = (("linear", _("Linear")), ("constant", _("Constant")))
    method = gds.ChoiceItem(_("Detrending method"), methods, default="linear")


@computation_function()
def detrending(src: SignalObj, p: DetrendingParam) -> SignalObj:
    """Detrend data with :py:func:`scipy.signal.detrend`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "detrending", f"method={p.method}")
    x, y = src.get_data()
    dst.set_xydata(x, sps.detrend(y, type=p.method))
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def xy_mode(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Simulate the X-Y mode of an oscilloscope.

    Use the first signal as the X-axis and the second signal as the Y-axis.

    Args:
        src1: First input signal (X-axis).
        src2: Second input signal (Y-axis).

    Returns:
        A signal object representing the X-Y mode.
    """
    dst = dst_2_to_1(src1, src2, "", "X-Y Mode")
    p = ResamplingParam()
    p.xmin = max(src1.x[0], src2.x[0])
    p.xmax = min(src1.x[-1], src2.x[-1])
    assert p.xmin < p.xmax, "X-Y mode: No overlap between signals."
    p.mode = "nbpts"
    p.nbpts = min(src1.x.size, src2.x.size)
    _, y1 = resampling(src1, p).get_data()
    _, y2 = resampling(src2, p).get_data()
    dst.set_xydata(y1, y2)
    dst.title = "{1} = f({0})"
    restore_data_outside_roi(dst, src1)
    return dst


@computation_function()
def convolution(src1: SignalObj, src2: SignalObj) -> SignalObj:
    """Compute convolution of two signals
    with :py:func:`scipy.signal.convolve`

    Args:
        src1: source signal 1
        src2: source signal 2

    Returns:
        Result signal object
    """
    dst = dst_2_to_1(src1, src2, "⊛")
    x1, y1 = src1.get_data()
    _x2, y2 = src2.get_data()
    ynew = np.real(sps.convolve(y1, y2, mode="same"))
    dst.set_xydata(x1, ynew)
    restore_data_outside_roi(dst, src1)
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


@computation_function()
def apply_window(src: SignalObj, p: WindowingParam) -> SignalObj:
    """Compute windowing (available methods: hamming, hanning, bartlett, blackman,
    tukey, rectangular) with :py:func:`sigima_.algorithms.signal.windowing.apply_window`

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
    dst = dst_1_to_1(src, "apply_window", suffix)  # type: ignore
    dst.y = windowing.apply_window(dst.y, p.method, p.alpha)  # type: ignore
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def reverse_x(src: SignalObj) -> SignalObj:
    """Reverse x-axis

    Args:
        src: source signal

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "reverse_x")
    dst.y = dst.y[::-1]
    return dst


class AngleUnitParam(gds.DataSet):
    """Choice of angle unit."""

    units = (("rad", _("Radian")), ("deg", _("Degree")))
    unit = gds.ChoiceItem(_("Angle unit"), units, default="rad")


@computation_function()
def to_polar(src: SignalObj, p: AngleUnitParam) -> SignalObj:
    """Convert cartesian coordinates to polar coordinates with
    :py:func:`sigima_.algorithms.coordinates.to_polar`.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.
    """
    dst = dst_1_to_1(src, "Polar coordinates", f"unit={p.unit}")
    x, y = src.get_data()
    r, theta = coordinates.to_polar(x, y, p.unit)
    dst.set_xydata(r, theta)
    return dst


@computation_function()
def to_cartesian(src: SignalObj, p: AngleUnitParam) -> SignalObj:
    """Convert polar coordinates to cartesian coordinates with
    :py:func:`sigima_.algorithms.coordinates.to_cartesian`.

    Args:
        src: Source signal.
        p: Parameters.

    Returns:
        Result signal object.

    .. note::

        This function assumes that the x-axis represents the radius and the y-axis
        represents the angle. Negative values are not allowed for the radius, and will
        be clipped to 0 (a warning will be raised).
    """
    dst = dst_1_to_1(src, "Cartesian coordinates", f"unit={p.unit}")
    r, theta = src.get_data()
    x, y = coordinates.to_cartesian(r, theta, p.unit)
    dst.set_xydata(x, y)
    return dst


class AllanVarianceParam(gds.DataSet):
    """Allan variance parameters"""

    max_tau = gds.IntItem("Max τ", default=100, min=1, unit="pts")


@computation_function()
def allan_variance(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Allan variance with
    :py:func:`sigima_.algorithms.signal.stability.allan_variance`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "allan_variance", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    avar = stability.allan_variance(x, y, tau_values)
    dst.set_xydata(tau_values, avar)
    return dst


@computation_function()
def allan_deviation(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Allan deviation with
    :py:func:`sigima_.algorithms.signal.stability.allan_deviation`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "allan_deviation", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    adev = stability.allan_deviation(x, y, tau_values)
    dst.set_xydata(tau_values, adev)
    return dst


@computation_function()
def overlapping_allan_variance(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Overlapping Allan variance.

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "overlapping_allan_variance", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    oavar = stability.overlapping_allan_variance(x, y, tau_values)
    dst.set_xydata(tau_values, oavar)
    return dst


@computation_function()
def modified_allan_variance(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Modified Allan variance.

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "modified_allan_variance", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    mavar = stability.modified_allan_variance(x, y, tau_values)
    dst.set_xydata(tau_values, mavar)
    return dst


@computation_function()
def hadamard_variance(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Hadamard variance.

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "hadamard_variance", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    hvar = stability.hadamard_variance(x, y, tau_values)
    dst.set_xydata(tau_values, hvar)
    return dst


@computation_function()
def total_variance(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Total variance.

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "total_variance", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    tvar = stability.total_variance(x, y, tau_values)
    dst.set_xydata(tau_values, tvar)
    return dst


@computation_function()
def time_deviation(src: SignalObj, p: AllanVarianceParam) -> SignalObj:
    """Compute Time Deviation (TDEV).

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_1_to_1(src, "time_deviation", f"max_tau={p.max_tau}")
    x, y = src.get_data()
    tau_values = np.arange(1, p.max_tau + 1)
    tdev = stability.time_deviation(x, y, tau_values)
    dst.set_xydata(tau_values, tdev)
    return dst


# MARK: compute_1_to_0 functions -------------------------------------------------------
# Functions with 1 input signal and 0 output signals (ResultShape or ResultProperties)
# --------------------------------------------------------------------------------------


def calc_resultshape(
    title: str,
    shape: Literal[
        "rectangle", "circle", "ellipse", "segment", "marker", "point", "polygon"
    ],
    obj: SignalObj,
    func: Callable,
    *args: Any,
    add_label: bool = False,
) -> ResultShape | None:
    """Calculate result shape by executing a computation function on a signal object,
    taking into account the signal ROIs.

    Args:
        title: result title
        shape: result shape kind
        obj: input image object
        func: computation function
        *args: computation function arguments
        add_label: if True, add a label item (and the geometrical shape) to plot
         (default to False)

    Returns:
        Result shape object or None if no result is found

    .. warning::

        The computation function must take either a single argument (the data) or
        multiple arguments (the data followed by the computation parameters).

        Moreover, the computation function must return a 1D NumPy array (or a list,
        or a tuple) containing the result of the computation.
    """
    res = []
    for i_roi in obj.iterate_roi_indices():
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
            results = np.array([0 if i_roi is None else i_roi] + results.tolist())
            res.append(results)
    if res:
        return ResultShape(title, np.vstack(res), shape, add_label=add_label)
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


@computation_function()
def fwhm(obj: SignalObj, param: FWHMParam) -> ResultShape | None:
    """Compute FWHM with :py:func:`sigima_.algorithms.signal.pulse.fwhm`

    Args:
        obj: source signal
        param: parameters

    Returns:
        Segment coordinates
    """
    return calc_resultshape(
        "fwhm",
        "segment",
        obj,
        pulse.fwhm,
        param.method,
        param.xmin,
        param.xmax,
        add_label=True,
    )


@computation_function()
def fw1e2(obj: SignalObj) -> ResultShape | None:
    """Compute FW at 1/e² with :py:func:`sigima_.algorithms.signal.pulse.fw1e2`

    Args:
        obj: source signal

    Returns:
        Segment coordinates
    """
    return calc_resultshape("fw1e2", "segment", obj, pulse.fw1e2, add_label=True)


class OrdinateParam(gds.DataSet):
    """Ordinate parameter"""

    y = gds.FloatItem(_("Ordinate"), default=0)


@computation_function()
def full_width_at_y(obj: SignalObj, p: OrdinateParam) -> ResultShape | None:
    """
    Compute full width at a given y value for a signal object.

    Args:
        obj: The signal object containing x and y data.
        p: The ordinate parameter dataset

    Returns:
        Segment coordinates
    """
    return calc_resultshape(
        "∆X", "segment", obj, pulse.full_width_at_y, p.y, add_label=True
    )


@computation_function()
def x_at_y(obj: SignalObj, p: OrdinateParam) -> ResultProperties:
    """
    Compute the smallest x-value at a given y-value for a signal object.

    Args:
        obj: The signal object containing x and y data.
        p: The parameter dataset for finding the abscissa.

    Returns:
         An object containing the x-value.
    """
    return calc_resultproperties(
        f"x|y={p.y}",
        obj,
        {
            "x = %g {.xunit}": lambda xy: features.find_first_x_at_y_value(
                xy[0], xy[1], p.y
            )
        },
    )


class AbscissaParam(gds.DataSet):
    """Abscissa parameter"""

    x = gds.FloatItem(_("Abscissa"), default=0)


@computation_function()
def y_at_x(obj: SignalObj, p: AbscissaParam) -> ResultProperties:
    """
    Compute the smallest y-value at a given x-value for a signal object.

    Args:
        obj: The signal object containing x and y data.
        p: The parameter dataset for finding the ordinate.

    Returns:
         An object containing the y-value.
    """
    return calc_resultproperties(
        f"y|x={p.x}",
        obj,
        {"y = %g {.yunit}": lambda xy: features.find_y_at_x_value(xy[0], xy[1], p.x)},
    )


@computation_function()
def stats(obj: SignalObj) -> ResultProperties:
    """Compute statistics on a signal

    Args:
        obj: source signal

    Returns:
        Result properties object
    """
    statfuncs = {
        "min(y) = %g {.yunit}": lambda xy: np.nanmin(xy[1]),
        "max(y) = %g {.yunit}": lambda xy: np.nanmax(xy[1]),
        "<y> = %g {.yunit}": lambda xy: np.nanmean(xy[1]),
        "median(y) = %g {.yunit}": lambda xy: np.nanmedian(xy[1]),
        "σ(y) = %g {.yunit}": lambda xy: np.nanstd(xy[1]),
        "<y>/σ(y)": lambda xy: np.nanmean(xy[1]) / np.nanstd(xy[1]),
        "peak-to-peak(y) = %g {.yunit}": lambda xy: np.nanmax(xy[1]) - np.nanmin(xy[1]),
        "Σ(y) = %g {.yunit}": lambda xy: np.nansum(xy[1]),
        "∫ydx": lambda xy: spt.trapezoid(xy[1], xy[0]),
    }
    return calc_resultproperties("stats", obj, statfuncs)


@computation_function()
def bandwidth_3db(obj: SignalObj) -> ResultShape | None:
    """Compute bandwidth at -3 dB with
    :py:func:`sigima_.algorithms.signal.misc.bandwidth`

    Args:
        obj: source signal

    Returns:
        Result properties with bandwidth
    """
    return calc_resultshape(
        "bandwidth", "segment", obj, features.bandwidth, 3.0, add_label=True
    )


class DynamicParam(gds.DataSet):
    """Parameters for dynamic range computation (ENOB, SNR, SINAD, THD, SFDR)"""

    full_scale = gds.FloatItem(_("Full scale"), default=0.16, min=0.0, unit="V")
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


@computation_function()
def dynamic_parameters(src: SignalObj, p: DynamicParam) -> ResultProperties:
    """Compute Dynamic parameters
    using the following functions:

    - Freq: :py:func:`sigima_.algorithms.signal.dynamic.sinus_frequency`
    - ENOB: :py:func:`sigima_.algorithms.signal.dynamic.enob`
    - SNR: :py:func:`sigima_.algorithms.signal.dynamic.snr`
    - SINAD: :py:func:`sigima_.algorithms.signal.dynamic.sinad`
    - THD: :py:func:`sigima_.algorithms.signal.dynamic.thd`
    - SFDR: :py:func:`sigima_.algorithms.signal.dynamic.sfdr`

    Args:
        src: source signal
        p: parameters

    Returns:
        Result properties with ENOB, SNR, SINAD, THD, SFDR
    """
    dsfx = f" = %g {p.unit}"
    funcs = {
        "Freq": lambda xy: dynamic.sinus_frequency(xy[0], xy[1]),
        "ENOB = %.1f bits": lambda xy: dynamic.enob(xy[0], xy[1], p.full_scale),
        "SNR" + dsfx: lambda xy: dynamic.snr(xy[0], xy[1], p.unit),
        "SINAD" + dsfx: lambda xy: dynamic.sinad(xy[0], xy[1], p.unit),
        "THD" + dsfx: lambda xy: dynamic.thd(
            xy[0], xy[1], p.full_scale, p.unit, p.nb_harm
        ),
        "SFDR" + dsfx: lambda xy: dynamic.sfdr(xy[0], xy[1], p.full_scale, p.unit),
    }
    return calc_resultproperties("ADC", src, funcs)


@computation_function()
def sampling_rate_period(obj: SignalObj) -> ResultProperties:
    """Compute sampling rate and period
    using the following functions:

    - fs: :py:func:`sigima_.algorithms.signal.dynamic.sampling_rate`
    - T: :py:func:`sigima_.algorithms.signal.dynamic.sampling_period`

    Args:
        obj: source signal

    Returns:
        Result properties with sampling rate and period
    """
    return calc_resultproperties(
        "sampling_rate_period",
        obj,
        {
            "fs = %g": lambda xy: dynamic.sampling_rate(xy[0]),
            "T = %g {.xunit}": lambda xy: dynamic.sampling_period(xy[0]),
        },
    )


@computation_function()
def contrast(obj: SignalObj) -> ResultProperties:
    """Compute contrast with :py:func:`sigima_.algorithms.signal.misc.contrast`"""
    return calc_resultproperties(
        "contrast",
        obj,
        {
            "contrast": lambda xy: features.contrast(xy[1]),
        },
    )


@computation_function()
def x_at_minmax(obj: SignalObj) -> ResultProperties:
    """
    Compute the smallest argument at the minima and the smallest argument at the maxima.

    Args:
        obj: The signal object.

    Returns:
        An object containing the x-values at the minima and the maxima.
    """
    return calc_resultproperties(
        "x@min,max",
        obj,
        {
            "X@Ymin = %g {.xunit}": lambda xy: xy[0][np.argmin(xy[1])],
            "X@Ymax = %g {.xunit}": lambda xy: xy[0][np.argmax(xy[1])],
        },
    )
