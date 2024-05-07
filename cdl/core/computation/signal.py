# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Signal computation objects (see parent package :mod:`cdl.core.computation`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# All dataset classes must also be imported in the cdl.core.computation.param module.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import guidata.dataset as gds
import numpy as np
import scipy.integrate as spt
import scipy.ndimage as spi
import scipy.optimize as spo
import scipy.signal as sps

from cdl.algorithms import fit
from cdl.algorithms.signal import (
    derivative,
    interpolate,
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
    HistogramParam,
    MovingAverageParam,
    MovingMedianParam,
    ThresholdParam,
    calc_resultproperties,
    new_signal_result,
)
from cdl.core.model.base import ResultProperties, ResultShape, ShapeTypes
from cdl.core.model.signal import SignalObj

VALID_DTYPES_STRLIST = SignalObj.get_valid_dtypenames()


def dst_11(src: SignalObj, name: str, suffix: str | None = None) -> SignalObj:
    """Create a result signal object, as returned by the callback function of the
    :func:`cdl.core.gui.processor.base.BaseProcessor.compute_11` method

    Args:
        src: source signal
        name: name of the function

    Returns:
        Result signal object
    """
    dst = src.copy(title=f"{name}({src.short_id})")
    if suffix is not None:
        dst.title += "|" + suffix
    return dst


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


def dst_n1n(src1: SignalObj, src2: SignalObj, name: str, suffix: str | None = None):
    """Create a result signal object, as returned by the callback function of the
    :func:`cdl.core.gui.processor.base.BaseProcessor.compute_n1n` method

    Args:
        src1: source signal 1
        src2: source signal 2
        name: name of the function

    Returns:
        Result signal object
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
    """Add **dst** and **src** signals and return **dst** signal modified in place

    Args:
        dst: destination signal
        src: source signal

    Returns:
        Modified **dst** signal
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
        Modified **dst** signal
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

    .. note::

        If uncertainty is available, it is propagated.

    Args:
        src1: source signal 1
        src2: source signal 2

    Returns:
        Result signal object **src1** - **src2**
    """
    dst = dst_n1n(src1, src2, "difference")
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
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_11(src, "normalize", f"ref={p.method}")
    x, y = src.get_data()
    dst.set_xydata(x, normalize(y, p.method))
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
    dst.set_xydata(x, derivative(x, y))
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
    dst.set_xydata(x, moving_average(y, p.n))
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


def compute_fft(src: SignalObj, p: FFTParam) -> SignalObj:
    """Compute FFT

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_11(src, "fft")
    x, y = src.get_data()
    dst.set_xydata(*xy_fft(x, y, shift=p.shift))
    return dst


def compute_ifft(src: SignalObj, p: FFTParam) -> SignalObj:
    """Compute iFFT

    Args:
        src: source signal
        p: parameters

    Returns:
        Result signal object
    """
    dst = dst_11(src, "ifft")
    x, y = src.get_data()
    dst.set_xydata(*xy_ifft(x, y, shift=p.shift))
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

    _methods = (
        ("linear", _("Linear")),
        ("spline", _("Spline")),
        ("quadratic", _("Quadratic")),
        ("cubic", _("Cubic")),
        ("barycentric", _("Barycentric")),
        ("pchip", _("PCHIP")),
    )
    method = gds.ChoiceItem(_("Interpolation method"), _methods, default="linear")
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
    ynew = interpolate(x1, y1, xnew, p.method, p.fill_value)
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
    ynew = interpolate(x, y, xnew, p.method, p.fill_value)
    dst.set_xydata(xnew, ynew)
    return dst


class DetrendingParam(gds.DataSet):
    """Detrending parameters"""

    _methods = (("linear", _("Linear")), ("constant", _("Constant")))
    method = gds.ChoiceItem(_("Detrending method"), _methods, default="linear")


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
    dst = dst_n1n(src1, src2, "convolution")
    x1, y1 = src1.get_data()
    _x2, y2 = src2.get_data()
    ynew = np.real(sps.convolve(y1, y2, mode="same"))
    dst.set_xydata(x1, ynew)
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

    fittypes = (
        ("GaussianModel", _("Gaussian")),
        ("LorentzianModel", _("Lorentzian")),
        ("VoigtModel", "Voigt"),
    )

    fittype = gds.ChoiceItem(_("Fit type"), fittypes, default="GaussianModel")


def __func_fwhm(data: np.ndarray, fittype: str) -> tuple[float, float, float, float]:
    """Compute FWHM"""
    x, y = data
    dx, dy, base = np.max(x) - np.min(x), np.max(y) - np.min(y), np.min(y)
    sigma, mu = dx * 0.1, xpeak(x, y)
    FitModelClass: fit.FitModel = getattr(fit, fittype)
    amp = FitModelClass.get_amp_from_amplitude(dy, sigma)

    def func(params):
        """Fitting model function"""
        # pylint: disable=cell-var-from-loop
        return y - FitModelClass.func(x, *params)

    (amp, sigma, mu, base), _ier = spo.leastsq(func, np.array([amp, sigma, mu, base]))
    return FitModelClass.half_max_segment(amp, sigma, mu, base)


def compute_fwhm(signal: SignalObj, param: FWHMParam) -> ResultShape:
    """Compute FWHM

    Args:
        signal: source signal
        param: parameters

    Returns:
        Segment coordinates
    """

    return calc_resultshape(
        "fwhm", ShapeTypes.SEGMENT, signal, __func_fwhm, param.fittype
    )


def __func_fw1e2(data: np.ndarray) -> tuple[float, float, float, float]:
    """Compute FW at 1/e²"""
    x, y = data
    dx, dy, base = np.max(x) - np.min(x), np.max(y) - np.min(y), np.min(y)
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
    yhm = fit.GaussianModel.amplitude(amp, sigma) / np.e**2 + base
    return mu - hw, yhm, mu + hw, yhm


def compute_fw1e2(signal: SignalObj) -> ResultShape:
    """Compute FW at 1/e²

    Args:
        signal: source signal

    Returns:
        Segment coordinates
    """

    return calc_resultshape("fw1e2", ShapeTypes.SEGMENT, signal, __func_fw1e2)


def compute_stats_func(obj: SignalObj) -> ResultProperties:
    """Compute statistics functions"""
    statfuncs = {
        "min(y)": lambda xy: xy[1].min(),
        "max(y)": lambda xy: xy[1].max(),
        "<y>": lambda xy: xy[1].mean(),
        "median(y)": lambda xy: np.median(xy[1]),
        "σ(y)": lambda xy: xy[1].std(),
        "<y>/σ(y)": lambda xy: xy[1].mean() / xy[1].std(),
        "peak-to-peak": lambda xy: xy[1].ptp(),
        "Σ(y)": lambda xy: xy[1].sum(),
        "∫ydx": lambda xy: np.trapz(xy[1], xy[0]),
    }
    return calc_resultproperties("stats", obj, statfuncs)
