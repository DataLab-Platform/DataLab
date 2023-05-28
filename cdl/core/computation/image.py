# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Image Computation module
--------------------------------

This module defines the image parameters and functions used by the
:mod:`cdl.core.gui.processor` module.

It is based on the :mod:`cdl.algorithms` module, which defines the algorithms
that are applied to the data, and on the :mod:`cdl.core.model` module, which
defines the data model.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import guidata.dataset.dataitems as gdi
import guidata.dataset.datatypes as gdt
import numpy as np
import pywt
import scipy.ndimage as spi
import scipy.signal as sps
from guiqwt.geometry import vector_rotation
from skimage import exposure, feature, filters, morphology
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, denoise_wavelet
from skimage.util.dtype import dtype_range

from cdl.algorithms.image import (
    BINNING_OPERATIONS,
    binning,
    find_blobs_dog,
    find_blobs_doh,
    find_blobs_log,
    find_blobs_opencv,
    flatfield,
    get_2d_peaks_coords,
    get_centroid_fourier,
    get_contour_shapes,
    get_enclosing_circle,
    get_hough_circle_peaks,
)
from cdl.config import _
from cdl.core.computation.base import (
    ClipParam,
    GaussianParam,
    MovingAverageParam,
    MovingMedianParam,
    ThresholdParam,
)
from cdl.core.model.base import BaseProcParam
from cdl.core.model.image import ImageObj, RoiDataGeometries, RoiDataItem

VALID_DTYPES_STRLIST = [
    dtype.__name__ for dtype in dtype_range if dtype in ImageObj.VALID_DTYPES
]


def dst_11(src: ImageObj, name: str, suffix: str | None = None) -> ImageObj:
    """Create result image object for compute_11 function

    Args:
        src (ImageObj): input image object
        name (str): name of the processing function

    Returns:
        ImageObj: output image object
    """
    dst = ImageObj()
    dst.title = f"{name}({src.short_id})"
    if suffix is not None:
        dst.title += "|" + suffix
    dst.copy_data_from(src)
    return dst


def dst_n1n(
    src1: ImageObj, src2: ImageObj, name: str, suffix: str | None = None
) -> ImageObj:
    """Create result image object for compute_n1n function

    Args:
        src1 (ImageObj): input image object
        src2 (ImageObj): input image object
        name (str): name of the processing function

    Returns:
        ImageObj: output image object
    """
    dst = ImageObj()
    dst.title = f"{name}({src1.short_id}, {src2.short_id})"
    if suffix is not None:
        dst.title += "|" + suffix
    dst.copy_data_from(src1)
    return dst


def compute_difference(src1: ImageObj, src2: ImageObj) -> ImageObj:
    """Compute difference between two images
    Args:
        src1 (ImageObj): input image object
        src2 (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_n1n(src1, src2, "difference")
    dst.data = src1.data - src2.data
    return dst


def compute_quadratic_difference(src1: ImageObj, src2: ImageObj) -> ImageObj:
    """Compute quadratic difference between two images
    Args:
        src1 (ImageObj): input image object
        src2 (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_n1n(src1, src2, "quadratic_difference")
    dst.data = (src1.data - src2.data) / np.sqrt(2.0)
    if np.issubdtype(dst.data.dtype, np.unsignedinteger):
        dst.data[src1.data < src2.data] = 0
    return dst


def compute_division(src1: ImageObj, src2: ImageObj) -> ImageObj:
    """Compute division between two images
    Args:
        src1 (ImageObj): input image object
        src2 (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_n1n(src1, src2, "division")
    dst.data = src1.data / np.array(src2.data, dtype=src1.data.dtype)
    return dst


class FlatFieldParam(BaseProcParam):
    """Flat-field parameters"""

    threshold = gdi.FloatItem(_("Threshold"), default=0.0)


def compute_flatfield(src1: ImageObj, src2: ImageObj, p: FlatFieldParam) -> ImageObj:
    """Compute flat field correction
    Args:
        src1 (ImageObj): raw data image object
        src2 (ImageObj): flat field image object
        p (FlatFieldParam): flat field parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_n1n(src1, src2, "flatfield", f"threshold={p.threshold}")
    dst.data = flatfield(src1.data, src2.data, p.threshold)
    return dst


class LogP1Param(gdt.DataSet):
    """Log10 parameters"""

    n = gdi.FloatItem("n")


def compute_logp1(src: ImageObj, p: LogP1Param) -> ImageObj:
    """Compute log10(z+n)
    Args:
        src (ImageObj): input image object
        p (LogP1Param): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "log_z_plus_n", f"n={p.n}")
    dst.data = np.log10(src.data + p.n)
    return dst


class RotateParam(gdt.DataSet):
    """Rotate parameters"""

    boundaries = ("constant", "nearest", "reflect", "wrap")
    prop = gdt.ValueProp(False)

    angle = gdi.FloatItem(f"{_('Angle')} (°)")
    mode = gdi.ChoiceItem(
        _("Mode"), list(zip(boundaries, boundaries)), default=boundaries[0]
    )
    cval = gdi.FloatItem(
        _("cval"),
        default=0.0,
        help=_(
            "Value used for points outside the "
            "boundaries of the input if mode is "
            "'constant'"
        ),
    )
    reshape = gdi.BoolItem(
        _("Reshape the output array"),
        default=False,
        help=_(
            "Reshape the output array "
            "so that the input array is "
            "contained completely in the output"
        ),
    )
    prefilter = gdi.BoolItem(_("Prefilter the input image"), default=True).set_prop(
        "display", store=prop
    )
    order = gdi.IntItem(
        _("Order"),
        default=3,
        min=0,
        max=5,
        help=_("Spline interpolation order"),
    ).set_prop("display", active=prop)


def rotate_obj_coords(
    angle: float, obj: ImageObj, orig: ImageObj, coords: np.ndarray
) -> None:
    """Apply rotation to coords associated to image obj
    Args:
        angle (float): rotation angle (in degrees)
        obj (ImageObj): image object
        orig (ImageObj): original image object
        coords (np.ndarray): coordinates to rotate
    Returns:
        np.ndarray: output data
    """
    for row in range(coords.shape[0]):
        for col in range(0, coords.shape[1], 2):
            x1, y1 = coords[row, col : col + 2]
            dx1 = x1 - orig.xc
            dy1 = y1 - orig.yc
            dx2, dy2 = vector_rotation(-angle * np.pi / 180.0, dx1, dy1)
            coords[row, col : col + 2] = dx2 + obj.xc, dy2 + obj.yc
    obj.roi = None


def rotate_obj_alpha(
    obj: ImageObj, orig: ImageObj, coords: np.ndarray, p: RotateParam
) -> None:
    """Apply rotation to coords associated to image obj"""
    rotate_obj_coords(p.angle, obj, orig, coords)


def compute_rotate(src: ImageObj, p: RotateParam) -> ImageObj:
    """Rotate data
    Args:
        src (ImageObj): input image object
        p (RotateParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "rotate", f"α={p.angle:.3f}°, mode='{p.mode}'")
    dst.data = spi.rotate(
        src.data,
        p.angle,
        reshape=p.reshape,
        order=p.order,
        mode=p.mode,
        cval=p.cval,
        prefilter=p.prefilter,
    )
    dst.transform_shapes(src, rotate_obj_alpha, p)
    return dst


def rotate_obj_90(dst: ImageObj, src: ImageObj, coords: np.ndarray) -> None:
    """Apply rotation to coords associated to image obj"""
    rotate_obj_coords(90.0, dst, src, coords)


def compute_rotate90(src: ImageObj) -> ImageObj:
    """Rotate data 90°
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "rotate90")
    dst.data = np.rot90(src.data)
    dst.transform_shapes(src, rotate_obj_90)
    return dst


def rotate_obj_270(dst: ImageObj, src: ImageObj, coords: np.ndarray) -> None:
    """Apply rotation to coords associated to image obj"""
    rotate_obj_coords(270.0, dst, src, coords)


def compute_rotate270(src: ImageObj) -> ImageObj:
    """Rotate data 270°
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "rotate270")
    dst.data = np.rot90(src.data, 3)
    dst.transform_shapes(src, rotate_obj_270)
    return dst


# pylint: disable=unused-argument
def hflip_coords(dst: ImageObj, src: ImageObj, coords: np.ndarray) -> None:
    """Apply HFlip to coords"""
    coords[:, ::2] = dst.x0 + dst.dx * dst.data.shape[1] - coords[:, ::2]
    dst.roi = None


def compute_fliph(src: ImageObj) -> ImageObj:
    """Flip data horizontally
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "fliph")
    dst.data = np.fliplr(src.data)
    dst.transform_shapes(src, hflip_coords)
    return dst


# pylint: disable=unused-argument
def vflip_coords(dst: ImageObj, src: ImageObj, coords: np.ndarray) -> None:
    """Apply VFlip to coords"""
    coords[:, 1::2] = dst.y0 + dst.dy * dst.data.shape[0] - coords[:, 1::2]
    dst.roi = None


def compute_flipv(src: ImageObj) -> ImageObj:
    """Flip data vertically
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "flipv")
    dst.data = np.flipud(src.data)
    dst.transform_shapes(src, vflip_coords)
    return dst


class GridParam(gdt.DataSet):
    """Grid parameters"""

    _prop = gdt.GetAttrProp("direction")
    _directions = (("col", _("columns")), ("row", _("rows")))
    direction = gdi.ChoiceItem(_("Distribute over"), _directions, radio=True).set_prop(
        "display", store=_prop
    )
    cols = gdi.IntItem(_("Columns"), default=1, nonzero=True).set_prop(
        "display", active=gdt.FuncProp(_prop, lambda x: x == "col")
    )
    rows = gdi.IntItem(_("Rows"), default=1, nonzero=True).set_prop(
        "display", active=gdt.FuncProp(_prop, lambda x: x == "row")
    )
    colspac = gdi.FloatItem(_("Column spacing"), default=0.0, min=0.0)
    rowspac = gdi.FloatItem(_("Row spacing"), default=0.0, min=0.0)


class ResizeParam(gdt.DataSet):
    """Resize parameters"""

    boundaries = ("constant", "nearest", "reflect", "wrap")
    prop = gdt.ValueProp(False)

    zoom = gdi.FloatItem(_("Zoom"))
    mode = gdi.ChoiceItem(
        _("Mode"), list(zip(boundaries, boundaries)), default=boundaries[0]
    )
    cval = gdi.FloatItem(
        _("cval"),
        default=0.0,
        help=_(
            "Value used for points outside the "
            "boundaries of the input if mode is "
            "'constant'"
        ),
    )
    prefilter = gdi.BoolItem(_("Prefilter the input image"), default=True).set_prop(
        "display", store=prop
    )
    order = gdi.IntItem(
        _("Order"),
        default=3,
        min=0,
        max=5,
        help=_("Spline interpolation order"),
    ).set_prop("display", active=prop)


def compute_resize(src: ImageObj, p: ResizeParam) -> ImageObj:
    """Zooming function
    Args:
        src (ImageObj): input image object
        p (ResizeParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "resize", f"zoom={p.zoom:.3f}")
    dst.data = spi.interpolation.zoom(
        src.data,
        p.zoom,
        order=p.order,
        mode=p.mode,
        cval=p.cval,
        prefilter=p.prefilter,
    )
    if dst.dx is not None and dst.dy is not None:
        dst.dx, dst.dy = dst.dx / p.zoom, dst.dy / p.zoom
    # TODO: [P2] Instead of removing geometric shapes, apply zoom
    dst.remove_all_shapes()
    return dst


class BinningParam(gdt.DataSet):
    """Binning parameters"""

    binning_x = gdi.IntItem(
        _("Cluster size (X)"),
        default=2,
        min=2,
        help=_("Number of adjacent pixels to be combined together along X-axis."),
    )
    binning_y = gdi.IntItem(
        _("Cluster size (Y)"),
        default=2,
        min=2,
        help=_("Number of adjacent pixels to be combined together along Y-axis."),
    )
    _operations = BINNING_OPERATIONS
    operation = gdi.ChoiceItem(
        _("Operation"),
        list(zip(_operations, _operations)),
        default=_operations[0],
    )
    _dtype_list = ["dtype"] + VALID_DTYPES_STRLIST
    dtype_str = gdi.ChoiceItem(
        _("Data type"),
        list(zip(_dtype_list, _dtype_list)),
        help=_("Output image data type."),
    )
    change_pixel_size = gdi.BoolItem(
        _("Change pixel size"),
        default=False,
        help=_("Change pixel size so that overall image size remains the same."),
    )


def compute_binning(src: ImageObj, param: BinningParam) -> ImageObj:
    """Binning function on data
    Args:
        src (ImageObj): input image object
        param (BinningParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(
        src,
        "binning",
        f"{param.binning_x}x{param.binning_y},{param.operation},"
        f"change_pixel_size={param.change_pixel_size}",
    )
    dst.data = binning(
        src.data,
        binning_x=param.binning_x,
        binning_y=param.binning_y,
        operation=param.operation,
        dtype=param.dtype_str,
    )
    if param.change_pixel_size:
        if src.dx is not None and src.dy is not None:
            dst.dx = src.dx * param.binning_x
            dst.dy = src.dy * param.binning_y
    else:
        # TODO: [P2] Instead of removing geometric shapes, apply zoom
        dst.remove_all_shapes()
    return dst


def extract_multiple_roi(src: ImageObj, group: gdt.DataSetGroup) -> ImageObj:
    """Extract multiple regions of interest from data
    Args:
        src (ImageObj): input image object
        group (gdt.DataSetGroup): parameters defining the regions of interest
    Returns:
        ImageObj: output image object
    """
    suffix = None
    if len(group.datasets) == 1:
        p = group.datasets[0]
        suffix = p.get_suffix()
    dst = dst_11(src, "extract_multiple_roi", suffix)
    if len(group.datasets) == 1:
        p = group.datasets[0]
        dst.data = src.data.copy()[p.y0 : p.y1, p.x0 : p.x1]
        return dst
    out = np.zeros_like(src.data)
    for p in group.datasets:
        slice1, slice2 = slice(p.y0, p.y1 + 1), slice(p.x0, p.x1 + 1)
        out[slice1, slice2] = src.data[slice1, slice2]
    x0 = min(p.x0 for p in group.datasets)
    y0 = min(p.y0 for p in group.datasets)
    x1 = max(p.x1 for p in group.datasets)
    y1 = max(p.y1 for p in group.datasets)
    dst.data = out[y0:y1, x0:x1]
    dst.x0 += min(p.x0 for p in group.datasets)
    dst.y0 += min(p.y0 for p in group.datasets)
    dst.roi = None
    return dst


def extract_single_roi(src: ImageObj, p: gdt.DataSet) -> ImageObj:
    """Extract single ROI
    Args:
        src (ImageObj): input image object
        p (gdt.DataSet): ROI parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "extract_single_roi", p.get_suffix())
    dst.data = src.data.copy()[p.y0 : p.y1, p.x0 : p.x1]
    dst.x0 += p.x0
    dst.y0 += p.y0
    dst.roi = None
    if p.geometry is RoiDataGeometries.CIRCLE:
        # Circular ROI
        dst.roi = p.get_single_roi()
    return dst


def compute_swap_axes(src: ImageObj) -> ImageObj:
    """Swap image axes
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "swap_axes")
    dst.data = np.transpose(src.data)
    src.remove_all_shapes()
    return dst


def compute_abs(src: ImageObj) -> ImageObj:
    """Compute absolute value
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "abs")
    dst.data = np.abs(src.data)
    return dst


def compute_log10(src: ImageObj) -> ImageObj:
    """Compute log10
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "log10")
    dst.data = np.log10(src.data)
    return dst


class ZCalibrateParam(gdt.DataSet):
    """Image linear calibration parameters"""

    a = gdi.FloatItem("a", default=1.0)
    b = gdi.FloatItem("b", default=0.0)


def compute_calibration(src: ImageObj, p: ZCalibrateParam) -> ImageObj:
    """Compute linear calibration
    Args:
        src (ImageObj): input image object
        param (ZCalibrateParam): calibration parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "calibration", f"z={p.a}*z+{p.b}")
    dst.data = p.a * src.data + p.b
    return dst


def compute_threshold(src: ImageObj, p: ThresholdParam) -> ImageObj:
    """Apply thresholding
    Args:
        src (ImageObj): input image object
        p (ThresholdParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "threshold", f"min={p.value} lsb")
    dst.data = np.clip(src.data, p.value, src.data.max())
    return dst


def compute_clip(src: ImageObj, p: ClipParam) -> ImageObj:
    """Apply clipping
    Args:
        src (ImageObj): input image object
        p (ClipParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "clip", f"max={p.value} lsb")
    dst.data = np.clip(src.data, src.data.min(), p.value)
    return dst


def compute_gaussian_filter(src: ImageObj, p: GaussianParam) -> ImageObj:
    """Compute gaussian filter
    Args:
        src (ImageObj): input image object
        p (GaussianParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "gaussian_filter", f"σ={p.sigma:.3f} pixels")
    dst.data = spi.gaussian_filter(src.data, sigma=p.sigma)
    return dst


def compute_moving_average(src: ImageObj, p: MovingAverageParam) -> ImageObj:
    """Compute moving average
    Args:
        src (ImageObj): input image object
        p (MovingAverageParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "moving_average", f"n={p.n}")
    dst.data = spi.uniform_filter(src.data, size=p.n, mode="constant")
    return dst


def compute_moving_median(src: ImageObj, p: MovingMedianParam) -> ImageObj:
    """Compute moving median
    Args:
        src (ImageObj): input image object
        p (MovingMedianParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "moving_median", f"n={p.n}")
    dst.data = sps.medfilt(src.data, kernel_size=p.n)
    return dst


def compute_wiener(src: ImageObj) -> ImageObj:
    """Compute Wiener filter
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "wiener")
    dst.data = sps.wiener(src.data)
    return dst


def compute_fft(src: ImageObj) -> ImageObj:
    """Compute FFT
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "fft")
    dst.data = np.fft.fft2(src.data)
    return dst


def compute_ifft(src: ImageObj) -> ImageObj:
    """Compute inverse FFT
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "ifft")
    dst.data = np.fft.ifft2(src.data)
    return dst


class ButterworthParam(gdt.DataSet):
    """Butterworth filter parameters"""

    cut_off = gdi.FloatItem(
        _("Cut-off frequency ratio"),
        default=0.5,
        min=0.0,
        max=1.0,
        help=_("Cut-off frequency ratio (0.0 - 1.0)."),
    )
    high_pass = gdi.BoolItem(
        _("High-pass filter"),
        default=False,
        help=_("If True, apply high-pass filter instead of low-pass."),
    )
    order = gdi.IntItem(
        _("Order"),
        default=2,
        min=1,
        help=_("Order of the Butterworth filter."),
    )


def compute_butterworth(src: ImageObj, p: ButterworthParam) -> ImageObj:
    """Compute Butterworth filter
    Args:
        src (ImageObj): input image object
        p (ButterworthParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(
        src,
        "butterworth",
        f"cut_off={p.cut_off:.3f}, order={p.order}, high_pass={p.high_pass}",
    )
    dst.data = filters.butterworth(src.data, p.cut_off, p.high_pass, p.order)
    return dst


class AdjustGammaParam(gdt.DataSet):
    """Gamma adjustment parameters"""

    gamma = gdi.FloatItem(
        _("Gamma"),
        default=1.0,
        min=0.0,
        help=_("Gamma correction factor (higher values give more contrast)."),
    )
    gain = gdi.FloatItem(
        _("Gain"),
        default=1.0,
        min=0.0,
        help=_("Gain factor (higher values give more contrast)."),
    )


def compute_adjust_gamma(src: ImageObj, p: AdjustGammaParam) -> ImageObj:
    """Gamma correction
    Args:
        src (ImageObj): input image object
        p (AdjustGammaParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "adjust_gamma", f"gamma={p.gamma}, gain={p.gain}")
    dst.data = exposure.adjust_gamma(src.data, gamma=p.gamma, gain=p.gain)
    return dst


class AdjustLogParam(gdt.DataSet):
    """Logarithmic adjustment parameters"""

    gain = gdi.FloatItem(
        _("Gain"),
        default=1.0,
        min=0.0,
        help=_("Gain factor (higher values give more contrast)."),
    )
    inv = gdi.BoolItem(
        _("Inverse"),
        default=False,
        help=_("If True, apply inverse logarithmic transformation."),
    )


def compute_adjust_log(src: ImageObj, p: AdjustLogParam) -> ImageObj:
    """Compute log correction
    Args:
        src (ImageObj): input image object
        p (AdjustLogParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "adjust_log", f"gain={p.gain}, inv={p.inv}")
    dst.data = exposure.adjust_log(src.data, gain=p.gain, inv=p.inv)
    return dst


class AdjustSigmoidParam(gdt.DataSet):
    """Sigmoid adjustment parameters"""

    cutoff = gdi.FloatItem(
        _("Cutoff"),
        default=0.5,
        min=0.0,
        max=1.0,
        help=_("Cutoff value (higher values give more contrast)."),
    )
    gain = gdi.FloatItem(
        _("Gain"),
        default=10.0,
        min=0.0,
        help=_("Gain factor (higher values give more contrast)."),
    )
    inv = gdi.BoolItem(
        _("Inverse"),
        default=False,
        help=_("If True, apply inverse sigmoid transformation."),
    )


def compute_adjust_sigmoid(src: ImageObj, p: AdjustSigmoidParam) -> ImageObj:
    """Compute sigmoid correction
    Args:
        src (ImageObj): input image object
        p (AdjustSigmoidParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(
        src, "adjust_sigmoid", f"cutoff={p.cutoff}, gain={p.gain}, inv={p.inv}"
    )
    dst.data = exposure.adjust_sigmoid(
        src.data, cutoff=p.cutoff, gain=p.gain, inv=p.inv
    )
    return dst


class RescaleIntensityParam(gdt.DataSet):
    """Intensity rescaling parameters"""

    _dtype_list = ["image", "dtype"] + VALID_DTYPES_STRLIST
    in_range = gdi.ChoiceItem(
        _("Input range"),
        list(zip(_dtype_list, _dtype_list)),
        default="image",
        help=_(
            "Min and max intensity values of input image ('image' refers to input "
            "image min/max levels, 'dtype' refers to input image data type range)."
        ),
    )
    out_range = gdi.ChoiceItem(
        _("Output range"),
        list(zip(_dtype_list, _dtype_list)),
        default="dtype",
        help=_(
            "Min and max intensity values of output image  ('image' refers to input "
            "image min/max levels, 'dtype' refers to input image data type range).."
        ),
    )


def compute_rescale_intensity(src: ImageObj, p: RescaleIntensityParam) -> ImageObj:
    """Rescale image intensity levels
    Args:
        src (ImageObj): input image object
        p (RescaleIntensityParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(
        src,
        "rescale_intensity",
        f"in_range={p.in_range}, out_range={p.out_range}",
    )
    dst.data = exposure.rescale_intensity(
        src.data, in_range=p.in_range, out_range=p.out_range
    )
    return dst


class EqualizeHistParam(gdt.DataSet):
    """Histogram equalization parameters"""

    nbins = gdi.IntItem(
        _("Number of bins"),
        min=1,
        default=256,
        help=_("Number of bins for image histogram."),
    )


def compute_equalize_hist(src: ImageObj, p: EqualizeHistParam) -> ImageObj:
    """Histogram equalization
    Args:
        src (ImageObj): input image object
        p (EqualizeHistParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "equalize_hist", f"nbins={p.nbins}")
    dst.data = exposure.equalize_hist(src.data, nbins=p.nbins)
    return dst


class EqualizeAdaptHistParam(EqualizeHistParam):
    """Adaptive histogram equalization parameters"""

    clip_limit = gdi.FloatItem(
        _("Clipping limit"),
        default=0.01,
        min=0.0,
        max=1.0,
        help=_("Clipping limit (higher values give more contrast)."),
    )


def compute_equalize_adapthist(src: ImageObj, p: EqualizeAdaptHistParam) -> ImageObj:
    """Adaptive histogram equalization
    Args:
        src (ImageObj): input image object
        p (EqualizeAdaptHistParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(
        src, "equalize_adapthist", f"nbins={p.nbins}, clip_limit={p.clip_limit}"
    )
    dst.data = exposure.equalize_adapthist(
        src.data, clip_limit=p.clip_limit, nbins=p.nbins
    )
    return dst


class DenoiseTVParam(gdt.DataSet):
    """Total Variation denoising parameters"""

    weight = gdi.FloatItem(
        _("Denoising weight"),
        default=0.1,
        min=0,
        nonzero=True,
        help=_(
            "The greater weight, the more denoising "
            "(at the expense of fidelity to input)."
        ),
    )
    eps = gdi.FloatItem(
        "Epsilon",
        default=0.0002,
        min=0,
        nonzero=True,
        help=_(
            "Relative difference of the value of the cost function that "
            "determines the stop criterion. The algorithm stops when: "
            "(E_(n-1) - E_n) < eps * E_0"
        ),
    )
    max_num_iter = gdi.IntItem(
        _("Max. iterations"),
        default=200,
        min=0,
        nonzero=True,
        help=_("Maximal number of iterations used for the optimization"),
    )


def compute_denoise_tv(src: ImageObj, p: DenoiseTVParam) -> ImageObj:
    """Compute Total Variation denoising
    Args:
        src (ImageObj): input image object
        p (DenoiseTVParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(
        src,
        "denoise_tv",
        f"weight={p.weight}, eps={p.eps}, max_num_iter={p.max_num_iter}",
    )
    dst.data = denoise_tv_chambolle(
        src.data, weight=p.weight, eps=p.eps, max_num_iter=p.max_num_iter
    )
    return dst


class DenoiseBilateralParam(gdt.DataSet):
    """Bilateral filter denoising parameters"""

    sigma_spatial = gdi.FloatItem(
        "σ<sub>spatial</sub>",
        default=1.0,
        min=0,
        nonzero=True,
        unit="pixels",
        help=_(
            "Standard deviation for range distance. "
            "A larger value results in averaging of pixels "
            "with larger spatial differences."
        ),
    )
    _modelist = ("constant", "edge", "symmetric", "reflect", "wrap")
    mode = gdi.ChoiceItem(
        _("Mode"), list(zip(_modelist, _modelist)), default="constant"
    )
    cval = gdi.FloatItem(
        "cval",
        default=0,
        help=_(
            "Used in conjunction with mode 'constant', "
            "the value outside the image boundaries."
        ),
    )


def compute_denoise_bilateral(src: ImageObj, p: DenoiseBilateralParam) -> ImageObj:
    """Compute bilateral filter denoising
    Args:
        src (ImageObj): input image object
        p (DenoiseBilateralParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(
        src,
        "denoise_bilateral",
        f"σspatial={p.sigma_spatial}, mode={p.mode}, cval={p.cval}",
    )
    dst.data = denoise_bilateral(
        src.data,
        sigma_spatial=p.sigma_spatial,
        mode=p.mode,
        cval=p.cval,
    )
    return dst


class DenoiseWaveletParam(gdt.DataSet):
    """Wavelet denoising parameters"""

    _wavelist = pywt.wavelist()
    wavelet = gdi.ChoiceItem(
        _("Wavelet"), list(zip(_wavelist, _wavelist)), default="sym9"
    )
    _modelist = ("soft", "hard")
    mode = gdi.ChoiceItem(_("Mode"), list(zip(_modelist, _modelist)), default="soft")
    _methlist = ("BayesShrink", "VisuShrink")
    method = gdi.ChoiceItem(
        _("Method"), list(zip(_methlist, _methlist)), default="VisuShrink"
    )


def compute_denoise_wavelet(src: ImageObj, p: DenoiseWaveletParam) -> ImageObj:
    """Compute Wavelet denoising
    Args:
        src (ImageObj): input image object
        p (DenoiseWaveletParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(
        src,
        "denoise_wavelet",
        f"wavelet={p.wavelet}, mode={p.mode}, method={p.method}",
    )
    dst.data = denoise_wavelet(
        src.data,
        wavelet=p.wavelet,
        mode=p.mode,
        method=p.method,
    )
    return dst


class MorphologyParam(gdt.DataSet):
    """White Top-Hat parameters"""

    radius = gdi.IntItem(
        _("Radius"), default=1, min=1, help=_("Footprint (disk) radius.")
    )


def compute_denoise_tophat(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Denoise using White Top-Hat
    Args:
        src (ImageObj): input image object
        p (MorphologyParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "denoise_tophat", f"radius={p.radius}")
    dst.data = src.data - morphology.white_tophat(src.data, morphology.disk(p.radius))
    return dst


def compute_white_tophat(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute White Top-Hat
    Args:
        src (ImageObj): input image object
        p (MorphologyParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "white_tophat", f"radius={p.radius}")
    dst.data = morphology.white_tophat(src.data, morphology.disk(p.radius))
    return dst


def compute_black_tophat(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute Black Top-Hat
    Args:
        src (ImageObj): input image object
        p (MorphologyParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "black_tophat", f"radius={p.radius}")
    dst.data = morphology.black_tophat(src.data, morphology.disk(p.radius))
    return dst


def compute_erosion(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute Erosion
    Args:
        src (ImageObj): input image object
        p (MorphologyParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "erosion", f"radius={p.radius}")
    dst.data = morphology.erosion(src.data, morphology.disk(p.radius))
    return dst


def compute_dilation(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute Dilation
    Args:
        src (ImageObj): input image object
        p (MorphologyParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "dilation", f"radius={p.radius}")
    dst.data = morphology.dilation(src.data, morphology.disk(p.radius))
    return dst


def compute_opening(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute morphological opening
    Args:
        src (ImageObj): input image object
        p (MorphologyParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "opening", f"radius={p.radius}")
    dst.data = morphology.opening(src.data, morphology.disk(p.radius))
    return dst


def compute_closing(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute morphological closing
    Args:
        src (ImageObj): input image object
        p (MorphologyParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "closing", f"radius={p.radius}")
    dst.data = morphology.closing(src.data, morphology.disk(p.radius))
    return dst


class CannyParam(gdt.DataSet):
    """Canny filter parameters"""

    sigma = gdi.FloatItem(
        "Sigma",
        default=1.0,
        unit="pixels",
        min=0,
        nonzero=True,
        help=_("Standard deviation of the Gaussian filter."),
    )
    low_threshold = gdi.FloatItem(
        _("Low threshold"),
        default=0.1,
        min=0,
        help=_("Lower bound for hysteresis thresholding (linking edges)."),
    )
    high_threshold = gdi.FloatItem(
        _("High threshold"),
        default=0.9,
        min=0,
        help=_("Upper bound for hysteresis thresholding (linking edges)."),
    )
    use_quantiles = gdi.BoolItem(
        _("Use quantiles"),
        default=True,
        help=_(
            "If True then treat low_threshold and high_threshold as quantiles "
            "of the edge magnitude image, rather than absolute edge magnitude "
            "values. If True then the thresholds must be in the range [0, 1]."
        ),
    )
    _modelist = ("reflect", "constant", "nearest", "mirror", "wrap")
    mode = gdi.ChoiceItem(
        _("Mode"), list(zip(_modelist, _modelist)), default="constant"
    )
    cval = gdi.FloatItem(
        "cval",
        default=0.0,
        help=_("Value to fill past edges of input if mode is constant."),
    )


def compute_canny(src: ImageObj, p: CannyParam) -> ImageObj:
    """Compute Canny filter
    Args:
        src (ImageObj): input image object
        p (CannyParam): parameters
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(
        src,
        "canny",
        f"sigma={p.sigma}, low_threshold={p.low_threshold}, "
        f"high_threshold={p.high_threshold}, use_quantiles={p.use_quantiles}, "
        f"mode={p.mode}, cval={p.cval}",
    )
    dst.data = np.array(
        feature.canny(
            src.data,
            sigma=p.sigma,
            low_threshold=p.low_threshold,
            high_threshold=p.high_threshold,
            use_quantiles=p.use_quantiles,
            mode=p.mode,
            cval=p.cval,
        ),
        dtype=np.uint8,
    )
    return dst


def compute_roberts(src: ImageObj) -> ImageObj:
    """Compute Roberts filter
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "roberts")
    dst.data = filters.roberts(src.data)
    return dst


def compute_prewitt(src: ImageObj) -> ImageObj:
    """Compute Prewitt filter
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "prewitt")
    dst.data = filters.prewitt(src.data)
    return dst


def compute_prewitt_h(src: ImageObj) -> ImageObj:
    """Compute horizontal Prewitt filter
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "prewitt_h")
    dst.data = filters.prewitt_h(src.data)
    return dst


def compute_prewitt_v(src: ImageObj) -> ImageObj:
    """Compute vertical Prewitt filter
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "prewitt_v")
    dst.data = filters.prewitt_v(src.data)
    return dst


def compute_sobel(src: ImageObj) -> ImageObj:
    """Compute Sobel filter
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "sobel")
    dst.data = filters.sobel(src.data)
    return dst


def compute_sobel_h(src: ImageObj) -> ImageObj:
    """Compute horizontal Sobel filter
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "sobel_h")
    dst.data = filters.sobel_h(src.data)
    return dst


def compute_sobel_v(src: ImageObj) -> ImageObj:
    """Compute vertical Sobel filter
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "sobel_v")
    dst.data = filters.sobel_v(src.data)
    return dst


def compute_scharr(src: ImageObj) -> ImageObj:
    """Compute Scharr filter
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "scharr")
    dst.data = filters.scharr(src.data)
    return dst


def compute_scharr_h(src: ImageObj) -> ImageObj:
    """Compute horizontal Scharr filter
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "scharr_h")
    dst.data = filters.scharr_h(src.data)
    return dst


def compute_scharr_v(src: ImageObj) -> ImageObj:
    """Compute vertical Scharr filter
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "scharr_v")
    dst.data = filters.scharr_v(src.data)
    return dst


def compute_farid(src: ImageObj) -> ImageObj:
    """Compute Farid filter
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "farid")
    dst.data = filters.farid(src.data)
    return dst


def compute_farid_h(src: ImageObj) -> ImageObj:
    """Compute horizontal Farid filter
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "farid_h")
    dst.data = filters.farid_h(src.data)
    return dst


def compute_farid_v(src: ImageObj) -> ImageObj:
    """Compute vertical Farid filter
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "farid_v")
    dst.data = filters.farid_v(src.data)
    return dst


def compute_laplace(src: ImageObj) -> ImageObj:
    """Compute Laplace filter
    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = dst_11(src, "laplace")
    dst.data = filters.laplace(src.data)
    return dst


def calc_with_osr(image: ImageObj, func: Callable, *args: Any) -> np.ndarray:
    """Exec computation taking into account image x0, y0, dx, dy and ROIs"""
    res = []
    for i_roi in image.iterate_roi_indexes():
        data_roi = image.get_data(i_roi)
        if args is None:
            coords = func(data_roi)
        else:
            coords = func(data_roi, *args)
        if coords.size:
            if image.roi is not None:
                x0, y0, _x1, _y1 = RoiDataItem(image.roi[i_roi]).get_rect()
                coords[:, ::2] += x0
                coords[:, 1::2] += y0
            coords[:, ::2] = image.dx * coords[:, ::2] + image.x0
            coords[:, 1::2] = image.dy * coords[:, 1::2] + image.y0
            idx = np.ones((coords.shape[0], 1)) * i_roi
            coords = np.hstack([idx, coords])
            res.append(coords)
    if res:
        return np.vstack(res)
    return None


def get_centroid_coords(data: np.ndarray) -> np.ndarray:
    """Return centroid coordinates
    Args:
        data (np.ndarray): input data
    Returns:
        np.ndarray: centroid coordinates
    """
    y, x = get_centroid_fourier(data)
    return np.array([(x, y)])


def compute_centroid(image: ImageObj) -> np.ndarray:
    """Compute centroid
    Args:
        image (ImageObj): input image
    Returns:
        np.ndarray: centroid coordinates
    """
    return calc_with_osr(image, get_centroid_coords)


def get_enclosing_circle_coords(data: np.ndarray) -> np.ndarray:
    """Return diameter coords for the circle contour enclosing image
    values above threshold (FWHM)
    Args:
        data (np.ndarray): input data
    Returns:
        np.ndarray: diameter coords
    """
    x, y, r = get_enclosing_circle(data)
    return np.array([[x - r, y, x + r, y]])


def compute_enclosing_circle(image: ImageObj) -> np.ndarray:
    """Compute minimum enclosing circle
    Args:
        image (ImageObj): input image
    Returns:
        np.ndarray: diameter coords
    """
    return calc_with_osr(image, get_enclosing_circle_coords)


class GenericDetectionParam(gdt.DataSet):
    """Generic detection parameters"""

    threshold = gdi.FloatItem(
        _("Relative threshold"),
        default=0.5,
        min=0.1,
        max=0.9,
        help=_(
            "Detection threshold, relative to difference between "
            "data maximum and minimum"
        ),
    )


class PeakDetectionParam(GenericDetectionParam):
    """Peak detection parameters"""

    size = gdi.IntItem(
        _("Neighborhoods size"),
        default=10,
        min=1,
        unit="pixels",
        help=_(
            "Size of the sliding window used in maximum/minimum filtering algorithm"
        ),
    )
    create_rois = gdi.BoolItem(_("Create regions of interest"), default=True)


def compute_peak_detection(image: ImageObj, p: PeakDetectionParam) -> np.ndarray:
    """Compute 2D peak detection
    Args:
        image (ImageObj): input image
        p (PeakDetectionParam): parameters
    Returns:
        np.ndarray: peak coordinates
    """
    return calc_with_osr(image, get_2d_peaks_coords, p.size, p.threshold)


class ContourShapeParam(GenericDetectionParam):
    """Contour shape parameters"""

    shapes = (
        ("ellipse", _("Ellipse")),
        ("circle", _("Circle")),
    )
    shape = gdi.ChoiceItem(_("Shape"), shapes, default="ellipse")


def compute_contour_shape(image: ImageObj, p: ContourShapeParam) -> np.ndarray:
    """Compute contour shape fit"""
    return calc_with_osr(image, get_contour_shapes, p.shape, p.threshold)


class HoughCircleParam(gdt.DataSet):
    """Circle Hough transform parameters"""

    min_radius = gdi.IntItem(
        _("Radius<sub>min</sub>"), unit="pixels", min=0, nonzero=True
    )
    max_radius = gdi.IntItem(
        _("Radius<sub>max</sub>"), unit="pixels", min=0, nonzero=True
    )
    min_distance = gdi.IntItem(_("Minimal distance"), min=0)


def compute_hough_circle_peaks(image: ImageObj, p: HoughCircleParam) -> np.ndarray:
    """Compute Hough circles
    Args:
        image (ImageObj): input image
        p (HoughCircleParam): parameters
    Returns:
        np.ndarray: circle coordinates
    """
    return calc_with_osr(
        image,
        get_hough_circle_peaks,
        p.min_radius,
        p.max_radius,
        None,
        p.min_distance,
    )


class BaseBlobParam(gdt.DataSet):
    """Base class for blob detection parameters"""

    min_sigma = gdi.FloatItem(
        "σ<sub>min</sub>",
        default=1.0,
        unit="pixels",
        min=0,
        nonzero=True,
        help=_(
            "The minimum standard deviation for Gaussian Kernel. "
            "Keep this low to detect smaller blobs."
        ),
    )
    max_sigma = gdi.FloatItem(
        "σ<sub>max</sub>",
        default=30.0,
        unit="pixels",
        min=0,
        nonzero=True,
        help=_(
            "The maximum standard deviation for Gaussian Kernel. "
            "Keep this high to detect larger blobs."
        ),
    )
    threshold_rel = gdi.FloatItem(
        _("Relative threshold"),
        default=0.2,
        min=0.0,
        max=1.0,
        help=_("Minimum intensity of blobs."),
    )
    overlap = gdi.FloatItem(
        _("Overlap"),
        default=0.5,
        min=0.0,
        max=1.0,
        help=_(
            "If two blobs overlap by a fraction greater than this value, the "
            "smaller blob is eliminated."
        ),
    )


class BlobDOGParam(BaseBlobParam):
    """Blob detection using Difference of Gaussian method"""

    exclude_border = gdi.BoolItem(
        _("Exclude border"),
        default=True,
        help=_("If True, exclude blobs from the border of the image."),
    )


def compute_blob_dog(image: ImageObj, p: BlobDOGParam) -> np.ndarray:
    """Compute blobs using Difference of Gaussian method
    Args:
        image (ImageObj): input image
        p (BlobDOGParam): parameters
    Returns:
        np.ndarray: blobs coordinates
    """
    return calc_with_osr(
        image,
        find_blobs_dog,
        p.min_sigma,
        p.max_sigma,
        p.overlap,
        p.threshold_rel,
        p.exclude_border,
    )


class BlobDOHParam(BaseBlobParam):
    """Blob detection using Determinant of Hessian method"""

    log_scale = gdi.BoolItem(
        _("Log scale"),
        default=False,
        help=_(
            "If set intermediate values of standard deviations are interpolated "
            "using a logarithmic scale to the base 10. "
            "If not, linear interpolation is used."
        ),
    )


def compute_blob_doh(image: ImageObj, p: BlobDOHParam) -> np.ndarray:
    """Compute blobs using Determinant of Hessian method
    Args:
        image (ImageObj): input image
        p (BlobDOHParam): parameters
    Returns:
        np.ndarray: blobs coordinates
    """
    return calc_with_osr(
        image,
        find_blobs_doh,
        p.min_sigma,
        p.max_sigma,
        p.overlap,
        p.log_scale,
        p.threshold_rel,
    )


class BlobLOGParam(BlobDOHParam):
    """Blob detection using Laplacian of Gaussian method"""

    exclude_border = gdi.BoolItem(
        _("Exclude border"),
        default=True,
        help=_("If True, exclude blobs from the border of the image."),
    )


def compute_blob_log(image: ImageObj, p: BlobLOGParam) -> np.ndarray:
    """Compute blobs using Laplacian of Gaussian method
    Args:
        image (ImageObj): input image
        p (BlobLOGParam): parameters
    Returns:
        np.ndarray: blobs coordinates
    """
    return calc_with_osr(
        image,
        find_blobs_log,
        p.min_sigma,
        p.max_sigma,
        p.overlap,
        p.log_scale,
        p.threshold_rel,
        p.exclude_border,
    )


class BlobOpenCVParam(gdt.DataSet):
    """Blob detection using OpenCV"""

    min_threshold = gdi.FloatItem(
        _("Min. threshold"),
        default=10.0,
        min=0.0,
        help=_(
            "The minimum threshold between local maxima and minima. "
            "This parameter does not affect the quality of the blobs, "
            "only the quantity. Lower thresholds result in larger "
            "numbers of blobs."
        ),
    )
    max_threshold = gdi.FloatItem(
        _("Max. threshold"),
        default=200.0,
        min=0.0,
        help=_(
            "The maximum threshold between local maxima and minima. "
            "This parameter does not affect the quality of the blobs, "
            "only the quantity. Lower thresholds result in larger "
            "numbers of blobs."
        ),
    )
    min_repeatability = gdi.IntItem(
        _("Min. repeatability"),
        default=2,
        min=1,
        help=_(
            "The minimum number of times a blob needs to be detected "
            "in a sequence of images to be considered valid."
        ),
    )
    min_dist_between_blobs = gdi.FloatItem(
        _("Min. distance between blobs"),
        default=10.0,
        min=0.0,
        help=_(
            "The minimum distance between two blobs. If blobs are found "
            "closer together than this distance, the smaller blob is removed."
        ),
    )
    _prop_col = gdt.ValueProp(False)
    filter_by_color = gdi.BoolItem(
        _("Filter by color"),
        default=True,
        help=_("If true, the image is filtered by color instead of intensity."),
    ).set_prop("display", store=_prop_col)
    blob_color = gdi.IntItem(
        _("Blob color"),
        default=0,
        help=_(
            "The color of the blobs to detect (0 for dark blobs, 255 for light blobs)."
        ),
    ).set_prop("display", active=_prop_col)
    _prop_area = gdt.ValueProp(False)
    filter_by_area = gdi.BoolItem(
        _("Filter by area"),
        default=True,
        help=_("If true, the image is filtered by blob area."),
    ).set_prop("display", store=_prop_area)
    min_area = gdi.FloatItem(
        _("Min. area"),
        default=25.0,
        min=0.0,
        help=_("The minimum blob area."),
    ).set_prop("display", active=_prop_area)
    max_area = gdi.FloatItem(
        _("Max. area"),
        default=500.0,
        min=0.0,
        help=_("The maximum blob area."),
    ).set_prop("display", active=_prop_area)
    _prop_circ = gdt.ValueProp(False)
    filter_by_circularity = gdi.BoolItem(
        _("Filter by circularity"),
        default=False,
        help=_("If true, the image is filtered by blob circularity."),
    ).set_prop("display", store=_prop_circ)
    min_circularity = gdi.FloatItem(
        _("Min. circularity"),
        default=0.8,
        min=0.0,
        max=1.0,
        help=_("The minimum circularity of the blobs."),
    ).set_prop("display", active=_prop_circ)
    max_circularity = gdi.FloatItem(
        _("Max. circularity"),
        default=1.0,
        min=0.0,
        max=1.0,
        help=_("The maximum circularity of the blobs."),
    ).set_prop("display", active=_prop_circ)
    _prop_iner = gdt.ValueProp(False)
    filter_by_inertia = gdi.BoolItem(
        _("Filter by inertia"),
        default=False,
        help=_("If true, the image is filtered by blob inertia."),
    ).set_prop("display", store=_prop_iner)
    min_inertia_ratio = gdi.FloatItem(
        _("Min. inertia ratio"),
        default=0.6,
        min=0.0,
        max=1.0,
        help=_("The minimum inertia ratio of the blobs."),
    ).set_prop("display", active=_prop_iner)
    max_inertia_ratio = gdi.FloatItem(
        _("Max. inertia ratio"),
        default=1.0,
        min=0.0,
        max=1.0,
        help=_("The maximum inertia ratio of the blobs."),
    ).set_prop("display", active=_prop_iner)
    _prop_conv = gdt.ValueProp(False)
    filter_by_convexity = gdi.BoolItem(
        _("Filter by convexity"),
        default=False,
        help=_("If true, the image is filtered by blob convexity."),
    ).set_prop("display", store=_prop_conv)
    min_convexity = gdi.FloatItem(
        _("Min. convexity"),
        default=0.8,
        min=0.0,
        max=1.0,
        help=_("The minimum convexity of the blobs."),
    ).set_prop("display", active=_prop_conv)
    max_convexity = gdi.FloatItem(
        _("Max. convexity"),
        default=1.0,
        min=0.0,
        max=1.0,
        help=_("The maximum convexity of the blobs."),
    ).set_prop("display", active=_prop_conv)


def compute_blob_opencv(image: ImageObj, p: BlobOpenCVParam) -> np.ndarray:
    """Compute blobs using OpenCV
    Args:
        image (ImageObj): input image
        p (BlobOpenCVParam): parameters
    Returns:
        np.ndarray: blobs coordinates
    """
    return calc_with_osr(
        image,
        find_blobs_opencv,
        p.min_threshold,
        p.max_threshold,
        p.min_repeatability,
        p.min_dist_between_blobs,
        p.filter_by_color,
        p.blob_color,
        p.filter_by_area,
        p.min_area,
        p.max_area,
        p.filter_by_circularity,
        p.min_circularity,
        p.max_circularity,
        p.filter_by_inertia,
        p.min_inertia_ratio,
        p.max_inertia_ratio,
        p.filter_by_convexity,
        p.min_convexity,
        p.max_convexity,
    )
