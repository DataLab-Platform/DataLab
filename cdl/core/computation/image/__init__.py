# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Image computation objects (see parent package :mod:`cdl.core.computation`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# MARK: Important note
# --------------------
# All `guidata.dataset.DataSet` classes must also be imported in the `cdl.param` module.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import guidata.dataset as gds
import numpy as np
import scipy.ndimage as spi
import scipy.signal as sps
from numpy import ma
from plotpy.mathutils.geometry import vector_rotation
from skimage import filters

from cdl.algorithms.image import (
    BINNING_OPERATIONS,
    binning,
    flatfield,
    get_centroid_fourier,
    get_enclosing_circle,
    get_hough_circle_peaks,
    get_radial_profile,
    normalize,
    z_fft,
    z_ifft,
)
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
from cdl.core.model.base import BaseProcParam, ResultProperties, ResultShape, ShapeTypes
from cdl.core.model.image import ImageObj, RoiDataGeometries, RoiDataItem
from cdl.core.model.signal import SignalObj

VALID_DTYPES_STRLIST = ImageObj.get_valid_dtypenames()


class Wrap11Func:
    """Wrap a 1 array → 1 array function to produce a 1 image → 1 image function,
    which can be used inside DataLab's infrastructure to perform computations with
    :class:`cdl.core.gui.processor.signal.ImageProcessor`.

    This wrapping mechanism using a class is necessary for the resulted function to be
    pickable by the ``multiprocessing`` module.

    The instance of this wrapper is callable and returns a :class:`cdl.obj.ImageObj`
    object.

    Example:

        >>> import numpy as np
        >>> from cdl.core.computation.signal import Wrap11Func
        >>> import cdl.obj
        >>> def add_noise(data):
        ...     return data + np.random.random(data.shape)
        >>> compute_add_noise = Wrap11Func(add_noise)
        >>> data= np.ones((100, 100))
        >>> ima0 = cdl.obj.create_image("Example", data)
        >>> ima1 = compute_add_noise(ima0)

    Args:
        func: 1 array → 1 array function
    """

    def __init__(self, func: Callable) -> None:
        self.func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__call__.__func__.__doc__ = self.func.__doc__

    def __call__(self, src: ImageObj) -> ImageObj:
        dst = dst_11(src, self.func.__name__)
        dst.data = self.func(src.data)
        return dst


def dst_11_signal(src: ImageObj, name: str, suffix: str | None = None) -> SignalObj:
    """Create a result signal object, as returned by the callback function of the
    :func:`cdl.core.gui.processor.base.BaseProcessor.compute_11` method

    Args:
        src: input image object
        name: name of the processing function

    Returns:
        Output signal object
    """
    return new_signal_result(
        src, name, suffix, (src.xunit, src.zunit), (src.xlabel, src.zlabel)
    )


# MARK: compute_n1 functions -----------------------------------------------------------
# Functions with N input images and 1 output image
# --------------------------------------------------------------------------------------
# Those functions are perfoming a computation on N input images and return a single
# output image. If we were only executing these functions locally, we would not need
# to define them here, but since we are using the multiprocessing module, we need to
# define them here so that they can be pickled and sent to the worker processes.
# Also, we need to systematically return the output image object, even if it is already
# modified in place, because the multiprocessing module will not be able to retrieve
# the modified object from the worker processes.


def compute_add(dst: ImageObj, src: ImageObj) -> ImageObj:
    """Add **dst** and **src** images and return **dst** image modified in place

    Args:
        dst: output image object
        src: input image object

    Returns:
        Output image object
    """
    dst.data += np.array(src.data, dtype=dst.data.dtype)
    return dst


def compute_product(dst: ImageObj, src: ImageObj) -> ImageObj:
    """Multiply **dst** and **src** images and return **dst** image modified in place

    Args:
        dst: output image object
        src: input image object

    Returns:
        Output image object
    """
    dst.data *= np.array(src.data, dtype=dst.data.dtype)
    return dst


def compute_add_constant(src: ImageObj, p: ConstantOperationParam) -> ImageObj:
    """Add **dst** and a constant value and return **dst** image modified in place

    Args:
        dst: output image object
        src: input image object
        p: constant value

    Returns:
        Result image object **src** + **p.value**
    """
    dst = dst_11(src, "", f"+{p.value}")
    dst.data += p.value
    return dst


def compute_difference_constant(src: ImageObj, p: ConstantOperationParam) -> ImageObj:
    """Subtract a constant value from an image

    Args:
        src: input image object
        p: constant value

    Returns:
        Result image object **src** - **p.value**
    """
    dst = dst_11(src, "", f"-{p.value}")
    dst.data -= p.value
    return dst


def compute_product_by_constant(src: ImageObj, p: ConstantOperationParam) -> ImageObj:
    """Multiply **dst** by a constant value and return **dst** image modified in place

    Args:
        dst: output image object
        src: input image object
        p: constant value

    Returns:
        Result image object **src** * **p.value**
    """
    dst = dst_11(src, "", f"*{p.value}")
    dst.data *= p.value
    return dst


def compute_divide_by_constant(src: ImageObj, p: ConstantOperationParam) -> ImageObj:
    """Divide an image by a constant value

    Args:
        src: input image object
        p: constant value

    Returns:
        Result image object **src** / **p.value**
    """
    dst = dst_11(src, "", f"/{p.value}")
    dst.data /= p.value
    return dst


# MARK: compute_n1n functions ----------------------------------------------------------
# Functions with N input images + 1 input image and N output images
# --------------------------------------------------------------------------------------


def compute_difference(src1: ImageObj, src2: ImageObj) -> ImageObj:
    """Compute difference between two images

    Args:
        src1: input image object
        src2: input image object

    Returns:
        Result image object **src1** - **src2**
    """
    dst = dst_n1n(src1, src2, "difference")
    dst.data = src1.data - src2.data
    return dst


def compute_quadratic_difference(src1: ImageObj, src2: ImageObj) -> ImageObj:
    """Compute quadratic difference between two images

    Args:
        src1: input image object
        src2: input image object

    Returns:
        Result image object (**src1** - **src2**) / sqrt(2.0)
    """
    dst = dst_n1n(src1, src2, "quadratic_difference")
    dst.data = (src1.data - src2.data) / np.sqrt(2.0)
    if np.issubdtype(dst.data.dtype, np.unsignedinteger):
        dst.data[src1.data < src2.data] = 0
    return dst


def compute_division(src1: ImageObj, src2: ImageObj) -> ImageObj:
    """Compute division between two images

    Args:
        src1: input image object
        src2: input image object

    Returns:
        Result image object **src1** / **src2**
    """
    dst = dst_n1n(src1, src2, "division")
    dst.data = src1.data / np.array(src2.data, dtype=src1.data.dtype)
    return dst


class FlatFieldParam(BaseProcParam):
    """Flat-field parameters"""

    threshold = gds.FloatItem(_("Threshold"), default=0.0)


def compute_flatfield(src1: ImageObj, src2: ImageObj, p: FlatFieldParam) -> ImageObj:
    """Compute flat field correction

    Args:
        src1: raw data image object
        src2: flat field image object
        p: flat field parameters

    Returns:
        Output image object
    """
    dst = dst_n1n(src1, src2, "flatfield", f"threshold={p.threshold}")
    dst.data = flatfield(src1.data, src2.data, p.threshold)
    return dst


# MARK: compute_11 functions -----------------------------------------------------------
# Functions with 1 input image and 1 output image
# --------------------------------------------------------------------------------------


def compute_normalize(src: ImageObj, p: NormalizeParam) -> ImageObj:
    """
    Normalize image data depending on its maximum.

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = dst_11(src, "normalize", suffix=f"ref={p.method}")
    dst.data = normalize(src.data, p.method)  # type: ignore
    return dst


class LogP1Param(gds.DataSet):
    """Log10 parameters"""

    n = gds.FloatItem("n")


def compute_logp1(src: ImageObj, p: LogP1Param) -> ImageObj:
    """Compute log10(z+n)

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "log_z_plus_n", f"n={p.n}")
    dst.data = np.log10(src.data + p.n)
    return dst


class RotateParam(gds.DataSet):
    """Rotate parameters"""

    boundaries = ("constant", "nearest", "reflect", "wrap")
    prop = gds.ValueProp(False)

    angle = gds.FloatItem(f"{_('Angle')} (°)")
    mode = gds.ChoiceItem(
        _("Mode"), list(zip(boundaries, boundaries)), default=boundaries[0]
    )
    cval = gds.FloatItem(
        _("cval"),
        default=0.0,
        help=_(
            "Value used for points outside the "
            "boundaries of the input if mode is "
            "'constant'"
        ),
    )
    reshape = gds.BoolItem(
        _("Reshape the output array"),
        default=False,
        help=_(
            "Reshape the output array "
            "so that the input array is "
            "contained completely in the output"
        ),
    )
    prefilter = gds.BoolItem(_("Prefilter the input image"), default=True).set_prop(
        "display", store=prop
    )
    order = gds.IntItem(
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
        angle: rotation angle (in degrees)
        obj: image object
        orig: original image object
        coords: coordinates to rotate

    Returns:
        Output data
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
        src: input image object
        p: parameters

    Returns:
        Output image object
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
        src: input image object

    Returns:
        Output image object
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
        src: input image object

    Returns:
        Output image object
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
        src: input image object

    Returns:
        Output image object
    """
    dst = Wrap11Func(np.fliplr)(src)
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
        src: input image object

    Returns:
        Output image object
    """
    dst = Wrap11Func(np.flipud)(src)
    dst.transform_shapes(src, vflip_coords)
    return dst


class GridParam(gds.DataSet):
    """Grid parameters"""

    _prop = gds.GetAttrProp("direction")
    _directions = (("col", _("columns")), ("row", _("rows")))
    direction = gds.ChoiceItem(_("Distribute over"), _directions, radio=True).set_prop(
        "display", store=_prop
    )
    cols = gds.IntItem(_("Columns"), default=1, nonzero=True).set_prop(
        "display", active=gds.FuncProp(_prop, lambda x: x == "col")
    )
    rows = gds.IntItem(_("Rows"), default=1, nonzero=True).set_prop(
        "display", active=gds.FuncProp(_prop, lambda x: x == "row")
    )
    colspac = gds.FloatItem(_("Column spacing"), default=0.0, min=0.0)
    rowspac = gds.FloatItem(_("Row spacing"), default=0.0, min=0.0)


class ResizeParam(gds.DataSet):
    """Resize parameters"""

    boundaries = ("constant", "nearest", "reflect", "wrap")
    prop = gds.ValueProp(False)

    zoom = gds.FloatItem(_("Zoom"))
    mode = gds.ChoiceItem(
        _("Mode"), list(zip(boundaries, boundaries)), default=boundaries[0]
    )
    cval = gds.FloatItem(
        _("cval"),
        default=0.0,
        help=_(
            "Value used for points outside the "
            "boundaries of the input if mode is "
            "'constant'"
        ),
    )
    prefilter = gds.BoolItem(_("Prefilter the input image"), default=True).set_prop(
        "display", store=prop
    )
    order = gds.IntItem(
        _("Order"),
        default=3,
        min=0,
        max=5,
        help=_("Spline interpolation order"),
    ).set_prop("display", active=prop)


def compute_resize(src: ImageObj, p: ResizeParam) -> ImageObj:
    """Zooming function

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
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
    return dst


class BinningParam(gds.DataSet):
    """Binning parameters"""

    binning_x = gds.IntItem(
        _("Cluster size (X)"),
        default=2,
        min=2,
        help=_("Number of adjacent pixels to be combined together along X-axis."),
    )
    binning_y = gds.IntItem(
        _("Cluster size (Y)"),
        default=2,
        min=2,
        help=_("Number of adjacent pixels to be combined together along Y-axis."),
    )
    _operations = BINNING_OPERATIONS
    operation = gds.ChoiceItem(
        _("Operation"),
        list(zip(_operations, _operations)),
        default=_operations[0],
    )
    _dtype_list = ["dtype"] + VALID_DTYPES_STRLIST
    dtype_str = gds.ChoiceItem(
        _("Data type"),
        list(zip(_dtype_list, _dtype_list)),
        help=_("Output image data type."),
    )
    change_pixel_size = gds.BoolItem(
        _("Change pixel size"),
        default=False,
        help=_("Change pixel size so that overall image size remains the same."),
    )


def compute_binning(src: ImageObj, param: BinningParam) -> ImageObj:
    """Binning function on data

    Args:
        src: input image object
        param: parameters

    Returns:
        Output image object
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
        dtype=None if param.dtype_str == "dtype" else param.dtype_str,
    )
    if param.change_pixel_size:
        if src.dx is not None and src.dy is not None:
            dst.dx = src.dx * param.binning_x
            dst.dy = src.dy * param.binning_y
    else:
        # TODO: [P2] Instead of removing geometric shapes, apply zoom
        dst.remove_all_shapes()
    return dst


def extract_multiple_roi(src: ImageObj, group: gds.DataSetGroup) -> ImageObj:
    """Extract multiple regions of interest from data

    Args:
        src: input image object
        group: parameters defining the regions of interest

    Returns:
        Output image object
    """
    x0 = max(min(p.x0 for p in group.datasets), 0)
    y0 = max(min(p.y0 for p in group.datasets), 0)
    x1 = max(p.x1 for p in group.datasets)
    y1 = max(p.y1 for p in group.datasets)
    suffix = None
    if len(group.datasets) == 1:
        p = group.datasets[0]
        suffix = p.get_suffix()
    dst = dst_11(src, "extract_multiple_roi", suffix)
    dst.x0 += x0 * src.dx
    dst.y0 += y0 * src.dy
    dst.roi = None
    if len(group.datasets) == 1:
        dst.data = src.data.copy()[y0:y1, x0:x1]
        return dst
    out = np.zeros_like(src.data)
    for p in group.datasets:
        slice1, slice2 = slice(max(p.y0, 0), p.y1 + 1), slice(max(p.x0, 0), p.x1 + 1)
        out[slice1, slice2] = src.data[slice1, slice2]
    dst.data = out[y0:y1, x0:x1]
    return dst


def extract_single_roi(src: ImageObj, p: gds.DataSet) -> ImageObj:
    """Extract single ROI

    Args:
        src: input image object
        p: ROI parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "extract_single_roi", p.get_suffix())
    x0, y0, x1, y1 = max(p.x0, 0), max(p.y0, 0), p.x1, p.y1
    dst.data = src.data.copy()[y0:y1, x0:x1]
    dst.x0 += x0 * src.dx
    dst.y0 += y0 * src.dy
    dst.roi = None
    if p.geometry is RoiDataGeometries.CIRCLE:
        # Circular ROI
        dst.roi = p.get_single_roi()
    return dst


class ProfileParam(gds.DataSet):
    """Horizontal or vertical profile parameters"""

    _prop = gds.GetAttrProp("direction")
    _directions = (("horizontal", _("horizontal")), ("vertical", _("vertical")))
    direction = gds.ChoiceItem(_("Direction"), _directions, radio=True).set_prop(
        "display", store=_prop
    )
    row = gds.IntItem(_("Row"), default=0, min=0).set_prop(
        "display", active=gds.FuncProp(_prop, lambda x: x == "horizontal")
    )
    col = gds.IntItem(_("Column"), default=0, min=0).set_prop(
        "display", active=gds.FuncProp(_prop, lambda x: x == "vertical")
    )


def compute_profile(src: ImageObj, p: ProfileParam) -> ImageObj:
    """Compute horizontal or vertical profile

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    data = src.get_masked_view()
    p.row = min(p.row, data.shape[0] - 1)
    p.col = min(p.col, data.shape[1] - 1)
    if p.direction == "horizontal":
        suffix, shape_index, pdata = f"row={p.row}", 1, data[p.row, :]
    else:
        suffix, shape_index, pdata = f"col={p.col}", 0, data[:, p.col]
    pdata: ma.MaskedArray
    x = np.arange(data.shape[shape_index])[~pdata.mask]
    y = np.array(pdata, dtype=float)[~pdata.mask]
    dst = dst_11_signal(src, "profile", suffix)
    dst.set_xydata(x, y)
    return dst


class AverageProfileParam(gds.DataSet):
    """Average horizontal or vertical profile parameters"""

    _directions = (("horizontal", _("horizontal")), ("vertical", _("vertical")))
    direction = gds.ChoiceItem(_("Direction"), _directions, radio=True)
    _hgroup_begin = gds.BeginGroup(_("Profile rectangular area"))
    row1 = gds.IntItem(_("Row 1"), default=0, min=0)
    row2 = gds.IntItem(_("Row 2"), default=-1, min=-1)
    col1 = gds.IntItem(_("Column 1"), default=0, min=0)
    col2 = gds.IntItem(_("Column 2"), default=-1, min=-1)
    _hgroup_end = gds.EndGroup(_("Profile rectangular area"))


def compute_average_profile(src: ImageObj, p: AverageProfileParam) -> ImageObj:
    """Compute horizontal or vertical average profile

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    data = src.get_masked_view()
    if p.row2 == -1:
        p.row2 = data.shape[0] - 1
    if p.col2 == -1:
        p.col2 = data.shape[1] - 1
    if p.row1 > p.row2:
        p.row1, p.row2 = p.row2, p.row1
    if p.col1 > p.col2:
        p.col1, p.col2 = p.col2, p.col1
    p.row1 = min(p.row1, data.shape[0] - 1)
    p.row2 = min(p.row2, data.shape[0] - 1)
    p.col1 = min(p.col1, data.shape[1] - 1)
    p.col2 = min(p.col2, data.shape[1] - 1)
    suffix = f"{p.direction}, rows=[{p.row1}, {p.row2}], cols=[{p.col1}, {p.col2}]"
    if p.direction == "horizontal":
        x, axis = np.arange(p.col1, p.col2 + 1), 0
    else:
        x, axis = np.arange(p.row1, p.row2 + 1), 1
    y = ma.mean(data[p.row1 : p.row2 + 1, p.col1 : p.col2 + 1], axis=axis)
    dst = dst_11_signal(src, "average_profile", suffix)
    dst.set_xydata(x, y)
    return dst


class RadialProfileParam(gds.DataSet):
    """Radial profile parameters"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__obj: ImageObj | None = None

    def update_from_image(self, obj: ImageObj) -> None:
        """Update parameters from image"""
        self.__obj = obj
        self.x0 = obj.xc
        self.y0 = obj.yc

    def choice_callback(self, item, value):
        """Callback for choice item"""
        if value == "centroid":
            self.y0, self.x0 = get_centroid_fourier(self.__obj.get_masked_view())
        elif value == "center":
            self.x0, self.y0 = self.__obj.xc, self.__obj.yc

    _prop = gds.GetAttrProp("center")
    center = gds.ChoiceItem(
        _("Center position"),
        (
            ("centroid", _("Image centroid")),
            ("center", _("Image center")),
            ("user", _("User-defined")),
        ),
        default="centroid",
    ).set_prop("display", store=_prop, callback=choice_callback)

    _func_prop = gds.FuncProp(_prop, lambda x: x == "user")
    _xyl = "<sub>" + _("Center") + "</sub>"
    x0 = gds.FloatItem(f"X{_xyl}", unit="pixel").set_prop("display", active=_func_prop)
    y0 = gds.FloatItem(f"X{_xyl}", unit="pixel").set_prop("display", active=_func_prop)


def compute_radial_profile(src: ImageObj, p: RadialProfileParam) -> SignalObj:
    """Compute radial profile around the centroid

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    data = src.get_masked_view()
    if p.center == "centroid":
        y0, x0 = get_centroid_fourier(data)
    elif p.center == "center":
        x0, y0 = src.xc, src.yc
    else:
        x0, y0 = p.x0, p.y0
    suffix = f"center=({x0:.3f}, {y0:.3f})"
    dst = dst_11_signal(src, "radial_profile", suffix)
    x, y = get_radial_profile(data, (x0, y0))
    dst.set_xydata(x, y)
    return dst


def compute_histogram(src: ImageObj, p: HistogramParam) -> SignalObj:
    """Compute histogram of the image data

    Args:
        src: input image object
        p: parameters

    Returns:
        Output signal object
    """
    data = src.get_masked_view().compressed()
    suffix = p.get_suffix(data)  # Also updates p.lower and p.upper
    y, bin_edges = np.histogram(data, bins=p.bins, range=(p.lower, p.upper))
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    dst = new_signal_result(
        src,
        "histogram",
        suffix=suffix,
        units=(src.zunit, ""),
        labels=(src.zlabel, _("Counts")),
    )
    dst.set_xydata(x, y)
    dst.metadata["shade"] = 0.5
    return dst


def compute_swap_axes(src: ImageObj) -> ImageObj:
    """Swap image axes

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = Wrap11Func(np.transpose)(src)
    # TODO: [P2] Instead of removing geometric shapes, apply swap
    dst.remove_all_shapes()
    return dst


def compute_abs(src: ImageObj) -> ImageObj:
    """Compute absolute value

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(np.abs)(src)


def compute_re(src: ImageObj) -> ImageObj:
    """Compute real part

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(np.real)(src)


def compute_im(src: ImageObj) -> ImageObj:
    """Compute imaginary part

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(np.imag)(src)


class DataTypeIParam(gds.DataSet):
    """Convert image data type parameters"""

    dtype_str = gds.ChoiceItem(
        _("Destination data type"),
        list(zip(VALID_DTYPES_STRLIST, VALID_DTYPES_STRLIST)),
        help=_("Output image data type."),
    )


def compute_astype(src: ImageObj, p: DataTypeIParam) -> ImageObj:
    """Convert image data type

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "astype", p.dtype_str)
    dst.data = src.data.astype(p.dtype_str)
    return dst


def compute_log10(src: ImageObj) -> ImageObj:
    """Compute log10

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(np.log10)(src)


def compute_exp(src: ImageObj) -> ImageObj:
    """Compute exponential

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(np.exp)(src)


class ZCalibrateParam(gds.DataSet):
    """Image linear calibration parameters"""

    a = gds.FloatItem("a", default=1.0)
    b = gds.FloatItem("b", default=0.0)


def compute_calibration(src: ImageObj, p: ZCalibrateParam) -> ImageObj:
    """Compute linear calibration

    Args:
        src: input image object
        p: calibration parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "calibration", f"z={p.a}*z+{p.b}")
    dst.data = p.a * src.data + p.b
    return dst


def compute_threshold(src: ImageObj, p: ThresholdParam) -> ImageObj:
    """Apply thresholding

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "threshold", f"min={p.value} lsb")
    dst.data = np.clip(src.data, p.value, src.data.max())
    return dst


def compute_clip(src: ImageObj, p: ClipParam) -> ImageObj:
    """Apply clipping

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "clip", f"max={p.value} lsb")
    dst.data = np.clip(src.data, src.data.min(), p.value)
    return dst


def compute_gaussian_filter(src: ImageObj, p: GaussianParam) -> ImageObj:
    """Compute gaussian filter

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "gaussian_filter", f"σ={p.sigma:.3f} pixels")
    dst.data = spi.gaussian_filter(src.data, sigma=p.sigma)
    return dst


def compute_moving_average(src: ImageObj, p: MovingAverageParam) -> ImageObj:
    """Compute moving average

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "moving_average", f"n={p.n}")
    dst.data = spi.uniform_filter(src.data, size=p.n, mode="constant")
    return dst


def compute_moving_median(src: ImageObj, p: MovingMedianParam) -> ImageObj:
    """Compute moving median

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "moving_median", f"n={p.n}")
    dst.data = sps.medfilt(src.data, kernel_size=p.n)
    return dst


def compute_wiener(src: ImageObj) -> ImageObj:
    """Compute Wiener filter

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(sps.wiener)(src)


def compute_fft(src: ImageObj, p: FFTParam) -> ImageObj:
    """Compute FFT

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "fft")
    dst.data = z_fft(src.data, shift=p.shift)
    return dst


def compute_ifft(src: ImageObj, p: FFTParam) -> ImageObj:
    """Compute inverse FFT

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "ifft")
    dst.data = z_ifft(src.data, shift=p.shift)
    return dst


class ButterworthParam(gds.DataSet):
    """Butterworth filter parameters"""

    cut_off = gds.FloatItem(
        _("Cut-off frequency ratio"),
        default=0.5,
        min=0.0,
        max=1.0,
        help=_("Cut-off frequency ratio (0.0 - 1.0)."),
    )
    high_pass = gds.BoolItem(
        _("High-pass filter"),
        default=False,
        help=_("If True, apply high-pass filter instead of low-pass."),
    )
    order = gds.IntItem(
        _("Order"),
        default=2,
        min=1,
        help=_("Order of the Butterworth filter."),
    )


def compute_butterworth(src: ImageObj, p: ButterworthParam) -> ImageObj:
    """Compute Butterworth filter

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(
        src,
        "butterworth",
        f"cut_off={p.cut_off:.3f}, order={p.order}, high_pass={p.high_pass}",
    )
    dst.data = filters.butterworth(src.data, p.cut_off, p.high_pass, p.order)
    return dst


# MARK: compute_10 functions -----------------------------------------------------------
# Functions with 1 input image and 0 output image
# --------------------------------------------------------------------------------------


def calc_resultshape(
    label: str, shapetype: ShapeTypes, obj: ImageObj, func: Callable, *args: Any
) -> ResultShape | None:
    """Calculate result shape by executing a computation function on an image object,
    taking into account the image origin (x0, y0), scale (dx, dy) and ROIs.

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

        Moreover, the computation function must return a single value or a NumPy array
        containing the result of the computation. This array contains the coordinates
        of points, polygons, circles or ellipses in the form [[x, y], ...], or
        [[x0, y0, x1, y1, ...], ...], or [[x0, y0, r], ...], or
        [[x0, y0, a, b, theta], ...].
    """
    res = []
    num_cols = []
    for i_roi in obj.iterate_roi_indexes():
        data_roi = obj.get_data(i_roi)
        if args is None:
            coords: np.ndarray = func(data_roi)
        else:
            coords: np.ndarray = func(data_roi, *args)

        # This is a very long condition, but it's still quite readable, so we keep it
        # as is and disable the pylint warning.
        #
        # pylint: disable=too-many-boolean-expressions
        if not isinstance(coords, np.ndarray) or (
            (
                coords.ndim != 2
                or coords.shape[1] < 2
                or (coords.shape[1] > 5 and coords.shape[1] % 2 != 0)
            )
            and coords.size > 0
        ):
            raise ValueError(
                f"Computation function {func.__name__} must return a NumPy array "
                f"containing coordinates of points, polygons, circles or ellipses "
                f"(in the form [[x, y], ...], or [[x0, y0, x1, y1, ...], ...], or "
                f"[[x0, y0, r], ...], or [[x0, y0, a, b, theta], ...]), or an empty "
                f"array."
            )

        if coords.size:
            if coords.shape[1] % 2 == 0:
                # Coordinates are in the form [x0, y0, x1, y1, ...]
                colx, coly = slice(None, None, 2), slice(1, None, 2)
            else:
                # Circle [x0, y0, r] or ellipse coordinates [x0, y0, a, b, theta]
                colx, coly = 0, 1
            if obj.roi is not None:
                x0, y0, _x1, _y1 = RoiDataItem(obj.roi[i_roi]).get_rect()
                coords[:, colx] += x0
                coords[:, coly] += y0
            coords[:, colx] = obj.dx * coords[:, colx] + obj.x0
            coords[:, coly] = obj.dy * coords[:, coly] + obj.y0
            idx = np.ones((coords.shape[0], 1)) * i_roi
            coords = np.hstack([idx, coords])
            res.append(coords)
            num_cols.append(coords.shape[1])
    if res:
        if len(set(num_cols)) != 1:
            # This happens when the number of columns is not the same for all ROIs.
            # As of now, this happens only for polygon contours.
            # We need to pad the arrays with NaNs.
            max_cols = max(num_cols)
            num_rows = sum(coords.shape[0] for coords in res)
            out = np.full((num_rows, max_cols), np.nan)
            row = 0
            for coords in res:
                out[row : row + coords.shape[0], : coords.shape[1]] = coords
                row += coords.shape[0]
            return out
        return ResultShape(label, np.vstack(res), shapetype)
    return None


def get_centroid_coords(data: np.ndarray) -> np.ndarray:
    """Return centroid coordinates

    Args:
        data: input data

    Returns:
        Centroid coordinates
    """
    y, x = get_centroid_fourier(data)
    return np.array([(x, y)])


def compute_centroid(image: ImageObj) -> ResultShape | None:
    """Compute centroid

    Args:
        image: input image

    Returns:
        Centroid coordinates
    """
    return calc_resultshape("centroid", ShapeTypes.MARKER, image, get_centroid_coords)


def get_enclosing_circle_coords(data: np.ndarray) -> np.ndarray:
    """Return diameter coords for the circle contour enclosing image
    values above threshold (FWHM)

    Args:
        data: input data

    Returns:
        Diameter coords
    """
    x, y, r = get_enclosing_circle(data)
    return np.array([[x, y, r]])


def compute_enclosing_circle(image: ImageObj) -> ResultShape | None:
    """Compute minimum enclosing circle

    Args:
        image: input image

    Returns:
        Diameter coords
    """
    return calc_resultshape(
        "enclosing_circle", ShapeTypes.CIRCLE, image, get_enclosing_circle_coords
    )


class HoughCircleParam(gds.DataSet):
    """Circle Hough transform parameters"""

    min_radius = gds.IntItem(
        _("Radius<sub>min</sub>"), unit="pixels", min=0, nonzero=True
    )
    max_radius = gds.IntItem(
        _("Radius<sub>max</sub>"), unit="pixels", min=0, nonzero=True
    )
    min_distance = gds.IntItem(_("Minimal distance"), min=0)


def compute_hough_circle_peaks(
    image: ImageObj, p: HoughCircleParam
) -> ResultShape | None:
    """Compute Hough circles

    Args:
        image: input image
        p: parameters

    Returns:
        Circle coordinates
    """
    return calc_resultshape(
        "hough_circle_peaks",
        ShapeTypes.CIRCLE,
        image,
        get_hough_circle_peaks,
        p.min_radius,
        p.max_radius,
        None,
        p.min_distance,
    )


def compute_stats_func(obj: ImageObj) -> ResultProperties:
    """Compute statistics functions"""
    statfuncs = {
        "min(z)": ma.min,
        "max(z)": ma.max,
        "<z>": ma.mean,
        "median(z)": ma.median,
        "σ(z)": ma.std,
        "<z>/σ(z)": lambda z: ma.mean(z) / ma.std(z),
        "peak-to-peak(z)": ma.ptp,
        "Σ(z)": ma.sum,
    }
    return calc_resultproperties("stats", obj, statfuncs)
