# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Image computation objects (see parent package :mod:`cdl.computation`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# MARK: Important note
# --------------------
# All `guidata.dataset.DataSet` classes must also be imported in the `cdl.param` module.

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import guidata.dataset as gds
import numpy as np
import scipy.ndimage as spi
import scipy.signal as sps
from numpy import ma
from plotpy.mathutils.geometry import vector_rotation

# Import as "csline" to avoid the function to be interpreted as a validation function
# in the context of DataLab's validation process:
from plotpy.panels.csection.csitem import compute_line_section as csline
from skimage import filters

import cdl.algorithms.image as alg
from cdl.algorithms.datatypes import clip_astype, is_integer_dtype
from cdl.computation.base import (
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
    dst_11,
    dst_n1n,
    new_signal_result,
)
from cdl.config import _
from cdl.obj import (
    BaseProcParam,
    ImageObj,
    ImageRoiDataItem,
    ResultProperties,
    ResultShape,
    ROI2DParam,
    RoiDataGeometries,
    SignalObj,
)

VALID_DTYPES_STRLIST = ImageObj.get_valid_dtypenames()


def restore_data_outside_roi(dst: ImageObj, src: ImageObj) -> None:
    """Restore data outside the Region Of Interest (ROI) of the input image after a
    computation, only if the input image has a ROI, if the data types are compatible,
    and if the shapes are the same.
    Otherwise, do nothing.

    Args:
        dst: output image object
        src: input image object
    """
    if (
        src.maskdata is not None
        and (dst.data.dtype == src.data.dtype or not is_integer_dtype(dst.data.dtype))
        and dst.data.shape == src.data.shape
    ):
        dst.data[src.maskdata] = src.data[src.maskdata]


class Wrap11Func:
    """Wrap a 1 array → 1 array function to produce a 1 image → 1 image function,
    which can be used inside DataLab's infrastructure to perform computations with
    :class:`cdl.core.gui.processor.image.ImageProcessor`.

    This wrapping mechanism using a class is necessary for the resulted function to be
    pickable by the ``multiprocessing`` module.

    The instance of this wrapper is callable and returns a :class:`cdl.obj.ImageObj`
    object.

    Example:

        >>> import numpy as np
        >>> from cdl.computation.image import Wrap11Func
        >>> import cdl.obj
        >>> def add_noise(data):
        ...     return data + np.random.random(data.shape)
        >>> compute_add_noise = Wrap11Func(add_noise)
        >>> data= np.ones((100, 100))
        >>> ima0 = cdl.obj.create_image("Example", data)
        >>> ima1 = compute_add_noise(ima0)

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

    def __call__(self, src: ImageObj) -> ImageObj:
        """Compute the function on the input image and return the result image

        Args:
            src: input image object

        Returns:
            Output image object
        """
        suffix = ", ".join(
            [str(arg) for arg in self.args]
            + [f"{k}={v}" for k, v in self.kwargs.items() if v is not None]
        )
        dst = dst_11(src, self.func.__name__, suffix)
        dst.data = self.func(src.data, *self.args, **self.kwargs)
        restore_data_outside_roi(dst, src)
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


def compute_addition(dst: ImageObj, src: ImageObj) -> ImageObj:
    """Add **dst** and **src** images and return **dst** image modified in place

    Args:
        dst: output image object
        src: input image object

    Returns:
        Output image object (modified in place)
    """
    dst.data = np.add(dst.data, src.data, dtype=float)
    restore_data_outside_roi(dst, src)
    return dst


def compute_product(dst: ImageObj, src: ImageObj) -> ImageObj:
    """Multiply **dst** and **src** images and return **dst** image modified in place

    Args:
        dst: output image object
        src: input image object

    Returns:
        Output image object (modified in place)
    """
    dst.data = np.multiply(dst.data, src.data, dtype=float)
    restore_data_outside_roi(dst, src)
    return dst


def compute_addition_constant(src: ImageObj, p: ConstantParam) -> ImageObj:
    """Add **dst** and a constant value and return the new result image object

    Args:
        src: input image object
        p: constant value

    Returns:
        Result image object **src** + **p.value** (new object)
    """
    # For the addition of a constant value, we convert the constant value to the same
    # data type as the input image, for consistency.
    value = np.array(p.value).astype(dtype=src.data.dtype)
    dst = dst_11(src, "+", str(value))
    dst.data = np.add(src.data, value, dtype=float)
    restore_data_outside_roi(dst, src)
    return dst


def compute_difference_constant(src: ImageObj, p: ConstantParam) -> ImageObj:
    """Subtract a constant value from an image and return the new result image object

    Args:
        src: input image object
        p: constant value

    Returns:
        Result image object **src** - **p.value** (new object)
    """
    # For the subtraction of a constant value, we convert the constant value to the same
    # data type as the input image, for consistency.
    value = np.array(p.value).astype(dtype=src.data.dtype)
    dst = dst_11(src, "-", str(value))
    dst.data = np.subtract(src.data, value, dtype=float)
    restore_data_outside_roi(dst, src)
    return dst


def compute_product_constant(src: ImageObj, p: ConstantParam) -> ImageObj:
    """Multiply **dst** by a constant value and return the new result image object

    Args:
        src: input image object
        p: constant value

    Returns:
        Result image object **src** * **p.value** (new object)
    """
    # For the multiplication by a constant value, we do not convert the constant value
    # to the same data type as the input image, because we want to allow the user to
    # multiply an image by a constant value of a different data type. The final data
    # type conversion ensures that the output image has the same data type as the input
    # image.
    dst = dst_11(src, "×", str(p.value))
    dst.data = np.multiply(src.data, p.value, dtype=float)
    restore_data_outside_roi(dst, src)
    return dst


def compute_division_constant(src: ImageObj, p: ConstantParam) -> ImageObj:
    """Divide an image by a constant value and return the new result image object

    Args:
        src: input image object
        p: constant value

    Returns:
        Result image object **src** / **p.value** (new object)
    """
    # For the division by a constant value, we do not convert the constant value to the
    # same data type as the input image, because we want to allow the user to divide an
    # image by a constant value of a different data type. The final data type conversion
    # ensures that the output image has the same data type as the input image.
    dst = dst_11(src, "/", str(p.value))
    dst.data = np.divide(src.data, p.value, dtype=float)
    restore_data_outside_roi(dst, src)
    return dst


# MARK: compute_n1n functions ----------------------------------------------------------
# Functions with N input images + 1 input image and N output images
# --------------------------------------------------------------------------------------


def compute_arithmetic(src1: ImageObj, src2: ImageObj, p: ArithmeticParam) -> ImageObj:
    """Compute arithmetic operation on two images

    Args:
        src1: input image object
        src2: input image object
        p: arithmetic parameters

    Returns:
        Result image object
    """
    initial_dtype = src1.data.dtype
    title = p.operation.replace("obj1", src1.short_id).replace("obj2", src2.short_id)
    dst = src1.copy(title=title)
    o, a, b = p.operator, p.factor, p.constant
    # Apply operator
    if o in ("×", "/") and a == 0.0:
        dst.data = np.ones_like(src1.data) * b
    elif o == "+":
        dst.data = np.add(src1.data, src2.data, dtype=float) * a + b
    elif o == "-":
        dst.data = np.subtract(src1.data, src2.data, dtype=float) * a + b
    elif o == "×":
        dst.data = np.multiply(src1.data, src2.data, dtype=float) * a + b
    elif o == "/":
        dst.data = np.divide(src1.data, src2.data, dtype=float) * a + b
    # Eventually convert to initial data type
    if p.restore_dtype:
        dst.data = clip_astype(dst.data, initial_dtype)
    restore_data_outside_roi(dst, src1)
    return dst


def compute_difference(src1: ImageObj, src2: ImageObj) -> ImageObj:
    """Compute difference between two images

    Args:
        src1: input image object
        src2: input image object

    Returns:
        Result image object **src1** - **src2** (new object)
    """
    dst = dst_n1n(src1, src2, "-")
    dst.data = np.subtract(src1.data, src2.data, dtype=float)
    restore_data_outside_roi(dst, src1)
    return dst


def compute_quadratic_difference(src1: ImageObj, src2: ImageObj) -> ImageObj:
    """Compute quadratic difference between two images

    Args:
        src1: input image object
        src2: input image object

    Returns:
        Result image object (**src1** - **src2**) / sqrt(2.0) (new object)
    """
    dst = dst_n1n(src1, src2, "quadratic_difference")
    dst.data = np.subtract(src1.data, src2.data, dtype=float) / np.sqrt(2.0)
    restore_data_outside_roi(dst, src1)
    return dst


def compute_division(src1: ImageObj, src2: ImageObj) -> ImageObj:
    """Compute division between two images

    Args:
        src1: input image object
        src2: input image object

    Returns:
        Result image object **src1** / **src2** (new object)
    """
    dst = dst_n1n(src1, src2, "/")
    dst.data = np.divide(src1.data, src2.data, dtype=float)
    restore_data_outside_roi(dst, src1)
    return dst


class FlatFieldParam(BaseProcParam):
    """Flat-field parameters"""

    threshold = gds.FloatItem(_("Threshold"), default=0.0)


def compute_flatfield(src1: ImageObj, src2: ImageObj, p: FlatFieldParam) -> ImageObj:
    """Compute flat field correction with :py:func:`cdl.algorithms.image.flatfield`

    Args:
        src1: raw data image object
        src2: flat field image object
        p: flat field parameters

    Returns:
        Output image object
    """
    dst = dst_n1n(src1, src2, "flatfield", f"threshold={p.threshold}")
    dst.data = alg.flatfield(src1.data, src2.data, p.threshold)
    restore_data_outside_roi(dst, src1)
    return dst


# MARK: compute_11 functions -----------------------------------------------------------
# Functions with 1 input image and 1 output image
# --------------------------------------------------------------------------------------


def compute_normalize(src: ImageObj, p: NormalizeParam) -> ImageObj:
    """
    Normalize image data depending on its maximum,
    with :py:func:`cdl.algorithms.image.normalize`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = dst_11(src, "normalize", suffix=f"ref={p.method}")
    dst.data = alg.normalize(src.data, p.method)  # type: ignore
    restore_data_outside_roi(dst, src)
    return dst


class LogP1Param(gds.DataSet):
    """Log10 parameters"""

    n = gds.FloatItem("n")


def compute_logp1(src: ImageObj, p: LogP1Param) -> ImageObj:
    """Compute log10(z+n) with :py:data:`numpy.log10`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "log_z_plus_n", f"n={p.n}")
    dst.data = np.log10(src.data + p.n)
    restore_data_outside_roi(dst, src)
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
    """Rotate data with :py:func:`scipy.ndimage.rotate`

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
    """Rotate data 90° with :py:func:`numpy.rot90`

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
    """Rotate data 270° with :py:func:`numpy.rot90`

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
    coords[:, ::2] = dst.x0 + dst.width - coords[:, ::2]
    dst.roi = None


def compute_fliph(src: ImageObj) -> ImageObj:
    """Flip data horizontally with :py:func:`numpy.fliplr`

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
    coords[:, 1::2] = dst.y0 + dst.height - coords[:, 1::2]
    dst.roi = None


def compute_flipv(src: ImageObj) -> ImageObj:
    """Flip data vertically with :py:func:`numpy.flipud`

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
    """Zooming function with :py:func:`scipy.ndimage.zoom`

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

    sx = gds.IntItem(
        _("Cluster size (X)"),
        default=2,
        min=2,
        help=_("Number of adjacent pixels to be combined together along X-axis."),
    )
    sy = gds.IntItem(
        _("Cluster size (Y)"),
        default=2,
        min=2,
        help=_("Number of adjacent pixels to be combined together along Y-axis."),
    )
    operations = alg.BINNING_OPERATIONS
    operation = gds.ChoiceItem(
        _("Operation"),
        list(zip(operations, operations)),
        default=operations[0],
    )
    dtypes = ["dtype"] + VALID_DTYPES_STRLIST
    dtype_str = gds.ChoiceItem(
        _("Data type"),
        list(zip(dtypes, dtypes)),
        help=_("Output image data type."),
    )
    change_pixel_size = gds.BoolItem(
        _("Change pixel size"),
        default=False,
        help=_("Change pixel size so that overall image size remains the same."),
    )


def compute_binning(src: ImageObj, param: BinningParam) -> ImageObj:
    """Binning function on data with :py:func:`cdl.algorithms.image.binning`

    Args:
        src: input image object
        param: parameters

    Returns:
        Output image object
    """
    dst = dst_11(
        src,
        "binning",
        f"{param.sx}x{param.sy},{param.operation},"
        f"change_pixel_size={param.change_pixel_size}",
    )
    dst.data = alg.binning(
        src.data,
        sx=param.sx,
        sy=param.sy,
        operation=param.operation,
        dtype=None if param.dtype_str == "dtype" else param.dtype_str,
    )
    if param.change_pixel_size:
        if src.dx is not None and src.dy is not None:
            dst.dx = src.dx * param.sx
            dst.dy = src.dy * param.sy
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
    # Initialize x0, y0 with maximum values:
    y0, x0 = src.data.shape
    # Initialize x1, y1 with minimum values:
    y1, x1 = 0, 0
    for p in group.datasets:
        p: ROI2DParam
        x0i, y0i, x1i, y1i = p.get_rect_indexes()
        x0, y0, x1, y1 = min(x0, x0i), min(y0, y0i), max(x1, x1i), max(y1, y1i)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, src.data.shape[1]), min(y1, src.data.shape[0])

    suffix = None
    if len(group.datasets) == 1:
        p = group.datasets[0]
        suffix = p.get_suffix()
    dst = dst_11(src, "extract_multiple_roi", suffix)
    dst.x0 += x0 * src.dx
    dst.y0 += y0 * src.dy
    dst.roi = None

    src2 = src.copy()
    src2.roi = src2.params_to_roidata(group)
    src2.data[src2.maskdata] = 0
    dst.data = src2.data[y0:y1, x0:x1]
    return dst


def extract_single_roi(src: ImageObj, p: ROI2DParam) -> ImageObj:
    """Extract single ROI

    Args:
        src: input image object
        p: ROI parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "extract_single_roi", p.get_suffix())
    dst.data = p.get_data(src).copy()
    x0, y0, _x1, _y1 = p.get_rect_indexes()
    dst.x0 += x0 * src.dx
    dst.y0 += y0 * src.dy
    dst.roi = None
    if p.geometry is RoiDataGeometries.CIRCLE:
        # Circular ROI
        dst.roi = p.get_single_roi()
    return dst


class LineProfileParam(gds.DataSet):
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


def compute_line_profile(src: ImageObj, p: LineProfileParam) -> ImageObj:
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


class SegmentProfileParam(gds.DataSet):
    """Segment profile parameters"""

    row1 = gds.IntItem(_("Start row"), default=0, min=0)
    col1 = gds.IntItem(_("Start column"), default=0, min=0)
    row2 = gds.IntItem(_("End row"), default=0, min=0)
    col2 = gds.IntItem(_("End column"), default=0, min=0)


def compute_segment_profile(src: ImageObj, p: SegmentProfileParam) -> ImageObj:
    """Compute segment profile

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    data = src.get_masked_view()
    p.row1 = min(p.row1, data.shape[0] - 1)
    p.col1 = min(p.col1, data.shape[1] - 1)
    p.row2 = min(p.row2, data.shape[0] - 1)
    p.col2 = min(p.col2, data.shape[1] - 1)
    suffix = f"({p.row1}, {p.col1})-({p.row2}, {p.col2})"
    x, y = csline(data, p.row1, p.col1, p.row2, p.col2)
    x, y = x[~np.isnan(y)], y[~np.isnan(y)]  # Remove NaN values
    dst = dst_11_signal(src, "segment_profile", suffix)
    dst.set_xydata(np.array(x, dtype=float), np.array(y, dtype=float))
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
            self.y0, self.x0 = alg.get_centroid_fourier(self.__obj.get_masked_view())
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
    with :py:func:`cdl.algorithms.image.get_radial_profile`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    data = src.get_masked_view()
    if p.center == "centroid":
        y0, x0 = alg.get_centroid_fourier(data)
    elif p.center == "center":
        x0, y0 = src.xc, src.yc
    else:
        x0, y0 = p.x0, p.y0
    suffix = f"center=({x0:.3f}, {y0:.3f})"
    dst = dst_11_signal(src, "radial_profile", suffix)
    x, y = alg.get_radial_profile(data, (x0, y0))
    dst.set_xydata(x, y)
    return dst


def compute_histogram(src: ImageObj, p: HistogramParam) -> SignalObj:
    """Compute histogram of the image data, with :py:func:`numpy.histogram`

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
    """Swap image axes with :py:func:`numpy.transpose`

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
    """Compute absolute value with :py:data:`numpy.absolute`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(np.absolute)(src)


def compute_re(src: ImageObj) -> ImageObj:
    """Compute real part with :py:func:`numpy.real`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(np.real)(src)


def compute_im(src: ImageObj) -> ImageObj:
    """Compute imaginary part with :py:func:`numpy.imag`

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
    """Convert image data type with :py:func:`cdl.algorithms.datatypes.clip_astype`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "clip_astype", p.dtype_str)
    dst.data = clip_astype(src.data, p.dtype_str)
    return dst


def compute_log10(src: ImageObj) -> ImageObj:
    """Compute log10 with :py:data:`numpy.log10`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(np.log10)(src)


def compute_exp(src: ImageObj) -> ImageObj:
    """Compute exponential with :py:data:`numpy.exp`

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
    restore_data_outside_roi(dst, src)
    return dst


def compute_clip(src: ImageObj, p: ClipParam) -> ImageObj:
    """Apply clipping with :py:func:`numpy.clip`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    return Wrap11Func(np.clip, a_min=p.lower, a_max=p.upper)(src)


def compute_offset_correction(src: ImageObj, p: ROI2DParam) -> ImageObj:
    """Apply offset correction

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "offset_correction", p.get_suffix())
    dst.data = src.data - p.get_data(src).mean()
    restore_data_outside_roi(dst, src)
    return dst


def compute_gaussian_filter(src: ImageObj, p: GaussianParam) -> ImageObj:
    """Compute gaussian filter with :py:func:`scipy.ndimage.gaussian_filter`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    return Wrap11Func(spi.gaussian_filter, sigma=p.sigma)(src)


def compute_moving_average(src: ImageObj, p: MovingAverageParam) -> ImageObj:
    """Compute moving average with :py:func:`scipy.ndimage.uniform_filter`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    return Wrap11Func(spi.uniform_filter, size=p.n, mode=p.mode)(src)


def compute_moving_median(src: ImageObj, p: MovingMedianParam) -> ImageObj:
    """Compute moving median with :py:func:`scipy.ndimage.median_filter`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    return Wrap11Func(spi.median_filter, size=p.n, mode=p.mode)(src)


def compute_wiener(src: ImageObj) -> ImageObj:
    """Compute Wiener filter with :py:func:`scipy.signal.wiener`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(sps.wiener)(src)


def compute_fft(src: ImageObj, p: FFTParam | None = None) -> ImageObj:
    """Compute FFT with :py:func:`cdl.algorithms.image.fft2d`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "fft")
    dst.data = alg.fft2d(src.data, shift=True if p is None else p.shift)
    dst.save_attr_to_metadata("xunit", "")
    dst.save_attr_to_metadata("yunit", "")
    dst.save_attr_to_metadata("zunit", "")
    dst.save_attr_to_metadata("xlabel", _("Frequency"))
    dst.save_attr_to_metadata("ylabel", _("Frequency"))
    return dst


def compute_ifft(src: ImageObj, p: FFTParam | None = None) -> ImageObj:
    """Compute inverse FFT with :py:func:`cdl.algorithms.image.ifft2d`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "ifft")
    dst.data = alg.ifft2d(src.data, shift=True if p is None else p.shift)
    dst.restore_attr_from_metadata("xunit", "")
    dst.restore_attr_from_metadata("yunit", "")
    dst.restore_attr_from_metadata("zunit", "")
    dst.restore_attr_from_metadata("xlabel", "")
    dst.restore_attr_from_metadata("ylabel", "")
    return dst


def compute_magnitude_spectrum(
    src: ImageObj, p: SpectrumParam | None = None
) -> ImageObj:
    """Compute magnitude spectrum
    with :py:func:`cdl.algorithms.image.magnitude_spectrum`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "magnitude_spectrum")
    log_scale = p is not None and p.log
    dst.data = alg.magnitude_spectrum(src.data, log_scale=log_scale)
    dst.xunit = dst.yunit = dst.zunit = ""
    dst.xlabel = dst.ylabel = _("Frequency")
    return dst


def compute_phase_spectrum(src: ImageObj) -> ImageObj:
    """Compute phase spectrum
    with :py:func:`cdl.algorithms.image.phase_spectrum`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = Wrap11Func(alg.phase_spectrum)(src)
    dst.xunit = dst.yunit = dst.zunit = ""
    dst.xlabel = dst.ylabel = _("Frequency")
    return dst


def compute_psd(src: ImageObj, p: SpectrumParam | None = None) -> ImageObj:
    """Compute power spectral density
    with :py:func:`cdl.algorithms.image.psd`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "psd")
    log_scale = p is not None and p.log
    dst.data = alg.psd(src.data, log_scale=log_scale)
    dst.xunit = dst.yunit = dst.zunit = ""
    dst.xlabel = dst.ylabel = _("Frequency")
    return dst


class ButterworthParam(gds.DataSet):
    """Butterworth filter parameters"""

    cut_off = gds.FloatItem(
        _("Cut-off frequency ratio"),
        default=0.005,
        min=0.0,
        max=0.5,
        help=_("Cut-off frequency ratio"),
    )
    high_pass = gds.BoolItem(
        _("High-pass filter"),
        default=False,
        help=_("If True, apply high-pass filter instead of low-pass"),
    )
    order = gds.IntItem(
        _("Order"),
        default=2,
        min=1,
        help=_("Order of the Butterworth filter"),
    )


def compute_butterworth(src: ImageObj, p: ButterworthParam) -> ImageObj:
    """Compute Butterworth filter with :py:func:`skimage.filters.butterworth`

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
    restore_data_outside_roi(dst, src)
    return dst


# MARK: compute_10 functions -----------------------------------------------------------
# Functions with 1 input image and 0 output image
# --------------------------------------------------------------------------------------


def calc_resultshape(
    title: str,
    shape: Literal[
        "rectangle", "circle", "ellipse", "segment", "marker", "point", "polygon"
    ],
    obj: ImageObj,
    func: Callable,
    *args: Any,
    add_label: bool = False,
) -> ResultShape | None:
    """Calculate result shape by executing a computation function on an image object,
    taking into account the image origin (x0, y0), scale (dx, dy) and ROIs.

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
                x0, y0, _x1, _y1 = ImageRoiDataItem(obj.roi[i_roi]).get_rect()
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
            array = np.full((num_rows, max_cols), np.nan)
            row = 0
            for coords in res:
                array[row : row + coords.shape[0], : coords.shape[1]] = coords
                row += coords.shape[0]
        else:
            array = np.vstack(res)
        return ResultShape(title, array, shape, add_label=add_label)
    return None


def get_centroid_coords(data: np.ndarray) -> np.ndarray:
    """Return centroid coordinates
    with :py:func:`cdl.algorithms.image.get_centroid_fourier`

    Args:
        data: input data

    Returns:
        Centroid coordinates
    """
    y, x = alg.get_centroid_fourier(data)
    return np.array([(x, y)])


def compute_centroid(image: ImageObj) -> ResultShape | None:
    """Compute centroid
    with :py:func:`cdl.algorithms.image.get_centroid_fourier`

    Args:
        image: input image

    Returns:
        Centroid coordinates
    """
    return calc_resultshape("centroid", "marker", image, get_centroid_coords)


def get_enclosing_circle_coords(data: np.ndarray) -> np.ndarray:
    """Return diameter coords for the circle contour enclosing image
    values above threshold (FWHM)

    Args:
        data: input data

    Returns:
        Diameter coords
    """
    x, y, r = alg.get_enclosing_circle(data)
    return np.array([[x, y, r]])


def compute_enclosing_circle(image: ImageObj) -> ResultShape | None:
    """Compute minimum enclosing circle
    with :py:func:`cdl.algorithms.image.get_enclosing_circle`

    Args:
        image: input image

    Returns:
        Diameter coords
    """
    return calc_resultshape(
        "enclosing_circle", "circle", image, get_enclosing_circle_coords
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
    with :py:func:`cdl.algorithms.image.get_hough_circle_peaks`

    Args:
        image: input image
        p: parameters

    Returns:
        Circle coordinates
    """
    return calc_resultshape(
        "hough_circle_peak",
        "circle",
        image,
        alg.get_hough_circle_peaks,
        p.min_radius,
        p.max_radius,
        None,
        p.min_distance,
    )


def compute_stats(obj: ImageObj) -> ResultProperties:
    """Compute statistics on an image

    Args:
        obj: input image object

    Returns:
        Result properties
    """
    statfuncs = {
        "min(z) = %g {.zunit}": ma.min,
        "max(z) = %g {.zunit}": ma.max,
        "<z> = %g {.zunit}": ma.mean,
        "median(z) = %g {.zunit}": ma.median,
        "σ(z) = %g {.zunit}": ma.std,
        "<z>/σ(z)": lambda z: ma.mean(z) / ma.std(z),
        "peak-to-peak(z) = %g {.zunit}": ma.ptp,
        "Σ(z) = %g {.zunit}": ma.sum,
    }
    return calc_resultproperties("stats", obj, statfuncs)
