# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Geometry computation module
---------------------------

This module implements geometric transformations and manipulations for images,
such as rotations, flips, resizing, axis swapping, binning, and padding.

Main features include:
- Rotation by arbitrary or fixed angles
- Horizontal and vertical flipping
- Resizing and binning of images
- Axis swapping and zero padding

These functions are useful for preparing and augmenting image data.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# - All `guidata.dataset.DataSet` parameter classes must also be imported
#   in the `sigima_.param` module.
# - All functions decorated by `computation_function` must be imported in the upper
#   level `sigima_.image` module.

from __future__ import annotations

import guidata.dataset as gds
import numpy as np
import scipy.ndimage as spi
from plotpy.mathutils.geometry import vector_rotation

import sigima_.algorithms.image as alg
from cdl.config import _
from sigima_ import computation_function
from sigima_.base import dst_1_to_1
from sigima_.image.base import Wrap1to1Func
from sigima_.model.image import ImageObj


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


@computation_function()
def rotate(src: ImageObj, p: RotateParam) -> ImageObj:
    """Rotate data with :py:func:`scipy.ndimage.rotate`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "rotate", f"α={p.angle:.3f}°, mode='{p.mode}'")
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


@computation_function()
def rotate90(src: ImageObj) -> ImageObj:
    """Rotate data 90° with :py:func:`numpy.rot90`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "rotate90")
    dst.data = np.rot90(src.data)
    dst.transform_shapes(src, rotate_obj_90)
    return dst


def rotate_obj_270(dst: ImageObj, src: ImageObj, coords: np.ndarray) -> None:
    """Apply rotation to coords associated to image obj"""
    rotate_obj_coords(270.0, dst, src, coords)


@computation_function()
def rotate270(src: ImageObj) -> ImageObj:
    """Rotate data 270° with :py:func:`numpy.rot90`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "rotate270")
    dst.data = np.rot90(src.data, 3)
    dst.transform_shapes(src, rotate_obj_270)
    return dst


def hflip_coords(dst: ImageObj, src: ImageObj, coords: np.ndarray) -> None:
    """Apply HFlip to coords"""
    coords[:, ::2] = dst.x0 + dst.width - coords[:, ::2]
    dst.roi = None


@computation_function()
def fliph(src: ImageObj) -> ImageObj:
    """Flip data horizontally with :py:func:`numpy.fliplr`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = Wrap1to1Func(np.fliplr)(src)
    dst.transform_shapes(src, hflip_coords)
    return dst


def vflip_coords(dst: ImageObj, src: ImageObj, coords: np.ndarray) -> None:
    """Apply VFlip to coords"""
    coords[:, 1::2] = dst.y0 + dst.height - coords[:, 1::2]
    dst.roi = None


@computation_function()
def flipv(src: ImageObj) -> ImageObj:
    """Flip data vertically with :py:func:`numpy.flipud`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = Wrap1to1Func(np.flipud)(src)
    dst.transform_shapes(src, vflip_coords)
    return dst


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


@computation_function()
def resize(src: ImageObj, p: ResizeParam) -> ImageObj:
    """Zooming function with :py:func:`scipy.ndimage.zoom`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "resize", f"zoom={p.zoom:.3f}")
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
    dtypes = ["dtype"] + ImageObj.get_valid_dtypenames()
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


@computation_function()
def binning(src: ImageObj, param: BinningParam) -> ImageObj:
    """Binning function on data with :py:func:`sigima_.algorithms.image.binning`

    Args:
        src: input image object
        param: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(
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


@computation_function()
def swap_axes(src: ImageObj) -> ImageObj:
    """Swap image axes with :py:func:`numpy.transpose`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = Wrap1to1Func(np.transpose)(src)
    # TODO: [P2] Instead of removing geometric shapes, apply swap
    dst.remove_all_shapes()
    return dst
