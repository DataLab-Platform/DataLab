# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Mathematical Operations Module
------------------------------

This module implements mathematical operations on images, such as inversion,
absolute value, real/imaginary part extraction, type casting, and exponentiation.

Main features include:
- Pixel-wise mathematical transformations (e.g., log, exp, abs, real, imag, log10)
- Type casting and other value-level operations

These functions enable flexible manipulation of image data at the value level.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# - All `guidata.dataset.DataSet` parameter classes must also be imported
#   in the `sigima_.param` module.
# - All functions decorated by `computation_function` must be imported in the upper
#   level `sigima_.computation.image` module.

from __future__ import annotations

import warnings

import guidata.dataset as gds
import numpy as np

from sigima_.algorithms.datatypes import clip_astype
from sigima_.computation import computation_function
from sigima_.computation.base import dst_1_to_1
from sigima_.computation.image.base import Wrap1to1Func, restore_data_outside_roi
from sigima_.config import _
from sigima_.obj.image import ImageObj


@computation_function()
def inverse(src: ImageObj) -> ImageObj:
    """Compute the inverse of an image and return the new result image object

    Args:
        src: input image object

    Returns:
        Result image object 1 / **src** (new object)
    """
    dst = dst_1_to_1(src, "inverse")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dst.data = np.reciprocal(src.data, dtype=float)
        dst.data[np.isinf(dst.data)] = np.nan
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def absolute(src: ImageObj) -> ImageObj:
    """Compute absolute value with :py:data:`numpy.absolute`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap1to1Func(np.absolute)(src)


@computation_function()
def real(src: ImageObj) -> ImageObj:
    """Compute real part with :py:func:`numpy.real`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap1to1Func(np.real)(src)


@computation_function()
def imag(src: ImageObj) -> ImageObj:
    """Compute imaginary part with :py:func:`numpy.imag`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap1to1Func(np.imag)(src)


@computation_function()
def log10(src: ImageObj) -> ImageObj:
    """Compute log10 with :py:data:`numpy.log10`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap1to1Func(np.log10)(src)


@computation_function()
def exp(src: ImageObj) -> ImageObj:
    """Compute exponential with :py:data:`numpy.exp`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap1to1Func(np.exp)(src)


class LogP1Param(gds.DataSet):
    """Log10 parameters"""

    n = gds.FloatItem("n")


@computation_function()
def logp1(src: ImageObj, p: LogP1Param) -> ImageObj:
    """Compute log10(z+n) with :py:data:`numpy.log10`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "log_z_plus_n", f"n={p.n}")
    dst.data = np.log10(src.data + p.n)
    restore_data_outside_roi(dst, src)
    return dst


class DataTypeIParam(gds.DataSet):
    """Convert image data type parameters"""

    dtype_str = gds.ChoiceItem(
        _("Destination data type"),
        list(zip(ImageObj.get_valid_dtypenames(), ImageObj.get_valid_dtypenames())),
        help=_("Output image data type."),
    )


@computation_function()
def astype(src: ImageObj, p: DataTypeIParam) -> ImageObj:
    """Convert image data type with :py:func:`sigima_.algorithms.datatypes.clip_astype`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "clip_astype", p.dtype_str)
    dst.data = clip_astype(src.data, p.dtype_str)
    return dst
