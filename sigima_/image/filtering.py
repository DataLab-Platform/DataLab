# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Filtering computation module
----------------------------

This module provides spatial and frequency-based filtering operations for images.

Main features include:
- Gaussian, median, moving average, Wiener, and Butterworth filters
- Noise reduction and image smoothing

Filtering functions are essential for enhancing image quality
and removing noise prior to further analysis.
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
import scipy.ndimage as spi
import scipy.signal as sps
from skimage import filters

from cdl.config import _
from sigima_ import computation_function
from sigima_.base import (
    GaussianParam,
    MovingAverageParam,
    MovingMedianParam,
    dst_1_to_1,
)
from sigima_.image.base import Wrap1to1Func, restore_data_outside_roi
from sigima_.obj.image import ImageObj


@computation_function()
def gaussian_filter(src: ImageObj, p: GaussianParam) -> ImageObj:
    """Compute gaussian filter with :py:func:`scipy.ndimage.gaussian_filter`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    return Wrap1to1Func(spi.gaussian_filter, sigma=p.sigma)(src)


@computation_function()
def moving_average(src: ImageObj, p: MovingAverageParam) -> ImageObj:
    """Compute moving average with :py:func:`scipy.ndimage.uniform_filter`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    return Wrap1to1Func(spi.uniform_filter, size=p.n, mode=p.mode)(src)


@computation_function()
def moving_median(src: ImageObj, p: MovingMedianParam) -> ImageObj:
    """Compute moving median with :py:func:`scipy.ndimage.median_filter`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    return Wrap1to1Func(spi.median_filter, size=p.n, mode=p.mode)(src)


@computation_function()
def wiener(src: ImageObj) -> ImageObj:
    """Compute Wiener filter with :py:func:`scipy.signal.wiener`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap1to1Func(sps.wiener)(src)


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


@computation_function()
def butterworth(src: ImageObj, p: ButterworthParam) -> ImageObj:
    """Compute Butterworth filter with :py:func:`skimage.filters.butterworth`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(
        src,
        "butterworth",
        f"cut_off={p.cut_off:.3f}, order={p.order}, high_pass={p.high_pass}",
    )
    dst.data = filters.butterworth(src.data, p.cut_off, p.high_pass, p.order)
    restore_data_outside_roi(dst, src)
    return dst
