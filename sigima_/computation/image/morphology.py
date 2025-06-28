# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Morphology computation module
-----------------------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# - All `guidata.dataset.DataSet` parameter classes must also be imported
#   in the `sigima_.param` module.
# - All functions decorated by `computation_function` must be imported in the upper
#   level `sigima_.computation.image` module.

from __future__ import annotations

import guidata.dataset as gds
from skimage import morphology

from sigima_.computation import computation_function
from sigima_.computation.base import dst_1_to_1
from sigima_.computation.image.base import restore_data_outside_roi
from sigima_.config import _
from sigima_.obj.image import ImageObj


class MorphologyParam(gds.DataSet):
    """White Top-Hat parameters"""

    radius = gds.IntItem(
        _("Radius"), default=1, min=1, help=_("Footprint (disk) radius.")
    )


@computation_function()
def white_tophat(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute White Top-Hat with :py:func:`skimage.morphology.white_tophat`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "white_tophat", f"radius={p.radius}")
    dst.data = morphology.white_tophat(src.data, morphology.disk(p.radius))
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def black_tophat(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute Black Top-Hat with :py:func:`skimage.morphology.black_tophat`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "black_tophat", f"radius={p.radius}")
    dst.data = morphology.black_tophat(src.data, morphology.disk(p.radius))
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def erosion(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute Erosion with :py:func:`skimage.morphology.erosion`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "erosion", f"radius={p.radius}")
    dst.data = morphology.erosion(src.data, morphology.disk(p.radius))
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def dilation(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute Dilation with :py:func:`skimage.morphology.dilation`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "dilation", f"radius={p.radius}")
    dst.data = morphology.dilation(src.data, morphology.disk(p.radius))
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def opening(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute morphological opening with :py:func:`skimage.morphology.opening`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "opening", f"radius={p.radius}")
    dst.data = morphology.opening(src.data, morphology.disk(p.radius))
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def closing(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute morphological closing with :py:func:`skimage.morphology.closing`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "closing", f"radius={p.radius}")
    dst.data = morphology.closing(src.data, morphology.disk(p.radius))
    restore_data_outside_roi(dst, src)
    return dst
