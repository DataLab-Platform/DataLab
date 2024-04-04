# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Morphology computation module
-----------------------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# All dataset classes must also be imported in the cdl.core.computation.param module.

from __future__ import annotations

import guidata.dataset as gds
from skimage import morphology

from cdl.config import _
from cdl.core.computation.image import dst_11
from cdl.core.model.image import ImageObj


class MorphologyParam(gds.DataSet):
    """White Top-Hat parameters"""

    radius = gds.IntItem(
        _("Radius"), default=1, min=1, help=_("Footprint (disk) radius.")
    )


def compute_white_tophat(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute White Top-Hat

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "white_tophat", f"radius={p.radius}")
    dst.data = morphology.white_tophat(src.data, morphology.disk(p.radius))
    return dst


def compute_black_tophat(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute Black Top-Hat

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "black_tophat", f"radius={p.radius}")
    dst.data = morphology.black_tophat(src.data, morphology.disk(p.radius))
    return dst


def compute_erosion(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute Erosion

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "erosion", f"radius={p.radius}")
    dst.data = morphology.erosion(src.data, morphology.disk(p.radius))
    return dst


def compute_dilation(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute Dilation

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "dilation", f"radius={p.radius}")
    dst.data = morphology.dilation(src.data, morphology.disk(p.radius))
    return dst


def compute_opening(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute morphological opening

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "opening", f"radius={p.radius}")
    dst.data = morphology.opening(src.data, morphology.disk(p.radius))
    return dst


def compute_closing(src: ImageObj, p: MorphologyParam) -> ImageObj:
    """Compute morphological closing

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "closing", f"radius={p.radius}")
    dst.data = morphology.closing(src.data, morphology.disk(p.radius))
    return dst
