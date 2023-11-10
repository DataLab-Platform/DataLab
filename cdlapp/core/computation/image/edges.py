# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
Edges computation module
------------------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# All dataset classes must also be imported in the cdlapp.core.computation.param module.

from __future__ import annotations

import guidata.dataset as gds
import numpy as np
from skimage import feature, filters

from cdlapp.config import _
from cdlapp.core.computation.image import dst_11
from cdlapp.core.model.image import ImageObj


class CannyParam(gds.DataSet):
    """Canny filter parameters"""

    sigma = gds.FloatItem(
        "Sigma",
        default=1.0,
        unit="pixels",
        min=0,
        nonzero=True,
        help=_("Standard deviation of the Gaussian filter."),
    )
    low_threshold = gds.FloatItem(
        _("Low threshold"),
        default=0.1,
        min=0,
        help=_("Lower bound for hysteresis thresholding (linking edges)."),
    )
    high_threshold = gds.FloatItem(
        _("High threshold"),
        default=0.9,
        min=0,
        help=_("Upper bound for hysteresis thresholding (linking edges)."),
    )
    use_quantiles = gds.BoolItem(
        _("Use quantiles"),
        default=True,
        help=_(
            "If True then treat low_threshold and high_threshold as quantiles "
            "of the edge magnitude image, rather than absolute edge magnitude "
            "values. If True then the thresholds must be in the range [0, 1]."
        ),
    )
    _modelist = ("reflect", "constant", "nearest", "mirror", "wrap")
    mode = gds.ChoiceItem(
        _("Mode"), list(zip(_modelist, _modelist)), default="constant"
    )
    cval = gds.FloatItem(
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
