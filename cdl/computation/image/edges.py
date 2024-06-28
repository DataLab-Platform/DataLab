# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Edges computation module
------------------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# All dataset classes must also be imported in the cdl.computation.param module.

from __future__ import annotations

import guidata.dataset as gds
import skimage
from skimage import feature, filters

from cdl.computation.image import Wrap11Func, dst_11
from cdl.config import _
from cdl.obj import ImageObj


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
    modes = ("reflect", "constant", "nearest", "mirror", "wrap")
    mode = gds.ChoiceItem(_("Mode"), list(zip(modes, modes)), default="constant")
    cval = gds.FloatItem(
        "cval",
        default=0.0,
        help=_("Value to fill past edges of input if mode is constant."),
    )


def compute_canny(src: ImageObj, p: CannyParam) -> ImageObj:
    """Compute Canny filter

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(
        src,
        "canny",
        f"sigma={p.sigma}, low_threshold={p.low_threshold}, "
        f"high_threshold={p.high_threshold}, use_quantiles={p.use_quantiles}, "
        f"mode={p.mode}, cval={p.cval}",
    )
    dst.data = skimage.util.img_as_ubyte(
        feature.canny(
            src.data,
            sigma=p.sigma,
            low_threshold=p.low_threshold,
            high_threshold=p.high_threshold,
            use_quantiles=p.use_quantiles,
            mode=p.mode,
            cval=p.cval,
        )
    )
    return dst


def compute_roberts(src: ImageObj) -> ImageObj:
    """Compute Roberts filter

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(filters.roberts)(src)


def compute_prewitt(src: ImageObj) -> ImageObj:
    """Compute Prewitt filter

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(filters.prewitt)(src)


def compute_prewitt_h(src: ImageObj) -> ImageObj:
    """Compute horizontal Prewitt filter

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(filters.prewitt_h)(src)


def compute_prewitt_v(src: ImageObj) -> ImageObj:
    """Compute vertical Prewitt filter

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(filters.prewitt_v)(src)


def compute_sobel(src: ImageObj) -> ImageObj:
    """Compute Sobel filter

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(filters.sobel)(src)


def compute_sobel_h(src: ImageObj) -> ImageObj:
    """Compute horizontal Sobel filter

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(filters.sobel_h)(src)


def compute_sobel_v(src: ImageObj) -> ImageObj:
    """Compute vertical Sobel filter

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(filters.sobel_v)(src)


def compute_scharr(src: ImageObj) -> ImageObj:
    """Compute Scharr filter

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(filters.scharr)(src)


def compute_scharr_h(src: ImageObj) -> ImageObj:
    """Compute horizontal Scharr filter

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(filters.scharr_h)(src)


def compute_scharr_v(src: ImageObj) -> ImageObj:
    """Compute vertical Scharr filter

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(filters.scharr_v)(src)


def compute_farid(src: ImageObj) -> ImageObj:
    """Compute Farid filter

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(filters.farid)(src)


def compute_farid_h(src: ImageObj) -> ImageObj:
    """Compute horizontal Farid filter

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(filters.farid_h)(src)


def compute_farid_v(src: ImageObj) -> ImageObj:
    """Compute vertical Farid filter

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(filters.farid_v)(src)


def compute_laplace(src: ImageObj) -> ImageObj:
    """Compute Laplace filter

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return Wrap11Func(filters.laplace)(src)
