# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Exposure computation module
---------------------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# All dataset classes must also be imported in the cdl.core.computation.param module.

from __future__ import annotations

import guidata.dataset as gds
from skimage import exposure

from cdl.config import _
from cdl.core.computation.image import VALID_DTYPES_STRLIST, dst_11
from cdl.core.model.image import ImageObj


class AdjustGammaParam(gds.DataSet):
    """Gamma adjustment parameters"""

    gamma = gds.FloatItem(
        _("Gamma"),
        default=1.0,
        min=0.0,
        help=_("Gamma correction factor (higher values give more contrast)."),
    )
    gain = gds.FloatItem(
        _("Gain"),
        default=1.0,
        min=0.0,
        help=_("Gain factor (higher values give more contrast)."),
    )


def compute_adjust_gamma(src: ImageObj, p: AdjustGammaParam) -> ImageObj:
    """Gamma correction

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "adjust_gamma", f"gamma={p.gamma}, gain={p.gain}")
    dst.data = exposure.adjust_gamma(src.data, gamma=p.gamma, gain=p.gain)
    return dst


class AdjustLogParam(gds.DataSet):
    """Logarithmic adjustment parameters"""

    gain = gds.FloatItem(
        _("Gain"),
        default=1.0,
        min=0.0,
        help=_("Gain factor (higher values give more contrast)."),
    )
    inv = gds.BoolItem(
        _("Inverse"),
        default=False,
        help=_("If True, apply inverse logarithmic transformation."),
    )


def compute_adjust_log(src: ImageObj, p: AdjustLogParam) -> ImageObj:
    """Compute log correction

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "adjust_log", f"gain={p.gain}, inv={p.inv}")
    dst.data = exposure.adjust_log(src.data, gain=p.gain, inv=p.inv)
    return dst


class AdjustSigmoidParam(gds.DataSet):
    """Sigmoid adjustment parameters"""

    cutoff = gds.FloatItem(
        _("Cutoff"),
        default=0.5,
        min=0.0,
        max=1.0,
        help=_("Cutoff value (higher values give more contrast)."),
    )
    gain = gds.FloatItem(
        _("Gain"),
        default=10.0,
        min=0.0,
        help=_("Gain factor (higher values give more contrast)."),
    )
    inv = gds.BoolItem(
        _("Inverse"),
        default=False,
        help=_("If True, apply inverse sigmoid transformation."),
    )


def compute_adjust_sigmoid(src: ImageObj, p: AdjustSigmoidParam) -> ImageObj:
    """Compute sigmoid correction

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(
        src, "adjust_sigmoid", f"cutoff={p.cutoff}, gain={p.gain}, inv={p.inv}"
    )
    dst.data = exposure.adjust_sigmoid(
        src.data, cutoff=p.cutoff, gain=p.gain, inv=p.inv
    )
    return dst


class RescaleIntensityParam(gds.DataSet):
    """Intensity rescaling parameters"""

    _dtype_list = ["image", "dtype"] + VALID_DTYPES_STRLIST
    in_range = gds.ChoiceItem(
        _("Input range"),
        list(zip(_dtype_list, _dtype_list)),
        default="image",
        help=_(
            "Min and max intensity values of input image ('image' refers to input "
            "image min/max levels, 'dtype' refers to input image data type range)."
        ),
    )
    out_range = gds.ChoiceItem(
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
        src: input image object
        p: parameters

    Returns:
        Output image object
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


class EqualizeHistParam(gds.DataSet):
    """Histogram equalization parameters"""

    nbins = gds.IntItem(
        _("Number of bins"),
        min=1,
        default=256,
        help=_("Number of bins for image histogram."),
    )


def compute_equalize_hist(src: ImageObj, p: EqualizeHistParam) -> ImageObj:
    """Histogram equalization

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(src, "equalize_hist", f"nbins={p.nbins}")
    dst.data = exposure.equalize_hist(src.data, nbins=p.nbins)
    return dst


class EqualizeAdaptHistParam(EqualizeHistParam):
    """Adaptive histogram equalization parameters"""

    clip_limit = gds.FloatItem(
        _("Clipping limit"),
        default=0.01,
        min=0.0,
        max=1.0,
        help=_("Clipping limit (higher values give more contrast)."),
    )


def compute_equalize_adapthist(src: ImageObj, p: EqualizeAdaptHistParam) -> ImageObj:
    """Adaptive histogram equalization

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_11(
        src, "equalize_adapthist", f"nbins={p.nbins}, clip_limit={p.clip_limit}"
    )
    dst.data = exposure.equalize_adapthist(
        src.data, clip_limit=p.clip_limit, nbins=p.nbins
    )
    return dst
