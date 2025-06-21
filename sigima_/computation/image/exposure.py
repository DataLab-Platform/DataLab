# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Exposure computation module
---------------------------

This module provides tools for adjusting and analyzing image exposure and contrast.

Main features include:
- Histogram computation and equalization
- Contrast adjustment and normalization
- Logarithmic and gamma correction

Exposure processing improves the visual quality and interpretability of images,
especially under variable lighting conditions.
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
from skimage import exposure

import sigima_.algorithms.image as alg
from cdl.config import _
from sigima_ import computation_function
from sigima_.computation.base import (
    ClipParam,
    HistogramParam,
    NormalizeParam,
    dst_1_to_1,
    dst_2_to_1,
    new_signal_result,
)
from sigima_.computation.image.base import Wrap1to1Func, restore_data_outside_roi
from sigima_.obj.base import BaseProcParam
from sigima_.obj.image import ImageObj, ROI2DParam
from sigima_.obj.signal import SignalObj


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


@computation_function()
def adjust_gamma(src: ImageObj, p: AdjustGammaParam) -> ImageObj:
    """Gamma correction with :py:func:`skimage.exposure.adjust_gamma`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "adjust_gamma", f"gamma={p.gamma}, gain={p.gain}")
    dst.data = exposure.adjust_gamma(src.data, gamma=p.gamma, gain=p.gain)
    restore_data_outside_roi(dst, src)
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


@computation_function()
def adjust_log(src: ImageObj, p: AdjustLogParam) -> ImageObj:
    """Compute log correction with :py:func:`skimage.exposure.adjust_log`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "adjust_log", f"gain={p.gain}, inv={p.inv}")
    dst.data = exposure.adjust_log(src.data, gain=p.gain, inv=p.inv)
    restore_data_outside_roi(dst, src)
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


@computation_function()
def adjust_sigmoid(src: ImageObj, p: AdjustSigmoidParam) -> ImageObj:
    """Compute sigmoid correction with :py:func:`skimage.exposure.adjust_sigmoid`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(
        src, "adjust_sigmoid", f"cutoff={p.cutoff}, gain={p.gain}, inv={p.inv}"
    )
    dst.data = exposure.adjust_sigmoid(
        src.data, cutoff=p.cutoff, gain=p.gain, inv=p.inv
    )
    restore_data_outside_roi(dst, src)
    return dst


class RescaleIntensityParam(gds.DataSet):
    """Intensity rescaling parameters"""

    _dtype_list = ["image", "dtype"] + ImageObj.get_valid_dtypenames()
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


@computation_function()
def rescale_intensity(src: ImageObj, p: RescaleIntensityParam) -> ImageObj:
    """Rescale image intensity levels
    with :py:func:`skimage.exposure.rescale_intensity`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(
        src,
        "rescale_intensity",
        f"in_range={p.in_range}, out_range={p.out_range}",
    )
    dst.data = exposure.rescale_intensity(
        src.data, in_range=p.in_range, out_range=p.out_range
    )
    restore_data_outside_roi(dst, src)
    return dst


class EqualizeHistParam(gds.DataSet):
    """Histogram equalization parameters"""

    nbins = gds.IntItem(
        _("Number of bins"),
        min=1,
        default=256,
        help=_("Number of bins for image histogram."),
    )


@computation_function()
def equalize_hist(src: ImageObj, p: EqualizeHistParam) -> ImageObj:
    """Histogram equalization with :py:func:`skimage.exposure.equalize_hist`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "equalize_hist", f"nbins={p.nbins}")
    dst.data = exposure.equalize_hist(src.data, nbins=p.nbins)
    restore_data_outside_roi(dst, src)
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


@computation_function()
def equalize_adapthist(src: ImageObj, p: EqualizeAdaptHistParam) -> ImageObj:
    """Adaptive histogram equalization
    with :py:func:`skimage.exposure.equalize_adapthist`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(
        src, "equalize_adapthist", f"nbins={p.nbins}, clip_limit={p.clip_limit}"
    )
    dst.data = exposure.equalize_adapthist(
        src.data, clip_limit=p.clip_limit, nbins=p.nbins
    )
    restore_data_outside_roi(dst, src)
    return dst


class FlatFieldParam(BaseProcParam):
    """Flat-field parameters"""

    threshold = gds.FloatItem(_("Threshold"), default=0.0)

    def update_from_obj(self, obj: ImageObj) -> None:
        """Update parameters from image"""
        self.set_from_datatype(obj.data.dtype)


@computation_function()
def flatfield(src1: ImageObj, src2: ImageObj, p: FlatFieldParam) -> ImageObj:
    """Compute flat field correction with :py:func:`sigima_.algorithms.image.flatfield`

    Args:
        src1: raw data image object
        src2: flat field image object
        p: flat field parameters

    Returns:
        Output image object
    """
    dst = dst_2_to_1(src1, src2, "flatfield", f"threshold={p.threshold}")
    dst.data = alg.flatfield(src1.data, src2.data, p.threshold)
    restore_data_outside_roi(dst, src1)
    return dst


# MARK: compute_1_to_1 functions -------------------------------------------------------
# Functions with 1 input image and 1 output image
# --------------------------------------------------------------------------------------


@computation_function()
def normalize(src: ImageObj, p: NormalizeParam) -> ImageObj:
    """
    Normalize image data depending on its maximum,
    with :py:func:`sigima_.algorithms.image.normalize`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "normalize", suffix=f"ref={p.method}")
    dst.data = alg.normalize(src.data, p.method)  # type: ignore
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def histogram(src: ImageObj, p: HistogramParam) -> SignalObj:
    """Compute histogram of the image data, with :py:func:`numpy.histogram`

    Args:
        src: input image object
        p: parameters

    Returns:
        Signal object with the histogram
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
    dst.set_metadata_option("shade", 0.5)
    return dst


class ZCalibrateParam(gds.DataSet):
    """Image linear calibration parameters"""

    a = gds.FloatItem("a", default=1.0)
    b = gds.FloatItem("b", default=0.0)


@computation_function()
def calibration(src: ImageObj, p: ZCalibrateParam) -> ImageObj:
    """Compute linear calibration

    Args:
        src: input image object
        p: calibration parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "calibration", f"z={p.a}*z+{p.b}")
    dst.data = p.a * src.data + p.b
    restore_data_outside_roi(dst, src)
    return dst


@computation_function()
def clip(src: ImageObj, p: ClipParam) -> ImageObj:
    """Apply clipping with :py:func:`numpy.clip`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    return Wrap1to1Func(np.clip, a_min=p.lower, a_max=p.upper)(src)


@computation_function()
def offset_correction(src: ImageObj, p: ROI2DParam) -> ImageObj:
    """Apply offset correction

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "offset_correction", p.get_suffix())
    dst.data = src.data - np.nanmean(p.get_data(src))
    restore_data_outside_roi(dst, src)
    return dst
