# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Image Computation module
--------------------------------

This module defines the image parameters and functions used by the
:mod:`cdl.core.gui.processor` module.

It is based on the :mod:`cdl.algorithms` module, which defines the algorithms
that are applied to the data, and on the :mod:`cdl.core.model` module, which
defines the data model.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import guidata.dataset.dataitems as gdi
import guidata.dataset.datatypes as gdt
import numpy as np
import pywt
import scipy.ndimage as spi
import scipy.signal as sps
from guiqwt.geometry import vector_rotation
from skimage import exposure, feature, filters, morphology
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, denoise_wavelet
from skimage.util.dtype import dtype_range

from cdl.algorithms.image import (
    BINNING_OPERATIONS,
    binning,
    find_blobs_dog,
    find_blobs_doh,
    find_blobs_log,
    find_blobs_opencv,
    flatfield,
    get_2d_peaks_coords,
    get_centroid_fourier,
    get_contour_shapes,
    get_enclosing_circle,
    get_hough_circle_peaks,
)
from cdl.config import _
from cdl.core.computation.base import (
    ClipParam,
    GaussianParam,
    MovingAverageParam,
    MovingMedianParam,
    ThresholdParam,
)
from cdl.core.model.base import BaseProcParam
from cdl.core.model.image import ImageObj, RoiDataItem

VALID_DTYPES_STRLIST = [
    dtype.__name__ for dtype in dtype_range if dtype in ImageObj.VALID_DTYPES
]


def compute_gaussian_filter(data: np.ndarray, p: GaussianParam) -> np.ndarray:
    """Compute gaussian filter
    Args:
        data (np.ndarray): input data
        p (GaussianParam): parameters
    Returns:
        np.ndarray: output data
    """
    return spi.gaussian_filter(data, sigma=p.sigma)


def compute_moving_average(data: np.ndarray, p: MovingAverageParam) -> np.ndarray:
    """Compute moving average
    Args:
        data (np.ndarray): input data
        p (MovingAverageParam): parameters
    Returns:
        np.ndarray: output data
    """
    return spi.uniform_filter(data, size=p.n, mode="constant")


def compute_moving_median(data: np.ndarray, p: MovingMedianParam) -> np.ndarray:
    """Compute moving median
    Args:
        data (np.ndarray): input data
        p (MovingMedianParam): parameters
    Returns:
        np.ndarray: output data
    """
    return sps.medfilt(data, kernel_size=p.n)


def compute_threshold(data: np.ndarray, p: ThresholdParam) -> np.ndarray:
    """Apply thresholding
    Args:
        data (np.ndarray): data
        p (ThresholdParam): parameters
    Returns:
        np.ndarray: thresholded data
    """
    return np.clip(data, p.value, data.max())


def compute_clip(data: np.ndarray, p: ClipParam) -> np.ndarray:
    """Apply clipping
    Args:
        data (np.ndarray): data
        p (ClipParam): parameters
    Returns:
        np.ndarray: clipped data"""
    return np.clip(data, data.min(), p.value)


class AdjustGammaParam(gdt.DataSet):
    """Gamma adjustment parameters"""

    gamma = gdi.FloatItem(
        _("Gamma"),
        default=1.0,
        min=0.0,
        help=_("Gamma correction factor (higher values give more contrast)."),
    )
    gain = gdi.FloatItem(
        _("Gain"),
        default=1.0,
        min=0.0,
        help=_("Gain factor (higher values give more contrast)."),
    )


def compute_adjust_gamma(data: np.ndarray, p: AdjustGammaParam) -> np.ndarray:
    """Gamma correction
    Args:
        data (np.ndarray): input data
        p (AdjustGammaParam): parameters
    Returns:
        np.ndarray: output data
    """
    return exposure.adjust_gamma(data, gamma=p.gamma, gain=p.gain)


class AdjustLogParam(gdt.DataSet):
    """Logarithmic adjustment parameters"""

    gain = gdi.FloatItem(
        _("Gain"),
        default=1.0,
        min=0.0,
        help=_("Gain factor (higher values give more contrast)."),
    )
    inv = gdi.BoolItem(
        _("Inverse"),
        default=False,
        help=_("If True, apply inverse logarithmic transformation."),
    )


def compute_adjust_log(data: np.ndarray, p: AdjustLogParam) -> np.ndarray:
    """Compute log correction
    Args:
        data (np.ndarray): input data
        p (AdjustLogParam): parameters
    Returns:
        np.ndarray: output data
    """
    return exposure.adjust_log(data, gain=p.gain, inv=p.inv)


class AdjustSigmoidParam(gdt.DataSet):
    """Sigmoid adjustment parameters"""

    cutoff = gdi.FloatItem(
        _("Cutoff"),
        default=0.5,
        min=0.0,
        max=1.0,
        help=_("Cutoff value (higher values give more contrast)."),
    )
    gain = gdi.FloatItem(
        _("Gain"),
        default=10.0,
        min=0.0,
        help=_("Gain factor (higher values give more contrast)."),
    )
    inv = gdi.BoolItem(
        _("Inverse"),
        default=False,
        help=_("If True, apply inverse sigmoid transformation."),
    )


def compute_adjust_sigmoid(data: np.ndarray, p: AdjustSigmoidParam) -> np.ndarray:
    """Compute sigmoid correction
    Args:
        data (np.ndarray): input data
        p (AdjustSigmoidParam): parameters
    Returns:
        np.ndarray: output data
    """
    return exposure.adjust_sigmoid(data, cutoff=p.cutoff, gain=p.gain, inv=p.inv)


class RescaleIntensityParam(gdt.DataSet):
    """Intensity rescaling parameters"""

    _dtype_list = ["image", "dtype"] + VALID_DTYPES_STRLIST
    in_range = gdi.ChoiceItem(
        _("Input range"),
        list(zip(_dtype_list, _dtype_list)),
        default="image",
        help=_(
            "Min and max intensity values of input image ('image' refers to input "
            "image min/max levels, 'dtype' refers to input image data type range)."
        ),
    )
    out_range = gdi.ChoiceItem(
        _("Output range"),
        list(zip(_dtype_list, _dtype_list)),
        default="dtype",
        help=_(
            "Min and max intensity values of output image  ('image' refers to input "
            "image min/max levels, 'dtype' refers to input image data type range).."
        ),
    )


def compute_rescale_intensity(data: np.ndarray, p: RescaleIntensityParam) -> np.ndarray:
    """Rescale image intensity levels
    Args:
        data (np.ndarray): input data
        p (RescaleIntensityParam): parameters
    Returns:
        np.ndarray: output data
    """
    return exposure.rescale_intensity(data, in_range=p.in_range, out_range=p.out_range)


class EqualizeHistParam(gdt.DataSet):
    """Histogram equalization parameters"""

    nbins = gdi.IntItem(
        _("Number of bins"),
        min=1,
        default=256,
        help=_("Number of bins for image histogram."),
    )


def compute_equalize_hist(data: np.ndarray, p: EqualizeHistParam) -> np.ndarray:
    """Histogram equalization
    Args:
        data (np.ndarray): input data
        p (EqualizeHistParam): parameters
    Returns:
        np.ndarray: output data
    """
    return exposure.equalize_hist(data, nbins=p.nbins)


class EqualizeAdaptHistParam(EqualizeHistParam):
    """Adaptive histogram equalization parameters"""

    clip_limit = gdi.FloatItem(
        _("Clipping limit"),
        default=0.01,
        min=0.0,
        max=1.0,
        help=_("Clipping limit (higher values give more contrast)."),
    )


def compute_equalize_adapthist(
    data: np.ndarray, p: EqualizeAdaptHistParam
) -> np.ndarray:
    """Adaptive histogram equalization
    Args:
        data (np.ndarray): input data
        p (EqualizeAdaptHistParam): parameters
    Returns:
        np.ndarray: output data
    """
    return exposure.equalize_adapthist(data, clip_limit=p.clip_limit, nbins=p.nbins)


class LogP1Param(gdt.DataSet):
    """Log10 parameters"""

    n = gdi.FloatItem("n")


def log_z_plus_n(data: np.ndarray, p: LogP1Param) -> np.ndarray:
    """Compute log10(z+n)
    Args:
        data (np.ndarray): input data
        p (LogP1Param): parameters
    Returns:
        np.ndarray: output data
    """
    return np.log10(data + p.n)


class RotateParam(gdt.DataSet):
    """Rotate parameters"""

    boundaries = ("constant", "nearest", "reflect", "wrap")
    prop = gdt.ValueProp(False)

    angle = gdi.FloatItem(f"{_('Angle')} (°)")
    mode = gdi.ChoiceItem(
        _("Mode"), list(zip(boundaries, boundaries)), default=boundaries[0]
    )
    cval = gdi.FloatItem(
        _("cval"),
        default=0.0,
        help=_(
            "Value used for points outside the "
            "boundaries of the input if mode is "
            "'constant'"
        ),
    )
    reshape = gdi.BoolItem(
        _("Reshape the output array"),
        default=False,
        help=_(
            "Reshape the output array "
            "so that the input array is "
            "contained completely in the output"
        ),
    )
    prefilter = gdi.BoolItem(_("Prefilter the input image"), default=True).set_prop(
        "display", store=prop
    )
    order = gdi.IntItem(
        _("Order"),
        default=3,
        min=0,
        max=5,
        help=_("Spline interpolation order"),
    ).set_prop("display", active=prop)


def compute_rotate(data: np.ndarray, p: RotateParam) -> np.ndarray:
    """Rotate data
    Args:
        data (np.ndarray): input data
        p (RotateParam): parameters
    Returns:
        np.ndarray: output data
    """
    return spi.rotate(
        data,
        p.angle,
        reshape=p.reshape,
        order=p.order,
        mode=p.mode,
        cval=p.cval,
        prefilter=p.prefilter,
    )


def rotate_obj_coords(
    angle: float, obj: ImageObj, orig: ImageObj, coords: np.ndarray
) -> None:
    """Apply rotation to coords associated to image obj
    Args:
        angle (float): rotation angle (in degrees)
        obj (ImageObj): image object
        orig (ImageObj): original image object
        coords (np.ndarray): coordinates to rotate
    Returns:
        np.ndarray: output data
    """
    for row in range(coords.shape[0]):
        for col in range(0, coords.shape[1], 2):
            x1, y1 = coords[row, col : col + 2]
            dx1 = x1 - orig.xc
            dy1 = y1 - orig.yc
            dx2, dy2 = vector_rotation(-angle * np.pi / 180.0, dx1, dy1)
            coords[row, col : col + 2] = dx2 + obj.xc, dy2 + obj.yc
    obj.roi = None


def rotate270(data: np.ndarray) -> np.ndarray:
    """Rotate data 270°
    Args:
        data (np.ndarray): input data
    Returns:
        np.ndarray: output data
    """
    return np.rot90(data, 3)


class GridParam(gdt.DataSet):
    """Grid parameters"""

    _prop = gdt.GetAttrProp("direction")
    _directions = (("col", _("columns")), ("row", _("rows")))
    direction = gdi.ChoiceItem(_("Distribute over"), _directions, radio=True).set_prop(
        "display", store=_prop
    )
    cols = gdi.IntItem(_("Columns"), default=1, nonzero=True).set_prop(
        "display", active=gdt.FuncProp(_prop, lambda x: x == "col")
    )
    rows = gdi.IntItem(_("Rows"), default=1, nonzero=True).set_prop(
        "display", active=gdt.FuncProp(_prop, lambda x: x == "row")
    )
    colspac = gdi.FloatItem(_("Column spacing"), default=0.0, min=0.0)
    rowspac = gdi.FloatItem(_("Row spacing"), default=0.0, min=0.0)


class ResizeParam(gdt.DataSet):
    """Resize parameters"""

    boundaries = ("constant", "nearest", "reflect", "wrap")
    prop = gdt.ValueProp(False)

    zoom = gdi.FloatItem(_("Zoom"))
    mode = gdi.ChoiceItem(
        _("Mode"), list(zip(boundaries, boundaries)), default=boundaries[0]
    )
    cval = gdi.FloatItem(
        _("cval"),
        default=0.0,
        help=_(
            "Value used for points outside the "
            "boundaries of the input if mode is "
            "'constant'"
        ),
    )
    prefilter = gdi.BoolItem(_("Prefilter the input image"), default=True).set_prop(
        "display", store=prop
    )
    order = gdi.IntItem(
        _("Order"),
        default=3,
        min=0,
        max=5,
        help=_("Spline interpolation order"),
    ).set_prop("display", active=prop)


def compute_resize(data: np.ndarray, p: ResizeParam) -> np.ndarray:
    """Zooming function
    Args:
        data (np.ndarray): input data
        p (ResizeParam): parameters
    Returns:
        np.ndarray: output data
    """
    return spi.interpolation.zoom(
        data,
        p.zoom,
        order=p.order,
        mode=p.mode,
        cval=p.cval,
        prefilter=p.prefilter,
    )


class BinningParam(gdt.DataSet):
    """Binning parameters"""

    binning_x = gdi.IntItem(
        _("Cluster size (X)"),
        default=2,
        min=2,
        help=_("Number of adjacent pixels to be combined together along X-axis."),
    )
    binning_y = gdi.IntItem(
        _("Cluster size (Y)"),
        default=2,
        min=2,
        help=_("Number of adjacent pixels to be combined together along Y-axis."),
    )
    _operations = BINNING_OPERATIONS
    operation = gdi.ChoiceItem(
        _("Operation"),
        list(zip(_operations, _operations)),
        default=_operations[0],
    )
    _dtype_list = ["dtype"] + VALID_DTYPES_STRLIST
    dtype_str = gdi.ChoiceItem(
        _("Data type"),
        list(zip(_dtype_list, _dtype_list)),
        help=_("Output image data type."),
    )
    change_pixel_size = gdi.BoolItem(
        _("Change pixel size"),
        default=False,
        help=_("Change pixel size so that overall image size remains the same."),
    )


def compute_binning(data: np.ndarray, param: BinningParam) -> np.ndarray:
    """Binning function on data
    Args:
        data (np.ndarray): input data
        param (BinningParam): parameters
    Returns:
        np.ndarray: output data
    """
    return binning(
        data,
        binning_x=param.binning_x,
        binning_y=param.binning_y,
        operation=param.operation,
        dtype=param.dtype_str,
    )


def extract_multiple_roi(data: np.ndarray, group: gdt.DataSetGroup) -> np.ndarray:
    """Extract multiple regions of interest from data
    Args:
        data (np.ndarray): input data
        group (gdt.DataSetGroup): parameters defining the regions of interest
    Returns:
        np.ndarray: output data
    """
    if len(group.datasets) == 1:
        p = group.datasets[0]
        return data.copy()[p.y0 : p.y1, p.x0 : p.x1]
    out = np.zeros_like(data)
    for p in group.datasets:
        slice1, slice2 = slice(p.y0, p.y1 + 1), slice(p.x0, p.x1 + 1)
        out[slice1, slice2] = data[slice1, slice2]
    x0 = min([p.x0 for p in group.datasets])
    y0 = min([p.y0 for p in group.datasets])
    x1 = max([p.x1 for p in group.datasets])
    y1 = max([p.y1 for p in group.datasets])
    return out[y0:y1, x0:x1]


def extract_single_roi(data: np.ndarray, p: gdt.DataSet) -> np.ndarray:
    """Extract single ROI
    Args:
        data (np.ndarray): data
        p (gdt.DataSet): ROI parameters
    Returns:
        np.ndarray: ROI data
    """
    return data.copy()[p.y0 : p.y1, p.x0 : p.x1]


class FlatFieldParam(BaseProcParam):
    """Flat-field parameters"""

    threshold = gdi.FloatItem(_("Threshold"), default=0.0)


def compute_flatfield(raw: np.ndarray, flat: np.ndarray, p: FlatFieldParam):
    """Compute flat field correction
    Args:
        raw (np.ndarray): raw data
        flat (np.ndarray): flat field data
        p (FlatFieldParam): flat field parameters
    Returns:
        np.ndarray: corrected data
    """
    return flatfield(raw, flat, p.threshold)


class ZCalibrateParam(gdt.DataSet):
    """Image linear calibration parameters"""

    a = gdi.FloatItem("a", default=1.0)
    b = gdi.FloatItem("b", default=0.0)


def compute_calibration(data: np.ndarray, param: ZCalibrateParam) -> np.ndarray:
    """Compute linear calibration
    Args:
        data (np.ndarray): data to calibrate
        param (ZCalibrateParam): calibration parameters
    Returns:
        np.ndarray: calibrated data
    """
    return param.a * data + param.b


class DenoiseTVParam(gdt.DataSet):
    """Total Variation denoising parameters"""

    weight = gdi.FloatItem(
        _("Denoising weight"),
        default=0.1,
        min=0,
        nonzero=True,
        help=_(
            "The greater weight, the more denoising "
            "(at the expense of fidelity to input)."
        ),
    )
    eps = gdi.FloatItem(
        "Epsilon",
        default=0.0002,
        min=0,
        nonzero=True,
        help=_(
            "Relative difference of the value of the cost function that "
            "determines the stop criterion. The algorithm stops when: "
            "(E_(n-1) - E_n) < eps * E_0"
        ),
    )
    max_num_iter = gdi.IntItem(
        _("Max. iterations"),
        default=200,
        min=0,
        nonzero=True,
        help=_("Maximal number of iterations used for the optimization"),
    )


def compute_denoise_tv(data: np.ndarray, p: DenoiseTVParam) -> np.ndarray:
    """Compute Total Variation denoising
    Args:
        data (np.ndarray): input data
        p (DenoiseTVParam): parameters
    Returns:
        np.ndarray: output data
    """
    return denoise_tv_chambolle(
        data, weight=p.weight, eps=p.eps, max_num_iter=p.max_num_iter
    )


class DenoiseBilateralParam(gdt.DataSet):
    """Bilateral filter denoising parameters"""

    sigma_spatial = gdi.FloatItem(
        "σ<sub>spatial</sub>",
        default=1.0,
        min=0,
        nonzero=True,
        unit="pixels",
        help=_(
            "Standard deviation for range distance. "
            "A larger value results in averaging of pixels "
            "with larger spatial differences."
        ),
    )
    _modelist = ("constant", "edge", "symmetric", "reflect", "wrap")
    mode = gdi.ChoiceItem(
        _("Mode"), list(zip(_modelist, _modelist)), default="constant"
    )
    cval = gdi.FloatItem(
        "cval",
        default=0,
        help=_(
            "Used in conjunction with mode 'constant', "
            "the value outside the image boundaries."
        ),
    )


def compute_denoise_bilateral(data: np.ndarray, p: DenoiseBilateralParam) -> np.ndarray:
    """Compute bilateral filter denoising
    Args:
        data (np.ndarray): input data
        p (DenoiseBilateralParam): parameters
    Returns:
        np.ndarray: output data
    """
    return denoise_bilateral(
        data,
        sigma_spatial=p.sigma_spatial,
        mode=p.mode,
        cval=p.cval,
    )


class DenoiseWaveletParam(gdt.DataSet):
    """Wavelet denoising parameters"""

    _wavelist = pywt.wavelist()
    wavelet = gdi.ChoiceItem(
        _("Wavelet"), list(zip(_wavelist, _wavelist)), default="sym9"
    )
    _modelist = ("soft", "hard")
    mode = gdi.ChoiceItem(_("Mode"), list(zip(_modelist, _modelist)), default="soft")
    _methlist = ("BayesShrink", "VisuShrink")
    method = gdi.ChoiceItem(
        _("Method"), list(zip(_methlist, _methlist)), default="VisuShrink"
    )


def compute_denoise_wavelet(data: np.ndarray, p: DenoiseWaveletParam) -> np.ndarray:
    """Compute Wavelet denoising
    Args:
        data (np.ndarray): input data
        p (DenoiseWaveletParam): parameters
    Returns:
        np.ndarray: output data
    """
    return denoise_wavelet(
        data,
        wavelet=p.wavelet,
        mode=p.mode,
        method=p.method,
    )


class MorphologyParam(gdt.DataSet):
    """White Top-Hat parameters"""

    radius = gdi.IntItem(
        _("Radius"), default=1, min=1, help=_("Footprint (disk) radius.")
    )


def compute_denoise_tophat(data: np.ndarray, p: MorphologyParam) -> np.ndarray:
    """Denoise using White Top-Hat
    Args:
        data (np.ndarray): input data
        p (MorphologyParam): parameters
    Returns:
        np.ndarray: output data
    """
    return data - morphology.white_tophat(data, morphology.disk(p.radius))


def compute_white_tophat(data: np.ndarray, p: MorphologyParam) -> np.ndarray:
    """Compute White Top-Hat
    Args:
        data (np.ndarray): input data
        p (MorphologyParam): parameters
    Returns:
        np.ndarray: output data
    """
    return morphology.white_tophat(data, morphology.disk(p.radius))


def compute_black_tophat(data: np.ndarray, p: MorphologyParam) -> np.ndarray:
    """Compute Black Top-Hat
    Args:
        data (np.ndarray): input data
        p (MorphologyParam): parameters
    Returns:
        np.ndarray: output data
    """
    return morphology.black_tophat(data, morphology.disk(p.radius))


def compute_erosion(data: np.ndarray, p: MorphologyParam) -> np.ndarray:
    """Compute Erosion
    Args:
        data (np.ndarray): input data
        p (MorphologyParam): parameters
    Returns:
        np.ndarray: output data
    """
    return morphology.erosion(data, morphology.disk(p.radius))


def compute_dilation(data: np.ndarray, p: MorphologyParam) -> np.ndarray:
    """Compute Dilation
    Args:
        data (np.ndarray): input data
        p (MorphologyParam): parameters
    Returns:
        np.ndarray: output data
    """
    return morphology.dilation(data, morphology.disk(p.radius))


def compute_opening(data: np.ndarray, p: MorphologyParam) -> np.ndarray:
    """Compute morphological opening
    Args:
        data (np.ndarray): input data
        p (MorphologyParam): parameters
    Returns:
        np.ndarray: output data
    """
    return morphology.opening(data, morphology.disk(p.radius))


def compute_closing(data: np.ndarray, p: MorphologyParam) -> np.ndarray:
    """Compute morphological closing
    Args:
        data (np.ndarray): input data
        p (MorphologyParam): parameters
    Returns:
        np.ndarray: output data
    """
    return morphology.closing(data, morphology.disk(p.radius))


class ButterworthParam(gdt.DataSet):
    """Butterworth filter parameters"""

    cut_off = gdi.FloatItem(
        _("Cut-off frequency ratio"),
        default=0.5,
        min=0.0,
        max=1.0,
        help=_("Cut-off frequency ratio (0.0 - 1.0)."),
    )
    high_pass = gdi.BoolItem(
        _("High-pass filter"),
        default=False,
        help=_("If True, apply high-pass filter instead of low-pass."),
    )
    order = gdi.IntItem(
        _("Order"),
        default=2,
        min=1,
        help=_("Order of the Butterworth filter."),
    )


def compute_butterworth(data: np.ndarray, p: ButterworthParam) -> np.ndarray:
    """Compute Butterworth filter
    Args:
        data (np.ndarray): input data
        p (ButterworthParam): parameters
    Returns:
        np.ndarray: output data
    """
    return filters.butterworth(data, p.cut_off, p.high_pass, p.order)


class CannyParam(gdt.DataSet):
    """Canny filter parameters"""

    sigma = gdi.FloatItem(
        "Sigma",
        default=1.0,
        unit="pixels",
        min=0,
        nonzero=True,
        help=_("Standard deviation of the Gaussian filter."),
    )
    low_threshold = gdi.FloatItem(
        _("Low threshold"),
        default=0.1,
        min=0,
        help=_("Lower bound for hysteresis thresholding (linking edges)."),
    )
    high_threshold = gdi.FloatItem(
        _("High threshold"),
        default=0.9,
        min=0,
        help=_("Upper bound for hysteresis thresholding (linking edges)."),
    )
    use_quantiles = gdi.BoolItem(
        _("Use quantiles"),
        default=True,
        help=_(
            "If True then treat low_threshold and high_threshold as quantiles "
            "of the edge magnitude image, rather than absolute edge magnitude "
            "values. If True then the thresholds must be in the range [0, 1]."
        ),
    )
    _modelist = ("reflect", "constant", "nearest", "mirror", "wrap")
    mode = gdi.ChoiceItem(
        _("Mode"), list(zip(_modelist, _modelist)), default="constant"
    )
    cval = gdi.FloatItem(
        "cval",
        default=0.0,
        help=_("Value to fill past edges of input if mode is constant."),
    )


def compute_canny(data: np.ndarray, p: CannyParam) -> np.ndarray:
    """Compute Canny filter
    Args:
        data (np.ndarray): input data
        p (CannyParam): parameters
    Returns:
        np.ndarray: output data
    """
    return np.array(
        feature.canny(
            data,
            sigma=p.sigma,
            low_threshold=p.low_threshold,
            high_threshold=p.high_threshold,
            use_quantiles=p.use_quantiles,
            mode=p.mode,
            cval=p.cval,
        ),
        dtype=np.uint8,
    )


def calc_with_osr(image: ImageObj, func: Callable, *args: Any) -> np.ndarray:
    """Exec computation taking into account image x0, y0, dx, dy and ROIs"""
    res = []
    for i_roi in image.iterate_roi_indexes():
        data_roi = image.get_data(i_roi)
        if args is None:
            coords = func(data_roi)
        else:
            coords = func(data_roi, *args)
        if coords.size:
            if image.roi is not None:
                x0, y0, _x1, _y1 = RoiDataItem(image.roi[i_roi]).get_rect()
                coords[:, ::2] += x0
                coords[:, 1::2] += y0
            coords[:, ::2] = image.dx * coords[:, ::2] + image.x0
            coords[:, 1::2] = image.dy * coords[:, 1::2] + image.y0
            idx = np.ones((coords.shape[0], 1)) * i_roi
            coords = np.hstack([idx, coords])
            res.append(coords)
    if res:
        return np.vstack(res)
    return None


def get_centroid_coords(data: np.ndarray) -> np.ndarray:
    """Return centroid coordinates
    Args:
        data (np.ndarray): input data
    Returns:
        np.ndarray: centroid coordinates
    """
    y, x = get_centroid_fourier(data)
    return np.array([(x, y)])


def compute_centroid(image: ImageObj) -> np.ndarray:
    """Compute centroid
    Args:
        image (ImageObj): input image
    Returns:
        np.ndarray: centroid coordinates
    """
    return calc_with_osr(image, get_centroid_coords)


def get_enclosing_circle_coords(data: np.ndarray) -> np.ndarray:
    """Return diameter coords for the circle contour enclosing image
    values above threshold (FWHM)
    Args:
        data (np.ndarray): input data
    Returns:
        np.ndarray: diameter coords
    """
    x, y, r = get_enclosing_circle(data)
    return np.array([[x - r, y, x + r, y]])


def compute_enclosing_circle(image: ImageObj) -> np.ndarray:
    """Compute minimum enclosing circle
    Args:
        image (ImageObj): input image
    Returns:
        np.ndarray: diameter coords
    """
    return calc_with_osr(image, get_enclosing_circle_coords)


class GenericDetectionParam(gdt.DataSet):
    """Generic detection parameters"""

    threshold = gdi.FloatItem(
        _("Relative threshold"),
        default=0.5,
        min=0.1,
        max=0.9,
        help=_(
            "Detection threshold, relative to difference between "
            "data maximum and minimum"
        ),
    )


class PeakDetectionParam(GenericDetectionParam):
    """Peak detection parameters"""

    size = gdi.IntItem(
        _("Neighborhoods size"),
        default=10,
        min=1,
        unit="pixels",
        help=_(
            "Size of the sliding window used in maximum/minimum filtering algorithm"
        ),
    )
    create_rois = gdi.BoolItem(_("Create regions of interest"), default=True)


def compute_peak_detection(image: ImageObj, p: PeakDetectionParam) -> np.ndarray:
    """Compute 2D peak detection
    Args:
        image (ImageObj): input image
        p (PeakDetectionParam): parameters
    Returns:
        np.ndarray: peak coordinates
    """
    return calc_with_osr(image, get_2d_peaks_coords, p.size, p.threshold)


class ContourShapeParam(GenericDetectionParam):
    """Contour shape parameters"""

    shapes = (
        ("ellipse", _("Ellipse")),
        ("circle", _("Circle")),
    )
    shape = gdi.ChoiceItem(_("Shape"), shapes, default="ellipse")


def compute_contour_shape(image: ImageObj, p: ContourShapeParam) -> np.ndarray:
    """Compute contour shape fit"""
    return calc_with_osr(image, get_contour_shapes, p.shape, p.threshold)


class HoughCircleParam(gdt.DataSet):
    """Circle Hough transform parameters"""

    min_radius = gdi.IntItem(
        _("Radius<sub>min</sub>"), unit="pixels", min=0, nonzero=True
    )
    max_radius = gdi.IntItem(
        _("Radius<sub>max</sub>"), unit="pixels", min=0, nonzero=True
    )
    min_distance = gdi.IntItem(_("Minimal distance"), min=0)


def compute_hough_circle_peaks(image: ImageObj, p: HoughCircleParam) -> np.ndarray:
    """Compute Hough circles
    Args:
        image (ImageObj): input image
        p (HoughCircleParam): parameters
    Returns:
        np.ndarray: circle coordinates
    """
    return calc_with_osr(
        image,
        get_hough_circle_peaks,
        p.min_radius,
        p.max_radius,
        None,
        p.min_distance,
    )


class BaseBlobParam(gdt.DataSet):
    """Base class for blob detection parameters"""

    min_sigma = gdi.FloatItem(
        "σ<sub>min</sub>",
        default=1.0,
        unit="pixels",
        min=0,
        nonzero=True,
        help=_(
            "The minimum standard deviation for Gaussian Kernel. "
            "Keep this low to detect smaller blobs."
        ),
    )
    max_sigma = gdi.FloatItem(
        "σ<sub>max</sub>",
        default=30.0,
        unit="pixels",
        min=0,
        nonzero=True,
        help=_(
            "The maximum standard deviation for Gaussian Kernel. "
            "Keep this high to detect larger blobs."
        ),
    )
    threshold_rel = gdi.FloatItem(
        _("Relative threshold"),
        default=0.2,
        min=0.0,
        max=1.0,
        help=_("Minimum intensity of blobs."),
    )
    overlap = gdi.FloatItem(
        _("Overlap"),
        default=0.5,
        min=0.0,
        max=1.0,
        help=_(
            "If two blobs overlap by a fraction greater than this value, the "
            "smaller blob is eliminated."
        ),
    )


class BlobDOGParam(BaseBlobParam):
    """Blob detection using Difference of Gaussian method"""

    exclude_border = gdi.BoolItem(
        _("Exclude border"),
        default=True,
        help=_("If True, exclude blobs from the border of the image."),
    )


def compute_blob_dog(image: ImageObj, p: BlobDOGParam) -> np.ndarray:
    """Compute blobs using Difference of Gaussian method
    Args:
        image (ImageObj): input image
        p (BlobDOGParam): parameters
    Returns:
        np.ndarray: blobs coordinates
    """
    return calc_with_osr(
        image,
        find_blobs_dog,
        p.min_sigma,
        p.max_sigma,
        p.overlap,
        p.threshold_rel,
        p.exclude_border,
    )


class BlobDOHParam(BaseBlobParam):
    """Blob detection using Determinant of Hessian method"""

    log_scale = gdi.BoolItem(
        _("Log scale"),
        default=False,
        help=_(
            "If set intermediate values of standard deviations are interpolated "
            "using a logarithmic scale to the base 10. "
            "If not, linear interpolation is used."
        ),
    )


def compute_blob_doh(image: ImageObj, p: BlobDOHParam) -> np.ndarray:
    """Compute blobs using Determinant of Hessian method
    Args:
        image (ImageObj): input image
        p (BlobDOHParam): parameters
    Returns:
        np.ndarray: blobs coordinates
    """
    return calc_with_osr(
        image,
        find_blobs_doh,
        p.min_sigma,
        p.max_sigma,
        p.overlap,
        p.log_scale,
        p.threshold_rel,
    )


class BlobLOGParam(BlobDOHParam):
    """Blob detection using Laplacian of Gaussian method"""

    exclude_border = gdi.BoolItem(
        _("Exclude border"),
        default=True,
        help=_("If True, exclude blobs from the border of the image."),
    )


def compute_blob_log(image: ImageObj, p: BlobLOGParam) -> np.ndarray:
    """Compute blobs using Laplacian of Gaussian method
    Args:
        image (ImageObj): input image
        p (BlobLOGParam): parameters
    Returns:
        np.ndarray: blobs coordinates
    """
    return calc_with_osr(
        image,
        find_blobs_log,
        p.min_sigma,
        p.max_sigma,
        p.overlap,
        p.log_scale,
        p.threshold_rel,
        p.exclude_border,
    )


class BlobOpenCVParam(gdt.DataSet):
    """Blob detection using OpenCV"""

    min_threshold = gdi.FloatItem(
        _("Min. threshold"),
        default=10.0,
        min=0.0,
        help=_(
            "The minimum threshold between local maxima and minima. "
            "This parameter does not affect the quality of the blobs, "
            "only the quantity. Lower thresholds result in larger "
            "numbers of blobs."
        ),
    )
    max_threshold = gdi.FloatItem(
        _("Max. threshold"),
        default=200.0,
        min=0.0,
        help=_(
            "The maximum threshold between local maxima and minima. "
            "This parameter does not affect the quality of the blobs, "
            "only the quantity. Lower thresholds result in larger "
            "numbers of blobs."
        ),
    )
    min_repeatability = gdi.IntItem(
        _("Min. repeatability"),
        default=2,
        min=1,
        help=_(
            "The minimum number of times a blob needs to be detected "
            "in a sequence of images to be considered valid."
        ),
    )
    min_dist_between_blobs = gdi.FloatItem(
        _("Min. distance between blobs"),
        default=10.0,
        min=0.0,
        help=_(
            "The minimum distance between two blobs. If blobs are found "
            "closer together than this distance, the smaller blob is removed."
        ),
    )
    _prop_col = gdt.ValueProp(False)
    filter_by_color = gdi.BoolItem(
        _("Filter by color"),
        default=True,
        help=_("If true, the image is filtered by color instead of intensity."),
    ).set_prop("display", store=_prop_col)
    blob_color = gdi.IntItem(
        _("Blob color"),
        default=0,
        help=_(
            "The color of the blobs to detect (0 for dark blobs, 255 for light blobs)."
        ),
    ).set_prop("display", active=_prop_col)
    _prop_area = gdt.ValueProp(False)
    filter_by_area = gdi.BoolItem(
        _("Filter by area"),
        default=True,
        help=_("If true, the image is filtered by blob area."),
    ).set_prop("display", store=_prop_area)
    min_area = gdi.FloatItem(
        _("Min. area"),
        default=25.0,
        min=0.0,
        help=_("The minimum blob area."),
    ).set_prop("display", active=_prop_area)
    max_area = gdi.FloatItem(
        _("Max. area"),
        default=500.0,
        min=0.0,
        help=_("The maximum blob area."),
    ).set_prop("display", active=_prop_area)
    _prop_circ = gdt.ValueProp(False)
    filter_by_circularity = gdi.BoolItem(
        _("Filter by circularity"),
        default=False,
        help=_("If true, the image is filtered by blob circularity."),
    ).set_prop("display", store=_prop_circ)
    min_circularity = gdi.FloatItem(
        _("Min. circularity"),
        default=0.8,
        min=0.0,
        max=1.0,
        help=_("The minimum circularity of the blobs."),
    ).set_prop("display", active=_prop_circ)
    max_circularity = gdi.FloatItem(
        _("Max. circularity"),
        default=1.0,
        min=0.0,
        max=1.0,
        help=_("The maximum circularity of the blobs."),
    ).set_prop("display", active=_prop_circ)
    _prop_iner = gdt.ValueProp(False)
    filter_by_inertia = gdi.BoolItem(
        _("Filter by inertia"),
        default=False,
        help=_("If true, the image is filtered by blob inertia."),
    ).set_prop("display", store=_prop_iner)
    min_inertia_ratio = gdi.FloatItem(
        _("Min. inertia ratio"),
        default=0.6,
        min=0.0,
        max=1.0,
        help=_("The minimum inertia ratio of the blobs."),
    ).set_prop("display", active=_prop_iner)
    max_inertia_ratio = gdi.FloatItem(
        _("Max. inertia ratio"),
        default=1.0,
        min=0.0,
        max=1.0,
        help=_("The maximum inertia ratio of the blobs."),
    ).set_prop("display", active=_prop_iner)
    _prop_conv = gdt.ValueProp(False)
    filter_by_convexity = gdi.BoolItem(
        _("Filter by convexity"),
        default=False,
        help=_("If true, the image is filtered by blob convexity."),
    ).set_prop("display", store=_prop_conv)
    min_convexity = gdi.FloatItem(
        _("Min. convexity"),
        default=0.8,
        min=0.0,
        max=1.0,
        help=_("The minimum convexity of the blobs."),
    ).set_prop("display", active=_prop_conv)
    max_convexity = gdi.FloatItem(
        _("Max. convexity"),
        default=1.0,
        min=0.0,
        max=1.0,
        help=_("The maximum convexity of the blobs."),
    ).set_prop("display", active=_prop_conv)


def compute_blob_opencv(image: ImageObj, p: BlobOpenCVParam) -> np.ndarray:
    """Compute blobs using OpenCV
    Args:
        image (ImageObj): input image
        p (BlobOpenCVParam): parameters
    Returns:
        np.ndarray: blobs coordinates
    """
    return calc_with_osr(
        image,
        find_blobs_opencv,
        p.min_threshold,
        p.max_threshold,
        p.min_repeatability,
        p.min_dist_between_blobs,
        p.filter_by_color,
        p.blob_color,
        p.filter_by_area,
        p.min_area,
        p.max_area,
        p.filter_by_circularity,
        p.min_circularity,
        p.max_circularity,
        p.filter_by_inertia,
        p.min_inertia_ratio,
        p.max_inertia_ratio,
        p.filter_by_convexity,
        p.min_convexity,
        p.max_convexity,
    )
