# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Image computation objects (see parent package :mod:`sigima_.computation`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# MARK: Important notes
# ---------------------
# - All `guidata.dataset.DataSet` classes must also be imported
#   in the `sigima_.param` module.
# - All functions decorated by `computation_function` defined in the other modules
#   of this package must be imported right here.

# pylint:disable=unused-import
# flake8: noqa

from __future__ import annotations

import guidata.dataset as gds

from cdl.config import _
from sigima_.computation.image.arithmetic import (
    addition,
    average,
    product,
    addition_constant,
    difference_constant,
    product_constant,
    division_constant,
    arithmetic,
    difference,
    quadratic_difference,
    division,
)
from sigima_.computation.image.detection import (
    BlobDOGParam,
    BlobDOHParam,
    BlobLOGParam,
    BlobOpenCVParam,
    ContourShapeParam,
    HoughCircleParam,
    Peak2DDetectionParam,
    blob_dog,
    blob_doh,
    blob_log,
    blob_opencv,
    contour_shape,
    hough_circle_peaks,
    peak_detection,
)
from sigima_.computation.image.edges import (
    CannyParam,
    canny,
    farid,
    farid_h,
    farid_v,
    laplace,
    prewitt,
    prewitt_h,
    prewitt_v,
    roberts,
    scharr,
    scharr_h,
    scharr_v,
    sobel,
    sobel_h,
    sobel_v,
)
from sigima_.computation.image.exposure import (
    AdjustGammaParam,
    AdjustLogParam,
    AdjustSigmoidParam,
    EqualizeAdaptHistParam,
    EqualizeHistParam,
    FlatFieldParam,
    RescaleIntensityParam,
    ZCalibrateParam,
    adjust_gamma,
    adjust_log,
    adjust_sigmoid,
    calibration,
    clip,
    equalize_adapthist,
    equalize_hist,
    flatfield,
    histogram,
    normalize,
    offset_correction,
    rescale_intensity,
)
from sigima_.computation.image.extraction import (
    AverageProfileParam,
    LineProfileParam,
    RadialProfileParam,
    SegmentProfileParam,
    average_profile,
    extract_roi,
    extract_rois,
    line_profile,
    radial_profile,
    segment_profile,
)
from sigima_.computation.image.filtering import (
    ButterworthParam,
    butterworth,
    gaussian_filter,
    moving_average,
    moving_median,
    wiener,
)
from sigima_.computation.image.fourier import (
    ZeroPadding2DParam,
    fft,
    ifft,
    magnitude_spectrum,
    phase_spectrum,
    psd,
    zero_padding,
)
from sigima_.computation.image.geometry import (
    BinningParam,
    ResizeParam,
    RotateParam,
    binning,
    fliph,
    flipv,
    resize,
    rotate,
    rotate90,
    rotate270,
    swap_axes,
)
from sigima_.computation.image.mathops import (
    DataTypeIParam,
    LogP1Param,
    absolute,
    astype,
    exp,
    imag,
    inverse,
    log10,
    logp1,
    real,
)
from sigima_.computation.image.measurement import centroid, enclosing_circle, stats
from sigima_.computation.image.morphology import (
    MorphologyParam,
    black_tophat,
    closing,
    dilation,
    erosion,
    opening,
    white_tophat,
)
from sigima_.computation.image.restoration import (
    DenoiseBilateralParam,
    DenoiseTVParam,
    DenoiseWaveletParam,
    denoise_bilateral,
    denoise_tophat,
    denoise_tv,
    denoise_wavelet,
)
from sigima_.computation.image.threshold import (
    ThresholdParam,
    threshold,
    threshold_isodata,
    threshold_li,
    threshold_mean,
    threshold_minimum,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)


class GridParam(gds.DataSet):
    """Grid parameters"""

    _prop = gds.GetAttrProp("direction")
    _directions = (("col", _("columns")), ("row", _("rows")))
    direction = gds.ChoiceItem(_("Distribute over"), _directions, radio=True).set_prop(
        "display", store=_prop
    )
    cols = gds.IntItem(_("Columns"), default=1, nonzero=True).set_prop(
        "display", active=gds.FuncProp(_prop, lambda x: x == "col")
    )
    rows = gds.IntItem(_("Rows"), default=1, nonzero=True).set_prop(
        "display", active=gds.FuncProp(_prop, lambda x: x == "row")
    )
    colspac = gds.FloatItem(_("Column spacing"), default=0.0, min=0.0)
    rowspac = gds.FloatItem(_("Row spacing"), default=0.0, min=0.0)
