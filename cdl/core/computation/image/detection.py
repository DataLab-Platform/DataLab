# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Blob detection computation module
---------------------------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# All dataset classes must also be imported in the cdl.core.computation.param module.

from __future__ import annotations

import guidata.dataset as gds
import numpy as np

from cdl.algorithms.image import (
    find_blobs_dog,
    find_blobs_doh,
    find_blobs_log,
    find_blobs_opencv,
    get_2d_peaks_coords,
    get_contour_shapes,
)
from cdl.config import _
from cdl.core.computation.image import calc_with_osr
from cdl.core.model.base import ShapeTypes
from cdl.core.model.image import ImageObj


class GenericDetectionParam(gds.DataSet):
    """Generic detection parameters"""

    threshold = gds.FloatItem(
        _("Relative threshold"),
        default=0.5,
        min=0.1,
        max=0.9,
        help=_(
            "Detection threshold, relative to difference between "
            "data maximum and minimum"
        ),
    )


class Peak2DDetectionParam(GenericDetectionParam):
    """Peak detection parameters"""

    size = gds.IntItem(
        _("Neighborhoods size"),
        default=10,
        min=1,
        unit="pixels",
        help=_(
            "Size of the sliding window used in maximum/minimum filtering algorithm"
        ),
    )
    create_rois = gds.BoolItem(_("Create regions of interest"), default=True)


def compute_peak_detection(image: ImageObj, p: Peak2DDetectionParam) -> np.ndarray:
    """Compute 2D peak detection

    Args:
        imageOutput: input image
        p: parameters

    Returns:
        Peak coordinates
    """
    return calc_with_osr(image, get_2d_peaks_coords, p.size, p.threshold)


class ContourShapeParam(GenericDetectionParam):
    """Contour shape parameters"""

    shapes = (
        ("ellipse", _("Ellipse")),
        ("circle", _("Circle")),
        ("polygon", _("Polygon")),
    )

    # The following item is used to store the 'shape type' and is implicitly accessed by
    # the `cdl.core.gui.processor.base.BaseProcessor.compute_10` method. The keys of the
    # item choices (i.e. the first element of each tuple of `shapes`) must match the
    # names of the `cdl.core.model.base.ShapeTypes` (when uppercased).
    assert {shape[0].upper() for shape in shapes}.issubset(
        set(ShapeTypes.__members__.keys())
    )
    shape = gds.ChoiceItem(_("Shape"), shapes, default="ellipse")


def compute_contour_shape(image: ImageObj, p: ContourShapeParam) -> np.ndarray:
    """Compute contour shape fit"""
    return calc_with_osr(image, get_contour_shapes, p.shape, p.threshold)


class BaseBlobParam(gds.DataSet):
    """Base class for blob detection parameters"""

    min_sigma = gds.FloatItem(
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
    max_sigma = gds.FloatItem(
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
    threshold_rel = gds.FloatItem(
        _("Relative threshold"),
        default=0.2,
        min=0.0,
        max=1.0,
        help=_("Minimum intensity of blobs."),
    )
    overlap = gds.FloatItem(
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

    exclude_border = gds.BoolItem(
        _("Exclude border"),
        default=True,
        help=_("If True, exclude blobs from the border of the image."),
    )


def compute_blob_dog(image: ImageObj, p: BlobDOGParam) -> np.ndarray:
    """Compute blobs using Difference of Gaussian method

    Args:
        imageOutput: input image
        p: parameters

    Returns:
        Blobs coordinates
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

    log_scale = gds.BoolItem(
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
        imageOutput: input image
        p: parameters

    Returns:
        Blobs coordinates
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

    exclude_border = gds.BoolItem(
        _("Exclude border"),
        default=True,
        help=_("If True, exclude blobs from the border of the image."),
    )


def compute_blob_log(image: ImageObj, p: BlobLOGParam) -> np.ndarray:
    """Compute blobs using Laplacian of Gaussian method

    Args:
        imageOutput: input image
        p: parameters

    Returns:
        Blobs coordinates
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


class BlobOpenCVParam(gds.DataSet):
    """Blob detection using OpenCV"""

    min_threshold = gds.FloatItem(
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
    max_threshold = gds.FloatItem(
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
    min_repeatability = gds.IntItem(
        _("Min. repeatability"),
        default=2,
        min=1,
        help=_(
            "The minimum number of times a blob needs to be detected "
            "in a sequence of images to be considered valid."
        ),
    )
    min_dist_between_blobs = gds.FloatItem(
        _("Min. distance between blobs"),
        default=10.0,
        min=0.0,
        nonzero=True,
        help=_(
            "The minimum distance between two blobs. If blobs are found "
            "closer together than this distance, the smaller blob is removed."
        ),
    )
    _prop_col = gds.ValueProp(False)
    filter_by_color = gds.BoolItem(
        _("Filter by color"),
        default=True,
        help=_("If true, the image is filtered by color instead of intensity."),
    ).set_prop("display", store=_prop_col)
    blob_color = gds.IntItem(
        _("Blob color"),
        default=0,
        help=_(
            "The color of the blobs to detect (0 for dark blobs, 255 for light blobs)."
        ),
    ).set_prop("display", active=_prop_col)
    _prop_area = gds.ValueProp(False)
    filter_by_area = gds.BoolItem(
        _("Filter by area"),
        default=True,
        help=_("If true, the image is filtered by blob area."),
    ).set_prop("display", store=_prop_area)
    min_area = gds.FloatItem(
        _("Min. area"),
        default=25.0,
        min=0.0,
        help=_("The minimum blob area."),
    ).set_prop("display", active=_prop_area)
    max_area = gds.FloatItem(
        _("Max. area"),
        default=500.0,
        min=0.0,
        help=_("The maximum blob area."),
    ).set_prop("display", active=_prop_area)
    _prop_circ = gds.ValueProp(False)
    filter_by_circularity = gds.BoolItem(
        _("Filter by circularity"),
        default=False,
        help=_("If true, the image is filtered by blob circularity."),
    ).set_prop("display", store=_prop_circ)
    min_circularity = gds.FloatItem(
        _("Min. circularity"),
        default=0.8,
        min=0.0,
        max=1.0,
        help=_("The minimum circularity of the blobs."),
    ).set_prop("display", active=_prop_circ)
    max_circularity = gds.FloatItem(
        _("Max. circularity"),
        default=1.0,
        min=0.0,
        max=1.0,
        help=_("The maximum circularity of the blobs."),
    ).set_prop("display", active=_prop_circ)
    _prop_iner = gds.ValueProp(False)
    filter_by_inertia = gds.BoolItem(
        _("Filter by inertia"),
        default=False,
        help=_("If true, the image is filtered by blob inertia."),
    ).set_prop("display", store=_prop_iner)
    min_inertia_ratio = gds.FloatItem(
        _("Min. inertia ratio"),
        default=0.6,
        min=0.0,
        max=1.0,
        help=_("The minimum inertia ratio of the blobs."),
    ).set_prop("display", active=_prop_iner)
    max_inertia_ratio = gds.FloatItem(
        _("Max. inertia ratio"),
        default=1.0,
        min=0.0,
        max=1.0,
        help=_("The maximum inertia ratio of the blobs."),
    ).set_prop("display", active=_prop_iner)
    _prop_conv = gds.ValueProp(False)
    filter_by_convexity = gds.BoolItem(
        _("Filter by convexity"),
        default=False,
        help=_("If true, the image is filtered by blob convexity."),
    ).set_prop("display", store=_prop_conv)
    min_convexity = gds.FloatItem(
        _("Min. convexity"),
        default=0.8,
        min=0.0,
        max=1.0,
        help=_("The minimum convexity of the blobs."),
    ).set_prop("display", active=_prop_conv)
    max_convexity = gds.FloatItem(
        _("Max. convexity"),
        default=1.0,
        min=0.0,
        max=1.0,
        help=_("The maximum convexity of the blobs."),
    ).set_prop("display", active=_prop_conv)


def compute_blob_opencv(image: ImageObj, p: BlobOpenCVParam) -> np.ndarray:
    """Compute blobs using OpenCV

    Args:
        imageOutput: input image
        p: parameters

    Returns:
        Blobs coordinates
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
