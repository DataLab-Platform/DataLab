# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Detection computation module
----------------------------

This module provides algorithms for detecting objects or patterns in images,
such as blobs, peaks, or custom structures.

Main features include:
- Blob and peak detection algorithms
- Support for object localization and counting

Detection algorithms are fundamental for many image analysis pipelines,
enabling automated extraction of regions or features of interest.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# All dataset classes must also be imported in the sigima_.param module.

from __future__ import annotations

import guidata.dataset as gds
import numpy as np

import sigima_.algorithms.image as alg
from sigima_.computation import computation_function
from sigima_.computation.image.base import calc_resultshape
from sigima_.config import _
from sigima_.obj.base import ResultShape, ShapeTypes
from sigima_.obj.image import ImageObj, create_image_roi


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
        default=None,
        check=False,  # Allow None value
        min=1,
        unit="pixels",
        help=_(
            "Size of the sliding window used in maximum/minimum filtering algorithm "
            "(if no value is provided, the algorithm will use a default size "
            "based on the image size). "
        ),
    )
    create_rois = gds.BoolItem(_("Create regions of interest"), default=True)


@computation_function()
def peak_detection(obj: ImageObj, p: Peak2DDetectionParam) -> ResultShape | None:
    """Compute 2D peak detection
    with :py:func:`sigima_.algorithms.image.get_2d_peaks_coords`

    Args:
        obj: input image
        p: parameters

    Returns:
        Peak coordinates
    """
    result = calc_resultshape(
        "peak", "point", obj, alg.get_2d_peaks_coords, p.size, p.threshold
    )
    if result is not None and p.create_rois and result.raw_data.shape[0] > 1:
        # Create a rectangular ROI around each peak, only if there are more than one
        # peak detected (otherwise, it would not make sense to create an ROI)
        dist = alg.distance_matrix(result.raw_data)
        dist_min = dist[dist != 0].min()
        assert dist_min > 0
        radius = int(0.5 * dist_min / np.sqrt(2) - 1)
        assert radius >= 1
        ymax, xmax = obj.data.shape
        coords = []
        for x, y in result.raw_data:
            x0, y0 = max(x - radius, 0), max(y - radius, 0)
            dx, dy = min(x + radius, xmax) - x0, min(y + radius, ymax) - y0
            coords.append([x0, y0, dx, dy])
        result.roi = create_image_roi("rectangle", coords, indices=True)
    return result


class ContourShapeParam(GenericDetectionParam):
    """Contour shape parameters"""

    shapes = (
        ("ellipse", _("Ellipse")),
        ("circle", _("Circle")),
        ("polygon", _("Polygon")),
    )

    # The following item is used to store the 'shape type' and is implicitly accessed
    # by the `cdl.gui.processor.base.BaseProcessor.compute_1_to_0` method
    # (see DataLab's main package).
    # The keys of the item choices (i.e. the first element of each tuple of `shapes`)
    # must match the names of the `sigima_.obj.base.ShapeTypes` (when uppercased).
    assert {shape[0].upper() for shape in shapes}.issubset(
        set(ShapeTypes.__members__.keys())
    )
    shape = gds.ChoiceItem(_("Shape"), shapes, default="ellipse")


@computation_function()
def contour_shape(image: ImageObj, p: ContourShapeParam) -> ResultShape | None:
    """Compute contour shape fit
    with :py:func:`sigima_.algorithms.image.get_contour_shapes`"""
    return calc_resultshape(
        "contour", p.shape, image, alg.get_contour_shapes, p.shape, p.threshold
    )


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


@computation_function()
def blob_dog(image: ImageObj, p: BlobDOGParam) -> ResultShape | None:
    """Compute blobs using Difference of Gaussian method
    with :py:func:`sigima_.algorithms.image.find_blobs_dog`

    Args:
        imageOutput: input image
        p: parameters

    Returns:
        Blobs coordinates
    """
    return calc_resultshape(
        "blob_dog",
        "circle",
        image,
        alg.find_blobs_dog,
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


@computation_function()
def blob_doh(image: ImageObj, p: BlobDOHParam) -> ResultShape | None:
    """Compute blobs using Determinant of Hessian method
    with :py:func:`sigima_.algorithms.image.find_blobs_doh`

    Args:
        imageOutput: input image
        p: parameters

    Returns:
        Blobs coordinates
    """
    return calc_resultshape(
        "blob_doh",
        "circle",
        image,
        alg.find_blobs_doh,
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


@computation_function()
def blob_log(image: ImageObj, p: BlobLOGParam) -> ResultShape | None:
    """Compute blobs using Laplacian of Gaussian method
    with :py:func:`sigima_.algorithms.image.find_blobs_log`

    Args:
        imageOutput: input image
        p: parameters

    Returns:
        Blobs coordinates
    """
    return calc_resultshape(
        "blob_log",
        "circle",
        image,
        alg.find_blobs_log,
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


@computation_function()
def blob_opencv(image: ImageObj, p: BlobOpenCVParam) -> ResultShape | None:
    """Compute blobs using OpenCV
    with :py:func:`sigima_.algorithms.image.find_blobs_opencv`

    Args:
        imageOutput: input image
        p: parameters

    Returns:
        Blobs coordinates
    """
    return calc_resultshape(
        "blob_opencv",
        "circle",
        image,
        alg.find_blobs_opencv,
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


class HoughCircleParam(gds.DataSet):
    """Circle Hough transform parameters"""

    min_radius = gds.IntItem(
        _("Radius<sub>min</sub>"), unit="pixels", min=0, nonzero=True
    )
    max_radius = gds.IntItem(
        _("Radius<sub>max</sub>"), unit="pixels", min=0, nonzero=True
    )
    min_distance = gds.IntItem(_("Minimal distance"), min=0)


@computation_function()
def hough_circle_peaks(image: ImageObj, p: HoughCircleParam) -> ResultShape | None:
    """Compute Hough circles
    with :py:func:`sigima_.algorithms.image.get_hough_circle_peaks`

    Args:
        image: input image
        p: parameters

    Returns:
        Circle coordinates
    """
    return calc_resultshape(
        "hough_circle_peak",
        "circle",
        image,
        alg.get_hough_circle_peaks,
        p.min_radius,
        p.max_radius,
        None,
        p.min_distance,
    )
