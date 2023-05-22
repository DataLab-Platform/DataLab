# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Image Processor GUI module
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import guidata.dataset.dataitems as gdi
import guidata.dataset.datatypes as gdt
import numpy as np
import pywt
import scipy.ndimage as spi
import scipy.signal as sps
from guiqwt.geometry import vector_rotation
from guiqwt.widgets.resizedialog import ResizeDialog
from numpy import ma
from qtpy import QtWidgets as QW
from skimage import exposure, feature, morphology
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, denoise_wavelet
from skimage.util.dtype import dtype_range

from cdl.config import APP_NAME, _
from cdl.core.computation.image import (
    BINNING_OPERATIONS,
    binning,
    distance_matrix,
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
from cdl.core.gui.processor.base import BaseProcessor, ClipParam, ThresholdParam
from cdl.core.model.base import BaseProcParam, ShapeTypes
from cdl.core.model.image import ImageObj, RoiDataGeometries, RoiDataItem
from cdl.utils.qthelpers import create_progress_bar, exec_dialog, qt_try_except

if TYPE_CHECKING:  # pragma: no cover
    from cdl.core.gui.processor.base import (
        GaussianParam,
        MovingAverageParam,
        MovingMedianParam,
    )

    Obj = ImageObj

VALID_DTYPES_STRLIST = [
    dtype.__name__ for dtype in dtype_range if dtype in ImageObj.VALID_DTYPES
]


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


class EqualizeHistParam(gdt.DataSet):
    """Histogram equalization parameters"""

    nbins = gdi.IntItem(
        _("Number of bins"),
        min=1,
        default=256,
        help=_("Number of bins for image histogram."),
    )


class EqualizeAdaptHistParam(EqualizeHistParam):
    """Adaptive histogram equalization parameters"""

    clip_limit = gdi.FloatItem(
        _("Clipping limit"),
        default=0.01,
        min=0.0,
        max=1.0,
        help=_("Clipping limit (higher values give more contrast)."),
    )


class LogP1Param(gdt.DataSet):
    """Log10 parameters"""

    n = gdi.FloatItem("n")


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


def rotate_obj_coords(
    angle: float, obj: ImageObj, orig: ImageObj, coords: np.ndarray
) -> None:
    """Apply rotation to coords associated to image obj"""
    for row in range(coords.shape[0]):
        for col in range(0, coords.shape[1], 2):
            x1, y1 = coords[row, col : col + 2]
            dx1 = x1 - orig.xc
            dy1 = y1 - orig.yc
            dx2, dy2 = vector_rotation(-angle * np.pi / 180.0, dx1, dy1)
            coords[row, col : col + 2] = dx2 + obj.xc, dy2 + obj.yc
    obj.roi = None


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


class FlatFieldParam(BaseProcParam):
    """Flat-field parameters"""

    threshold = gdi.FloatItem(_("Threshold"), default=0.0)


class ZCalibrateParam(gdt.DataSet):
    """Image linear calibration parameters"""

    a = gdi.FloatItem("a", default=1.0)
    b = gdi.FloatItem("b", default=0.0)


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


class MorphologyParam(gdt.DataSet):
    """White Top-Hat parameters"""

    radius = gdi.IntItem(
        _("Radius"), default=1, min=1, help=_("Footprint (disk) radius.")
    )


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


class ContourShapeParam(GenericDetectionParam):
    """Contour shape parameters"""

    shapes = (
        ("ellipse", _("Ellipse")),
        ("circle", _("Circle")),
    )
    shape = gdi.ChoiceItem(_("Shape"), shapes, default="ellipse")


class HoughCircleParam(gdt.DataSet):
    """Circle Hough transform parameters"""

    min_radius = gdi.IntItem(
        _("Radius<sub>min</sub>"), unit="pixels", min=0, nonzero=True
    )
    max_radius = gdi.IntItem(
        _("Radius<sub>max</sub>"), unit="pixels", min=0, nonzero=True
    )
    min_distance = gdi.IntItem(_("Minimal distance"), min=0)


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


class BlobLOGParam(BlobDOHParam):
    """Blob detection using Laplacian of Gaussian method"""

    exclude_border = gdi.BoolItem(
        _("Exclude border"),
        default=True,
        help=_("If True, exclude blobs from the border of the image."),
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


class ImageProcessor(BaseProcessor):
    """Object handling image processing: operations, processing, computing"""

    # pylint: disable=duplicate-code

    EDIT_ROI_PARAMS = True

    def compute_logp1(self, param: LogP1Param | None = None) -> None:
        """Compute base 10 logarithm"""
        edit, param = self.init_param(param, LogP1Param, "Log10(z+n)")
        self.compute_11(
            "Log10(z+n)",
            lambda z, p: np.log10(z + p.n),
            param,
            suffix=lambda p: f"n={p.n}",
            edit=edit,
        )

    def rotate_arbitrarily(self, param: RotateParam | None = None) -> None:
        """Rotate data arbitrarily"""
        edit, param = self.init_param(param, RotateParam, "Rotate")

        def rotate_xy(
            obj: ImageObj, orig: ImageObj, coords: np.ndarray, p: RotateParam
        ) -> None:
            """Apply rotation to coords"""
            rotate_obj_coords(p.angle, obj, orig, coords)

        self.compute_11(
            "Rotate",
            lambda x, p: spi.rotate(
                x,
                p.angle,
                reshape=p.reshape,
                order=p.order,
                mode=p.mode,
                cval=p.cval,
                prefilter=p.prefilter,
            ),
            param,
            suffix=lambda p: f"α={p.angle:.3f}°, mode='{p.mode}'",
            func_obj=lambda obj, orig, p: obj.transform_shapes(orig, rotate_xy, p),
            edit=edit,
        )

    def rotate_90(self) -> None:
        """Rotate data 90°"""

        def rotate_xy(obj: ImageObj, orig: ImageObj, coords: np.ndarray) -> None:
            """Apply rotation to coords"""
            rotate_obj_coords(90.0, obj, orig, coords)

        self.compute_11(
            "Rotate90",
            np.rot90,
            func_obj=lambda obj, orig: obj.transform_shapes(orig, rotate_xy),
        )

    def rotate_270(self) -> None:
        """Rotate data 270°"""

        def rotate_xy(obj: ImageObj, orig: ImageObj, coords: np.ndarray) -> None:
            """Apply rotation to coords"""
            rotate_obj_coords(270.0, obj, orig, coords)

        self.compute_11(
            "Rotate270",
            lambda x: np.rot90(x, 3),
            func_obj=lambda obj, orig: obj.transform_shapes(orig, rotate_xy),
        )

    def flip_horizontally(self) -> None:
        """Flip data horizontally"""

        # pylint: disable=unused-argument
        def hflip_coords(obj: ImageObj, orig: ImageObj, coords: np.ndarray) -> None:
            """Apply HFlip to coords"""
            coords[:, ::2] = obj.x0 + obj.dx * obj.data.shape[1] - coords[:, ::2]
            obj.roi = None

        self.compute_11(
            "HFlip",
            np.fliplr,
            func_obj=lambda obj, orig: obj.transform_shapes(orig, hflip_coords),
        )

    def flip_vertically(self) -> None:
        """Flip data vertically"""

        # pylint: disable=unused-argument
        def vflip_coords(obj: ImageObj, orig: ImageObj, coords: np.ndarray) -> None:
            """Apply VFlip to coords"""
            coords[:, 1::2] = obj.y0 + obj.dy * obj.data.shape[0] - coords[:, 1::2]
            obj.roi = None

        self.compute_11(
            "VFlip",
            np.flipud,
            func_obj=lambda obj, orig: obj.transform_shapes(orig, vflip_coords),
        )

    def distribute_on_grid(self, param: GridParam | None = None) -> None:
        """Distribute images on a grid"""
        title = _("Distribute on grid")
        edit, param = self.init_param(param, GridParam, title)
        if edit and not param.edit(parent=self.panel.parent()):
            return
        objs = self.panel.objview.get_sel_objects(include_groups=True)
        g_row, g_col, x0, y0, x0_0, y0_0 = 0, 0, 0.0, 0.0, 0.0, 0.0
        delta_x0, delta_y0 = 0.0, 0.0
        with create_progress_bar(self.panel, title, max_=len(objs)) as progress:
            for i_row, obj in enumerate(objs):
                progress.setValue(i_row + 1)
                QW.QApplication.processEvents()
                if progress.wasCanceled():
                    break
                if i_row == 0:
                    x0_0, y0_0 = x0, y0 = obj.x0, obj.y0
                else:
                    delta_x0, delta_y0 = x0 - obj.x0, y0 - obj.y0
                    obj.x0 += delta_x0
                    obj.y0 += delta_y0

                    # pylint: disable=unused-argument
                    def translate_coords(obj, orig, coords):
                        """Apply translation to coords"""
                        coords[:, ::2] += delta_x0
                        coords[:, 1::2] += delta_y0

                    obj.transform_shapes(None, translate_coords)
                if param.direction == "row":
                    # Distributing images over rows
                    sign = np.sign(param.rows)
                    g_row = (g_row + sign) % param.rows
                    y0 += (obj.dy * obj.data.shape[0] + param.rowspac) * sign
                    if g_row == 0:
                        g_col += 1
                        x0 += obj.dx * obj.data.shape[1] + param.colspac
                        y0 = y0_0
                else:
                    # Distributing images over columns
                    sign = np.sign(param.cols)
                    g_col = (g_col + sign) % param.cols
                    x0 += (obj.dx * obj.data.shape[1] + param.colspac) * sign
                    if g_col == 0:
                        g_row += 1
                        x0 = x0_0
                        y0 += obj.dy * obj.data.shape[0] + param.rowspac
        self.panel.SIG_UPDATE_PLOT_ITEMS.emit()

    def reset_positions(self) -> None:
        """Reset image positions"""
        x0_0, y0_0 = 0.0, 0.0
        delta_x0, delta_y0 = 0.0, 0.0
        objs = self.panel.objview.get_sel_objects(include_groups=True)
        for i_row, obj in enumerate(objs):
            if i_row == 0:
                x0_0, y0_0 = obj.x0, obj.y0
            else:
                delta_x0, delta_y0 = x0_0 - obj.x0, y0_0 - obj.y0
                obj.x0 += delta_x0
                obj.y0 += delta_y0

                # pylint: disable=unused-argument
                def translate_coords(obj, orig, coords):
                    """Apply translation to coords"""
                    coords[:, ::2] += delta_x0
                    coords[:, 1::2] += delta_y0

                obj.transform_shapes(None, translate_coords)
        self.panel.SIG_UPDATE_PLOT_ITEMS.emit()

    def resize(self, param: ResizeParam | None = None) -> None:
        """Resize image"""
        obj0 = self.panel.objview.get_sel_objects()[0]
        for obj in self.panel.objview.get_sel_objects():
            if obj.size != obj0.size:
                QW.QMessageBox.warning(
                    self.panel.parent(),
                    APP_NAME,
                    _("Warning:")
                    + "\n"
                    + _("Selected images do not have the same size"),
                )

        edit, param = self.init_param(param, ResizeParam, _("Resize"))
        if edit:
            original_size = obj0.size
            dlg = ResizeDialog(
                self.plotwidget,
                new_size=original_size,
                old_size=original_size,
                text=_("Destination size:"),
            )
            if not exec_dialog(dlg):
                return
            param.zoom = dlg.get_zoom()

        # pylint: disable=unused-argument
        def func_obj(obj, orig: ImageObj, param: ResizeParam) -> None:
            """Zooming function"""
            if obj.dx is not None and obj.dy is not None:
                obj.dx, obj.dy = obj.dx / param.zoom, obj.dy / param.zoom
            # TODO: [P2] Instead of removing geometric shapes, apply zoom
            obj.remove_all_shapes()

        self.compute_11(
            "Zoom",
            lambda x, p: spi.interpolation.zoom(
                x,
                p.zoom,
                order=p.order,
                mode=p.mode,
                cval=p.cval,
                prefilter=p.prefilter,
            ),
            param,
            suffix=lambda p: f"zoom={p.zoom:.3f}",
            func_obj=func_obj,
            edit=edit,
        )

    def rebin(self, param: BinningParam | None = None) -> None:
        """Binning image"""
        edit = param is None
        obj0 = self.panel.objview.get_sel_objects(include_groups=True)[0]
        input_dtype_str = str(obj0.data.dtype)
        edit, param = self.init_param(param, BinningParam, _("Binning"))
        if edit:
            param.dtype_str = input_dtype_str
        if param.dtype_str is None:
            param.dtype_str = input_dtype_str

        # pylint: disable=unused-argument
        def func_obj(obj: ImageObj, orig: ImageObj, param: BinningParam):
            """Binning function"""
            if param.change_pixel_size:
                if obj.dx is not None and obj.dy is not None:
                    obj.dx *= param.binning_x
                    obj.dy *= param.binning_y
                # TODO: [P2] Instead of removing geometric shapes, apply zoom
                obj.remove_all_shapes()

        self.compute_11(
            "PixelBinning",
            lambda x, p: binning(
                x,
                binning_x=p.binning_x,
                binning_y=p.binning_y,
                operation=p.operation,
                dtype=p.dtype_str,
            ),
            param,
            suffix=lambda p: f"{p.binning_x}x{p.binning_y},{p.operation},"
            f"change_pixel_size={p.change_pixel_size}",
            func_obj=func_obj,
            edit=edit,
        )

    def extract_roi(
        self, roidata: np.ndarray | None = None, singleobj: bool | None = None
    ) -> None:
        """Extract Region Of Interest (ROI) from data"""
        roieditordata = self._get_roieditordata(roidata, singleobj)
        if roieditordata is None or roieditordata.is_empty:
            return
        obj = self.panel.objview.get_sel_objects()[0]
        group = obj.roidata_to_params(roieditordata.roidata)

        if roieditordata.singleobj:

            def suffix_func(group: gdt.DataSetGroup):
                if len(group.datasets) == 1:
                    p = group.datasets[0]
                    return p.get_suffix()
                return ""

            def extract_roi_func(data: np.ndarray, group: gdt.DataSetGroup):
                """Extract ROI function on data"""
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

            def extract_roi_func_obj(
                image: ImageObj, orig: ImageObj, group: gdt.DataSetGroup
            ):  # pylint: disable=unused-argument
                """Extract ROI function on object"""
                image.x0 += min([p.x0 for p in group.datasets])
                image.y0 += min([p.y0 for p in group.datasets])
                image.roi = None

            self.compute_11(
                "ROI",
                extract_roi_func,
                group,
                suffix=suffix_func,
                func_obj=extract_roi_func_obj,
                edit=False,
            )

        else:

            def extract_roi_func_obj(
                image: ImageObj, orig: ImageObj, p: gdt.DataSet
            ):  # pylint: disable=unused-argument
                """Extract ROI function on object"""
                image.x0 += p.x0
                image.y0 += p.y0
                image.roi = None
                if p.geometry is RoiDataGeometries.CIRCLE:
                    # Circular ROI
                    image.roi = p.get_single_roi()

            self.compute_1n(
                [f"ROI{iroi}" for iroi in range(len(group.datasets))],
                lambda z, p: z.copy()[p.y0 : p.y1, p.x0 : p.x1],
                group.datasets,
                suffix=lambda p: p.get_suffix(),
                func_obj=extract_roi_func_obj,
                edit=False,
            )

    def swap_axes(self) -> None:
        """Swap data axes"""
        self.compute_11(
            "SwapAxes",
            lambda z: z.T,
            func_obj=lambda obj, _orig: obj.remove_all_shapes(),
        )

    def compute_abs(self) -> None:
        """Compute absolute value"""
        self.compute_11("Abs", np.abs)

    def compute_log10(self) -> None:
        """Compute Log10"""
        self.compute_11("Log10", np.log10)

    @qt_try_except()
    def flat_field_correction(
        self, obj2: Obj | None = None, param: FlatFieldParam | None = None
    ) -> None:
        """Compute flat field correction"""
        edit, param = self.init_param(param, FlatFieldParam, _("Flat field"))
        if edit:
            obj = self.panel.objview.get_sel_objects()[0]
            param.set_from_datatype(obj.data.dtype)
        self.compute_n1n(
            _("FlatField"),
            obj2,
            _("flat field image"),
            func=lambda raw, flat, p: flatfield(raw, flat, p.threshold),
            param=param,
            suffix=lambda p: "threshold={p.threshold}",
            edit=edit,
        )

    # ------Image Processing
    def get_11_func_args(self, orig: ImageObj, param: gdt.DataSet) -> tuple[Any]:
        """Get 11 function args: 1 object in --> 1 object out"""
        if param is None:
            return (orig.data,)
        return (orig.data, param)

    def set_11_func_result(self, new_obj: ImageObj, result: np.ndarray) -> None:
        """Set 11 function result: 1 object in --> 1 object out"""
        new_obj.data = result

    @qt_try_except()
    def compute_calibration(self, param: ZCalibrateParam | None = None) -> None:
        """Compute data linear calibration"""
        edit, param = self.init_param(
            param, ZCalibrateParam, _("Linear calibration"), "y = a.x + b"
        )
        self.compute_11(
            "LinearCal",
            lambda x, p: p.a * x + p.b,
            param,
            suffix=lambda p: "z={p.a}*z+{p.b}",
            edit=edit,
        )

    @qt_try_except()
    def compute_threshold(self, param: ThresholdParam | None = None) -> None:
        """Compute threshold clipping"""
        edit, param = self.init_param(param, ThresholdParam, _("Thresholding"))
        self.compute_11(
            "Threshold",
            lambda x, p: np.clip(x, p.value, x.max()),
            param,
            suffix=lambda p: f"min={p.value} lsb",
            edit=edit,
        )

    @qt_try_except()
    def compute_clip(self, param: ClipParam | None = None) -> None:
        """Compute maximum data clipping"""
        edit, param = self.init_param(param, ClipParam, _("Clipping"))
        self.compute_11(
            "Clip",
            lambda x, p: np.clip(x, x.min(), p.value),
            param,
            suffix=lambda p: f"max={p.value} lsb",
            edit=edit,
        )

    @qt_try_except()
    def rescale_intensity(self, param: RescaleIntensityParam | None = None) -> None:
        """Rescale image intensity levels"""
        edit, param = self.init_param(
            param, RescaleIntensityParam, _("Rescale intensity")
        )
        self.compute_11(
            "RescaleIntensity",
            lambda x, p: exposure.rescale_intensity(
                x, in_range=p.in_range, out_range=p.out_range
            ),
            param,
            suffix=lambda p: f"in_range={p.in_range},out_range={p.out_range}",
            edit=edit,
        )

    @qt_try_except()
    def equalize_hist(self, param: EqualizeHistParam | None = None) -> None:
        """Histogram equalization"""
        edit, param = self.init_param(
            param, EqualizeHistParam, _("Histogram equalization")
        )
        self.compute_11(
            "EqualizeHist",
            lambda x, p: exposure.equalize_hist(x, nbins=p.nbins),
            param,
            suffix=lambda p: f"nbins={p.nbins}",
            edit=edit,
        )

    @qt_try_except()
    def equalize_adapthist(self, param: EqualizeAdaptHistParam | None = None) -> None:
        """Adaptive histogram equalization"""
        edit, param = self.init_param(
            param, EqualizeAdaptHistParam, _("Adaptive histogram equalization")
        )
        self.compute_11(
            "EqualizeAdaptHist",
            lambda x, p: exposure.equalize_adapthist(
                x, clip_limit=p.clip_limit, nbins=p.nbins
            ),
            param,
            suffix=lambda p: f"clip_limit={p.clip_limit},nbins={p.nbins}",
            edit=edit,
        )

    @staticmethod
    # pylint: disable=arguments-differ
    def func_gaussian_filter(x: np.ndarray, p: GaussianParam) -> np.ndarray:
        """Compute gaussian filter"""
        return spi.gaussian_filter(x, p.sigma)

    @qt_try_except()
    def compute_fft(self) -> None:
        """Compute FFT"""
        self.compute_11("FFT", np.fft.fft2)

    @qt_try_except()
    def compute_ifft(self) -> None:
        "Compute iFFT" ""
        self.compute_11("iFFT", np.fft.ifft2)

    @staticmethod
    # pylint: disable=arguments-differ
    def func_moving_average(x: np.ndarray, p: MovingAverageParam) -> np.ndarray:
        """Moving average computing function"""
        return spi.uniform_filter(x, size=p.n, mode="constant")

    @staticmethod
    # pylint: disable=arguments-differ
    def func_moving_median(x: np.ndarray, p: MovingMedianParam) -> np.ndarray:
        """Moving median computing function"""
        return sps.medfilt(x, kernel_size=p.n)

    @qt_try_except()
    def compute_wiener(self) -> None:
        """Compute Wiener filter"""
        self.compute_11("WienerFilter", sps.wiener)

    @qt_try_except()
    def compute_denoise_tv(self, param: DenoiseTVParam | None = None) -> None:
        """Compute Total Variation denoising"""
        edit, param = self.init_param(
            param, DenoiseTVParam, _("Total variation denoising")
        )
        self.compute_11(
            "TV_Chambolle",
            lambda x, p: denoise_tv_chambolle(
                x, weight=p.weight, eps=p.eps, max_num_iter=p.max_num_iter
            ),
            param,
            suffix=lambda p: f"weight={p.weight},eps={p.eps},maxn={p.max_num_iter}",
            edit=edit,
        )

    @qt_try_except()
    def compute_denoise_bilateral(
        self, param: DenoiseBilateralParam | None = None
    ) -> None:
        """Compute bilateral filter denoising"""
        edit, param = self.init_param(
            param, DenoiseBilateralParam, _("Bilateral filtering")
        )
        self.compute_11(
            "DenoiseBilateral",
            lambda x, p: denoise_bilateral(
                x, sigma_spatial=p.sigma_spatial, mode=p.mode, cval=p.cval
            ),
            param,
            suffix=lambda p: f"σspatial={p.sigma_spatial},mode={p.mode},cval={p.cval}",
            edit=edit,
        )

    @qt_try_except()
    def compute_denoise_wavelet(self, param: DenoiseWaveletParam | None = None) -> None:
        """Compute Wavelet denoising"""
        edit, param = self.init_param(
            param, DenoiseWaveletParam, _("Wavelet denoising")
        )
        self.compute_11(
            "DenoiseWavelet",
            lambda x, p: denoise_wavelet(
                x,
                wavelet=p.wavelet,
                mode=p.mode,
                method=p.method,
            ),
            param,
            suffix=lambda p: f"wavelet={p.wavelet},mode={p.mode},method={p.method}",
            edit=edit,
        )

    @qt_try_except()
    def compute_denoise_tophat(self, param: MorphologyParam | None = None) -> None:
        """Denoise using White Top-Hat"""
        edit, param = self.init_param(param, MorphologyParam, _("Denoise / Top-Hat"))
        self.compute_11(
            "DenoiseWhiteTopHat",
            lambda x, p: x - morphology.white_tophat(x, morphology.disk(p.radius)),
            param,
            suffix=lambda p: f"radius={p.radius}",
            edit=edit,
        )

    def _morph(
        self, param: MorphologyParam | None, func: Callable, title: str, name: str
    ) -> None:
        """Compute morphological transform"""
        edit, param = self.init_param(param, MorphologyParam, title)
        self.compute_11(
            name,
            lambda x, p: func(x, morphology.disk(p.radius)),
            param,
            suffix=lambda p: f"radius={p.radius}",
            edit=edit,
        )

    @qt_try_except()
    def compute_white_tophat(self, param: MorphologyParam | None = None) -> None:
        """Compute White Top-Hat"""
        self._morph(
            param, morphology.white_tophat, _("White Top-Hat"), "WhiteTopHatDisk"
        )

    @qt_try_except()
    def compute_black_tophat(self, param: MorphologyParam | None = None) -> None:
        """Compute Black Top-Hat"""
        self._morph(
            param, morphology.black_tophat, _("Black Top-Hat"), "BlackTopHatDisk"
        )

    @qt_try_except()
    def compute_erosion(self, param: MorphologyParam | None = None) -> None:
        """Compute Erosion"""
        self._morph(param, morphology.erosion, _("Erosion"), "ErosionDisk")

    @qt_try_except()
    def compute_dilation(self, param: MorphologyParam | None = None) -> None:
        """Compute Dilation"""
        self._morph(param, morphology.dilation, _("Dilation"), "DilationDisk")

    @qt_try_except()
    def compute_opening(self, param: MorphologyParam | None = None) -> None:
        """Compute morphological opening"""
        self._morph(param, morphology.opening, _("Opening"), "OpeningDisk")

    @qt_try_except()
    def compute_closing(self, param: MorphologyParam | None = None) -> None:
        """Compute morphological closing"""
        self._morph(param, morphology.closing, _("Closing"), "ClosingDisk")

    @qt_try_except()
    def compute_canny(self, param: CannyParam | None = None) -> None:
        """Denoise using White Top-Hat"""
        edit, param = self.init_param(param, CannyParam, _("Canny filter"))
        self.compute_11(
            "Canny",
            lambda x, p: np.array(
                feature.canny(
                    x,
                    sigma=p.sigma,
                    low_threshold=p.low_threshold,
                    high_threshold=p.high_threshold,
                    use_quantiles=p.use_quantiles,
                    mode=p.mode,
                    cval=p.cval,
                ),
                dtype=np.uint8,
            ),
            param,
            suffix=lambda p: f"sigma={p.sigma},low_threshold={p.low_threshold},"
            f"high_threshold={p.high_threshold},use_quantiles={p.use_quantiles},"
            f"mode={p.mode},cval={p.cval}",
            edit=edit,
        )

    # ------Image Computing
    @qt_try_except()
    def compute_centroid(self) -> None:
        """Compute image centroid"""

        def get_centroid_coords(data: np.ndarray) -> np.ndarray:
            """Return centroid coordinates"""
            y, x = get_centroid_fourier(data)
            return np.array([(x, y)])

        def centroid(image: ImageObj) -> np.ndarray:
            """Compute centroid"""
            return calc_with_osr(image, get_centroid_coords)

        self.compute_10("Centroid", centroid, ShapeTypes.MARKER)

    @qt_try_except()
    def compute_enclosing_circle(self) -> None:
        """Compute minimum enclosing circle"""

        def get_enclosing_circle_coords(data: np.ndarray) -> np.ndarray:
            """Return diameter coords for the circle contour enclosing image
            values above threshold (FWHM)"""
            x, y, r = get_enclosing_circle(data)
            return np.array([[x - r, y, x + r, y]])

        def enclosing_circle(image: ImageObj):
            """Compute minimum enclosing circle"""
            return calc_with_osr(image, get_enclosing_circle_coords)

        # TODO: [P2] Find a way to add the circle to the computing results
        #  as in "enclosingcircle_test.py"
        self.compute_10("MinEnclosCircle", enclosing_circle, ShapeTypes.CIRCLE)

    @qt_try_except()
    def compute_peak_detection(self, param: PeakDetectionParam | None = None) -> None:
        """Compute 2D peak detection"""

        def peak_detection(image: ImageObj, p: PeakDetectionParam) -> np.ndarray:
            """Compute centroid"""
            return calc_with_osr(image, get_2d_peaks_coords, p.size, p.threshold)

        edit, param = self.init_param(param, PeakDetectionParam, _("Peak detection"))
        if edit:
            data = self.panel.objview.get_sel_objects()[0].data
            param.size = max(min(data.shape) // 40, 50)

        results = self.compute_10(
            _("Peaks"), peak_detection, ShapeTypes.POINT, param, edit=edit
        )
        if results is not None and param.create_rois and len(results.items()) > 1:
            with create_progress_bar(
                self.panel, _("Create regions of interest"), max_=len(results)
            ) as progress:
                for idx, (oid, result) in enumerate(results.items()):
                    progress.setValue(idx + 1)
                    QW.QApplication.processEvents()
                    if progress.wasCanceled():
                        break
                    obj = self.panel.objmodel[oid]
                    dist = distance_matrix(result.data)
                    dist_min = dist[dist != 0].min()
                    assert dist_min > 0
                    radius = int(0.5 * dist_min / np.sqrt(2) - 1)
                    assert radius >= 1
                    roicoords = []
                    ymax, xmax = obj.data.shape
                    for x, y in result.data:
                        coords = [
                            max(x - radius, 0),
                            max(y - radius, 0),
                            min(x + radius, xmax),
                            min(y + radius, ymax),
                        ]
                        roicoords.append(coords)
                    obj.roi = np.array(roicoords, int)
                    self.SIG_ADD_SHAPE.emit(obj.uuid)
                    self.panel.selection_changed()
                    self.panel.SIG_UPDATE_PLOT_ITEM.emit(obj.uuid)

    @qt_try_except()
    def compute_contour_shape(self, param: ContourShapeParam | None = None) -> None:
        """Compute contour shape fit"""

        def contour_shape(image: ImageObj, p: ContourShapeParam) -> np.ndarray:
            """Compute contour shape fit"""
            return calc_with_osr(image, get_contour_shapes, p.shape, p.threshold)

        edit, param = self.init_param(param, ContourShapeParam, _("Contour"))
        shapetype = ShapeTypes.CIRCLE if param.shape == "circle" else ShapeTypes.ELLIPSE
        self.compute_10("Contour", contour_shape, shapetype, param, edit=edit)

    @qt_try_except()
    def compute_hough_circle_peaks(self, param: HoughCircleParam | None = None) -> None:
        """Compute peak detection based on a circle Hough transform"""

        def hough_circles(image: ImageObj, p: HoughCircleParam) -> np.ndarray:
            """Compute Hough circles"""
            return calc_with_osr(
                image,
                get_hough_circle_peaks,
                p.min_radius,
                p.max_radius,
                None,
                p.min_distance,
            )

        edit, param = self.init_param(param, HoughCircleParam, _("Hough circles"))
        self.compute_10("Circles", hough_circles, ShapeTypes.CIRCLE, param, edit=edit)

    @qt_try_except()
    def compute_blob_dog(self, param: BlobDOGParam | None = None) -> None:
        """Compute blob detection using Difference of Gaussian method"""

        def blobs(image: ImageObj, p: BlobDOGParam) -> np.ndarray:
            """Compute blobs"""
            return calc_with_osr(
                image,
                find_blobs_dog,
                p.min_sigma,
                p.max_sigma,
                p.overlap,
                p.threshold_rel,
                p.exclude_border,
            )

        edit, param = self.init_param(param, BlobDOGParam, _("Blob detection (DOG)"))
        self.compute_10("BlobsDOG", blobs, ShapeTypes.CIRCLE, param, edit=edit)

    @qt_try_except()
    def compute_blob_doh(self, param: BlobDOHParam | None = None) -> None:
        """Compute blob detection using Determinant of Hessian method"""

        def blobs(image: ImageObj, p: BlobDOHParam) -> np.ndarray:
            """Compute blobs"""
            return calc_with_osr(
                image,
                find_blobs_doh,
                p.min_sigma,
                p.max_sigma,
                p.overlap,
                p.log_scale,
                p.threshold_rel,
            )

        edit, param = self.init_param(param, BlobDOHParam, _("Blob detection (DOH)"))
        self.compute_10("BlobsDOH", blobs, ShapeTypes.CIRCLE, param, edit=edit)

    @qt_try_except()
    def compute_blob_log(self, param: BlobLOGParam | None = None) -> None:
        """Compute blob detection using Laplacian of Gaussian method"""

        def blobs(image: ImageObj, p: BlobLOGParam) -> np.ndarray:
            """Compute blobs"""
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

        edit, param = self.init_param(param, BlobLOGParam, _("Blob detection (LOG)"))
        self.compute_10("BlobsLOG", blobs, ShapeTypes.CIRCLE, param, edit=edit)

    @qt_try_except()
    def compute_blob_opencv(self, param: BlobOpenCVParam | None = None) -> None:
        """Compute blob detection using OpenCV"""

        def blobs(image: ImageObj, p: BlobOpenCVParam) -> np.ndarray:
            """Compute blobs"""
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

        edit, param = self.init_param(
            param, BlobOpenCVParam, _("Blob detection (OpenCV)")
        )
        self.compute_10("BlobsOpenCV", blobs, ShapeTypes.CIRCLE, param, edit=edit)

    def _get_stat_funcs(self) -> list[tuple[str, Callable[[np.ndarray], float]]]:
        """Return statistics functions list"""
        # Be careful to use systematically functions adapted to masked arrays
        # (e.g. numpy.ma median, and *not* numpy.median)
        return [
            ("min(z)", lambda z: z.min()),
            ("max(z)", lambda z: z.max()),
            ("<z>", lambda z: z.mean()),
            ("Median(z)", ma.median),
            ("σ(z)", lambda z: z.std()),
            ("Σ(z)", lambda z: z.sum()),
            ("<z>/σ(z)", lambda z: z.mean() / z.std()),
        ]
