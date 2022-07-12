# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT Image Processor GUI module
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...


import numpy as np
import scipy.ndimage as spi
import scipy.signal as sps
from guidata.dataset.dataitems import BoolItem, ChoiceItem, FloatItem, IntItem
from guidata.dataset.datatypes import DataSet, DataSetGroup, ValueProp
from guiqwt.widgets.resizedialog import ResizeDialog
from qtpy import QtWidgets as QW

from codraft.config import APP_NAME, _
from codraft.core.computation.image import (
    distance_matrix,
    flatfield,
    get_2d_peaks_coords,
    get_centroid_fourier,
    get_contour_shapes,
    get_enclosing_circle,
)
from codraft.core.gui.processor.base import (
    BaseProcessor,
    ClipParam,
    ThresholdParam,
)
from codraft.core.model.base import BaseProcParam, ResultShape, ShapeTypes
from codraft.core.model.image import ImageParam, RoiDataItem
from codraft.utils.qthelpers import qt_try_except


class LogP1Param(DataSet):
    """Log10 parameters"""

    n = FloatItem("n")


class RotateParam(DataSet):
    """Rotate parameters"""

    boundaries = ("constant", "nearest", "reflect", "wrap")
    prop = ValueProp(False)

    angle = FloatItem(f"{_('Angle')} (°)")
    mode = ChoiceItem(
        _("Mode"), list(zip(boundaries, boundaries)), default=boundaries[0]
    )
    cval = FloatItem(
        _("cval"),
        default=0.0,
        help=_(
            "Value used for points outside the "
            "boundaries of the input if mode is "
            "'constant'"
        ),
    )
    reshape = BoolItem(
        _("Reshape the output array"),
        default=True,
        help=_(
            "Reshape the output array "
            "so that the input array is "
            "contained completely in the output"
        ),
    )
    prefilter = BoolItem(_("Prefilter the input image"), default=True).set_prop(
        "display", store=prop
    )
    order = IntItem(
        _("Order"),
        default=3,
        min=0,
        max=5,
        help=_("Spline interpolation order"),
    ).set_prop("display", active=prop)


class ResizeParam(DataSet):
    """Resize parameters"""

    boundaries = ("constant", "nearest", "reflect", "wrap")
    prop = ValueProp(False)

    zoom = FloatItem(_("Zoom"))
    mode = ChoiceItem(
        _("Mode"), list(zip(boundaries, boundaries)), default=boundaries[0]
    )
    cval = FloatItem(
        _("cval"),
        default=0.0,
        help=_(
            "Value used for points outside the "
            "boundaries of the input if mode is "
            "'constant'"
        ),
    )
    prefilter = BoolItem(_("Prefilter the input image"), default=True).set_prop(
        "display", store=prop
    )
    order = IntItem(
        _("Order"),
        default=3,
        min=0,
        max=5,
        help=_("Spline interpolation order"),
    ).set_prop("display", active=prop)


class FlatFieldParam(BaseProcParam):
    """Flat-field parameters"""

    threshold = FloatItem(_("Threshold"), default=0.0)


class CalibrateParam(DataSet):
    """Linear calibration parameters"""

    a = FloatItem("a", default=1.0)
    b = FloatItem("b", default=0.0)


class PeakDetectionParam(DataSet):
    """Peak detection parameters"""

    size = IntItem(
        _("Neighborhoods size"),
        default=10,
        min=1,
        unit="pixels",
        help=_(
            "Size of the sliding window used in maximum/minimum filtering algorithm"
        ),
    )
    threshold = FloatItem(
        _("Relative threshold"),
        default=0.5,
        min=0.1,
        max=0.9,
        help=_(
            "Detection threshold, relative to difference between "
            "data maximum and minimum"
        ),
    )
    create_rois = BoolItem(_("Create regions of interest"))


class ContourShapeParam(DataSet):
    """Contour shape parameters"""

    shapes = (
        ("ellipse", _("Ellipse")),
        ("circle", _("Circle")),
    )

    shape = ChoiceItem(_("Shape"), shapes, default="ellipse")


class ImageProcessor(BaseProcessor):
    """Object handling image processing: operations, processing, computing"""

    # pylint: disable=duplicate-code

    EDIT_ROI_PARAMS = True

    def compute_logp1(self, param: LogP1Param = None) -> None:
        """Compute base 10 logarithm"""
        edit = param is None
        if edit:
            param = LogP1Param("Log10(z+n)")
        self.compute_11(
            "Log10(z+n)",
            lambda z, p: np.log10(z + p.n),
            param,
            suffix=lambda p: f"n={p.n}",
            edit=edit,
        )

    def rotate_arbitrarily(self, param: RotateParam = None) -> None:
        """Rotate data arbitrarily"""
        edit = param is None
        if edit:
            param = RotateParam(_("Rotation"))
        # TODO: [P2] Instead of removing geometric shapes, apply rotation
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
            func_obj=lambda obj, _param: obj.remove_resultshapes(),
            edit=edit,
        )

    def rotate_90(self):
        """Rotate data 90°"""
        # TODO: [P2] Instead of removing geometric shapes, apply 90° rotation
        self.compute_11(
            "Rotate90",
            np.rot90,
            func_obj=lambda obj: obj.remove_resultshapes(),
        )

    def rotate_270(self):
        """Rotate data 270°"""
        # TODO: [P2] Instead of removing geometric shapes, apply 270° rotation
        self.compute_11(
            "Rotate270",
            lambda x: np.rot90(x, 3),
            func_obj=lambda obj: obj.remove_resultshapes(),
        )

    def flip_horizontally(self):
        """Flip data horizontally"""
        # TODO: [P2] Instead of removing geometric shapes, apply horizontal flip
        self.compute_11(
            "HFlip",
            np.fliplr,
            func_obj=lambda obj: obj.remove_resultshapes(),
        )

    def flip_vertically(self):
        """Flip data vertically"""
        # TODO: [P2] Instead of removing geometric shapes, apply vertical flip
        self.compute_11(
            "VFlip",
            np.flipud,
            func_obj=lambda obj: obj.remove_resultshapes(),
        )

    def resize_image(self, param: ResizeParam = None) -> None:
        """Resize image"""
        obj0 = self.objlist.get_sel_object(0)
        for obj in self.objlist.get_sel_objects():
            if obj.size != obj0.size:
                QW.QMessageBox.warning(
                    self.panel.parent(),
                    APP_NAME,
                    _("Warning:")
                    + "\n"
                    + _("Selected images do not have the same size"),
                )

        edit = param is None
        if edit:
            original_size = obj0.size
            dlg = ResizeDialog(
                self.plotwidget,
                new_size=original_size,
                old_size=original_size,
                text=_("Destination size:"),
            )
            if not dlg.exec():
                return
            param = ResizeParam(_("Resize"))
            param.zoom = dlg.get_zoom()

        def func_obj(obj, param):
            """Zooming function"""
            if obj.dx is not None and obj.dy is not None:
                obj.dx, obj.dy = obj.dx / param.zoom, obj.dy / param.zoom
            # TODO: [P2] Instead of removing geometric shapes, apply zoom
            obj.remove_resultshapes()

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

    def extract_roi(self, roidata: np.ndarray = None) -> None:
        """Extract Region Of Interest (ROI) from data"""
        if roidata is None:
            roidata = self.edit_regions_of_interest(update=False)
            if roidata is None:
                return

        def suffix_func(group: DataSetGroup):
            if len(group.datasets) == 1:
                p = group.datasets[0]
                return p.get_suffix()
            return ""

        def extract_roi_func(data: np.ndarray, group: DataSetGroup):
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

        def extract_roi_func_obj(image: ImageParam, group: DataSetGroup):
            """Extract ROI function on object"""
            image.x0 += min([p.x0 for p in group.datasets])
            image.y0 += min([p.y0 for p in group.datasets])
            image.remove_resultshapes()

        obj = self.objlist.get_sel_object()
        group = obj.roidata_to_params(roidata)

        # TODO: [P2] Instead of removing geometric shapes, apply ROI extract
        self.compute_11(
            "ROI",
            extract_roi_func,
            group,
            suffix=suffix_func,
            func_obj=extract_roi_func_obj,
            edit=False,
        )

    def swap_axes(self):
        """Swap data axes"""
        self.compute_11(
            "SwapAxes",
            lambda z: z.T,
            func_obj=lambda obj: obj.remove_resultshapes(),
        )

    def compute_abs(self):
        """Compute absolute value"""
        self.compute_11("Abs", np.abs)

    def compute_log10(self):
        """Compute Log10"""
        self.compute_11("Log10", np.log10)

    @qt_try_except()
    def flat_field_correction(self, param: FlatFieldParam = None) -> None:
        """Compute flat field correction"""
        edit = param is None
        rawdata = self.objlist.get_sel_object().data
        flatdata = self.objlist.get_sel_object(1).data
        if edit:
            param = FlatFieldParam(_("Flat field"))
            param.set_from_datatype(rawdata.dtype)
        if not edit or param.edit(self.panel.parent()):
            rows = self.objlist.get_selected_rows()
            robj = self.panel.create_object()
            robj.title = (
                "FlatField("
                + (",".join([f"{self.prefix}{row:03d}" for row in rows]))
                + f",threshold={param.threshold})"
            )
            robj.data = flatfield(rawdata, flatdata, param.threshold)
            self.panel.add_object(robj)

    # ------Image Processing
    def apply_11_func(self, obj, orig, func, param, message):
        """Apply 11 function: 1 object in --> 1 object out"""

        # (self is used by @qt_try_except)
        # pylint: disable=unused-argument
        @qt_try_except(message)
        def apply_11_func_callback(self, obj, orig, func, param):
            """Apply 11 function callback: 1 object in --> 1 object out"""
            if param is None:
                obj.data = func(orig.data)
            else:
                obj.data = func(orig.data, param)

        return apply_11_func_callback(self, obj, orig, func, param)

    @qt_try_except()
    def calibrate(self, param: CalibrateParam = None) -> None:
        """Compute data linear calibration"""
        edit = param is None
        if edit:
            param = CalibrateParam(_("Linear calibration"), "y = a.x + b")
        self.compute_11(
            "LinearCal",
            lambda x, p: p.a * x + p.b,
            param,
            suffix=lambda p: "z={p.a}*z+{p.b}",
            edit=edit,
        )

    @qt_try_except()
    def compute_threshold(self, param: ThresholdParam = None) -> None:
        """Compute threshold clipping"""
        edit = param is None
        if edit:
            param = ThresholdParam(_("Thresholding"))
        self.compute_11(
            "Threshold",
            lambda x, p: np.clip(x, p.value, x.max()),
            param,
            suffix=lambda p: f"min={p.value} lsb",
            edit=edit,
        )

    @qt_try_except()
    def compute_clip(self, param: ClipParam = None) -> None:
        """Compute maximum data clipping"""
        edit = param is None
        if edit:
            param = ClipParam(_("Clipping"))
        self.compute_11(
            "Clip",
            lambda x, p: np.clip(x, x.min(), p.value),
            param,
            suffix=lambda p: f"max={p.value} lsb",
            edit=edit,
        )

    @staticmethod
    def func_gaussian_filter(x, p):  # pylint: disable=arguments-differ
        """Compute gaussian filter"""
        return spi.gaussian_filter(x, p.sigma)

    @qt_try_except()
    def compute_fft(self):
        """Compute FFT"""
        self.compute_11("FFT", np.fft.fft2)

    @qt_try_except()
    def compute_ifft(self):
        "Compute iFFT" ""
        self.compute_11("iFFT", np.fft.ifft2)

    @staticmethod
    def func_moving_average(x, p):  # pylint: disable=arguments-differ
        """Moving average computing function"""
        return spi.uniform_filter(x, size=p.n, mode="constant")

    @staticmethod
    def func_moving_median(x, p):  # pylint: disable=arguments-differ
        """Moving median computing function"""
        return sps.medfilt(x, kernel_size=p.n)

    @qt_try_except()
    def compute_wiener(self):
        """Compute Wiener filter"""
        self.compute_11("WienerFilter", sps.wiener)

    # ------Image Computing
    def apply_10_func(self, orig, func, param, message) -> ResultShape:
        """Apply 10 function: 1 object in --> 0 object out (scalar result)"""

        # (self is used by @qt_try_except)
        # pylint: disable=unused-argument
        @qt_try_except(message)
        def apply_10_func_callback(self, orig, func, param):
            """Apply 10 function cb: 1 object in --> 0 object out (scalar result)"""
            if param is None:
                return func(orig)
            return func(orig, param)

        return apply_10_func_callback(self, orig, func, param)

    @staticmethod
    def __apply_origin_size_roi(image, func) -> np.ndarray:
        """Exec computation taking into account image x0, y0, dx, dy and ROIs"""
        res = []
        for i_roi in image.iterate_roi_indexes():
            coords = func(image.get_data(i_roi))
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

    @qt_try_except()
    def compute_centroid(self):
        """Compute image centroid"""

        def get_centroid_coords(data: np.ndarray):
            """Return centroid coordinates"""
            y, x = get_centroid_fourier(data)
            return np.array([(x, y)])

        def centroid(image: ImageParam):
            """Compute centroid"""
            res = self.__apply_origin_size_roi(image, get_centroid_coords)
            if res is not None:
                return image.add_resultshape("Centroid", ShapeTypes.MARKER, res)
            return None

        self.compute_10(_("Centroid"), centroid)

    @qt_try_except()
    def compute_enclosing_circle(self):
        """Compute minimum enclosing circle"""

        def get_enclosing_circle_coords(data: np.ndarray):
            """Return diameter coords for the circle contour enclosing image
            values above threshold (FWHM)"""
            x, y, r = get_enclosing_circle(data)
            return np.array([[x - r, y, x + r, y]])

        def enclosing_circle(image: ImageParam):
            """Compute minimum enclosing circle"""
            res = self.__apply_origin_size_roi(image, get_enclosing_circle_coords)
            if res is not None:
                return image.add_resultshape("MinEnclosCircle", ShapeTypes.CIRCLE, res)
            return None

        # TODO: [P2] Find a way to add the circle to the computing results
        #  as in "enclosingcircle_test.py"
        self.compute_10(_("MinEnclosingCircle"), enclosing_circle)

    @qt_try_except()
    def compute_peak_detection(self, param: PeakDetectionParam = None) -> None:
        """Compute 2D peak detection"""

        def peak_detection(image: ImageParam, p: PeakDetectionParam):
            """Compute centroid"""
            res = self.__apply_origin_size_roi(image, get_2d_peaks_coords)
            if res is not None:
                return image.add_resultshape("Peaks", ShapeTypes.POINT, res)
            return None

        edit = param is None
        if edit:
            data = self.objlist.get_sel_object().data
            param = PeakDetectionParam()
            param.size = max(min(data.shape) // 40, 50)

        results = self.compute_10(_("Peaks"), peak_detection, param, edit=edit)
        if param.create_rois:
            for row, result in results.items():
                obj = self.objlist[row]
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
                self.SIG_ADD_SHAPE.emit(row)
                self.panel.current_item_changed(row)
                self.panel.SIG_REFRESH_PLOT.emit()

    @qt_try_except()
    def compute_contour_shape(self, param: ContourShapeParam = None) -> None:
        """Compute contour shape fit"""

        def contour_shape(image: ImageParam, p: ContourShapeParam):
            """Compute contour shape fit"""
            res = self.__apply_origin_size_roi(image, get_contour_shapes)
            if res is not None:
                shape = ShapeTypes.CIRCLE if p.shape == "circle" else ShapeTypes.ELLIPSE
                return image.add_resultshape("Contour", shape, res)
            return None

        edit = param is None
        if edit:
            param = ContourShapeParam()
        self.compute_10(_("Contour"), contour_shape, param, edit=edit)

    def _get_stat_funcs(self):
        """Return statistics functions list"""
        return [
            ("min(z)", lambda z: z.min()),
            ("max(z)", lambda z: z.max()),
            ("<z>", lambda z: z.mean()),
            ("Median(z)", np.median),
            ("σ(z)", lambda z: z.std()),
            ("Σ(z)", lambda z: z.sum()),
            ("<z>/σ(z)", lambda z: z.mean() / z.std()),
        ]
