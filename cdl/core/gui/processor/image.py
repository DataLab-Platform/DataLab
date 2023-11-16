# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Image Processor
---------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from guidata.qthelpers import exec_dialog
from numpy import ma
from plotpy.widgets.resizedialog import ResizeDialog
from qtpy import QtWidgets as QW

import cdl.core.computation.base as cpb
import cdl.core.computation.image as cpi
import cdl.core.computation.image.detection as cpi_det
import cdl.core.computation.image.edges as cpi_edg
import cdl.core.computation.image.exposure as cpi_exp
import cdl.core.computation.image.morphology as cpi_mor
import cdl.core.computation.image.restoration as cpi_res
import cdl.param
from cdl.algorithms.image import distance_matrix
from cdl.config import APP_NAME, Conf, _
from cdl.core.gui.processor.base import BaseProcessor
from cdl.core.model.base import ShapeTypes
from cdl.core.model.image import ImageObj
from cdl.utils.qthelpers import create_progress_bar, qt_try_except


class ImageProcessor(BaseProcessor):
    """Object handling image processing: operations, processing, computing"""

    # pylint: disable=duplicate-code

    EDIT_ROI_PARAMS = True

    @qt_try_except()
    def compute_sum(self) -> None:
        """Compute sum"""
        self.compute_n1("Σ", cpi.compute_add, title=_("Sum"))

    @qt_try_except()
    def compute_average(self) -> None:
        """Compute average"""

        def func_objs(new_obj: ImageObj, old_objs: list[ImageObj]) -> None:
            """Finalize average computation"""
            new_obj.data = new_obj.data / float(len(old_objs))

        self.compute_n1("μ", cpi.compute_add, func_objs=func_objs, title=_("Average"))

    @qt_try_except()
    def compute_product(self) -> None:
        """Compute product"""
        self.compute_n1("Π", cpi.compute_product, title=_("Product"))

    @qt_try_except()
    def compute_logp1(self, param: cdl.param.LogP1Param | None = None) -> None:
        """Compute base 10 logarithm"""
        self.compute_11(cpi.compute_logp1, param, cpi.LogP1Param, title="Log10")

    @qt_try_except()
    def compute_rotate(self, param: cdl.param.RotateParam | None = None) -> None:
        """Rotate data arbitrarily"""
        self.compute_11(cpi.compute_rotate, param, cpi.RotateParam, title="Rotate")

    @qt_try_except()
    def compute_rotate90(self) -> None:
        """Rotate data 90°"""
        self.compute_11(cpi.compute_rotate90, title="Rotate90")

    @qt_try_except()
    def compute_rotate270(self) -> None:
        """Rotate data 270°"""
        self.compute_11(cpi.compute_rotate270, title="Rotate270")

    @qt_try_except()
    def compute_fliph(self) -> None:
        """Flip data horizontally"""
        self.compute_11(cpi.compute_fliph, title="HFlip")

    @qt_try_except()
    def compute_flipv(self) -> None:
        """Flip data vertically"""
        self.compute_11(cpi.compute_flipv, title="VFlip")

    @qt_try_except()
    def distribute_on_grid(self, param: cdl.param.GridParam | None = None) -> None:
        """Distribute images on a grid"""
        title = _("Distribute on grid")
        edit, param = self.init_param(param, cpi.GridParam, title)
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
        self.panel.SIG_REFRESH_PLOT.emit("selected", True)

    @qt_try_except()
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
        self.panel.SIG_REFRESH_PLOT.emit("selected", True)

    @qt_try_except()
    def compute_resize(self, param: cdl.param.ResizeParam | None = None) -> None:
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
        edit, param = self.init_param(param, cpi.ResizeParam, _("Resize"))
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
        self.compute_11(cpi.compute_resize, param, title=_("Resize"), edit=edit)

    @qt_try_except()
    def compute_binning(self, param: cdl.param.BinningParam | None = None) -> None:
        """Binning image"""
        edit = param is None
        obj0 = self.panel.objview.get_sel_objects(include_groups=True)[0]
        input_dtype_str = str(obj0.data.dtype)
        title = _("Binning")
        edit, param = self.init_param(param, cpi.BinningParam, title)
        if edit:
            param.dtype_str = input_dtype_str
        if param.dtype_str is None:
            param.dtype_str = input_dtype_str
        self.compute_11(cpi.compute_binning, param, title=title, edit=edit)

    @qt_try_except()
    def compute_roi_extraction(
        self, param: cdl.param.ROIDataParam | None = None
    ) -> None:
        """Extract Region Of Interest (ROI) from data"""
        param = self._get_roidataparam(param)
        if param is None or param.is_empty:
            return
        obj = self.panel.objview.get_sel_objects()[0]
        group = obj.roidata_to_params(param.roidata)
        if param.singleobj:
            self.compute_11(cpi.extract_multiple_roi, group, title=_("Extract ROI"))
        else:
            self.compute_1n(cpi.extract_single_roi, group.datasets, "ROI", edit=False)

    @qt_try_except()
    def compute_swap_axes(self) -> None:
        """Swap data axes"""
        self.compute_11(cpi.compute_swap_axes, title=_("Swap axes"))

    @qt_try_except()
    def compute_abs(self) -> None:
        """Compute absolute value"""
        self.compute_11(cpi.compute_abs, title=_("Absolute value"))

    @qt_try_except()
    def compute_log10(self) -> None:
        """Compute Log10"""
        self.compute_11(cpi.compute_log10, title="Log10")

    @qt_try_except()
    def compute_difference(self, obj2: ImageObj | None = None) -> None:
        """Compute difference between two images"""
        self.compute_n1n(
            obj2,
            _("image to subtract"),
            cpi.compute_difference,
            title=_("Difference"),
        )

    @qt_try_except()
    def compute_quadratic_difference(self, obj2: ImageObj | None = None) -> None:
        """Compute quadratic difference between two images"""
        self.compute_n1n(
            obj2,
            _("image to subtract"),
            cpi.compute_quadratic_difference,
            title=_("Quadratic difference"),
        )

    @qt_try_except()
    def compute_division(self, obj2: ImageObj | None = None) -> None:
        """Compute division between two images"""
        self.compute_n1n(
            obj2,
            _("divider"),
            cpi.compute_division,
            title=_("Division"),
        )

    @qt_try_except()
    def compute_flatfield(
        self,
        obj2: ImageObj | None = None,
        param: cdl.core.computation.param.FlatFieldParam | None = None,
    ) -> None:
        """Compute flat field correction"""
        edit, param = self.init_param(param, cpi.FlatFieldParam, _("Flat field"))
        if edit:
            obj = self.panel.objview.get_sel_objects()[0]
            param.set_from_datatype(obj.data.dtype)
        self.compute_n1n(
            obj2,
            _("flat field image"),
            cpi.compute_flatfield,
            param=param,
            title=_("Flat field correction"),
            edit=edit,
        )

    # ------Image Processing
    @qt_try_except()
    def compute_calibration(
        self, param: cdl.param.ZCalibrateParam | None = None
    ) -> None:
        """Compute data linear calibration"""
        self.compute_11(
            cpi.compute_calibration,
            param,
            cpi.ZCalibrateParam,
            _("Linear calibration"),
            "y = a.x + b",
        )

    @qt_try_except()
    def compute_threshold(self, param: cpb.ThresholdParam | None = None) -> None:
        """Compute threshold clipping"""
        self.compute_11(
            cpi.compute_threshold,
            param,
            cpb.ThresholdParam,
            _("Thresholding"),
        )

    @qt_try_except()
    def compute_clip(self, param: cpb.ClipParam | None = None) -> None:
        """Compute maximum data clipping"""
        self.compute_11(
            cpi.compute_clip,
            param,
            cpb.ClipParam,
            _("Clipping"),
        )

    @qt_try_except()
    def compute_gaussian_filter(self, param: cpb.GaussianParam | None = None) -> None:
        """Compute gaussian filter"""
        self.compute_11(
            cpi.compute_gaussian_filter, param, cpb.GaussianParam, _("Gaussian filter")
        )

    @qt_try_except()
    def compute_moving_average(
        self, param: cpb.MovingAverageParam | None = None
    ) -> None:
        """Compute moving average"""
        self.compute_11(
            cpi.compute_moving_average,
            param,
            cpb.MovingAverageParam,
            _("Moving average"),
        )

    @qt_try_except()
    def compute_moving_median(self, param: cpb.MovingMedianParam | None = None) -> None:
        """Compute moving median"""
        self.compute_11(
            cpi.compute_moving_median,
            param,
            cpb.MovingMedianParam,
            _("Moving median"),
        )

    @qt_try_except()
    def compute_wiener(self) -> None:
        """Compute Wiener filter"""
        self.compute_11(cpi.compute_wiener, title=_("Wiener filter"))

    @qt_try_except()
    def compute_fft(self, param: cdl.param.FFTParam | None = None) -> None:
        """Compute FFT"""
        if param is None:
            param = cpb.FFTParam.create(shift=Conf.proc.fft_shift_enabled.get())
        self.compute_11(cpi.compute_fft, param, title="FFT", edit=False)

    @qt_try_except()
    def compute_ifft(self, param: cdl.param.FFTParam | None = None) -> None:
        "Compute iFFT" ""
        if param is None:
            param = cpb.FFTParam.create(shift=Conf.proc.fft_shift_enabled.get())
        self.compute_11(cpi.compute_ifft, param, title="iFFT", edit=False)

    @qt_try_except()
    def compute_butterworth(
        self, param: cdl.param.ButterworthParam | None = None
    ) -> None:
        """Compute Butterworth filter"""
        self.compute_11(
            cpi.compute_butterworth,
            param,
            cpi.ButterworthParam,
            _("Butterworth filter"),
        )

    @qt_try_except()
    def compute_adjust_gamma(
        self, param: cdl.param.AdjustGammaParam | None = None
    ) -> None:
        """Compute gamma correction"""
        self.compute_11(
            cpi_exp.compute_adjust_gamma,
            param,
            cpi_exp.AdjustGammaParam,
            _("Gamma correction"),
        )

    @qt_try_except()
    def compute_adjust_log(self, param: cdl.param.AdjustLogParam | None = None) -> None:
        """Compute log correction"""
        self.compute_11(
            cpi_exp.compute_adjust_log,
            param,
            cpi_exp.AdjustLogParam,
            _("Log correction"),
        )

    @qt_try_except()
    def compute_adjust_sigmoid(
        self,
        param: cdl.param.AdjustSigmoidParam | None = None,
    ) -> None:
        """Compute sigmoid correction"""
        self.compute_11(
            cpi_exp.compute_adjust_sigmoid,
            param,
            cpi_exp.AdjustSigmoidParam,
            _("Sigmoid correction"),
        )

    @qt_try_except()
    def compute_rescale_intensity(
        self,
        param: cdl.param.RescaleIntensityParam | None = None,
    ) -> None:
        """Rescale image intensity levels"""
        self.compute_11(
            cpi_exp.compute_rescale_intensity,
            param,
            cpi_exp.RescaleIntensityParam,
            _("Rescale intensity"),
        )

    @qt_try_except()
    def compute_equalize_hist(
        self, param: cdl.param.EqualizeHistParam | None = None
    ) -> None:
        """Histogram equalization"""
        self.compute_11(
            cpi_exp.compute_equalize_hist,
            param,
            cpi_exp.EqualizeHistParam,
            _("Histogram equalization"),
        )

    @qt_try_except()
    def compute_equalize_adapthist(
        self,
        param: cdl.param.EqualizeAdaptHistParam | None = None,
    ) -> None:
        """Adaptive histogram equalization"""
        self.compute_11(
            cpi_exp.compute_equalize_adapthist,
            param,
            cpi_exp.EqualizeAdaptHistParam,
            _("Adaptive histogram equalization"),
        )

    @qt_try_except()
    def compute_denoise_tv(self, param: cdl.param.DenoiseTVParam | None = None) -> None:
        """Compute Total Variation denoising"""
        self.compute_11(
            cpi_res.compute_denoise_tv,
            param,
            cpi_res.DenoiseTVParam,
            _("Total variation denoising"),
        )

    @qt_try_except()
    def compute_denoise_bilateral(
        self,
        param: cdl.param.DenoiseBilateralParam | None = None,
    ) -> None:
        """Compute bilateral filter denoising"""
        self.compute_11(
            cpi_res.compute_denoise_bilateral,
            param,
            cpi_res.DenoiseBilateralParam,
            _("Bilateral filter denoising"),
        )

    @qt_try_except()
    def compute_denoise_wavelet(
        self,
        param: cdl.param.DenoiseWaveletParam | None = None,
    ) -> None:
        """Compute Wavelet denoising"""
        self.compute_11(
            cpi_res.compute_denoise_wavelet,
            param,
            cpi_res.DenoiseWaveletParam,
            _("Wavelet denoising"),
        )

    @qt_try_except()
    def compute_denoise_tophat(
        self, param: cdl.param.MorphologyParam | None = None
    ) -> None:
        """Denoise using White Top-Hat"""
        self.compute_11(
            cpi_res.compute_denoise_tophat,
            param,
            cpi_mor.MorphologyParam,
            _("Denoise / Top-Hat"),
        )

    @qt_try_except()
    def compute_all_denoise(self, params: list | None = None) -> None:
        """Compute all denoising filters"""
        if params is not None:
            assert len(params) == 4, "Wrong number of parameters (4 expected)"
        funcs = [
            cpi_res.compute_denoise_tv,
            cpi_res.compute_denoise_bilateral,
            cpi_res.compute_denoise_wavelet,
            cpi_res.compute_denoise_tophat,
        ]
        edit = params is None
        if edit:
            params = []
            for paramclass, title in (
                (cpi_res.DenoiseTVParam, _("Total variation denoising")),
                (cpi_res.DenoiseBilateralParam, _("Bilateral filter denoising")),
                (cpi_res.DenoiseWaveletParam, _("Wavelet denoising")),
                (cpi_mor.MorphologyParam, _("Denoise / Top-Hat")),
            ):
                param = paramclass(title)
                self.update_param_defaults(param)
                params.append(param)
        self.compute_1n(funcs, params, "Denoise", edit=edit)

    @qt_try_except()
    def compute_white_tophat(
        self, param: cdl.param.MorphologyParam | None = None
    ) -> None:
        """Compute White Top-Hat"""
        self.compute_11(
            cpi_mor.compute_white_tophat,
            param,
            cpi_mor.MorphologyParam,
            _("White Top-Hat"),
        )

    @qt_try_except()
    def compute_black_tophat(
        self, param: cdl.param.MorphologyParam | None = None
    ) -> None:
        """Compute Black Top-Hat"""
        self.compute_11(
            cpi_mor.compute_black_tophat,
            param,
            cpi_mor.MorphologyParam,
            _("Black Top-Hat"),
        )

    @qt_try_except()
    def compute_erosion(self, param: cdl.param.MorphologyParam | None = None) -> None:
        """Compute Erosion"""
        self.compute_11(
            cpi_mor.compute_erosion,
            param,
            cpi_mor.MorphologyParam,
            _("Erosion"),
        )

    @qt_try_except()
    def compute_dilation(self, param: cdl.param.MorphologyParam | None = None) -> None:
        """Compute Dilation"""
        self.compute_11(
            cpi_mor.compute_dilation,
            param,
            cpi_mor.MorphologyParam,
            _("Dilation"),
        )

    @qt_try_except()
    def compute_opening(self, param: cdl.param.MorphologyParam | None = None) -> None:
        """Compute morphological opening"""
        self.compute_11(
            cpi_mor.compute_opening,
            param,
            cpi_mor.MorphologyParam,
            _("Opening"),
        )

    @qt_try_except()
    def compute_closing(self, param: cdl.param.MorphologyParam | None = None) -> None:
        """Compute morphological closing"""
        self.compute_11(
            cpi_mor.compute_closing,
            param,
            cpi_mor.MorphologyParam,
            _("Closing"),
        )

    @qt_try_except()
    def compute_all_morphology(
        self, param: cdl.param.MorphologyParam | None = None
    ) -> None:
        """Compute all morphology filters"""
        if param is None:
            param = cpi_mor.MorphologyParam()
            if not param.edit(parent=self.panel.parent()):
                return
        funcs = [
            cpi_mor.compute_white_tophat,
            cpi_mor.compute_black_tophat,
            cpi_mor.compute_erosion,
            cpi_mor.compute_dilation,
            cpi_mor.compute_opening,
            cpi_mor.compute_closing,
        ]
        self.compute_1n(funcs, [param] * len(funcs), "Morph", edit=False)

    @qt_try_except()
    def compute_canny(self, param: cdl.param.CannyParam | None = None) -> None:
        """Compute Canny filter"""
        self.compute_11(
            cpi_edg.compute_canny,
            param,
            cpi_edg.CannyParam,
            _("Canny filter"),
        )

    @qt_try_except()
    def compute_roberts(self) -> None:
        """Compute Roberts filter"""
        self.compute_11(cpi_edg.compute_roberts, title=_("Roberts filter"))

    @qt_try_except()
    def compute_prewitt(self) -> None:
        """Compute Prewitt filter"""
        self.compute_11(cpi_edg.compute_prewitt, title=_("Prewitt filter"))

    @qt_try_except()
    def compute_prewitt_h(self) -> None:
        """Compute Prewitt filter (horizontal)"""
        self.compute_11(
            cpi_edg.compute_prewitt_h,
            title=_("Prewitt filter (horizontal)"),
        )

    @qt_try_except()
    def compute_prewitt_v(self) -> None:
        """Compute Prewitt filter (vertical)"""
        self.compute_11(
            cpi_edg.compute_prewitt_v,
            title=_("Prewitt filter (vertical)"),
        )

    @qt_try_except()
    def compute_sobel(self) -> None:
        """Compute Sobel filter"""
        self.compute_11(cpi_edg.compute_sobel, title=_("Sobel filter"))

    @qt_try_except()
    def compute_sobel_h(self) -> None:
        """Compute Sobel filter (horizontal)"""
        self.compute_11(
            cpi_edg.compute_sobel_h,
            title=_("Sobel filter (horizontal)"),
        )

    @qt_try_except()
    def compute_sobel_v(self) -> None:
        """Compute Sobel filter (vertical)"""
        self.compute_11(
            cpi_edg.compute_sobel_v,
            title=_("Sobel filter (vertical)"),
        )

    @qt_try_except()
    def compute_scharr(self) -> None:
        """Compute Scharr filter"""
        self.compute_11(cpi_edg.compute_scharr, title=_("Scharr filter"))

    @qt_try_except()
    def compute_scharr_h(self) -> None:
        """Compute Scharr filter (horizontal)"""
        self.compute_11(
            cpi_edg.compute_scharr_h,
            title=_("Scharr filter (horizontal)"),
        )

    @qt_try_except()
    def compute_scharr_v(self) -> None:
        """Compute Scharr filter (vertical)"""
        self.compute_11(
            cpi_edg.compute_scharr_v,
            title=_("Scharr filter (vertical)"),
        )

    @qt_try_except()
    def compute_farid(self) -> None:
        """Compute Farid filter"""
        self.compute_11(cpi_edg.compute_farid, title=_("Farid filter"))

    @qt_try_except()
    def compute_farid_h(self) -> None:
        """Compute Farid filter (horizontal)"""
        self.compute_11(
            cpi_edg.compute_farid_h,
            title=_("Farid filter (horizontal)"),
        )

    @qt_try_except()
    def compute_farid_v(self) -> None:
        """Compute Farid filter (vertical)"""
        self.compute_11(
            cpi_edg.compute_farid_v,
            title=_("Farid filter (vertical)"),
        )

    @qt_try_except()
    def compute_laplace(self) -> None:
        """Compute Laplace filter"""
        self.compute_11(cpi_edg.compute_laplace, title=_("Laplace filter"))

    @qt_try_except()
    def compute_all_edges(self) -> None:
        """Compute all edges"""
        funcs = [
            cpi_edg.compute_roberts,
            cpi_edg.compute_prewitt,
            cpi_edg.compute_prewitt_h,
            cpi_edg.compute_prewitt_v,
            cpi_edg.compute_sobel,
            cpi_edg.compute_sobel_h,
            cpi_edg.compute_sobel_v,
            cpi_edg.compute_scharr,
            cpi_edg.compute_scharr_h,
            cpi_edg.compute_scharr_v,
            cpi_edg.compute_farid,
            cpi_edg.compute_farid_h,
            cpi_edg.compute_farid_v,
            cpi_edg.compute_laplace,
        ]
        self.compute_1n(funcs, None, "Edges")

    # ------Image Computing
    @qt_try_except()
    def compute_centroid(self) -> None:
        """Compute image centroid"""
        self.compute_10(cpi.compute_centroid, ShapeTypes.MARKER, title=_("Centroid"))

    @qt_try_except()
    def compute_enclosing_circle(self) -> None:
        """Compute minimum enclosing circle"""
        # TODO: [P2] Find a way to add the circle to the computing results
        #  as in "enclosingcircle_test.py"
        self.compute_10(
            cpi.compute_enclosing_circle, ShapeTypes.CIRCLE, title=_("Enclosing circle")
        )

    @qt_try_except()
    def compute_peak_detection(
        self, param: cdl.param.Peak2DDetectionParam | None = None
    ) -> None:
        """Compute 2D peak detection"""
        edit, param = self.init_param(
            param, cpi_det.Peak2DDetectionParam, _("Peak detection")
        )
        if edit:
            data = self.panel.objview.get_sel_objects()[0].data
            param.size = max(min(data.shape) // 40, 50)

        results = self.compute_10(
            cpi_det.compute_peak_detection,
            ShapeTypes.POINT,
            param,
            edit=edit,
            title=_("Peak detection"),
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
                    self.panel.SIG_REFRESH_PLOT.emit(obj.uuid, True)

    @qt_try_except()
    def compute_contour_shape(
        self, param: cdl.param.ContourShapeParam | None = None
    ) -> None:
        """Compute contour shape fit"""
        edit, param = self.init_param(param, cpi_det.ContourShapeParam, _("Contour"))
        shapetype = {
            "ellipse": ShapeTypes.ELLIPSE,
            "circle": ShapeTypes.CIRCLE,
            "polygon": ShapeTypes.POLYGON,
        }[param.shape]
        self.compute_10(
            cpi_det.compute_contour_shape,
            shapetype,
            param,
            title=_("Contour"),
            edit=edit,
        )

    @qt_try_except()
    def compute_hough_circle_peaks(
        self, param: cdl.param.HoughCircleParam | None = None
    ) -> None:
        """Compute peak detection based on a circle Hough transform"""
        self.compute_10(
            cpi.compute_hough_circle_peaks,
            ShapeTypes.CIRCLE,
            param,
            cpi.HoughCircleParam,
            title=_("Hough circles"),
        )

    @qt_try_except()
    def compute_blob_dog(self, param: cdl.param.BlobDOGParam | None = None) -> None:
        """Compute blob detection using Difference of Gaussian method"""
        self.compute_10(
            cpi_det.compute_blob_dog,
            ShapeTypes.CIRCLE,
            param,
            cpi_det.BlobDOGParam,
            title=_("Blob detection (DOG)"),
        )

    @qt_try_except()
    def compute_blob_doh(self, param: cdl.param.BlobDOHParam | None = None) -> None:
        """Compute blob detection using Determinant of Hessian method"""
        self.compute_10(
            cpi_det.compute_blob_doh,
            ShapeTypes.CIRCLE,
            param,
            cpi_det.BlobDOHParam,
            title=_("Blob detection (DOH)"),
        )

    @qt_try_except()
    def compute_blob_log(self, param: cdl.param.BlobLOGParam | None = None) -> None:
        """Compute blob detection using Laplacian of Gaussian method"""
        self.compute_10(
            cpi_det.compute_blob_log,
            ShapeTypes.CIRCLE,
            param,
            cpi_det.BlobLOGParam,
            title=_("Blob detection (LOG)"),
        )

    @qt_try_except()
    def compute_blob_opencv(
        self,
        param: cdl.param.BlobOpenCVParam | None = None,
    ) -> None:
        """Compute blob detection using OpenCV"""
        self.compute_10(
            cpi_det.compute_blob_opencv,
            ShapeTypes.CIRCLE,
            param,
            cpi_det.BlobOpenCVParam,
            title=_("Blob detection (OpenCV)"),
        )

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
