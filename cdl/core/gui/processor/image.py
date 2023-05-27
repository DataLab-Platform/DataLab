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
from typing import Any

import guidata.dataset.datatypes as gdt
import numpy as np
import scipy.signal as sps
from guiqwt.widgets.resizedialog import ResizeDialog
from numpy import ma
from qtpy import QtWidgets as QW
from skimage import filters

import cdl.core.computation.base as cpb
import cdl.core.computation.image as cpi
from cdl.algorithms.image import distance_matrix
from cdl.config import APP_NAME, _
from cdl.core.gui.processor.base import BaseProcessor
from cdl.core.model.base import ShapeTypes
from cdl.core.model.image import ImageObj, RoiDataGeometries
from cdl.utils.qthelpers import create_progress_bar, exec_dialog, qt_try_except


class ImageProcessor(BaseProcessor):
    """Object handling image processing: operations, processing, computing"""

    # pylint: disable=duplicate-code

    EDIT_ROI_PARAMS = True

    def compute_logp1(self, param: cpi.LogP1Param | None = None) -> None:
        """Compute base 10 logarithm"""
        edit, param = self.init_param(param, cpi.LogP1Param, "Log10(z+n)")
        self.compute_11(
            "Log10(z+n)",
            cpi.log_z_plus_n,
            param,
            suffix=lambda p: f"n={p.n}",
            edit=edit,
        )

    def compute_rotate(self, param: cpi.RotateParam | None = None) -> None:
        """Rotate data arbitrarily"""
        edit, param = self.init_param(param, cpi.RotateParam, "Rotate")

        def rotate_obj(
            obj: ImageObj, orig: ImageObj, coords: np.ndarray, p: cpi.RotateParam
        ) -> None:
            """Apply rotation to coords"""
            cpi.rotate_obj_coords(p.angle, obj, orig, coords)

        self.compute_11(
            "Rotate",
            cpi.compute_rotate,
            param,
            suffix=lambda p: f"α={p.angle:.3f}°, mode='{p.mode}'",
            func_obj=lambda obj, orig, p: obj.transform_shapes(orig, rotate_obj, p),
            edit=edit,
        )

    def compute_rotate90(self) -> None:
        """Rotate data 90°"""

        def rotate_obj(obj: ImageObj, orig: ImageObj, coords: np.ndarray) -> None:
            """Apply rotation to coords"""
            cpi.rotate_obj_coords(90.0, obj, orig, coords)

        self.compute_11(
            "Rotate90",
            np.rot90,
            func_obj=lambda obj, orig: obj.transform_shapes(orig, rotate_obj),
        )

    def compute_rotate270(self) -> None:
        """Rotate data 270°"""

        def rotate_obj(obj: ImageObj, orig: ImageObj, coords: np.ndarray) -> None:
            """Apply rotation to coords"""
            cpi.rotate_obj_coords(270.0, obj, orig, coords)

        self.compute_11(
            "Rotate270",
            cpi.rotate270,
            func_obj=lambda obj, orig: obj.transform_shapes(orig, rotate_obj),
        )

    def compute_fliph(self) -> None:
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

    def compute_flipv(self) -> None:
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

    def distribute_on_grid(self, param: cpi.GridParam | None = None) -> None:
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

    def compute_resize(self, param: cpi.ResizeParam | None = None) -> None:
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

        # pylint: disable=unused-argument
        def func_obj(obj, orig: ImageObj, param: cpi.ResizeParam) -> None:
            """Zooming function"""
            if obj.dx is not None and obj.dy is not None:
                obj.dx, obj.dy = obj.dx / param.zoom, obj.dy / param.zoom
            # TODO: [P2] Instead of removing geometric shapes, apply zoom
            obj.remove_all_shapes()

        self.compute_11(
            "Zoom",
            cpi.compute_resize,
            param,
            suffix=lambda p: f"zoom={p.zoom:.3f}",
            func_obj=func_obj,
            edit=edit,
        )

    def compute_binning(self, param: cpi.BinningParam | None = None) -> None:
        """Binning image"""
        edit = param is None
        obj0 = self.panel.objview.get_sel_objects(include_groups=True)[0]
        input_dtype_str = str(obj0.data.dtype)
        edit, param = self.init_param(param, cpi.BinningParam, _("Binning"))
        if edit:
            param.dtype_str = input_dtype_str
        if param.dtype_str is None:
            param.dtype_str = input_dtype_str

        # pylint: disable=unused-argument
        def func_obj(obj: ImageObj, orig: ImageObj, param: cpi.BinningParam):
            """Binning function"""
            if param.change_pixel_size:
                if obj.dx is not None and obj.dy is not None:
                    obj.dx *= param.binning_x
                    obj.dy *= param.binning_y
                # TODO: [P2] Instead of removing geometric shapes, apply zoom
                obj.remove_all_shapes()

        self.compute_11(
            "PixelBinning",
            cpi.compute_binning,
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

            def extract_roi_func_obj(
                image: ImageObj, orig: ImageObj, group: gdt.DataSetGroup
            ):  # pylint: disable=unused-argument
                """Extract ROI function on object"""
                image.x0 += min([p.x0 for p in group.datasets])
                image.y0 += min([p.y0 for p in group.datasets])
                image.roi = None

            self.compute_11(
                "ROI",
                cpi.extract_multiple_roi,
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
                cpi.extract_single_roi,
                group.datasets,
                suffix=lambda p: p.get_suffix(),
                func_obj=extract_roi_func_obj,
                edit=False,
            )

    def compute_swap_axes(self) -> None:
        """Swap data axes"""
        self.compute_11(
            "SwapAxes",
            np.transpose,
            func_obj=lambda obj, _orig: obj.remove_all_shapes(),
        )

    def compute_abs(self) -> None:
        """Compute absolute value"""
        self.compute_11("Abs", np.abs)

    def compute_log10(self) -> None:
        """Compute Log10"""
        self.compute_11("Log10", np.log10)

    @qt_try_except()
    def compute_flatfield(
        self, obj2: ImageObj | None = None, param: cpi.FlatFieldParam | None = None
    ) -> None:
        """Compute flat field correction"""
        edit, param = self.init_param(param, cpi.FlatFieldParam, _("Flat field"))
        if edit:
            obj = self.panel.objview.get_sel_objects()[0]
            param.set_from_datatype(obj.data.dtype)
        self.compute_n1n(
            _("FlatField"),
            obj2,
            _("flat field image"),
            func=cpi.compute_flatfield,
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
    def compute_calibration(self, param: cpi.ZCalibrateParam | None = None) -> None:
        """Compute data linear calibration"""
        edit, param = self.init_param(
            param, cpi.ZCalibrateParam, _("Linear calibration"), "y = a.x + b"
        )
        self.compute_11(
            "LinearCal",
            cpi.compute_calibration,
            param,
            suffix=lambda p: "z={p.a}*z+{p.b}",
            edit=edit,
        )

    @qt_try_except()
    def compute_threshold(self, param: cpb.ThresholdParam | None = None) -> None:
        """Compute threshold clipping"""
        edit, param = self.init_param(param, cpb.ThresholdParam, _("Thresholding"))
        self.compute_11(
            "Threshold",
            cpi.compute_threshold,
            param,
            suffix=lambda p: f"min={p.value} lsb",
            edit=edit,
        )

    @qt_try_except()
    def compute_clip(self, param: cpb.ClipParam | None = None) -> None:
        """Compute maximum data clipping"""
        edit, param = self.init_param(param, cpb.ClipParam, _("Clipping"))
        self.compute_11(
            "Clip",
            cpi.compute_clip,
            param,
            suffix=lambda p: f"max={p.value} lsb",
            edit=edit,
        )

    @qt_try_except()
    def compute_adjust_gamma(self, param: cpi.AdjustGammaParam | None = None) -> None:
        """Compute gamma correction"""
        edit, param = self.init_param(
            param, cpi.AdjustGammaParam, _("Gamma correction")
        )
        self.compute_11(
            "Gamma",
            cpi.compute_adjust_gamma,
            param,
            suffix=lambda p: f"γ={p.gamma},gain={p.gain}",
            edit=edit,
        )

    @qt_try_except()
    def compute_adjust_log(self, param: cpi.AdjustLogParam | None = None) -> None:
        """Compute log correction"""
        edit, param = self.init_param(param, cpi.AdjustLogParam, _("Log correction"))
        self.compute_11(
            "Log",
            cpi.compute_adjust_log,
            param,
            suffix=lambda p: f"gain={p.gain},inv={p.inv}",
            edit=edit,
        )

    @qt_try_except()
    def compute_adjust_sigmoid(
        self, param: cpi.AdjustSigmoidParam | None = None
    ) -> None:
        """Compute sigmoid correction"""
        edit, param = self.init_param(
            param, cpi.AdjustSigmoidParam, _("Sigmoid correction")
        )
        self.compute_11(
            "Sigmoid",
            cpi.compute_adjust_sigmoid,
            param,
            suffix=lambda p: f"cutoff={p.cutoff},gain={p.gain},inv={p.inv}",
            edit=edit,
        )

    @qt_try_except()
    def compute_rescale_intensity(
        self, param: cpi.RescaleIntensityParam | None = None
    ) -> None:
        """Rescale image intensity levels"""
        edit, param = self.init_param(
            param, cpi.RescaleIntensityParam, _("Rescale intensity")
        )
        self.compute_11(
            "RescaleIntensity",
            cpi.compute_rescale_intensity,
            param,
            suffix=lambda p: f"in_range={p.in_range},out_range={p.out_range}",
            edit=edit,
        )

    @qt_try_except()
    def compute_equalize_hist(self, param: cpi.EqualizeHistParam | None = None) -> None:
        """Histogram equalization"""
        edit, param = self.init_param(
            param, cpi.EqualizeHistParam, _("Histogram equalization")
        )
        self.compute_11(
            "EqualizeHist",
            cpi.compute_equalize_hist,
            param,
            suffix=lambda p: f"nbins={p.nbins}",
            edit=edit,
        )

    @qt_try_except()
    def compute_equalize_adapthist(
        self, param: cpi.EqualizeAdaptHistParam | None = None
    ) -> None:
        """Adaptive histogram equalization"""
        edit, param = self.init_param(
            param, cpi.EqualizeAdaptHistParam, _("Adaptive histogram equalization")
        )
        self.compute_11(
            "EqualizeAdaptHist",
            cpi.compute_equalize_adapthist,
            param,
            suffix=lambda p: f"clip_limit={p.clip_limit},nbins={p.nbins}",
            edit=edit,
        )

    @qt_try_except()
    def compute_gaussian_filter(self, param: cpb.GaussianParam | None = None) -> None:
        """Compute gaussian filter"""
        edit, param = self.init_param(param, cpb.GaussianParam, _("Gaussian filter"))
        self.compute_11(
            "GaussianFilter",
            cpi.compute_gaussian_filter,
            param,
            suffix=lambda p: f"σ={p.sigma:.3f} pixels",
            edit=edit,
        )

    @qt_try_except()
    def compute_moving_average(
        self, param: cpb.MovingAverageParam | None = None
    ) -> None:
        """Compute moving average"""
        edit, param = self.init_param(
            param, cpb.MovingAverageParam, _("Moving average")
        )
        self.compute_11(
            "MovAvg",
            cpi.compute_moving_average,
            param,
            suffix=lambda p: f"n={p.n}",
            edit=edit,
        )

    @qt_try_except()
    def compute_moving_median(self, param: cpb.MovingMedianParam | None = None) -> None:
        """Compute moving median"""
        edit, param = self.init_param(param, cpb.MovingMedianParam, _("Moving median"))
        self.compute_11(
            "MovMed",
            cpi.compute_moving_median,
            param,
            suffix=lambda p: f"n={p.n}",
            edit=edit,
        )

    @qt_try_except()
    def compute_wiener(self) -> None:
        """Compute Wiener filter"""
        self.compute_11("WienerFilter", sps.wiener)

    @qt_try_except()
    def compute_fft(self) -> None:
        """Compute FFT"""
        self.compute_11("FFT", np.fft.fft2)

    @qt_try_except()
    def compute_ifft(self) -> None:
        "Compute iFFT" ""
        self.compute_11("iFFT", np.fft.ifft2)

    @qt_try_except()
    def compute_denoise_tv(self, param: cpi.DenoiseTVParam | None = None) -> None:
        """Compute Total Variation denoising"""
        edit, param = self.init_param(
            param, cpi.DenoiseTVParam, _("Total variation denoising")
        )
        self.compute_11(
            "TV_Chambolle",
            cpi.compute_denoise_tv,
            param,
            suffix=lambda p: f"weight={p.weight},eps={p.eps},maxn={p.max_num_iter}",
            edit=edit,
        )

    @qt_try_except()
    def compute_denoise_bilateral(
        self, param: cpi.DenoiseBilateralParam | None = None
    ) -> None:
        """Compute bilateral filter denoising"""
        edit, param = self.init_param(
            param, cpi.DenoiseBilateralParam, _("Bilateral filtering")
        )
        self.compute_11(
            "DenoiseBilateral",
            cpi.compute_denoise_bilateral,
            param,
            suffix=lambda p: f"σspatial={p.sigma_spatial},mode={p.mode},cval={p.cval}",
            edit=edit,
        )

    @qt_try_except()
    def compute_denoise_wavelet(
        self, param: cpi.DenoiseWaveletParam | None = None
    ) -> None:
        """Compute Wavelet denoising"""
        edit, param = self.init_param(
            param, cpi.DenoiseWaveletParam, _("Wavelet denoising")
        )
        self.compute_11(
            "DenoiseWavelet",
            cpi.compute_denoise_wavelet,
            param,
            suffix=lambda p: f"wavelet={p.wavelet},mode={p.mode},method={p.method}",
            edit=edit,
        )

    @qt_try_except()
    def compute_denoise_tophat(self, param: cpi.MorphologyParam | None = None) -> None:
        """Denoise using White Top-Hat"""
        edit, param = self.init_param(
            param, cpi.MorphologyParam, _("Denoise / Top-Hat")
        )
        self.compute_11(
            "DenoiseWhiteTopHat",
            cpi.compute_denoise_tophat,
            param,
            suffix=lambda p: f"radius={p.radius}",
            edit=edit,
        )

    @qt_try_except()
    def compute_white_tophat(self, param: cpi.MorphologyParam | None = None) -> None:
        """Compute White Top-Hat"""
        edit, param = self.init_param(param, cpi.MorphologyParam, _("White Top-Hat"))
        self.compute_11(
            "WhiteTopHatDisk",
            cpi.compute_white_tophat,
            param,
            suffix=lambda p: f"radius={p.radius}",
            edit=edit,
        )

    @qt_try_except()
    def compute_black_tophat(self, param: cpi.MorphologyParam | None = None) -> None:
        """Compute Black Top-Hat"""
        edit, param = self.init_param(param, cpi.MorphologyParam, _("Black Top-Hat"))
        self.compute_11(
            "BlackTopHatDisk",
            cpi.compute_black_tophat,
            param,
            suffix=lambda p: f"radius={p.radius}",
            edit=edit,
        )

    @qt_try_except()
    def compute_erosion(self, param: cpi.MorphologyParam | None = None) -> None:
        """Compute Erosion"""
        edit, param = self.init_param(param, cpi.MorphologyParam, _("Erosion"))
        self.compute_11(
            "ErosionDisk",
            cpi.compute_erosion,
            param,
            suffix=lambda p: f"radius={p.radius}",
            edit=edit,
        )

    @qt_try_except()
    def compute_dilation(self, param: cpi.MorphologyParam | None = None) -> None:
        """Compute Dilation"""
        edit, param = self.init_param(param, cpi.MorphologyParam, _("Dilation"))
        self.compute_11(
            "DilationDisk",
            cpi.compute_dilation,
            param,
            suffix=lambda p: f"radius={p.radius}",
            edit=edit,
        )

    @qt_try_except()
    def compute_opening(self, param: cpi.MorphologyParam | None = None) -> None:
        """Compute morphological opening"""
        edit, param = self.init_param(param, cpi.MorphologyParam, _("Opening"))
        self.compute_11(
            "OpeningDisk",
            cpi.compute_opening,
            param,
            suffix=lambda p: f"radius={p.radius}",
            edit=edit,
        )

    @qt_try_except()
    def compute_closing(self, param: cpi.MorphologyParam | None = None) -> None:
        """Compute morphological closing"""
        edit, param = self.init_param(param, cpi.MorphologyParam, _("Closing"))
        self.compute_11(
            "ClosingDisk",
            cpi.compute_closing,
            param,
            suffix=lambda p: f"radius={p.radius}",
            edit=edit,
        )

    @qt_try_except()
    def compute_butterworth(self, param: cpi.ButterworthParam | None = None) -> None:
        """Compute Butterworth filter"""
        edit, param = self.init_param(
            param, cpi.ButterworthParam, _("Butterworth filter")
        )
        self.compute_11(
            "Butterworth",
            cpi.compute_butterworth,
            param,
            suffix=lambda p: f"cut_off={p.cut_off},"
            "high_pass={p.high_pass},order={p.order}",
            edit=edit,
        )

    @qt_try_except()
    def compute_canny(self, param: cpi.CannyParam | None = None) -> None:
        """Denoise using White Top-Hat"""
        edit, param = self.init_param(param, cpi.CannyParam, _("Canny filter"))
        self.compute_11(
            "Canny",
            cpi.compute_canny,
            param,
            suffix=lambda p: f"sigma={p.sigma},low_threshold={p.low_threshold},"
            f"high_threshold={p.high_threshold},use_quantiles={p.use_quantiles},"
            f"mode={p.mode},cval={p.cval}",
            edit=edit,
        )

    @qt_try_except()
    def compute_roberts(self) -> None:
        """Compute Roberts filter"""
        self.compute_11("Roberts", filters.roberts)

    @qt_try_except()
    def compute_prewitt(self) -> None:
        """Compute Prewitt filter"""
        self.compute_11("Prewitt", filters.prewitt)

    @qt_try_except()
    def compute_prewitt_h(self) -> None:
        """Compute Prewitt filter (horizontal)"""
        self.compute_11("Prewitt_H", filters.prewitt_h)

    @qt_try_except()
    def compute_prewitt_v(self) -> None:
        """Compute Prewitt filter (vertical)"""
        self.compute_11("Prewitt_V", filters.prewitt_v)

    @qt_try_except()
    def compute_sobel(self) -> None:
        """Compute Sobel filter"""
        self.compute_11("Sobel", filters.sobel)

    @qt_try_except()
    def compute_sobel_h(self) -> None:
        """Compute Sobel filter (horizontal)"""
        self.compute_11("Sobel_H", filters.sobel_h)

    @qt_try_except()
    def compute_sobel_v(self) -> None:
        """Compute Sobel filter (vertical)"""
        self.compute_11("Sobel_V", filters.sobel_v)

    @qt_try_except()
    def compute_scharr(self) -> None:
        """Compute Scharr filter"""
        self.compute_11("Scharr", filters.scharr)

    @qt_try_except()
    def compute_scharr_h(self) -> None:
        """Compute Scharr filter (horizontal)"""
        self.compute_11("Scharr_H", filters.scharr_h)

    @qt_try_except()
    def compute_scharr_v(self) -> None:
        """Compute Scharr filter (vertical)"""
        self.compute_11("Scharr_V", filters.scharr_v)

    @qt_try_except()
    def compute_farid(self) -> None:
        """Compute Farid filter"""
        self.compute_11("Farid", filters.farid)

    @qt_try_except()
    def compute_farid_h(self) -> None:
        """Compute Farid filter (horizontal)"""
        self.compute_11("Farid_H", filters.farid_h)

    @qt_try_except()
    def compute_farid_v(self) -> None:
        """Compute Farid filter (vertical)"""
        self.compute_11("Farid_V", filters.farid_v)

    @qt_try_except()
    def compute_laplace(self) -> None:
        """Compute Laplace filter"""
        self.compute_11("Laplace", filters.laplace)

    # ------Image Computing
    @qt_try_except()
    def compute_centroid(self) -> None:
        """Compute image centroid"""
        self.compute_10("Centroid", cpi.compute_centroid, ShapeTypes.MARKER)

    @qt_try_except()
    def compute_enclosing_circle(self) -> None:
        """Compute minimum enclosing circle"""
        # TODO: [P2] Find a way to add the circle to the computing results
        #  as in "enclosingcircle_test.py"
        self.compute_10(
            "MinEnclosCircle", cpi.compute_enclosing_circle, ShapeTypes.CIRCLE
        )

    @qt_try_except()
    def compute_peak_detection(
        self, param: cpi.PeakDetectionParam | None = None
    ) -> None:
        """Compute 2D peak detection"""
        edit, param = self.init_param(
            param, cpi.PeakDetectionParam, _("Peak detection")
        )
        if edit:
            data = self.panel.objview.get_sel_objects()[0].data
            param.size = max(min(data.shape) // 40, 50)

        results = self.compute_10(
            _("Peaks"), cpi.compute_peak_detection, ShapeTypes.POINT, param, edit=edit
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
    def compute_contour_shape(self, param: cpi.ContourShapeParam | None = None) -> None:
        """Compute contour shape fit"""
        edit, param = self.init_param(param, cpi.ContourShapeParam, _("Contour"))
        shapetype = ShapeTypes.CIRCLE if param.shape == "circle" else ShapeTypes.ELLIPSE
        self.compute_10(
            "Contour", cpi.compute_contour_shape, shapetype, param, edit=edit
        )

    @qt_try_except()
    def compute_hough_circle_peaks(
        self, param: cpi.HoughCircleParam | None = None
    ) -> None:
        """Compute peak detection based on a circle Hough transform"""
        edit, param = self.init_param(param, cpi.HoughCircleParam, _("Hough circles"))
        self.compute_10(
            "Circles",
            cpi.compute_hough_circle_peaks,
            ShapeTypes.CIRCLE,
            param,
            edit=edit,
        )

    @qt_try_except()
    def compute_blob_dog(self, param: cpi.BlobDOGParam | None = None) -> None:
        """Compute blob detection using Difference of Gaussian method"""
        edit, param = self.init_param(
            param, cpi.BlobDOGParam, _("Blob detection (DOG)")
        )
        self.compute_10(
            "BlobsDOG", cpi.compute_blob_dog, ShapeTypes.CIRCLE, param, edit=edit
        )

    @qt_try_except()
    def compute_blob_doh(self, param: cpi.BlobDOHParam | None = None) -> None:
        """Compute blob detection using Determinant of Hessian method"""
        edit, param = self.init_param(
            param, cpi.BlobDOHParam, _("Blob detection (DOH)")
        )
        self.compute_10(
            "BlobsDOH", cpi.compute_blob_doh, ShapeTypes.CIRCLE, param, edit=edit
        )

    @qt_try_except()
    def compute_blob_log(self, param: cpi.BlobLOGParam | None = None) -> None:
        """Compute blob detection using Laplacian of Gaussian method"""
        edit, param = self.init_param(
            param, cpi.BlobLOGParam, _("Blob detection (LOG)")
        )
        self.compute_10(
            "BlobsLOG", cpi.compute_blob_log, ShapeTypes.CIRCLE, param, edit=edit
        )

    @qt_try_except()
    def compute_blob_opencv(self, param: cpi.BlobOpenCVParam | None = None) -> None:
        """Compute blob detection using OpenCV"""
        edit, param = self.init_param(
            param, cpi.BlobOpenCVParam, _("Blob detection (OpenCV)")
        )
        self.compute_10(
            "BlobsOpenCV", cpi.compute_blob_opencv, ShapeTypes.CIRCLE, param, edit=edit
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
