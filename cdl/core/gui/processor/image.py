# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Image processor object (see parent package :mod:`cdl.core.gui.processor`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
from guidata.qthelpers import exec_dialog
from plotpy.widgets.resizedialog import ResizeDialog
from qtpy import QtWidgets as QW

import cdl.computation.base as cpb
import cdl.computation.image as cpi
import cdl.computation.image.detection as cpi_det
import cdl.computation.image.edges as cpi_edg
import cdl.computation.image.exposure as cpi_exp
import cdl.computation.image.morphology as cpi_mor
import cdl.computation.image.restoration as cpi_res
import cdl.computation.image.threshold as cpi_thr
import cdl.param
from cdl.algorithms.image import distance_matrix
from cdl.config import APP_NAME, Conf, _
from cdl.core.gui.processor.base import BaseProcessor
from cdl.core.gui.profiledialog import ProfileExtractionDialog
from cdl.core.model.base import ResultProperties, ResultShape
from cdl.core.model.image import ImageObj, ROI2DParam, RoiDataGeometries
from cdl.utils.qthelpers import create_progress_bar, qt_try_except
from cdl.widgets import imagebackground


class ImageProcessor(BaseProcessor):
    """Object handling image processing: operations, processing, computing"""

    # pylint: disable=duplicate-code

    @qt_try_except()
    def compute_normalize(self, param: cpb.NormalizeParam | None = None) -> None:
        """Normalize data with :py:func:`cdl.computation.image.compute_normalize`"""
        self.compute_11(
            cpi.compute_normalize,
            param=param,
            paramclass=cpb.NormalizeParam,
            title=_("Normalize"),
        )

    @qt_try_except()
    def compute_sum(self) -> None:
        """Compute sum with :py:func:`cdl.computation.image.compute_addition`"""
        self.compute_n1("Σ", cpi.compute_addition, title=_("Sum"))

    @qt_try_except()
    def compute_addition_constant(
        self, param: cpb.ConstantOperationParam | None = None
    ) -> None:
        """Compute sum with a constant using
        :py:func:`cdl.computation.image.compute_addition_constant`"""
        self.compute_11(
            cpi.compute_addition_constant,
            param,
            paramclass=cpb.ConstantOperationParam,
            title=_("Add constant"),
            edit=True,
        )

    @qt_try_except()
    def compute_average(self) -> None:
        """Compute average with :py:func:`cdl.computation.image.compute_addition`
        and dividing by the number of images"""

        def func_objs(new_obj: ImageObj, old_objs: list[ImageObj]) -> None:
            """Finalize average computation"""
            new_obj.data = new_obj.data / float(len(old_objs))

        self.compute_n1(
            "μ", cpi.compute_addition, func_objs=func_objs, title=_("Average")
        )

    @qt_try_except()
    def compute_product(self) -> None:
        """Compute product with :py:func:`cdl.computation.image.compute_product`"""
        self.compute_n1("Π", cpi.compute_product, title=_("Product"))

    @qt_try_except()
    def compute_product_constant(
        self, param: cpb.ConstantOperationParam | None = None
    ) -> None:
        """Compute product with a constant using
        :py:func:`cdl.computation.image.compute_product_constant`"""
        self.compute_11(
            cpi.compute_product_constant,
            param,
            paramclass=cpb.ConstantOperationParam,
            title=_("Product with constant"),
            edit=True,
        )

    @qt_try_except()
    def compute_logp1(self, param: cdl.param.LogP1Param | None = None) -> None:
        """Compute base 10 logarithm using
        :py:func:`cdl.computation.image.compute_logp1`"""
        self.compute_11(cpi.compute_logp1, param, cpi.LogP1Param, title="Log10")

    @qt_try_except()
    def compute_rotate(self, param: cdl.param.RotateParam | None = None) -> None:
        """Rotate data arbitrarily using
        :py:func:`cdl.computation.image.compute_rotate`"""
        self.compute_11(cpi.compute_rotate, param, cpi.RotateParam, title="Rotate")

    @qt_try_except()
    def compute_rotate90(self) -> None:
        """Rotate data 90° with :py:func:`cdl.computation.image.compute_rotate90`"""
        self.compute_11(cpi.compute_rotate90, title="Rotate90")

    @qt_try_except()
    def compute_rotate270(self) -> None:
        """Rotate data 270° with :py:func:`cdl.computation.image.compute_rotate270`"""
        self.compute_11(cpi.compute_rotate270, title="Rotate270")

    @qt_try_except()
    def compute_fliph(self) -> None:
        """Flip data horizontally using
        :py:func:`cdl.computation.image.compute_fliph`"""
        self.compute_11(cpi.compute_fliph, title="HFlip")

    @qt_try_except()
    def compute_flipv(self) -> None:
        """Flip data vertically with :py:func:`cdl.computation.image.compute_flipv`"""
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
        """Resize image with :py:func:`cdl.computation.image.compute_resize`"""
        obj0 = self.panel.objview.get_sel_objects(include_groups=True)[0]
        for obj in self.panel.objview.get_sel_objects():
            if obj.data.shape != obj0.data.shape:
                QW.QMessageBox.warning(
                    self.panel.parent(),
                    APP_NAME,
                    _("Warning:")
                    + "\n"
                    + _("Selected images do not have the same size"),
                )
        edit, param = self.init_param(param, cpi.ResizeParam, _("Resize"))
        if edit:
            original_size = obj0.data.shape
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
        """Binning image with :py:func:`cdl.computation.image.compute_binning`"""
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
        """Extract Region Of Interest (ROI) from data, using:

        - :py:func:`cdl.computation.image.extract_single_roi` for single ROI
        - :py:func:`cdl.computation.image.extract_multiple_roi` for multiple ROIs"""
        param = self._get_roidataparam(param)
        if param is None or param.is_empty:
            return
        obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
        group = obj.roidata_to_params(param.roidata)
        if param.singleobj and len(group.datasets) > 1:
            # Extract multiple ROIs into a single object (remove all the ROIs),
            # if the "Extract all regions of interest into a single image object"
            # option is checked and if there are more than one ROI
            self.compute_11(cpi.extract_multiple_roi, group, title=_("Extract ROI"))
        else:
            # Extract each ROI into a separate object (keep the ROI in the case of
            # a circular ROI), if the "Extract all regions of interest into a single
            # image object" option is not checked or if there is only one ROI
            # (See Issue #31)
            self.compute_1n(cpi.extract_single_roi, group.datasets, "ROI", edit=False)

    @qt_try_except()
    def compute_line_profile(
        self, param: cdl.param.LineProfileParam | None = None
    ) -> None:
        """Compute profile along a vertical or horizontal line
        with :py:func:`cdl.computation.image.compute_line_profile`"""
        title = _("Profile")
        add_initial_shape = self.has_param_defaults(cdl.param.LineProfileParam)
        edit, param = self.init_param(param, cpi.LineProfileParam, title)
        if edit:
            options = self.panel.plothandler.get_current_plot_options()
            dlg = ProfileExtractionDialog(
                "line", param, options, self.panel.parent(), add_initial_shape
            )
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg.set_obj(obj)
            if not exec_dialog(dlg):
                return
        self.compute_11(cpi.compute_line_profile, param, title=title, edit=False)

    @qt_try_except()
    def compute_segment_profile(
        self, param: cdl.param.SegmentProfileParam | None = None
    ):
        """Compute profile along a segment
        with :py:func:`cdl.computation.image.compute_segment_profile`"""
        title = _("Profile")
        add_initial_shape = self.has_param_defaults(cdl.param.SegmentProfileParam)
        edit, param = self.init_param(param, cpi.SegmentProfileParam, title)
        if edit:
            options = self.panel.plothandler.get_current_plot_options()
            dlg = ProfileExtractionDialog(
                "segment", param, options, self.panel.parent(), add_initial_shape
            )
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg.set_obj(obj)
            if not exec_dialog(dlg):
                return
        self.compute_11(cpi.compute_segment_profile, param, title=title, edit=False)

    @qt_try_except()
    def compute_average_profile(
        self, param: cdl.param.AverageProfileParam | None = None
    ) -> None:
        """Compute average profile
        with :py:func:`cdl.computation.image.compute_average_profile`"""
        title = _("Average profile")
        add_initial_shape = self.has_param_defaults(cdl.param.AverageProfileParam)
        edit, param = self.init_param(param, cpi.AverageProfileParam, title)
        if edit:
            options = self.panel.plothandler.get_current_plot_options()
            dlg = ProfileExtractionDialog(
                "rectangle", param, options, self.panel.parent(), add_initial_shape
            )
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg.set_obj(obj)
            if not exec_dialog(dlg):
                return
        self.compute_11(cpi.compute_average_profile, param, title=title, edit=False)

    @qt_try_except()
    def compute_radial_profile(
        self, param: cdl.param.RadialProfileParam | None = None
    ) -> None:
        """Compute radial profile
        with :py:func:`cdl.computation.image.compute_radial_profile`"""
        title = _("Radial profile")
        edit, param = self.init_param(param, cpi.RadialProfileParam, title)
        if edit:
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            param.update_from_image(obj)
        self.compute_11(cpi.compute_radial_profile, param, title=title, edit=edit)

    @qt_try_except()
    def compute_histogram(self, param: cdl.param.HistogramParam | None = None) -> None:
        """Compute histogram with :py:func:`cdl.computation.image.compute_histogram`"""
        self.compute_11(
            cpi.compute_histogram, param, cpi.HistogramParam, title=_("Histogram")
        )

    @qt_try_except()
    def compute_swap_axes(self) -> None:
        """Swap data axes with :py:func:`cdl.computation.image.compute_swap_axes`"""
        self.compute_11(cpi.compute_swap_axes, title=_("Swap axes"))

    @qt_try_except()
    def compute_abs(self) -> None:
        """Compute absolute value with :py:func:`cdl.computation.image.compute_abs`"""
        self.compute_11(cpi.compute_abs, title=_("Absolute value"))

    @qt_try_except()
    def compute_re(self) -> None:
        """Compute real part with :py:func:`cdl.computation.image.compute_re`"""
        self.compute_11(cpi.compute_re, title=_("Real part"))

    @qt_try_except()
    def compute_im(self) -> None:
        """Compute imaginary part with :py:func:`cdl.computation.image.compute_im`"""
        self.compute_11(cpi.compute_im, title=_("Imaginary part"))

    @qt_try_except()
    def compute_astype(self, param: cdl.param.DataTypeIParam | None = None) -> None:
        """Convert data type with :py:func:`cdl.computation.image.compute_astype`"""
        self.compute_11(
            cpi.compute_astype, param, cpi.DataTypeIParam, title=_("Convert data type")
        )

    @qt_try_except()
    def compute_log10(self) -> None:
        """Compute Log10 with :py:func:`cdl.computation.image.compute_log10`"""
        self.compute_11(cpi.compute_log10, title="Log10")

    @qt_try_except()
    def compute_exp(self) -> None:
        """Compute Log10 with :py:func:`cdl.computation.image.compute_exp`"""
        self.compute_11(cpi.compute_exp, title=_("Exponential"))

    @qt_try_except()
    def compute_difference(self, obj2: ImageObj | None = None) -> None:
        """Compute difference between two images
        with :py:func:`cdl.computation.image.compute_difference`"""
        self.compute_n1n(
            obj2,
            _("image to subtract"),
            cpi.compute_difference,
            title=_("Difference"),
        )

    @qt_try_except()
    def compute_difference_constant(
        self, param: cpb.ConstantOperationParam | None = None
    ) -> None:
        """Compute difference with a constant
        with :py:func:`cdl.computation.image.compute_difference_constant`"""
        self.compute_11(
            cpi.compute_difference_constant,
            param,
            paramclass=cpb.ConstantOperationParam,
            title=_("Difference with constant"),
            edit=True,
        )

    @qt_try_except()
    def compute_quadratic_difference(self, obj2: ImageObj | None = None) -> None:
        """Compute quadratic difference between two images
        with :py:func:`cdl.computation.image.compute_quadratic_difference`"""
        self.compute_n1n(
            obj2,
            _("image to subtract"),
            cpi.compute_quadratic_difference,
            title=_("Quadratic difference"),
        )

    @qt_try_except()
    def compute_division(self, obj2: ImageObj | None = None) -> None:
        """Compute division between two images
        with :py:func:`cdl.computation.image.compute_division`"""
        self.compute_n1n(
            obj2,
            _("divider"),
            cpi.compute_division,
            title=_("Division"),
        )

    @qt_try_except()
    def compute_division_constant(
        self, param: cpb.ConstantOperationParam | None = None
    ) -> None:
        """Compute division by a constant
        with :py:func:`cdl.computation.image.compute_division_constant`"""
        self.compute_11(
            cpi.compute_division_constant,
            param,
            paramclass=cpb.ConstantOperationParam,
            title=_("Division by constant"),
            edit=True,
        )

    @qt_try_except()
    def compute_flatfield(
        self,
        obj2: ImageObj | None = None,
        param: cdl.param.FlatFieldParam | None = None,
    ) -> None:
        """Compute flat field correction
        with :py:func:`cdl.computation.image.compute_flatfield`"""
        edit, param = self.init_param(param, cpi.FlatFieldParam, _("Flat field"))
        if edit:
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
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
        """Compute data linear calibration
        with :py:func:`cdl.computation.image.compute_calibration`"""
        self.compute_11(
            cpi.compute_calibration,
            param,
            cpi.ZCalibrateParam,
            _("Linear calibration"),
            "y = a.x + b",
        )

    @qt_try_except()
    def compute_clip(self, param: cpb.ClipParam | None = None) -> None:
        """Compute maximum data clipping
        with :py:func:`cdl.computation.image.compute_clip`"""
        self.compute_11(
            cpi.compute_clip,
            param,
            cpb.ClipParam,
            _("Clipping"),
        )

    @qt_try_except()
    def compute_offset_correction(self, param: ROI2DParam | None = None) -> None:
        """Compute offset correction
        with :py:func:`cdl.computation.image.compute_offset_correction`"""
        obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
        if param is None:
            dlg = imagebackground.ImageBackgroundDialog(obj, parent=self.panel.parent())
            if exec_dialog(dlg):
                param = ROI2DParam.create(geometry=RoiDataGeometries.RECTANGLE)
                param.xr0, param.yr0, param.xr1, param.yr1 = dlg.get_index_range()
            else:
                return
        self.compute_11(cpi.compute_offset_correction, param)

    @qt_try_except()
    def compute_gaussian_filter(self, param: cpb.GaussianParam | None = None) -> None:
        """Compute gaussian filter
        with :py:func:`cdl.computation.image.compute_gaussian_filter`"""
        self.compute_11(
            cpi.compute_gaussian_filter, param, cpb.GaussianParam, _("Gaussian filter")
        )

    @qt_try_except()
    def compute_moving_average(
        self, param: cpb.MovingAverageParam | None = None
    ) -> None:
        """Compute moving average
        with :py:func:`cdl.computation.image.compute_moving_average`"""
        self.compute_11(
            cpi.compute_moving_average,
            param,
            cpb.MovingAverageParam,
            _("Moving average"),
        )

    @qt_try_except()
    def compute_moving_median(self, param: cpb.MovingMedianParam | None = None) -> None:
        """Compute moving median
        with :py:func:`cdl.computation.image.compute_moving_median`"""
        self.compute_11(
            cpi.compute_moving_median,
            param,
            cpb.MovingMedianParam,
            _("Moving median"),
        )

    @qt_try_except()
    def compute_wiener(self) -> None:
        """Compute Wiener filter
        with :py:func:`cdl.computation.image.compute_wiener`"""
        self.compute_11(cpi.compute_wiener, title=_("Wiener filter"))

    @qt_try_except()
    def compute_fft(self, param: cdl.param.FFTParam | None = None) -> None:
        """Compute FFT with :py:func:`cdl.computation.image.compute_fft`"""
        if param is None:
            param = cpb.FFTParam.create(shift=Conf.proc.fft_shift_enabled.get())
        self.compute_11(cpi.compute_fft, param, title="FFT", edit=False)

    @qt_try_except()
    def compute_ifft(self, param: cdl.param.FFTParam | None = None) -> None:
        """Compute iFFT with :py:func:`cdl.computation.image.compute_ifft`"""
        if param is None:
            param = cpb.FFTParam.create(shift=Conf.proc.fft_shift_enabled.get())
        self.compute_11(cpi.compute_ifft, param, title="iFFT", edit=False)

    @qt_try_except()
    def compute_magnitude_spectrum(
        self, param: cdl.param.SpectrumParam | None = None
    ) -> None:
        """Compute magnitude spectrum
        with :py:func:`cdl.computation.image.compute_magnitude_spectrum`"""
        self.compute_11(
            cpi.compute_magnitude_spectrum,
            param,
            cpi.SpectrumParam,
            _("Magnitude spectrum"),
        )

    @qt_try_except()
    def compute_phase_spectrum(self) -> None:
        """Compute phase spectrum
        with :py:func:`cdl.computation.image.compute_phase_spectrum`"""
        self.compute_11(cpi.compute_phase_spectrum, title="Phase spectrum")

    @qt_try_except()
    def compute_psd(self, param: cdl.param.SpectrumParam | None = None) -> None:
        """Compute Power Spectral Density (PSD)
        with :py:func:`cdl.computation.image.compute_psd`"""
        self.compute_11(cpi.compute_psd, param, cpi.SpectrumParam, _("PSD"))

    @qt_try_except()
    def compute_butterworth(
        self, param: cdl.param.ButterworthParam | None = None
    ) -> None:
        """Compute Butterworth filter
        with :py:func:`cdl.computation.image.compute_butterworth`"""
        self.compute_11(
            cpi.compute_butterworth,
            param,
            cpi.ButterworthParam,
            _("Butterworth filter"),
        )

    @qt_try_except()
    def compute_threshold(self, param: cdl.param.ThresholdParam | None = None) -> None:
        """Compute parametric threshold
        with :py:func:`cdl.computation.image.threshold.compute_threshold`"""
        self.compute_11(
            cpi_thr.compute_threshold,
            param,
            cpi_thr.ThresholdParam,
            _("Parametric threshold"),
        )

    @qt_try_except()
    def compute_threshold_isodata(self) -> None:
        """Compute threshold using Isodata algorithm
        with :py:func:`cdl.computation.image.threshold.compute_threshold_isodata`"""
        self.compute_11(cpi_thr.compute_threshold_isodata, title="ISODATA")

    @qt_try_except()
    def compute_threshold_li(self) -> None:
        """Compute threshold using Li algorithm
        with :py:func:`cdl.computation.image.threshold.compute_threshold_li`"""
        self.compute_11(cpi_thr.compute_threshold_li, title="Li")

    @qt_try_except()
    def compute_threshold_mean(self) -> None:
        """Compute threshold using Mean algorithm
        with :py:func:`cdl.computation.image.threshold.compute_threshold_mean`"""
        self.compute_11(cpi_thr.compute_threshold_mean, title=_("Mean"))

    @qt_try_except()
    def compute_threshold_minimum(self) -> None:
        """Compute threshold using Minimum algorithm
        with :py:func:`cdl.computation.image.threshold.compute_threshold_minimum`"""
        self.compute_11(cpi_thr.compute_threshold_minimum, title=_("Minimum"))

    @qt_try_except()
    def compute_threshold_otsu(self) -> None:
        """Compute threshold using Otsu algorithm
        with :py:func:`cdl.computation.image.threshold.compute_threshold_otsu`"""
        self.compute_11(cpi_thr.compute_threshold_otsu, title="Otsu")

    @qt_try_except()
    def compute_threshold_triangle(self) -> None:
        """Compute threshold using Triangle algorithm
        with :py:func:`cdl.computation.image.threshold.compute_threshold_triangle`"""
        self.compute_11(cpi_thr.compute_threshold_triangle, title=_("Triangle"))

    @qt_try_except()
    def compute_threshold_yen(self) -> None:
        """Compute threshold using Yen algorithm
        with :py:func:`cdl.computation.image.threshold.compute_threshold_yen`"""
        self.compute_11(cpi_thr.compute_threshold_yen, title="Yen")

    @qt_try_except()
    def compute_all_threshold(self) -> None:
        """Compute all threshold algorithms
        using the following functions:

        - :py:func:`cdl.computation.image.threshold.compute_threshold_isodata`
        - :py:func:`cdl.computation.image.threshold.compute_threshold_li`
        - :py:func:`cdl.computation.image.threshold.compute_threshold_mean`
        - :py:func:`cdl.computation.image.threshold.compute_threshold_minimum`
        - :py:func:`cdl.computation.image.threshold.compute_threshold_otsu`
        - :py:func:`cdl.computation.image.threshold.compute_threshold_triangle`
        - :py:func:`cdl.computation.image.threshold.compute_threshold_yen`
        """
        self.compute_1n(
            [
                cpi_thr.compute_threshold_isodata,
                cpi_thr.compute_threshold_li,
                cpi_thr.compute_threshold_mean,
                cpi_thr.compute_threshold_minimum,
                cpi_thr.compute_threshold_otsu,
                cpi_thr.compute_threshold_triangle,
                cpi_thr.compute_threshold_yen,
            ],
            None,
            "Threshold",
            edit=False,
        )

    @qt_try_except()
    def compute_adjust_gamma(
        self, param: cdl.param.AdjustGammaParam | None = None
    ) -> None:
        """Compute gamma correction
        with :py:func:`cdl.computation.image.exposure.compute_adjust_gamma`"""
        self.compute_11(
            cpi_exp.compute_adjust_gamma,
            param,
            cpi_exp.AdjustGammaParam,
            _("Gamma correction"),
        )

    @qt_try_except()
    def compute_adjust_log(self, param: cdl.param.AdjustLogParam | None = None) -> None:
        """Compute log correction
        with :py:func:`cdl.computation.image.exposure.compute_adjust_log`"""
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
        """Compute sigmoid correction
        with :py:func:`cdl.computation.image.exposure.compute_adjust_sigmoid`"""
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
        """Rescale image intensity levels
        with :py:func`cdl.computation.image.exposure.compute_rescale_intensity`"""
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
        """Histogram equalization
        with :py:func:`cdl.computation.image.exposure.compute_equalize_hist`"""
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
        """Adaptive histogram equalization
        with :py:func:`cdl.computation.image.exposure.compute_equalize_adapthist`"""
        self.compute_11(
            cpi_exp.compute_equalize_adapthist,
            param,
            cpi_exp.EqualizeAdaptHistParam,
            _("Adaptive histogram equalization"),
        )

    @qt_try_except()
    def compute_denoise_tv(self, param: cdl.param.DenoiseTVParam | None = None) -> None:
        """Compute Total Variation denoising
        with :py:func:`cdl.computation.image.restoration.compute_denoise_tv`"""
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
        """Compute bilateral filter denoising
        with :py:func:`cdl.computation.image.restoration.compute_denoise_bilateral`"""
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
        """Compute Wavelet denoising
        with :py:func:`cdl.computation.image.restoration.compute_denoise_wavelet`"""
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
        """Denoise using White Top-Hat
        with :py:func:`cdl.computation.image.restoration.compute_denoise_tophat`"""
        self.compute_11(
            cpi_res.compute_denoise_tophat,
            param,
            cpi_mor.MorphologyParam,
            _("Denoise / Top-Hat"),
        )

    @qt_try_except()
    def compute_all_denoise(self, params: list | None = None) -> None:
        """Compute all denoising filters
        using the following functions:

        - :py:func:`cdl.computation.image.restoration.compute_denoise_tv`
        - :py:func:`cdl.computation.image.restoration.compute_denoise_bilateral`
        - :py:func:`cdl.computation.image.restoration.compute_denoise_wavelet`
        - :py:func:`cdl.computation.image.restoration.compute_denoise_tophat`
        """
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
        """Compute White Top-Hat
        with :py:func:`cdl.computation.image.morphology.compute_white_tophat`"""
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
        """Compute Black Top-Hat
        with :py:func:`cdl.computation.image.morphology.compute_black_tophat`"""
        self.compute_11(
            cpi_mor.compute_black_tophat,
            param,
            cpi_mor.MorphologyParam,
            _("Black Top-Hat"),
        )

    @qt_try_except()
    def compute_erosion(self, param: cdl.param.MorphologyParam | None = None) -> None:
        """Compute Erosion
        with :py:func:`cdl.computation.image.morphology.compute_erosion`"""
        self.compute_11(
            cpi_mor.compute_erosion,
            param,
            cpi_mor.MorphologyParam,
            _("Erosion"),
        )

    @qt_try_except()
    def compute_dilation(self, param: cdl.param.MorphologyParam | None = None) -> None:
        """Compute Dilation
        with :py:func:`cdl.computation.image.morphology.compute_dilation`"""
        self.compute_11(
            cpi_mor.compute_dilation,
            param,
            cpi_mor.MorphologyParam,
            _("Dilation"),
        )

    @qt_try_except()
    def compute_opening(self, param: cdl.param.MorphologyParam | None = None) -> None:
        """Compute morphological opening
        with :py:func:`cdl.computation.image.morphology.compute_opening`"""
        self.compute_11(
            cpi_mor.compute_opening,
            param,
            cpi_mor.MorphologyParam,
            _("Opening"),
        )

    @qt_try_except()
    def compute_closing(self, param: cdl.param.MorphologyParam | None = None) -> None:
        """Compute morphological closing
        with :py:func:`cdl.computation.image.morphology.compute_closing`"""
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
        """Compute all morphology filters
        using the following functions:

        - :py:func:`cdl.computation.image.morphology.compute_white_tophat`
        - :py:func:`cdl.computation.image.morphology.compute_black_tophat`
        - :py:func:`cdl.computation.image.morphology.compute_erosion`
        - :py:func:`cdl.computation.image.morphology.compute_dilation`
        - :py:func:`cdl.computation.image.morphology.compute_opening`
        - :py:func:`cdl.computation.image.morphology.compute_closing`
        """
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
        """Compute Canny filter
        with :py:func:`cdl.computation.image.edges.compute_canny`"""
        self.compute_11(
            cpi_edg.compute_canny,
            param,
            cpi_edg.CannyParam,
            _("Canny filter"),
        )

    @qt_try_except()
    def compute_roberts(self) -> None:
        """Compute Roberts filter
        with :py:func:`cdl.computation.image.edges.compute_roberts`"""
        self.compute_11(cpi_edg.compute_roberts, title=_("Roberts filter"))

    @qt_try_except()
    def compute_prewitt(self) -> None:
        """Compute Prewitt filter
        with :py:func:`cdl.computation.image.edges.compute_prewitt`"""
        self.compute_11(cpi_edg.compute_prewitt, title=_("Prewitt filter"))

    @qt_try_except()
    def compute_prewitt_h(self) -> None:
        """Compute Prewitt filter (horizontal)
        with :py:func:`cdl.computation.image.edges.compute_prewitt_h`"""
        self.compute_11(
            cpi_edg.compute_prewitt_h,
            title=_("Prewitt filter (horizontal)"),
        )

    @qt_try_except()
    def compute_prewitt_v(self) -> None:
        """Compute Prewitt filter (vertical)
        with :py:func:`cdl.computation.image.edges.compute_prewitt_v`"""
        self.compute_11(
            cpi_edg.compute_prewitt_v,
            title=_("Prewitt filter (vertical)"),
        )

    @qt_try_except()
    def compute_sobel(self) -> None:
        """Compute Sobel filter
        with :py:func:`cdl.computation.image.edges.compute_sobel`"""
        self.compute_11(cpi_edg.compute_sobel, title=_("Sobel filter"))

    @qt_try_except()
    def compute_sobel_h(self) -> None:
        """Compute Sobel filter (horizontal)
        with :py:func:`cdl.computation.image.edges.compute_sobel_h`"""
        self.compute_11(
            cpi_edg.compute_sobel_h,
            title=_("Sobel filter (horizontal)"),
        )

    @qt_try_except()
    def compute_sobel_v(self) -> None:
        """Compute Sobel filter (vertical)
        with :py:func:`cdl.computation.image.edges.compute_sobel_v`"""
        self.compute_11(
            cpi_edg.compute_sobel_v,
            title=_("Sobel filter (vertical)"),
        )

    @qt_try_except()
    def compute_scharr(self) -> None:
        """Compute Scharr filter
        with :py:func:`cdl.computation.image.edges.compute_scharr`"""
        self.compute_11(cpi_edg.compute_scharr, title=_("Scharr filter"))

    @qt_try_except()
    def compute_scharr_h(self) -> None:
        """Compute Scharr filter (horizontal)
        with :py:func:`cdl.computation.image.edges.compute_scharr_h`"""
        self.compute_11(
            cpi_edg.compute_scharr_h,
            title=_("Scharr filter (horizontal)"),
        )

    @qt_try_except()
    def compute_scharr_v(self) -> None:
        """Compute Scharr filter (vertical)
        with :py:func:`cdl.computation.image.edges.compute_scharr_v`"""
        self.compute_11(
            cpi_edg.compute_scharr_v,
            title=_("Scharr filter (vertical)"),
        )

    @qt_try_except()
    def compute_farid(self) -> None:
        """Compute Farid filter
        with :py:func:`cdl.computation.image.edges.compute_farid`"""
        self.compute_11(cpi_edg.compute_farid, title=_("Farid filter"))

    @qt_try_except()
    def compute_farid_h(self) -> None:
        """Compute Farid filter (horizontal)
        with :py:func:`cdl.computation.image.edges.compute_farid_h`"""
        self.compute_11(
            cpi_edg.compute_farid_h,
            title=_("Farid filter (horizontal)"),
        )

    @qt_try_except()
    def compute_farid_v(self) -> None:
        """Compute Farid filter (vertical)
        with :py:func:`cdl.computation.image.edges.compute_farid_v`"""
        self.compute_11(
            cpi_edg.compute_farid_v,
            title=_("Farid filter (vertical)"),
        )

    @qt_try_except()
    def compute_laplace(self) -> None:
        """Compute Laplace filter
        with :py:func:`cdl.computation.image.edges.compute_laplace`"""
        self.compute_11(cpi_edg.compute_laplace, title=_("Laplace filter"))

    @qt_try_except()
    def compute_all_edges(self) -> None:
        """Compute all edges filters
        using the following functions:

        - :py:func:`cdl.computation.image.edges.compute_roberts`
        - :py:func:`cdl.computation.image.edges.compute_prewitt`
        - :py:func:`cdl.computation.image.edges.compute_prewitt_h`
        - :py:func:`cdl.computation.image.edges.compute_prewitt_v`
        - :py:func:`cdl.computation.image.edges.compute_sobel`
        - :py:func:`cdl.computation.image.edges.compute_sobel_h`
        - :py:func:`cdl.computation.image.edges.compute_sobel_v`
        - :py:func:`cdl.computation.image.edges.compute_scharr`
        - :py:func:`cdl.computation.image.edges.compute_scharr_h`
        - :py:func:`cdl.computation.image.edges.compute_scharr_v`
        - :py:func:`cdl.computation.image.edges.compute_farid`
        - :py:func:`cdl.computation.image.edges.compute_farid_h`
        - :py:func:`cdl.computation.image.edges.compute_farid_v`
        - :py:func:`cdl.computation.image.edges.compute_laplace`
        """
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
    def compute_stats(self) -> dict[str, ResultProperties]:
        """Compute data statistics
        with :py:func:`cdl.computation.image.compute_stats`"""
        return self.compute_10(cpi.compute_stats, title=_("Statistics"))

    @qt_try_except()
    def compute_centroid(self) -> dict[str, ResultShape]:
        """Compute image centroid
        with :py:func:`cdl.computation.image.compute_centroid`"""
        return self.compute_10(cpi.compute_centroid, title=_("Centroid"))

    @qt_try_except()
    def compute_enclosing_circle(self) -> dict[str, ResultShape]:
        """Compute minimum enclosing circle
        with :py:func:`cdl.computation.image.compute_enclosing_circle`"""
        return self.compute_10(
            cpi.compute_enclosing_circle, title=_("Enclosing circle")
        )

    @qt_try_except()
    def compute_peak_detection(
        self, param: cdl.param.Peak2DDetectionParam | None = None
    ) -> dict[str, ResultShape]:
        """Compute 2D peak detection
        with :py:func:`cdl.computation.image.compute_peak_detection`"""
        edit, param = self.init_param(
            param, cpi_det.Peak2DDetectionParam, _("Peak detection")
        )
        if edit:
            data = self.panel.objview.get_sel_objects(include_groups=True)[0].data
            param.size = max(min(data.shape) // 40, 50)

        results = self.compute_10(
            cpi_det.compute_peak_detection,
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
                    dist = distance_matrix(result.raw_data)
                    dist_min = dist[dist != 0].min()
                    assert dist_min > 0
                    radius = int(0.5 * dist_min / np.sqrt(2) - 1)
                    assert radius >= 1
                    roicoords = []
                    ymax, xmax = obj.data.shape
                    for x, y in result.raw_data:
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
        return results

    @qt_try_except()
    def compute_contour_shape(
        self, param: cdl.param.ContourShapeParam | None = None
    ) -> dict[str, ResultShape]:
        """Compute contour shape fit
        with :py:func:`cdl.computation.image.detection.compute_contour_shape`"""
        edit, param = self.init_param(param, cpi_det.ContourShapeParam, _("Contour"))
        return self.compute_10(
            cpi_det.compute_contour_shape,
            param=param,
            title=_("Contour"),
            edit=edit,
        )

    @qt_try_except()
    def compute_hough_circle_peaks(
        self, param: cdl.param.HoughCircleParam | None = None
    ) -> dict[str, ResultShape]:
        """Compute peak detection based on a circle Hough transform
        with :py:func:`cdl.computation.image.compute_hough_circle_peaks`"""
        return self.compute_10(
            cpi.compute_hough_circle_peaks,
            param,
            cpi.HoughCircleParam,
            title=_("Hough circles"),
        )

    @qt_try_except()
    def compute_blob_dog(
        self, param: cdl.param.BlobDOGParam | None = None
    ) -> dict[str, ResultShape]:
        """Compute blob detection using Difference of Gaussian method
        with :py:func:`cdl.computation.image.detection.compute_blob_dog`"""
        return self.compute_10(
            cpi_det.compute_blob_dog,
            param,
            cpi_det.BlobDOGParam,
            title=_("Blob detection (DOG)"),
        )

    @qt_try_except()
    def compute_blob_doh(
        self, param: cdl.param.BlobDOHParam | None = None
    ) -> dict[str, ResultShape]:
        """Compute blob detection using Determinant of Hessian method
        with :py:func:`cdl.computation.image.detection.compute_blob_doh`"""
        return self.compute_10(
            cpi_det.compute_blob_doh,
            param,
            cpi_det.BlobDOHParam,
            title=_("Blob detection (DOH)"),
        )

    @qt_try_except()
    def compute_blob_log(
        self, param: cdl.param.BlobLOGParam | None = None
    ) -> dict[str, ResultShape]:
        """Compute blob detection using Laplacian of Gaussian method
        with :py:func:`cdl.computation.image.detection.compute_blob_log`"""
        return self.compute_10(
            cpi_det.compute_blob_log,
            param,
            cpi_det.BlobLOGParam,
            title=_("Blob detection (LOG)"),
        )

    @qt_try_except()
    def compute_blob_opencv(
        self,
        param: cdl.param.BlobOpenCVParam | None = None,
    ) -> dict[str, ResultShape]:
        """Compute blob detection using OpenCV
        with :py:func:`cdl.computation.image.detection.compute_blob_opencv`"""
        return self.compute_10(
            cpi_det.compute_blob_opencv,
            param,
            cpi_det.BlobOpenCVParam,
            title=_("Blob detection (OpenCV)"),
        )
