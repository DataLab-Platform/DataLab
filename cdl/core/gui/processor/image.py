# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Image processor object (see parent package :mod:`cdl.core.gui.processor`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING

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
from cdl.config import APP_NAME, _
from cdl.core.gui.processor.base import BaseProcessor
from cdl.core.gui.profiledialog import ProfileExtractionDialog
from cdl.core.model.base import ResultShape
from cdl.core.model.image import ImageROI, ROI2DParam, create_image_roi
from cdl.utils.qthelpers import create_progress_bar, qt_try_except
from cdl.widgets import imagebackground

if TYPE_CHECKING:
    import guidata.dataset as gds


class ImageProcessor(BaseProcessor[ImageROI]):
    """Object handling image processing: operations, processing, analysis"""

    # pylint: disable=duplicate-code

    def register_computations(self) -> None:
        """Register image computations"""

        # TODO: Check if validation process has to be adapted to the new registering
        # mechanism: is it currently relying on the scanning of "compute_*" methods
        # of ImageProcessor? If that's so, it must be changed.

        # MARK: OPERATION
        self.register_n_to_1(cpi.compute_sum, _("Sum"), icon_name="sum.svg")
        self.register_n_to_1(cpi.compute_average, _("Average"), icon_name="average.svg")
        self.register_2_to_1(
            cpi.compute_difference,
            _("Difference"),
            icon_name="difference.svg",
            obj2_name=_("image to subtract"),
        )
        self.register_2_to_1(
            cpi.compute_quadratic_difference,
            _("Quadratic Difference"),
            icon_name="quadratic_difference.svg",
            obj2_name=_("image to subtract"),
        )
        self.register_n_to_1(cpi.compute_product, _("Product"), icon_name="product.svg")
        self.register_2_to_1(
            cpi.compute_division,
            _("Division"),
            icon_name="division.svg",
            obj2_name=_("divider"),
        )
        self.register_1_to_1(cpi.compute_inverse, _("Inverse"), icon_name="inverse.svg")
        self.register_2_to_1(
            cpi.compute_arithmetic,
            _("Arithmetic"),
            paramclass=cpb.ArithmeticParam,
            icon_name="arithmetic.svg",
            obj2_name=_("signal to operate with"),
        )
        self.register_1_to_1(
            cpi.compute_addition_constant,
            _("Add constant"),
            paramclass=cpb.ConstantParam,
            icon_name="constant_add.svg",
        )
        self.register_1_to_1(
            cpi.compute_difference_constant,
            _("Subtract constant"),
            paramclass=cpb.ConstantParam,
            icon_name="constant_subtract.svg",
        )
        self.register_1_to_1(
            cpi.compute_product_constant,
            _("Multiply by constant"),
            paramclass=cpb.ConstantParam,
            icon_name="constant_multiply.svg",
        )
        self.register_1_to_1(
            cpi.compute_division_constant,
            _("Divide by constant"),
            paramclass=cpb.ConstantParam,
            icon_name="constant_divide.svg",
        )
        self.register_1_to_1(cpi.compute_abs, _("Absolute value"), icon_name="abs.svg")
        self.register_1_to_1(cpi.compute_re, _("Real part"), icon_name="re.svg")
        self.register_1_to_1(cpi.compute_im, _("Imaginary part"), icon_name="im.svg")
        self.register_1_to_1(
            cpi.compute_astype,
            _("Convert data type"),
            paramclass=cdl.param.DataTypeSParam,
            icon_name="convert_dtype.svg",
        )
        self.register_1_to_1(cpi.compute_exp, _("Exponential"), icon_name="exp.svg")
        self.register_1_to_1(
            cpi.compute_log10, _("Logarithm (base 10)"), icon_name="log10.svg"
        )
        self.register_1_to_1(cpi.compute_logp1, "Log10(z+n)")
        self.register_2_to_1(
            cpi.compute_flatfield,
            _("Flat-field correction"),
            cpi.FlatFieldParam,
            obj2_name=_("flat field image"),
        )
        # Flip or rotation
        self.register_1_to_1(
            cpi.compute_fliph, _("Flip horizontally"), icon_name="flip_horizontally.svg"
        )
        self.register_1_to_1(
            cpi.compute_swap_axes, _("Flip diagonally"), icon_name="swap_x_y.svg"
        )
        self.register_1_to_1(
            cpi.compute_flipv, _("Flip vertically"), icon_name="flip_vertically.svg"
        )
        self.register_1_to_1(
            cpi.compute_rotate270,
            _("Rotate %s right") % "90°",
            icon_name="rotate_right.svg",
        )
        self.register_1_to_1(
            cpi.compute_rotate90,
            _("Rotate %s left") % "90°",
            icon_name="rotate_left.svg",
        )
        self.register_1_to_1(cpi.compute_rotate, _("Rotate by..."), cpi.RotateParam)
        # Intensity profiles
        self.register_1_to_1(
            cpi.compute_line_profile,
            _("Line profile"),
            cpi.LineProfileParam,
            icon_name="profile.svg",
            edit=False,
        )
        self.register_1_to_1(
            cpi.compute_segment_profile,
            _("Segment profile"),
            cpi.SegmentProfileParam,
            icon_name="profile_segment.svg",
            edit=False,
        )
        self.register_1_to_1(
            cpi.compute_average_profile,
            _("Average profile"),
            cpi.AverageProfileParam,
            icon_name="profile_average.svg",
            edit=False,
        )
        self.register_1_to_1(
            cpi.compute_radial_profile,
            _("Radial profile"),
            cpi.RadialProfileParam,
            icon_name="profile_radial.svg",
        )

        # MARK: PROCESSING
        # Axis transformation
        self.register_1_to_1(
            cpi.compute_calibration, _("Linear calibration"), cpi.ZCalibrateParam
        )
        self.register_1_to_1(
            cpi.compute_swap_axes, _("Swap X/Y axes"), icon_name="swap_x_y.svg"
        )
        # Level adjustment
        self.register_1_to_1(
            cpi.compute_normalize,
            _("Normalize"),
            paramclass=cpb.NormalizeParam,
            icon_name="normalize.svg",
        )
        self.register_1_to_1(cpi.compute_clip, _("Clipping"), cpi.ClipParam, "clip.svg")
        self.register_1_to_1(
            cpi.compute_offset_correction,
            _("Offset correction"),
            ROI2DParam,
            comment=_("Evaluate and subtract the offset value from the data"),
            icon_name="offset_correction.svg",
        )
        # Noise reduction
        self.register_1_to_1(
            cpi.compute_gaussian_filter, _("Gaussian filter"), cpb.GaussianParam
        )
        self.register_1_to_1(
            cpi.compute_moving_average, _("Moving average"), cpb.MovingAverageParam
        )
        self.register_1_to_1(
            cpi.compute_moving_median, _("Moving median"), cpb.MovingMedianParam
        )
        self.register_1_to_1(cpi.compute_wiener, _("Wiener filter"))
        # Fourier analysis
        self.register_1_to_1(
            cpi.compute_zero_padding,
            _("Zero padding"),
            cpi.ZeroPadding2DParam,
            comment=_(
                "Zero padding is used to increase the frequency resolution of the FFT"
            ),
        )
        self.register_1_to_1(
            cpi.compute_fft,
            _("FFT"),
            cpb.FFTParam,
            comment=_(
                "Fast Fourier Transform (FFT) is an estimation of the "
                "Discrete Fourier Transform (DFT). "
                "Results are complex numbers, but only the real part is plotted."
            ),
            edit=False,
        )
        self.register_1_to_1(
            cpi.compute_ifft,
            _("Inverse FFT"),
            cpb.FFTParam,
            comment=_(
                "Inverse Fast Fourier Transform (IFFT) is an estimation of the "
                "Inverse Discrete Fourier Transform (IDFT). "
                "Results are complex numbers, but only the real part is plotted."
            ),
            edit=False,
        )
        self.register_1_to_1(
            cpi.compute_magnitude_spectrum,
            _("Magnitude spectrum"),
            paramclass=cdl.param.SpectrumParam,
            comment=_(
                "Magnitude spectrum is the absolute value of the FFT result. "
                "It is a measure of the amplitude of the frequency components."
            ),
        )
        self.register_1_to_1(
            cpi.compute_phase_spectrum,
            _("Phase spectrum"),
            comment=_(
                "Phase spectrum is the angle of the FFT result. "
                "It is a measure of the phase of the frequency components."
            ),
        )
        self.register_1_to_1(
            cpi.compute_psd,
            _("Power spectral density"),
            paramclass=cdl.param.SpectrumParam,
            comment=_(
                "Power spectral density (PSD) is the square of the magnitude spectrum. "
                "It is a measure of the power of the frequency components."
            ),
        )
        # Thresholding
        self.register_1_to_1(
            cpi_thr.compute_threshold,
            _("Parametric thresholding"),
            cpi_thr.ThresholdParam,
            comment=_(
                "Parametric thresholding allows to select a thresholding method "
                "and a threshold value."
            ),
        )
        self.register_1_to_1(
            cpi_thr.compute_threshold_isodata, _("ISODATA thresholding")
        )
        self.register_1_to_1(cpi_thr.compute_threshold_li, _("Li thresholding"))
        self.register_1_to_1(cpi_thr.compute_threshold_mean, _("Mean thresholding"))
        self.register_1_to_1(
            cpi_thr.compute_threshold_minimum, _("Minimum thresholding")
        )
        self.register_1_to_1(cpi_thr.compute_threshold_otsu, _("Otsu thresholding"))
        self.register_1_to_1(
            cpi_thr.compute_threshold_triangle, _("Triangle thresholding")
        )
        self.register_1_to_1(cpi_thr.compute_threshold_yen, _("Li thresholding"))
        # Exposure
        self.register_1_to_1(
            cpi_exp.compute_adjust_gamma,
            _("Gamma correction"),
            cpi_exp.AdjustGammaParam,
        )
        self.register_1_to_1(
            cpi_exp.compute_adjust_log,
            _("Logarithmic correction"),
            cpi_exp.AdjustLogParam,
        )
        self.register_1_to_1(
            cpi_exp.compute_adjust_sigmoid,
            _("Sigmoid correction"),
            cpi_exp.AdjustSigmoidParam,
        )
        self.register_1_to_1(
            cpi_exp.compute_equalize_hist,
            _("Histogram equalization"),
            cpi_exp.EqualizeHistParam,
        )
        self.register_1_to_1(
            cpi_exp.compute_equalize_adapthist,
            _("Adaptive histogram equalization"),
            cpi_exp.EqualizeAdaptHistParam,
        )
        self.register_1_to_1(
            cpi_exp.compute_rescale_intensity,
            _("Intensity rescaling"),
            cpi_exp.RescaleIntensityParam,
        )
        # Restoration
        self.register_1_to_1(
            cpi_res.compute_denoise_tv,
            _("Total variation denoising"),
            cpi_res.DenoiseTVParam,
        )
        self.register_1_to_1(
            cpi_res.compute_denoise_bilateral,
            _("Bilateral filter denoising"),
            cpi_res.DenoiseBilateralParam,
        )
        self.register_1_to_1(
            cpi_res.compute_denoise_wavelet,
            _("Wavelet denoising"),
            cpi_res.DenoiseWaveletParam,
        )
        self.register_1_to_1(
            cpi_res.compute_denoise_tophat,
            _("White Top-hat denoising"),
            cpi_res.MorphologyParam,
        )
        # Morphology
        self.register_1_to_1(
            cpi_mor.compute_white_tophat,
            _("White Top-Hat (disk)"),
            cpi_mor.MorphologyParam,
        )
        self.register_1_to_1(
            cpi_mor.compute_black_tophat,
            _("Black Top-Hat (disk)"),
            cpi_mor.MorphologyParam,
        )
        self.register_1_to_1(
            cpi_mor.compute_erosion,
            _("Erosion (disk)"),
            cpi_mor.MorphologyParam,
        )
        self.register_1_to_1(
            cpi_mor.compute_dilation,
            _("Dilation (disk)"),
            cpi_mor.MorphologyParam,
        )
        self.register_1_to_1(
            cpi_mor.compute_opening,
            _("Opening (disk)"),
            cpi_mor.MorphologyParam,
        )
        self.register_1_to_1(
            cpi_mor.compute_closing,
            _("Closing (disk)"),
            cpi_mor.MorphologyParam,
        )
        # Edges
        self.register_1_to_1(cpi_edg.compute_roberts, _("Roberts filter"))
        self.register_1_to_1(cpi_edg.compute_prewitt, _("Prewitt filter"))
        self.register_1_to_1(
            cpi_edg.compute_prewitt_h, _("Prewitt filter (horizontal)")
        )
        self.register_1_to_1(cpi_edg.compute_prewitt_v, _("Prewitt filter (vertical)"))
        self.register_1_to_1(cpi_edg.compute_sobel, _("Sobel filter"))
        self.register_1_to_1(cpi_edg.compute_sobel_h, _("Sobel filter (horizontal)"))
        self.register_1_to_1(cpi_edg.compute_sobel_v, _("Sobel filter (vertical)"))
        self.register_1_to_1(cpi_edg.compute_scharr, _("Scharr filter"))
        self.register_1_to_1(cpi_edg.compute_scharr_h, _("Scharr filter (horizontal)"))
        self.register_1_to_1(cpi_edg.compute_scharr_v, _("Scharr filter (vertical)"))
        self.register_1_to_1(cpi_edg.compute_farid, _("Farid filter"))
        self.register_1_to_1(cpi_edg.compute_farid_h, _("Farid filter (horizontal)"))
        self.register_1_to_1(cpi_edg.compute_farid_v, _("Farid filter (vertical)"))
        self.register_1_to_1(cpi_edg.compute_laplace, _("Laplace filter"))
        self.register_1_to_1(
            cpi_edg.compute_canny, _("Canny filter"), cpi_edg.CannyParam
        )
        # Other processing
        self.register_1_to_1(
            cpi.compute_butterworth, _("Butterworth filter"), cpi.ButterworthParam
        )
        self.register_1_to_n(cpi.compute_extract_roi, "ROI", icon_name="roi.svg")
        self.register_1_to_1(
            cpi.compute_resize, _("Resize"), cpi.ResizeParam, icon_name="resize.svg"
        )
        self.register_1_to_1(
            cpi.compute_binning,
            _("Pixel binning"),
            cpi.BinningParam,
            icon_name="binning.svg",
        )

        # MARK: ANALYSIS
        self.register_1_to_0(cpi.compute_stats, _("Statistics"), icon_name="stats.svg")
        self.register_1_to_1(
            cpi.compute_histogram,
            _("Histogram"),
            paramclass=cpi.HistogramParam,
            icon_name="histogram.svg",
        )
        self.register_1_to_0(
            cpi.compute_centroid, _("Centroid"), comment=_("Compute image centroid")
        )
        self.register_1_to_0(
            cpi.compute_enclosing_circle,
            _("Minimum enclosing circle center"),
            comment=_("Compute smallest enclosing circle center"),
        )
        self.register_1_to_0(
            cpi_det.compute_contour_shape,
            _("Contour detection"),
            cpi_det.ContourShapeParam,
            comment=_("Compute contour shape fit"),
        )
        self.register_1_to_0(
            cpi.compute_hough_circle_peaks,
            _("Circle Hough transform"),
            cpi.HoughCircleParam,
            comment=_("Detect circular shapes using circle Hough transform"),
        )
        # Blob detection
        self.register_1_to_0(
            cpi_det.compute_blob_dog,
            _("Blob detection (DOG)"),
            cpi_det.BlobDOGParam,
            comment=_("Detect blobs using Difference of Gaussian (DOG) method"),
        )
        self.register_1_to_0(
            cpi_det.compute_blob_doh,
            _("Blob detection (DOH)"),
            cpi_det.BlobDOHParam,
            comment=_("Detect blobs using Difference of Gaussian (DOH) method"),
        )
        self.register_1_to_0(
            cpi_det.compute_blob_log,
            _("Blob detection (LOG)"),
            cpi_det.BlobLOGParam,
            comment=_("Detect blobs using Laplacian of Gaussian (LOG) method"),
        )
        self.register_1_to_0(
            cpi_det.compute_blob_opencv,
            _("Blob detection (OpenCV)"),
            cpi_det.BlobOpenCVParam,
            comment=_("Detect blobs using OpenCV SimpleBlobDetector"),
        )

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
        self.compute("resize", param, title=_("Resize"), edit=edit)

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
        self.compute("binning", param, title=title, edit=edit)

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
        self.compute("line_profile", param, title=title, edit=False)

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
        self.compute("segment_profile", param, title=title, edit=False)

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
        self.compute("average_profile", param, title=title, edit=False)

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
            param.update_from_obj(obj)
        self.compute("radial_profile", param, title=title, edit=edit)

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
                    y0 += (obj.height + param.rowspac) * sign
                    if g_row == 0:
                        g_col += 1
                        x0 += obj.width + param.colspac
                        y0 = y0_0
                else:
                    # Distributing images over columns
                    sign = np.sign(param.cols)
                    g_col = (g_col + sign) % param.cols
                    x0 += (obj.width + param.colspac) * sign
                    if g_col == 0:
                        g_row += 1
                        x0 = x0_0
                        y0 += obj.height + param.rowspac
        self.panel.refresh_plot("selected", True, False)

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
        self.panel.refresh_plot("selected", True, False)

    # ------Image Processing
    @qt_try_except()
    def compute_offset_correction(self, param: ROI2DParam | None = None) -> None:
        """Compute offset correction
        with :py:func:`cdl.computation.image.compute_offset_correction`"""
        obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
        if param is None:
            dlg = imagebackground.ImageBackgroundDialog(obj, parent=self.panel.parent())
            if exec_dialog(dlg):
                param = ROI2DParam.create(geometry="rectangle")
                x0, y0, x1, y1 = dlg.get_rect_coords()
                param.x0, param.y0, param.dx, param.dy = x0, y0, x1 - x0, y1 - y0
            else:
                return
        self.compute("offset_correction", param)

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
        self.compute_multiple_1_to_1(
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
        self.compute_multiple_1_to_1(funcs, params, "Denoise", edit=edit)

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
        self.compute_multiple_1_to_1(funcs, [param] * len(funcs), "Morph", edit=False)

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
        self.compute_multiple_1_to_1(funcs, None, "Edges")

    @qt_try_except()
    def _extract_multiple_roi_in_single_object(self, group: gds.DataSetGroup) -> None:
        """Extract multiple Regions Of Interest (ROIs) from data in a single object"""
        self.compute_1_to_1(cpi.compute_extract_rois, group, title=_("Extract ROI"))

    # ------Image Analysis
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

        results = self.compute_1_to_0(
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
                    ymax, xmax = obj.data.shape
                    coords = []
                    for x, y in result.raw_data:
                        x0, y0 = max(x - radius, 0), max(y - radius, 0)
                        dx, dy = min(x + radius, xmax) - x0, min(y + radius, ymax) - y0
                        coords.append([x0, y0, dx, dy])
                    obj.roi = create_image_roi("rectangle", coords, indices=True)
                    self.SIG_ADD_SHAPE.emit(obj.uuid)
                    self.panel.refresh_plot(obj.uuid, True, False)
        return results
