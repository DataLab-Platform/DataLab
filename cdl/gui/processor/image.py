# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Image processor object (see parent package :mod:`cdl.gui.processor`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from guidata.qthelpers import exec_dialog
from plotpy.widgets.resizedialog import ResizeDialog
from qtpy import QtWidgets as QW

import sigima_.base as sb
import sigima_.image as si
import sigima_.param
from cdl.config import APP_NAME, _
from cdl.gui.processor.base import BaseProcessor
from cdl.gui.profiledialog import ProfileExtractionDialog
from cdl.objectmodel import get_uuid
from cdl.utils.qthelpers import create_progress_bar, qt_try_except
from cdl.widgets import imagebackground
from sigima_ import ImageROI, ResultShape, ROI2DParam, create_image_roi
from sigima_.algorithms.image import distance_matrix

if TYPE_CHECKING:
    import guidata.dataset as gds


class ImageProcessor(BaseProcessor[ImageROI]):
    """Object handling image processing: operations, processing, analysis"""

    # pylint: disable=duplicate-code

    def register_computations(self) -> None:
        """Register image computations"""
        # MARK: OPERATION
        self.register_n_to_1(si.addition, _("Sum"), icon_name="sum.svg")
        self.register_n_to_1(si.average, _("Average"), icon_name="average.svg")
        self.register_2_to_1(
            si.difference,
            _("Difference"),
            icon_name="difference.svg",
            obj2_name=_("image to subtract"),
        )
        self.register_2_to_1(
            si.quadratic_difference,
            _("Quadratic Difference"),
            icon_name="quadratic_difference.svg",
            obj2_name=_("image to subtract"),
        )
        self.register_n_to_1(si.product, _("Product"), icon_name="product.svg")
        self.register_2_to_1(
            si.division,
            _("Division"),
            icon_name="division.svg",
            obj2_name=_("divider"),
        )
        self.register_1_to_1(si.inverse, _("Inverse"), icon_name="inverse.svg")
        self.register_2_to_1(
            si.arithmetic,
            _("Arithmetic"),
            paramclass=sb.ArithmeticParam,
            icon_name="arithmetic.svg",
            obj2_name=_("signal to operate with"),
        )
        self.register_1_to_1(
            si.addition_constant,
            _("Add constant"),
            paramclass=sb.ConstantParam,
            icon_name="constant_add.svg",
        )
        self.register_1_to_1(
            si.difference_constant,
            _("Subtract constant"),
            paramclass=sb.ConstantParam,
            icon_name="constant_subtract.svg",
        )
        self.register_1_to_1(
            si.product_constant,
            _("Multiply by constant"),
            paramclass=sb.ConstantParam,
            icon_name="constant_multiply.svg",
        )
        self.register_1_to_1(
            si.division_constant,
            _("Divide by constant"),
            paramclass=sb.ConstantParam,
            icon_name="constant_divide.svg",
        )
        self.register_1_to_1(si.absolute, _("Absolute value"), icon_name="abs.svg")
        self.register_1_to_1(si.real, _("Real part"), icon_name="re.svg")
        self.register_1_to_1(si.imag, _("Imaginary part"), icon_name="im.svg")
        self.register_1_to_1(
            si.astype,
            _("Convert data type"),
            paramclass=sigima_.param.DataTypeSParam,
            icon_name="convert_dtype.svg",
        )
        self.register_1_to_1(si.exp, _("Exponential"), icon_name="exp.svg")
        self.register_1_to_1(si.log10, _("Logarithm (base 10)"), icon_name="log10.svg")
        self.register_1_to_1(si.logp1, "Log10(z+n)")
        self.register_2_to_1(
            si.flatfield,
            _("Flat-field correction"),
            si.FlatFieldParam,
            obj2_name=_("flat field image"),
        )
        # Flip or rotation
        self.register_1_to_1(
            si.fliph,
            _("Flip horizontally"),
            icon_name="flip_horizontally.svg",
        )
        self.register_1_to_1(
            si.swap_axes,
            _("Flip diagonally"),
            icon_name="swap_x_y.svg",
        )
        self.register_1_to_1(
            si.flipv,
            _("Flip vertically"),
            icon_name="flip_vertically.svg",
        )
        self.register_1_to_1(
            si.rotate270,
            _("Rotate %s right") % "90°",
            icon_name="rotate_right.svg",
        )
        self.register_1_to_1(
            si.rotate90,
            _("Rotate %s left") % "90°",
            icon_name="rotate_left.svg",
        )
        self.register_1_to_1(
            si.rotate,
            _("Rotate by..."),
            si.RotateParam,
        )
        # Intensity profiles
        self.register_1_to_1(
            si.line_profile,
            _("Line profile"),
            si.LineProfileParam,
            icon_name="profile.svg",
            edit=False,
        )
        self.register_1_to_1(
            si.segment_profile,
            _("Segment profile"),
            si.SegmentProfileParam,
            icon_name="profile_segment.svg",
            edit=False,
        )
        self.register_1_to_1(
            si.average_profile,
            _("Average profile"),
            si.AverageProfileParam,
            icon_name="profile_average.svg",
            edit=False,
        )
        self.register_1_to_1(
            si.radial_profile,
            _("Radial profile"),
            si.RadialProfileParam,
            icon_name="profile_radial.svg",
        )

        # MARK: PROCESSING
        # Axis transformation
        self.register_1_to_1(
            si.calibration, _("Linear calibration"), si.ZCalibrateParam
        )
        self.register_1_to_1(
            si.swap_axes,
            _("Swap X/Y axes"),
            icon_name="swap_x_y.svg",
        )
        # Level adjustment
        self.register_1_to_1(
            si.normalize,
            _("Normalize"),
            paramclass=sb.NormalizeParam,
            icon_name="normalize.svg",
        )
        self.register_1_to_1(si.clip, _("Clipping"), sb.ClipParam, "clip.svg")
        self.register_1_to_1(
            si.offset_correction,
            _("Offset correction"),
            ROI2DParam,
            comment=_("Evaluate and subtract the offset value from the data"),
            icon_name="offset_correction.svg",
        )
        # Noise reduction
        self.register_1_to_1(
            si.gaussian_filter,
            _("Gaussian filter"),
            sb.GaussianParam,
        )
        self.register_1_to_1(
            si.moving_average,
            _("Moving average"),
            sb.MovingAverageParam,
        )
        self.register_1_to_1(
            si.moving_median,
            _("Moving median"),
            sb.MovingMedianParam,
        )
        self.register_1_to_1(si.wiener, _("Wiener filter"))
        # Fourier analysis
        self.register_1_to_1(
            si.zero_padding,
            _("Zero padding"),
            si.ZeroPadding2DParam,
            comment=_(
                "Zero padding is used to increase the frequency resolution of the FFT"
            ),
        )
        self.register_1_to_1(
            si.fft,
            _("FFT"),
            sb.FFTParam,
            comment=_(
                "Fast Fourier Transform (FFT) is an estimation of the "
                "Discrete Fourier Transform (DFT). "
                "Results are complex numbers, but only the real part is plotted."
            ),
            edit=False,
        )
        self.register_1_to_1(
            si.ifft,
            _("Inverse FFT"),
            sb.FFTParam,
            comment=_(
                "Inverse Fast Fourier Transform (IFFT) is an estimation of the "
                "Inverse Discrete Fourier Transform (IDFT). "
                "Results are complex numbers, but only the real part is plotted."
            ),
            edit=False,
        )
        self.register_1_to_1(
            si.magnitude_spectrum,
            _("Magnitude spectrum"),
            paramclass=sigima_.param.SpectrumParam,
            comment=_(
                "Magnitude spectrum is the absolute value of the FFT result. "
                "It is a measure of the amplitude of the frequency components."
            ),
        )
        self.register_1_to_1(
            si.phase_spectrum,
            _("Phase spectrum"),
            comment=_(
                "Phase spectrum is the angle of the FFT result. "
                "It is a measure of the phase of the frequency components."
            ),
        )
        self.register_1_to_1(
            si.psd,
            _("Power spectral density"),
            paramclass=sigima_.param.SpectrumParam,
            comment=_(
                "Power spectral density (PSD) is the square of the magnitude spectrum. "
                "It is a measure of the power of the frequency components."
            ),
        )
        # Thresholding
        self.register_1_to_1(
            si.threshold,
            _("Parametric thresholding"),
            si.ThresholdParam,
            comment=_(
                "Parametric thresholding allows to select a thresholding method "
                "and a threshold value."
            ),
        )
        self.register_1_to_1(si.threshold_isodata, _("ISODATA thresholding"))
        self.register_1_to_1(si.threshold_li, _("Li thresholding"))
        self.register_1_to_1(si.threshold_mean, _("Mean thresholding"))
        self.register_1_to_1(si.threshold_minimum, _("Minimum thresholding"))
        self.register_1_to_1(si.threshold_otsu, _("Otsu thresholding"))
        self.register_1_to_1(si.threshold_triangle, _("Triangle thresholding"))
        self.register_1_to_1(si.threshold_yen, _("Li thresholding"))
        # Exposure
        self.register_1_to_1(
            si.adjust_gamma,
            _("Gamma correction"),
            si.AdjustGammaParam,
        )
        self.register_1_to_1(
            si.adjust_log,
            _("Logarithmic correction"),
            si.AdjustLogParam,
        )
        self.register_1_to_1(
            si.adjust_sigmoid,
            _("Sigmoid correction"),
            si.AdjustSigmoidParam,
        )
        self.register_1_to_1(
            si.equalize_hist,
            _("Histogram equalization"),
            si.EqualizeHistParam,
        )
        self.register_1_to_1(
            si.equalize_adapthist,
            _("Adaptive histogram equalization"),
            si.EqualizeAdaptHistParam,
        )
        self.register_1_to_1(
            si.rescale_intensity,
            _("Intensity rescaling"),
            si.RescaleIntensityParam,
        )
        # Restoration
        self.register_1_to_1(
            si.denoise_tv,
            _("Total variation denoising"),
            si.DenoiseTVParam,
        )
        self.register_1_to_1(
            si.denoise_bilateral,
            _("Bilateral filter denoising"),
            si.DenoiseBilateralParam,
        )
        self.register_1_to_1(
            si.denoise_wavelet,
            _("Wavelet denoising"),
            si.DenoiseWaveletParam,
        )
        self.register_1_to_1(
            si.denoise_tophat,
            _("White Top-hat denoising"),
            si.MorphologyParam,
        )
        # Morphology
        self.register_1_to_1(
            si.white_tophat,
            _("White Top-Hat (disk)"),
            si.MorphologyParam,
        )
        self.register_1_to_1(
            si.black_tophat,
            _("Black Top-Hat (disk)"),
            si.MorphologyParam,
        )
        self.register_1_to_1(
            si.erosion,
            _("Erosion (disk)"),
            si.MorphologyParam,
        )
        self.register_1_to_1(
            si.dilation,
            _("Dilation (disk)"),
            si.MorphologyParam,
        )
        self.register_1_to_1(
            si.opening,
            _("Opening (disk)"),
            si.MorphologyParam,
        )
        self.register_1_to_1(
            si.closing,
            _("Closing (disk)"),
            si.MorphologyParam,
        )
        # Edges
        self.register_1_to_1(si.roberts, _("Roberts filter"))
        self.register_1_to_1(si.prewitt, _("Prewitt filter"))
        self.register_1_to_1(si.prewitt_h, _("Prewitt filter (horizontal)"))
        self.register_1_to_1(si.prewitt_v, _("Prewitt filter (vertical)"))
        self.register_1_to_1(si.sobel, _("Sobel filter"))
        self.register_1_to_1(si.sobel_h, _("Sobel filter (horizontal)"))
        self.register_1_to_1(si.sobel_v, _("Sobel filter (vertical)"))
        self.register_1_to_1(si.scharr, _("Scharr filter"))
        self.register_1_to_1(si.scharr_h, _("Scharr filter (horizontal)"))
        self.register_1_to_1(si.scharr_v, _("Scharr filter (vertical)"))
        self.register_1_to_1(si.farid, _("Farid filter"))
        self.register_1_to_1(si.farid_h, _("Farid filter (horizontal)"))
        self.register_1_to_1(si.farid_v, _("Farid filter (vertical)"))
        self.register_1_to_1(si.laplace, _("Laplace filter"))
        self.register_1_to_1(si.canny, _("Canny filter"), si.CannyParam)
        # Other processing
        self.register_1_to_1(
            si.butterworth,
            _("Butterworth filter"),
            si.ButterworthParam,
        )
        self.register_1_to_n(si.extract_roi, "ROI", icon_name="roi.svg")
        self.register_1_to_1(
            si.resize,
            _("Resize"),
            si.ResizeParam,
            icon_name="resize.svg",
        )
        self.register_1_to_1(
            si.binning,
            _("Pixel binning"),
            si.BinningParam,
            icon_name="binning.svg",
        )

        # MARK: ANALYSIS
        self.register_1_to_0(si.stats, _("Statistics"), icon_name="stats.svg")
        self.register_1_to_1(
            si.histogram,
            _("Histogram"),
            paramclass=sb.HistogramParam,
            icon_name="histogram.svg",
        )
        self.register_1_to_0(
            si.centroid,
            _("Centroid"),
            comment=_("Compute image centroid"),
        )
        self.register_1_to_0(
            si.enclosing_circle,
            _("Minimum enclosing circle center"),
            comment=_("Compute smallest enclosing circle center"),
        )
        self.register_1_to_0(
            si.contour_shape,
            _("Contour detection"),
            si.ContourShapeParam,
            comment=_("Compute contour shape fit"),
        )
        self.register_1_to_0(
            si.peak_detection,
            _("Peak detection"),
            si.Peak2DDetectionParam,
            comment=_("Detect peaks in the image"),
        )
        self.register_1_to_0(
            si.hough_circle_peaks,
            _("Circle Hough transform"),
            si.HoughCircleParam,
            comment=_("Detect circular shapes using circle Hough transform"),
        )
        # Blob detection
        self.register_1_to_0(
            si.blob_dog,
            _("Blob detection (DOG)"),
            si.BlobDOGParam,
            comment=_("Detect blobs using Difference of Gaussian (DOG) method"),
        )
        self.register_1_to_0(
            si.blob_doh,
            _("Blob detection (DOH)"),
            si.BlobDOHParam,
            comment=_("Detect blobs using Difference of Gaussian (DOH) method"),
        )
        self.register_1_to_0(
            si.blob_log,
            _("Blob detection (LOG)"),
            si.BlobLOGParam,
            comment=_("Detect blobs using Laplacian of Gaussian (LOG) method"),
        )
        self.register_1_to_0(
            si.blob_opencv,
            _("Blob detection (OpenCV)"),
            si.BlobOpenCVParam,
            comment=_("Detect blobs using OpenCV SimpleBlobDetector"),
        )

    @qt_try_except()
    def compute_resize(self, param: sigima_.param.ResizeParam | None = None) -> None:
        """Resize image with :py:func:`sigima_.image.resize`"""
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
        edit, param = self.init_param(param, si.ResizeParam, _("Resize"))
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
        self.run_feature("resize", param, title=_("Resize"), edit=edit)

    @qt_try_except()
    def compute_binning(self, param: sigima_.param.BinningParam | None = None) -> None:
        """Binning image with :py:func:`sigima_.image.binning`"""
        edit = param is None
        obj0 = self.panel.objview.get_sel_objects(include_groups=True)[0]
        input_dtype_str = str(obj0.data.dtype)
        title = _("Binning")
        edit, param = self.init_param(param, si.BinningParam, title)
        if edit:
            param.dtype_str = input_dtype_str
        if param.dtype_str is None:
            param.dtype_str = input_dtype_str
        self.run_feature("binning", param, title=title, edit=edit)

    @qt_try_except()
    def compute_line_profile(
        self, param: sigima_.param.LineProfileParam | None = None
    ) -> None:
        """Compute profile along a vertical or horizontal line
        with :py:func:`sigima_.image.line_profile`"""
        title = _("Profile")
        add_initial_shape = self.has_param_defaults(sigima_.param.LineProfileParam)
        edit, param = self.init_param(param, si.LineProfileParam, title)
        if edit:
            options = self.panel.plothandler.get_current_plot_options()
            dlg = ProfileExtractionDialog(
                "line", param, options, self.panel.parent(), add_initial_shape
            )
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg.set_obj(obj)
            if not exec_dialog(dlg):
                return
        self.run_feature("line_profile", param, title=title, edit=False)

    @qt_try_except()
    def compute_segment_profile(
        self, param: sigima_.param.SegmentProfileParam | None = None
    ):
        """Compute profile along a segment
        with :py:func:`sigima_.image.segment_profile`"""
        title = _("Profile")
        add_initial_shape = self.has_param_defaults(sigima_.param.SegmentProfileParam)
        edit, param = self.init_param(param, si.SegmentProfileParam, title)
        if edit:
            options = self.panel.plothandler.get_current_plot_options()
            dlg = ProfileExtractionDialog(
                "segment", param, options, self.panel.parent(), add_initial_shape
            )
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg.set_obj(obj)
            if not exec_dialog(dlg):
                return
        self.run_feature("segment_profile", param, title=title, edit=False)

    @qt_try_except()
    def compute_average_profile(
        self, param: sigima_.param.AverageProfileParam | None = None
    ) -> None:
        """Compute average profile
        with :py:func:`sigima_.image.average_profile`"""
        title = _("Average profile")
        add_initial_shape = self.has_param_defaults(sigima_.param.AverageProfileParam)
        edit, param = self.init_param(param, si.AverageProfileParam, title)
        if edit:
            options = self.panel.plothandler.get_current_plot_options()
            dlg = ProfileExtractionDialog(
                "rectangle", param, options, self.panel.parent(), add_initial_shape
            )
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg.set_obj(obj)
            if not exec_dialog(dlg):
                return
        self.run_feature("average_profile", param, title=title, edit=False)

    @qt_try_except()
    def compute_radial_profile(
        self, param: sigima_.param.RadialProfileParam | None = None
    ) -> None:
        """Compute radial profile
        with :py:func:`sigima_.image.radial_profile`"""
        title = _("Radial profile")
        edit, param = self.init_param(param, si.RadialProfileParam, title)
        if edit:
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            param.update_from_obj(obj)
        self.run_feature("radial_profile", param, title=title, edit=edit)

    @qt_try_except()
    def distribute_on_grid(self, param: sigima_.param.GridParam | None = None) -> None:
        """Distribute images on a grid"""
        title = _("Distribute on grid")
        edit, param = self.init_param(param, si.GridParam, title)
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
        with :py:func:`sigima_.image.offset_correction`"""
        obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
        if param is None:
            dlg = imagebackground.ImageBackgroundDialog(obj, parent=self.panel.parent())
            if exec_dialog(dlg):
                param = ROI2DParam.create(geometry="rectangle")
                x0, y0, x1, y1 = dlg.get_rect_coords()
                param.x0, param.y0, param.dx, param.dy = x0, y0, x1 - x0, y1 - y0
            else:
                return
        self.run_feature("offset_correction", param)

    @qt_try_except()
    def compute_all_threshold(self) -> None:
        """Compute all threshold algorithms
        using the following functions:

        - :py:func:`sigima_.image.threshold.threshold_isodata`
        - :py:func:`sigima_.image.threshold.threshold_li`
        - :py:func:`sigima_.image.threshold.threshold_mean`
        - :py:func:`sigima_.image.threshold.threshold_minimum`
        - :py:func:`sigima_.image.threshold.threshold_otsu`
        - :py:func:`sigima_.image.threshold.threshold_triangle`
        - :py:func:`sigima_.image.threshold.threshold_yen`
        """
        self.compute_multiple_1_to_1(
            [
                si.threshold_isodata,
                si.threshold_li,
                si.threshold_mean,
                si.threshold_minimum,
                si.threshold_otsu,
                si.threshold_triangle,
                si.threshold_yen,
            ],
            None,
            "Threshold",
            edit=False,
        )

    @qt_try_except()
    def compute_all_denoise(self, params: list | None = None) -> None:
        """Compute all denoising filters
        using the following functions:

        - :py:func:`sigima_.image.restoration.denoise_tv`
        - :py:func:`sigima_.image.restoration.denoise_bilateral`
        - :py:func:`sigima_.image.restoration.denoise_wavelet`
        - :py:func:`sigima_.image.restoration.denoise_tophat`
        """
        if params is not None:
            assert len(params) == 4, "Wrong number of parameters (4 expected)"
        funcs = [
            si.denoise_tv,
            si.denoise_bilateral,
            si.denoise_wavelet,
            si.denoise_tophat,
        ]
        edit = params is None
        if edit:
            params = []
            for paramclass, title in (
                (si.DenoiseTVParam, _("Total variation denoising")),
                (si.DenoiseBilateralParam, _("Bilateral filter denoising")),
                (si.DenoiseWaveletParam, _("Wavelet denoising")),
                (si.MorphologyParam, _("Denoise / Top-Hat")),
            ):
                param = paramclass(title)
                self.update_param_defaults(param)
                params.append(param)
        self.compute_multiple_1_to_1(funcs, params, "Denoise", edit=edit)

    @qt_try_except()
    def compute_all_morphology(
        self, param: sigima_.param.MorphologyParam | None = None
    ) -> None:
        """Compute all morphology filters
        using the following functions:

        - :py:func:`sigima_.image.morphology.white_tophat`
        - :py:func:`sigima_.image.morphology.black_tophat`
        - :py:func:`sigima_.image.morphology.erosion`
        - :py:func:`sigima_.image.morphology.dilation`
        - :py:func:`sigima_.image.morphology.opening`
        - :py:func:`sigima_.image.morphology.closing`
        """
        if param is None:
            param = si.MorphologyParam()
            if not param.edit(parent=self.panel.parent()):
                return
        funcs = [
            si.white_tophat,
            si.black_tophat,
            si.erosion,
            si.dilation,
            si.opening,
            si.closing,
        ]
        self.compute_multiple_1_to_1(funcs, [param] * len(funcs), "Morph", edit=False)

    @qt_try_except()
    def compute_all_edges(self) -> None:
        """Compute all edges filters
        using the following functions:

        - :py:func:`sigima_.image.edges.roberts`
        - :py:func:`sigima_.image.edges.prewitt`
        - :py:func:`sigima_.image.edges.prewitt_h`
        - :py:func:`sigima_.image.edges.prewitt_v`
        - :py:func:`sigima_.image.edges.sobel`
        - :py:func:`sigima_.image.edges.sobel_h`
        - :py:func:`sigima_.image.edges.sobel_v`
        - :py:func:`sigima_.image.edges.scharr`
        - :py:func:`sigima_.image.edges.scharr_h`
        - :py:func:`sigima_.image.edges.scharr_v`
        - :py:func:`sigima_.image.edges.farid`
        - :py:func:`sigima_.image.edges.farid_h`
        - :py:func:`sigima_.image.edges.farid_v`
        - :py:func:`sigima_.image.edges.laplace`
        """
        funcs = [
            si.roberts,
            si.prewitt,
            si.prewitt_h,
            si.prewitt_v,
            si.sobel,
            si.sobel_h,
            si.sobel_v,
            si.scharr,
            si.scharr_h,
            si.scharr_v,
            si.farid,
            si.farid_h,
            si.farid_v,
            si.laplace,
        ]
        self.compute_multiple_1_to_1(funcs, None, "Edges")

    @qt_try_except()
    def _extract_multiple_roi_in_single_object(self, group: gds.DataSetGroup) -> None:
        """Extract multiple Regions Of Interest (ROIs) from data in a single object"""
        self.compute_1_to_1(si.extract_rois, group, title=_("Extract ROI"))

    # ------Image Analysis
    @qt_try_except()
    def compute_peak_detection(
        self, param: sigima_.param.Peak2DDetectionParam | None = None
    ) -> dict[str, ResultShape]:
        """Compute 2D peak detection
        with :py:func:`sigima_.image.peak_detection`"""
        edit, param = self.init_param(
            param, si.Peak2DDetectionParam, _("Peak detection")
        )
        if edit:
            data = self.panel.objview.get_sel_objects(include_groups=True)[0].data
            param.size = max(min(data.shape) // 40, 50)

        results = self.run_feature("peak_detection", param, edit=edit)
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
                    self.SIG_ADD_SHAPE.emit(get_uuid(obj))
                    self.panel.refresh_plot(get_uuid(obj), True, False)
        return results
