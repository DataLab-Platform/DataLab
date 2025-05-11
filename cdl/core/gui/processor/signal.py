# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Signal processor object (see parent package :mod:`cdl.core.gui.processor`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import re
from collections.abc import Callable

import guidata.dataset as gds
import numpy as np
from guidata.qthelpers import exec_dialog

import cdl.computation.base as cpb
import cdl.computation.signal as cps
import cdl.param
from cdl.config import _
from cdl.core.gui.processor.base import BaseProcessor
from cdl.core.model.base import ResultProperties, ResultShape
from cdl.core.model.signal import ROI1DParam, SignalObj, SignalROI, create_signal
from cdl.utils.qthelpers import qt_try_except
from cdl.widgets import (
    fitdialog,
    signalbaseline,
    signalcursor,
    signaldeltax,
    signalpeak,
)


class SignalProcessor(BaseProcessor[SignalROI]):
    """Object handling signal processing: operations, processing, analysis"""

    # pylint: disable=duplicate-code

    def register_computations(self) -> None:
        """Register signal computations"""

        # TODO: Check if validation process has to be adapted to the new registering
        # mechanism: is it currently relying on the scanning of "compute_*" methods
        # of SignalProcessor? If that's so, it must be changed.

        # MARK: OPERATION
        self.register_n_to_1(cps.compute_sum, _("Sum"), icon_name="sum.svg")
        self.register_n_to_1(cps.compute_average, _("Average"), icon_name="average.svg")
        self.register_2_to_1(
            cps.compute_difference,
            _("Difference"),
            icon_name="difference.svg",
            obj2_name=_("signal to subtract"),
        )
        self.register_2_to_1(
            cps.compute_quadratic_difference,
            _("Quadratic Difference"),
            icon_name="quadratic_difference.svg",
            obj2_name=_("signal to subtract"),
        )
        self.register_n_to_1(cps.compute_product, _("Product"), icon_name="product.svg")
        self.register_2_to_1(
            cps.compute_division,
            _("Division"),
            icon_name="division.svg",
            obj2_name=_("divider"),
        )
        self.register_1_to_1(cps.compute_inverse, _("Inverse"), icon_name="inverse.svg")
        self.register_2_to_1(
            cps.compute_arithmetic,
            _("Arithmetic"),
            paramclass=cpb.ArithmeticParam,
            icon_name="arithmetic.svg",
            obj2_name=_("signal to operate with"),
        )
        self.register_1_to_1(
            cps.compute_addition_constant,
            _("Add constant"),
            paramclass=cpb.ConstantParam,
            icon_name="constant_add.svg",
        )
        self.register_1_to_1(
            cps.compute_difference_constant,
            _("Subtract constant"),
            paramclass=cpb.ConstantParam,
            icon_name="constant_subtract.svg",
        )
        self.register_1_to_1(
            cps.compute_product_constant,
            _("Multiply by constant"),
            paramclass=cpb.ConstantParam,
            icon_name="constant_multiply.svg",
        )
        self.register_1_to_1(
            cps.compute_division_constant,
            _("Divide by constant"),
            paramclass=cpb.ConstantParam,
            icon_name="constant_divide.svg",
        )
        self.register_1_to_1(cps.compute_abs, _("Absolute value"), icon_name="abs.svg")
        self.register_1_to_1(cps.compute_re, _("Real part"), icon_name="re.svg")
        self.register_1_to_1(cps.compute_im, _("Imaginary part"), icon_name="im.svg")
        self.register_1_to_1(
            cps.compute_astype,
            _("Convert data type"),
            paramclass=cdl.param.DataTypeSParam,
            icon_name="convert_dtype.svg",
        )
        self.register_1_to_1(cps.compute_exp, _("Exponential"), icon_name="exp.svg")
        self.register_1_to_1(
            cps.compute_log10, _("Logarithm (base 10)"), icon_name="log10.svg"
        )
        self.register_1_to_1(cps.compute_sqrt, _("Square root"), icon_name="sqrt.svg")
        self.register_1_to_1(
            cps.compute_derivative, _("Derivative"), icon_name="derivative.svg"
        )
        self.register_1_to_1(
            cps.compute_integral, _("Integral"), icon_name="integral.svg"
        )
        self.register_2_to_1(
            cps.compute_convolution,
            _("Convolution"),
            icon_name="convolution.svg",
            obj2_name=_("signal to convolve with"),
        )

        # MARK: PROCESSING
        # Axis transformation
        self.register_1_to_1(
            cps.compute_calibration, _("Linear calibration"), cps.XYCalibrateParam
        )
        self.register_1_to_1(
            cps.compute_swap_axes, _("Swap X/Y axes"), icon_name="swap_x_y.svg"
        )
        self.register_1_to_1(
            cps.compute_reverse_x, _("Reverse X-axis"), icon_name="reverse_signal_x.svg"
        )
        self.register_1_to_1(
            cps.compute_cartesian2polar,
            _("Convert to polar coordinates"),
            paramclass=cdl.param.AngleUnitParam,
        )
        self.register_1_to_1(
            cps.compute_polar2cartesian,
            _("Convert to cartesian coordinates"),
            paramclass=cdl.param.AngleUnitParam,
        )
        # Level adjustment
        self.register_1_to_1(
            cps.compute_normalize, _("Normalize"), cpb.NormalizeParam, "normalize.svg"
        )
        self.register_1_to_1(cps.compute_clip, _("Clipping"), cpb.ClipParam, "clip.svg")
        self.register_1_to_1(
            cps.compute_offset_correction,
            _("Offset correction"),
            icon_name="offset_correction.svg",
            comment=_("Evaluate and subtract the offset value from the data"),
        )
        # Noise reduction
        self.register_1_to_1(
            cps.compute_gaussian_filter, _("Gaussian filter"), cpb.GaussianParam
        )
        self.register_1_to_1(
            cps.compute_moving_average, _("Moving average"), cpb.MovingAverageParam
        )
        self.register_1_to_1(
            cps.compute_moving_median, _("Moving median"), cpb.MovingMedianParam
        )
        self.register_1_to_1(cps.compute_wiener, _("Wiener filter"))
        # Fourier analysis
        self.register_1_to_1(
            cps.compute_zero_padding,
            _("Zero padding"),
            cps.ZeroPadding1DParam,
            comment=_(
                "Zero padding is used to increase the frequency resolution of the FFT"
            ),
        )
        self.register_1_to_1(
            cps.compute_fft,
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
            cps.compute_ifft,
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
            cps.compute_magnitude_spectrum,
            _("Magnitude spectrum"),
            paramclass=cdl.param.SpectrumParam,
            comment=_(
                "Magnitude spectrum is the absolute value of the FFT result. "
                "It is a measure of the amplitude of the frequency components."
            ),
        )
        self.register_1_to_1(
            cps.compute_phase_spectrum,
            _("Phase spectrum"),
            comment=_(
                "Phase spectrum is the angle of the FFT result. "
                "It is a measure of the phase of the frequency components."
            ),
        )
        self.register_1_to_1(
            cps.compute_psd,
            _("Power spectral density"),
            paramclass=cdl.param.SpectrumParam,
            comment=_(
                "Power spectral density (PSD) is the square of the magnitude spectrum. "
                "It is a measure of the power of the frequency components."
            ),
        )

        self.register_1_to_1(
            cps.compute_power,
            _("Power"),
            paramclass=cdl.param.PowerParam,
            icon_name="power.svg",
        )
        self.register_1_to_1(
            cps.compute_peak_detection,
            _("Peak detection"),
            paramclass=cdl.param.PeakDetectionParam,
            icon_name="peak_detect.svg",
        )
        # Frequency filters
        self.register_1_to_1(
            cps.compute_lowpass,
            _("Low-pass filter"),
            cdl.param.LowPassFilterParam,
            "lowpass.svg",
        )
        self.register_1_to_1(
            cps.compute_highpass,
            _("High-pass filter"),
            cdl.param.HighPassFilterParam,
            "highpass.svg",
        )
        self.register_1_to_1(
            cps.compute_bandpass,
            _("Band-pass filter"),
            cdl.param.BandPassFilterParam,
            "bandpass.svg",
        )
        self.register_1_to_1(
            cps.compute_bandstop,
            _("Band-stop filter"),
            cdl.param.BandStopFilterParam,
            "bandstop.svg",
        )
        # Other processing
        self.register_1_to_1(
            cps.compute_windowing,
            _("Windowing"),
            paramclass=cdl.param.WindowingParam,
            icon_name="windowing.svg",
            comment=_(
                "Apply a window function (or apodization): Hanning, Hamming, ..."
            ),
        )
        self.register_1_to_1(
            cps.compute_detrending,
            _("Detrending"),
            cps.DetrendingParam,
            icon_name="detrending.svg",
        )
        self.register_2_to_1(
            cps.compute_interpolation,
            _("Interpolation"),
            paramclass=cdl.param.InterpolationParam,
            obj2_name=_("signal for X values"),
            icon_name="interpolation.svg",
        )

        self.register_1_to_1(
            cps.compute_resampling,
            _("Resampling"),
            cps.ResamplingParam,
            icon_name="resampling.svg",
        )
        # Stability analysis
        self.register_1_to_1(
            cps.compute_allan_variance,
            _("Allan variance"),
            paramclass=cdl.param.AllanVarianceParam,
        )
        self.register_1_to_1(
            cps.compute_allan_deviation,
            _("Allan deviation"),
            paramclass=cdl.param.AllanVarianceParam,
        )
        self.register_1_to_1(
            cps.compute_overlapping_allan_variance,
            _("Overlapping Allan variance"),
            paramclass=cdl.param.AllanVarianceParam,
        )
        self.register_1_to_1(
            cps.compute_modified_allan_variance,
            _("Modified Allan variance"),
            paramclass=cdl.param.AllanVarianceParam,
        )
        self.register_1_to_1(
            cps.compute_hadamard_variance,
            _("Hadamard variance"),
            paramclass=cdl.param.AllanVarianceParam,
        )
        self.register_1_to_1(
            cps.compute_modified_allan_variance,
            _("Modified Allan variance"),
            paramclass=cdl.param.AllanVarianceParam,
        )
        self.register_1_to_1(
            cps.compute_hadamard_variance,
            _("Hadamard variance"),
            paramclass=cdl.param.AllanVarianceParam,
        )
        self.register_1_to_1(
            cps.compute_modified_allan_variance,
            _("Modified Allan variance"),
            paramclass=cdl.param.AllanVarianceParam,
        )
        self.register_1_to_1(
            cps.compute_hadamard_variance,
            _("Hadamard variance"),
            paramclass=cdl.param.AllanVarianceParam,
        )
        self.register_1_to_1(
            cps.compute_total_variance,
            _("Total variance"),
            paramclass=cdl.param.AllanVarianceParam,
        )
        self.register_1_to_1(
            cps.compute_time_deviation,
            _("Time deviation"),
            paramclass=cdl.param.AllanVarianceParam,
        )
        # Other processing
        self.register_2_to_1(
            cps.compute_xy_mode,
            _("X-Y mode"),
            obj2_name=_("Y-signal of the X-Y mode"),
            comment=_("Plot one signal as a fonction of the other one"),
        )
        self.register_1_to_n(cps.compute_extract_roi, "ROI", icon_name="roi.svg")

        # MARK: ANALYSIS
        self.register_1_to_0(cps.compute_stats, _("Statistics"), icon_name="stats.svg")
        self.register_1_to_1(
            cps.compute_histogram,
            _("Histogram"),
            paramclass=cps.HistogramParam,
            icon_name="histogram.svg",
        )
        self.register_1_to_0(
            cps.compute_fwhm,
            _("Full width at half-maximum"),
            paramclass=cps.FWHMParam,
            icon_name="fwhm.svg",
        )
        self.register_1_to_0(
            cps.compute_fw1e2,
            _("Full width at") + " 1/e²",
            icon_name="fw1e2.svg",
        )
        self.register_1_to_0(
            cps.compute_full_width_at_y,
            _("Full width at y=..."),
            paramclass=cps.OrdinateParam,
            comment=_("Compute the full width at a given y value"),
        )
        self.register_1_to_0(
            cps.compute_x_at_y,
            _("First abscissa at y=..."),
            paramclass=cps.OrdinateParam,
            comment=_(
                "Compute the first abscissa at a given y value (linear interpolation)"
            ),
        )
        self.register_1_to_0(
            cps.compute_y_at_x,
            _("Ordinate at x=..."),
            paramclass=cps.AbscissaParam,
            comment=_("Compute the ordinate at a given x value (linear interpolation)"),
        )
        self.register_1_to_0(
            cps.compute_x_at_minmax,
            _("Abscissa of the minimum and maximum"),
            comment=_(
                "Compute the smallest argument of the minima and the smallest "
                "argument of the maxima"
            ),
        )
        self.register_1_to_0(
            cps.compute_sampling_rate_period,
            _("Sampling rate and period"),
            comment=_(
                "Compute sampling rate and period for a constant sampling signal"
            ),
        )
        self.register_1_to_0(
            cps.compute_dynamic_parameters,
            _("Dynamic parameters"),
            paramclass=cps.DynamicParam,
            comment=_("Compute dynamic parameters: ENOB, SNR, SINAD, THD, ..."),
        )
        self.register_1_to_0(
            cps.compute_bandwidth_3db,
            _("Bandwidth at -3dB"),
            comment=_(
                "Compute bandwidth at -3dB assuming a low-pass filter "
                "already expressed in dB"
            ),
        )
        self.register_1_to_0(
            cps.compute_contrast,
            _("Contrast"),
            comment=_(
                "Compute contrast of a signal, i.e. (max-min)/(max+min), "
                "e.g. for an image profile"
            ),
        )

    @qt_try_except()
    def compute_offset_correction(self, param: ROI1DParam | None = None) -> None:
        """Compute offset correction
        with :py:func:`cdl.computation.signal.compute_offset_correction`"""
        obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
        if param is None:
            dlg = signalbaseline.SignalBaselineDialog(obj, parent=self.panel.parent())
            if exec_dialog(dlg):
                param = ROI1DParam()
                param.xmin, param.xmax = dlg.get_x_range()
            else:
                return
        self.compute("offset_correction", param)

    @qt_try_except()
    def compute_all_stability(
        self, param: cdl.param.AllanVarianceParam | None = None
    ) -> None:
        """Compute all stability analysis features
        using the following functions:

        - :py:func:`cdl.computation.signal.compute_allan_variance`
        - :py:func:`cdl.computation.signal.compute_allan_deviation`
        - :py:func:`cdl.computation.signal.compute_overlapping_allan_variance`
        - :py:func:`cdl.computation.signal.compute_modified_allan_variance`
        - :py:func:`cdl.computation.signal.compute_hadamard_variance`
        - :py:func:`cdl.computation.signal.compute_total_variance`
        - :py:func:`cdl.computation.signal.compute_time_deviation`
        """
        if param is None:
            param = cps.AllanVarianceParam()
            if not param.edit(parent=self.panel.parent()):
                return
        funcs = [
            cps.compute_allan_variance,
            cps.compute_allan_deviation,
            cps.compute_overlapping_allan_variance,
            cps.compute_modified_allan_variance,
            cps.compute_hadamard_variance,
            cps.compute_total_variance,
            cps.compute_time_deviation,
        ]
        self.compute_multiple_1_to_1(
            funcs, [param] * len(funcs), "Stability", edit=False
        )

    @qt_try_except()
    def compute_peak_detection(
        self, param: cdl.param.PeakDetectionParam | None = None
    ) -> None:
        """Detect peaks from data
        with :py:func:`cdl.computation.signal.compute_peak_detection`"""
        obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
        edit, param = self.init_param(
            param, cps.PeakDetectionParam, _("Peak detection")
        )
        if edit:
            dlg = signalpeak.SignalPeakDetectionDialog(obj, parent=self.panel.parent())
            if exec_dialog(dlg):
                param.threshold = int(dlg.get_threshold() * 100)
                param.min_dist = dlg.get_min_dist()
            else:
                return
        self.compute("peak_detection", param)

    @qt_try_except()
    def compute_polyfit(
        self, param: cdl.param.PolynomialFitParam | None = None
    ) -> None:
        """Compute polynomial fitting curve"""
        txt = _("Polynomial fit")
        edit, param = self.init_param(param, cps.PolynomialFitParam, txt)
        if not edit or param.edit(self.panel.parent()):
            dlgfunc = fitdialog.polynomialfit

            def polynomialfit(x, y, parent=None):
                """Polynomial fit dialog function"""
                return dlgfunc(x, y, param.degree, parent=parent)

            self.compute_fit(txt, polynomialfit)

    def __row_compute_fit(
        self, obj: SignalObj, name: str, fitdlgfunc: Callable
    ) -> None:
        """Curve fitting computing sub-method"""
        output = fitdlgfunc(obj.x, obj.y, parent=self.panel.parent())
        if output is not None:
            y, params = output
            params: list[fitdialog.FitParam]
            pvalues = {}
            for param in params:
                if re.match(r"[\S\_]*\d{2}$", param.name):
                    shname = param.name[:-2]
                    value = pvalues.get(shname, np.array([]))
                    pvalues[shname] = np.array(list(value) + [param.value])
                else:
                    pvalues[param.name] = param.value
            # Creating new signal
            metadata = {fitdlgfunc.__name__: pvalues}
            signal = create_signal(f"{name}({obj.title})", obj.x, y, metadata=metadata)
            # Creating new plot item
            self.panel.add_object(signal)

    @qt_try_except()
    def compute_fit(self, title: str, fitdlgfunc: Callable) -> None:
        """Compute fitting curve using an interactive dialog

        Args:
            title: Title of the dialog
            fitdlgfunc: Fitting dialog function
        """
        for obj in self.panel.objview.get_sel_objects():
            self.__row_compute_fit(obj, title, fitdlgfunc)

    @qt_try_except()
    def compute_multigaussianfit(self) -> None:
        """Compute multi-Gaussian fitting curve using an interactive dialog"""
        fitdlgfunc = fitdialog.multigaussianfit
        for obj in self.panel.objview.get_sel_objects():
            dlg = signalpeak.SignalPeakDetectionDialog(obj, parent=self.panel)
            if exec_dialog(dlg):
                # Computing x, y
                peaks = dlg.get_peak_indices()

                def multigaussianfit(x, y, parent=None):
                    """Multi-Gaussian fit dialog function"""
                    # pylint: disable=cell-var-from-loop
                    return fitdlgfunc(x, y, peaks, parent=parent)

                self.__row_compute_fit(obj, _("Multi-Gaussian fit"), multigaussianfit)

    @qt_try_except()
    def _extract_multiple_roi_in_single_object(self, group: gds.DataSetGroup) -> None:
        """Extract multiple Regions Of Interest (ROIs) from data in a single object"""
        self.compute_1_to_1(cps.compute_extract_rois, group, title=_("Extract ROI"))

    # ------Signal Analysis

    @qt_try_except()
    def compute_full_width_at_y(
        self, param: cdl.param.OrdinateParam | None = None
    ) -> dict[str, ResultShape] | None:
        """Compute full width at a given y
        with :py:func:`cdl.computation.signal.compute_full_width_at_y`"""
        if param is None:
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg = signaldeltax.SignalDeltaXDialog(obj, parent=self.panel.parent())
            if exec_dialog(dlg):
                param = cps.OrdinateParam()
                param.y = dlg.get_y_value()
            else:
                return None
        return self.compute("full_width_at_y", param)

    @qt_try_except()
    def compute_x_at_y(
        self, param: cps.OrdinateParam | None = None
    ) -> dict[str, ResultProperties] | None:
        """Compute x at y with :py:func:`cdl.computation.signal.compute_x_at_y`."""
        if param is None:
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg = signalcursor.SignalCursorDialog(
                obj, cursor_orientation="horizontal", parent=self.panel.parent()
            )
            if exec_dialog(dlg):
                param = cps.OrdinateParam()
                param.y = dlg.get_y_value()
            else:
                return None
        return self.compute("x_at_y", param)

    @qt_try_except()
    def compute_y_at_x(
        self, param: cps.AbscissaParam | None = None
    ) -> dict[str, ResultProperties] | None:
        """Compute y at x with :py:func:`cdl.computation.signal.compute_y_at_x`."""
        if param is None:
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg = signalcursor.SignalCursorDialog(
                obj, cursor_orientation="vertical", parent=self.panel.parent()
            )
            if exec_dialog(dlg):
                param = cps.AbscissaParam()
                param.x = dlg.get_x_value()
            else:
                return None
        return self.compute("y_at_x", param)
