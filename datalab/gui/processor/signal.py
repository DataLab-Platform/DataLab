# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Signal processor object (see parent package :mod:`datalab.gui.processor`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
from guidata.qthelpers import exec_dialog

import sigima.params
import sigima.proc.base as sigima_base
import sigima.proc.signal as sigima_signal
from datalab.config import _
from datalab.gui.processor.base import BaseProcessor
from datalab.utils.qthelpers import qt_try_except
from datalab.widgets import (
    fitdialog,
    signalbaseline,
    signalcursor,
    signaldeltax,
    signalpeak,
)
from sigima.objects import (
    ResultProperties,
    ResultShape,
    ROI1DParam,
    SignalObj,
    SignalROI,
    create_signal,
)


class SignalProcessor(BaseProcessor[SignalROI, ROI1DParam]):
    """Object handling signal processing: operations, processing, analysis"""

    # pylint: disable=duplicate-code

    def register_computations(self) -> None:
        """Register signal computations"""
        # MARK: OPERATION
        self.register_n_to_1(sigima_signal.addition, _("Sum"), icon_name="sum.svg")
        self.register_n_to_1(
            sigima_signal.average, _("Average"), icon_name="average.svg"
        )
        self.register_2_to_1(
            sigima_signal.difference,
            _("Difference"),
            icon_name="difference.svg",
            obj2_name=_("signal to subtract"),
        )
        self.register_2_to_1(
            sigima_signal.quadratic_difference,
            _("Quadratic Difference"),
            icon_name="quadratic_difference.svg",
            obj2_name=_("signal to subtract"),
        )
        self.register_n_to_1(
            sigima_signal.product, _("Product"), icon_name="product.svg"
        )
        self.register_2_to_1(
            sigima_signal.division,
            _("Division"),
            icon_name="division.svg",
            obj2_name=_("divider"),
        )
        self.register_1_to_1(
            sigima_signal.inverse, _("Inverse"), icon_name="inverse.svg"
        )
        self.register_2_to_1(
            sigima_signal.arithmetic,
            _("Arithmetic"),
            paramclass=sigima_base.ArithmeticParam,
            icon_name="arithmetic.svg",
            obj2_name=_("signal to operate with"),
        )
        self.register_1_to_1(
            sigima_signal.addition_constant,
            _("Add constant"),
            paramclass=sigima_base.ConstantParam,
            icon_name="constant_add.svg",
        )
        self.register_1_to_1(
            sigima_signal.difference_constant,
            _("Subtract constant"),
            paramclass=sigima_base.ConstantParam,
            icon_name="constant_subtract.svg",
        )
        self.register_1_to_1(
            sigima_signal.product_constant,
            _("Multiply by constant"),
            paramclass=sigima_base.ConstantParam,
            icon_name="constant_multiply.svg",
        )
        self.register_1_to_1(
            sigima_signal.division_constant,
            _("Divide by constant"),
            paramclass=sigima_base.ConstantParam,
            icon_name="constant_divide.svg",
        )
        self.register_1_to_1(
            sigima_signal.absolute, _("Absolute value"), icon_name="abs.svg"
        )
        self.register_1_to_1(sigima_signal.real, _("Real part"), icon_name="re.svg")
        self.register_1_to_1(
            sigima_signal.imag, _("Imaginary part"), icon_name="im.svg"
        )
        self.register_1_to_1(
            sigima_signal.astype,
            _("Convert data type"),
            paramclass=sigima.params.DataTypeSParam,
            icon_name="convert_dtype.svg",
        )
        self.register_1_to_1(sigima_signal.exp, _("Exponential"), icon_name="exp.svg")
        self.register_1_to_1(
            sigima_signal.log10, _("Logarithm (base 10)"), icon_name="log10.svg"
        )
        self.register_1_to_1(sigima_signal.sqrt, _("Square root"), icon_name="sqrt.svg")
        self.register_1_to_1(
            sigima_signal.derivative, _("Derivative"), icon_name="derivative.svg"
        )
        self.register_1_to_1(
            sigima_signal.integral, _("Integral"), icon_name="integral.svg"
        )
        self.register_2_to_1(
            sigima_signal.convolution,
            _("Convolution"),
            icon_name="convolution.svg",
            obj2_name=_("signal to convolve with"),
        )

        # MARK: PROCESSING
        # Axis transformation
        self.register_1_to_1(
            sigima_signal.calibration,
            _("Linear calibration"),
            sigima_signal.XYCalibrateParam,
            comment=_(
                "Apply linear calibration to the X or Y axis:\n"
                "  • x' = ax + b\n"
                "  • y' = ay + b"
            ),
        )
        self.register_1_to_1(
            sigima_signal.swap_axes, _("Swap X/Y axes"), icon_name="swap_x_y.svg"
        )
        self.register_1_to_1(
            sigima_signal.reverse_x,
            _("Reverse X-axis"),
            icon_name="reverse_signal_x.svg",
        )
        self.register_1_to_1(
            sigima_signal.to_polar,
            _("Convert to polar coordinates"),
            paramclass=sigima.params.AngleUnitParam,
        )
        self.register_1_to_1(
            sigima_signal.to_cartesian,
            _("Convert to cartesian coordinates"),
            paramclass=sigima.params.AngleUnitParam,
        )
        # Level adjustment
        self.register_1_to_1(
            sigima_signal.normalize,
            _("Normalize"),
            sigima_base.NormalizeParam,
            "normalize.svg",
        )
        self.register_1_to_1(
            sigima_signal.clip, _("Clipping"), sigima_base.ClipParam, "clip.svg"
        )
        self.register_1_to_1(
            sigima_signal.offset_correction,
            _("Offset correction"),
            icon_name="offset_correction.svg",
            comment=_("Evaluate and subtract the offset value from the data"),
        )
        # Noise reduction
        self.register_1_to_1(
            sigima_signal.gaussian_filter,
            _("Gaussian filter"),
            sigima_base.GaussianParam,
        )
        self.register_1_to_1(
            sigima_signal.moving_average,
            _("Moving average"),
            sigima_base.MovingAverageParam,
        )
        self.register_1_to_1(
            sigima_signal.moving_median,
            _("Moving median"),
            sigima_base.MovingMedianParam,
        )
        self.register_1_to_1(sigima_signal.wiener, _("Wiener filter"))
        # Fourier analysis
        self.register_1_to_1(
            sigima_signal.zero_padding,
            _("Zero padding"),
            sigima_signal.ZeroPadding1DParam,
            comment=_(
                "Zero padding is used to increase the frequency resolution of the FFT"
            ),
        )
        self.register_1_to_1(
            sigima_signal.fft,
            _("FFT"),
            sigima_base.FFTParam,
            comment=_(
                "Fast Fourier Transform (FFT) is an estimation of the "
                "Discrete Fourier Transform (DFT). "
                "Results are complex numbers, but only the real part is plotted."
            ),
            edit=False,
        )
        self.register_1_to_1(
            sigima_signal.ifft,
            _("Inverse FFT"),
            comment=_(
                "Inverse Fast Fourier Transform (IFFT) is an estimation of the "
                "Inverse Discrete Fourier Transform (IDFT). "
                "Results are complex numbers, but only the real part is plotted."
            ),
            edit=False,
        )
        self.register_1_to_1(
            sigima_signal.magnitude_spectrum,
            _("Magnitude spectrum"),
            paramclass=sigima.params.SpectrumParam,
            comment=_(
                "Magnitude spectrum is the absolute value of the FFT result. "
                "It is a measure of the amplitude of the frequency components."
            ),
        )
        self.register_1_to_1(
            sigima_signal.phase_spectrum,
            _("Phase spectrum"),
            comment=_(
                "Phase spectrum is the angle of the FFT result. "
                "It is a measure of the phase of the frequency components."
            ),
        )
        self.register_1_to_1(
            sigima_signal.psd,
            _("Power spectral density"),
            paramclass=sigima.params.SpectrumParam,
            comment=_(
                "Power spectral density (PSD) is the square of the magnitude spectrum. "
                "It is a measure of the power of the frequency components."
            ),
        )

        self.register_1_to_1(
            sigima_signal.power,
            _("Power"),
            paramclass=sigima.params.PowerParam,
            icon_name="power.svg",
        )
        self.register_1_to_1(
            sigima_signal.peak_detection,
            _("Peak detection"),
            paramclass=sigima.params.PeakDetectionParam,
            icon_name="peak_detect.svg",
        )
        # Frequency filters
        self.register_1_to_1(
            sigima_signal.lowpass,
            _("Low-pass filter"),
            sigima.params.LowPassFilterParam,
            "lowpass.svg",
        )
        self.register_1_to_1(
            sigima_signal.highpass,
            _("High-pass filter"),
            sigima.params.HighPassFilterParam,
            "highpass.svg",
        )
        self.register_1_to_1(
            sigima_signal.bandpass,
            _("Band-pass filter"),
            sigima.params.BandPassFilterParam,
            "bandpass.svg",
        )
        self.register_1_to_1(
            sigima_signal.bandstop,
            _("Band-stop filter"),
            sigima.params.BandStopFilterParam,
            "bandstop.svg",
        )
        # Other processing
        self.register_1_to_1(
            sigima_signal.apply_window,
            _("Windowing"),
            paramclass=sigima.params.WindowingParam,
            icon_name="windowing.svg",
            comment=_(
                "Apply a window function (or apodization): Hanning, Hamming, ..."
            ),
        )
        self.register_1_to_1(
            sigima_signal.detrending,
            _("Detrending"),
            sigima_signal.DetrendingParam,
            icon_name="detrending.svg",
        )
        self.register_2_to_1(
            sigima_signal.interpolate,
            _("Interpolation"),
            paramclass=sigima.params.InterpolationParam,
            obj2_name=_("signal for X values"),
            icon_name="interpolation.svg",
        )

        self.register_1_to_1(
            sigima_signal.resampling,
            _("Resampling"),
            sigima_signal.ResamplingParam,
            icon_name="resampling.svg",
        )
        # Stability analysis
        self.register_1_to_1(
            sigima_signal.allan_variance,
            _("Allan variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sigima_signal.allan_deviation,
            _("Allan deviation"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sigima_signal.overlapping_allan_variance,
            _("Overlapping Allan variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sigima_signal.modified_allan_variance,
            _("Modified Allan variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sigima_signal.hadamard_variance,
            _("Hadamard variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sigima_signal.modified_allan_variance,
            _("Modified Allan variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sigima_signal.hadamard_variance,
            _("Hadamard variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sigima_signal.modified_allan_variance,
            _("Modified Allan variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sigima_signal.hadamard_variance,
            _("Hadamard variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sigima_signal.total_variance,
            _("Total variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sigima_signal.time_deviation,
            _("Time deviation"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        # Other processing
        self.register_2_to_1(
            sigima_signal.xy_mode,
            _("X-Y mode"),
            obj2_name=_("Y-signal of the X-Y mode"),
            comment=_("Plot one signal as a fonction of the other one"),
        )
        self.register_1_to_n(sigima_signal.extract_roi, "ROI", icon_name="roi.svg")

        # MARK: ANALYSIS
        self.register_1_to_0(
            sigima_signal.stats, _("Statistics"), icon_name="stats.svg"
        )
        self.register_1_to_1(
            sigima_signal.histogram,
            _("Histogram"),
            paramclass=sigima_signal.HistogramParam,
            icon_name="histogram.svg",
        )
        self.register_1_to_0(
            sigima_signal.fwhm,
            _("Full width at half-maximum"),
            paramclass=sigima_signal.FWHMParam,
            icon_name="fwhm.svg",
        )
        self.register_1_to_0(
            sigima_signal.fw1e2,
            _("Full width at") + " 1/e²",
            icon_name="fw1e2.svg",
        )
        self.register_1_to_0(
            sigima_signal.full_width_at_y,
            _("Full width at y=..."),
            paramclass=sigima_signal.OrdinateParam,
            comment=_("Compute the full width at a given y value"),
        )
        self.register_1_to_0(
            sigima_signal.x_at_y,
            _("First abscissa at y=..."),
            paramclass=sigima_signal.OrdinateParam,
            comment=_(
                "Compute the first abscissa at a given y value (linear interpolation)"
            ),
        )
        self.register_1_to_0(
            sigima_signal.y_at_x,
            _("Ordinate at x=..."),
            paramclass=sigima_signal.AbscissaParam,
            comment=_("Compute the ordinate at a given x value (linear interpolation)"),
        )
        self.register_1_to_0(
            sigima_signal.x_at_minmax,
            _("Abscissa of the minimum and maximum"),
            comment=_(
                "Compute the smallest argument of the minima and the smallest "
                "argument of the maxima"
            ),
        )
        self.register_1_to_0(
            sigima_signal.sampling_rate_period,
            _("Sampling rate and period"),
            comment=_(
                "Compute sampling rate and period for a constant sampling signal"
            ),
        )
        self.register_1_to_0(
            sigima_signal.dynamic_parameters,
            _("Dynamic parameters"),
            paramclass=sigima_signal.DynamicParam,
            comment=_("Compute dynamic parameters: ENOB, SNR, SINAD, THD, ..."),
        )
        self.register_1_to_0(
            sigima_signal.bandwidth_3db,
            _("Bandwidth at -3dB"),
            comment=_(
                "Compute bandwidth at -3dB assuming a low-pass filter "
                "already expressed in dB"
            ),
        )
        self.register_1_to_0(
            sigima_signal.contrast,
            _("Contrast"),
            comment=_(
                "Compute contrast of a signal, i.e. (max-min)/(max+min), "
                "e.g. for an image profile"
            ),
        )

    @qt_try_except()
    def compute_offset_correction(self, param: ROI1DParam | None = None) -> None:
        """Compute offset correction
        with :py:func:`sigima.proc.signal.offset_correction`"""
        obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
        if param is None:
            dlg = signalbaseline.SignalBaselineDialog(obj, parent=self.panel.parent())
            if exec_dialog(dlg):
                param = ROI1DParam()
                param.xmin, param.xmax = dlg.get_x_range()
            else:
                return
        self.run_feature("offset_correction", param)

    @qt_try_except()
    def compute_all_stability(
        self, param: sigima.params.AllanVarianceParam | None = None
    ) -> None:
        """Compute all stability analysis features
        using the following functions:

        - :py:func:`sigima.proc.signal.allan_variance`
        - :py:func:`sigima.proc.signal.allan_deviation`
        - :py:func:`sigima.proc.signal.overlapping_allan_variance`
        - :py:func:`sigima.proc.signal.modified_allan_variance`
        - :py:func:`sigima.proc.signal.hadamard_variance`
        - :py:func:`sigima.proc.signal.total_variance`
        - :py:func:`sigima.proc.signal.time_deviation`
        """
        if param is None:
            param = sigima_signal.AllanVarianceParam()
            if not param.edit(parent=self.panel.parent()):
                return
        funcs = [
            sigima_signal.allan_variance,
            sigima_signal.allan_deviation,
            sigima_signal.overlapping_allan_variance,
            sigima_signal.modified_allan_variance,
            sigima_signal.hadamard_variance,
            sigima_signal.total_variance,
            sigima_signal.time_deviation,
        ]
        self.compute_multiple_1_to_1(
            funcs, [param] * len(funcs), "Stability", edit=False
        )

    @qt_try_except()
    def compute_peak_detection(
        self, param: sigima.params.PeakDetectionParam | None = None
    ) -> None:
        """Detect peaks from data
        with :py:func:`sigima.proc.signal.peak_detection`"""
        obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
        edit, param = self.init_param(
            param, sigima_signal.PeakDetectionParam, _("Peak detection")
        )
        if edit:
            dlg = signalpeak.SignalPeakDetectionDialog(obj, parent=self.panel.parent())
            if exec_dialog(dlg):
                param.threshold = int(dlg.get_threshold() * 100)
                param.min_dist = dlg.get_min_dist()
            else:
                return
        self.run_feature("peak_detection", param)

    @qt_try_except()
    def compute_polyfit(
        self, param: sigima.params.PolynomialFitParam | None = None
    ) -> None:
        """Compute polynomial fitting curve"""
        txt = _("Polynomial fit")
        edit, param = self.init_param(param, sigima_signal.PolynomialFitParam, txt)
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
    def _extract_multiple_roi_in_single_object(self, params: list[ROI1DParam]) -> None:
        """Extract multiple Regions Of Interest (ROIs) from data in a single object"""
        # TODO: This `compute_1_to_1` call is not ideal, as it passes a list of
        # parameter sets (`params` is a list of `DataSet` objects) instead of a single
        # parameter set as expected by the method. Currently, the method implementation
        # is compatible with this call, and it simply passes the second argument through
        # to the `extract_rois` function. However, this should be rectified in the
        # future to ensure that the method signature and its usage are consistent.
        self.compute_1_to_1(sigima_signal.extract_rois, params, title=_("Extract ROI"))

    # ------Signal Analysis

    @qt_try_except()
    def compute_full_width_at_y(
        self, param: sigima.params.OrdinateParam | None = None
    ) -> dict[str, ResultShape] | None:
        """Compute full width at a given y
        with :py:func:`sigima.proc.signal.full_width_at_y`"""
        if param is None:
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg = signaldeltax.SignalDeltaXDialog(obj, parent=self.panel.parent())
            if exec_dialog(dlg):
                param = sigima_signal.OrdinateParam()
                param.y = dlg.get_y_value()
            else:
                return None
        return self.run_feature("full_width_at_y", param)

    @qt_try_except()
    def compute_x_at_y(
        self, param: sigima_signal.OrdinateParam | None = None
    ) -> dict[str, ResultProperties] | None:
        """Compute x at y with :py:func:`sigima.proc.signal.x_at_y`."""
        if param is None:
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg = signalcursor.SignalCursorDialog(
                obj, cursor_orientation="horizontal", parent=self.panel.parent()
            )
            if exec_dialog(dlg):
                param = sigima_signal.OrdinateParam()
                param.y = dlg.get_y_value()
            else:
                return None
        return self.run_feature("x_at_y", param)

    @qt_try_except()
    def compute_y_at_x(
        self, param: sigima_signal.AbscissaParam | None = None
    ) -> dict[str, ResultProperties] | None:
        """Compute y at x with :py:func:`sigima.proc.signal.y_at_x`."""
        if param is None:
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg = signalcursor.SignalCursorDialog(
                obj, cursor_orientation="vertical", parent=self.panel.parent()
            )
            if exec_dialog(dlg):
                param = sigima_signal.AbscissaParam()
                param.x = dlg.get_x_value()
            else:
                return None
        return self.run_feature("y_at_x", param)
