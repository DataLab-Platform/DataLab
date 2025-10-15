# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Signal processor object (see parent package :mod:`datalab.gui.processor`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import sigima.params
import sigima.proc.base as sigima_base
import sigima.proc.signal as sips
from guidata.qthelpers import exec_dialog
from sigima.objects import (
    NormalDistributionParam,
    PoissonDistributionParam,
    ROI1DParam,
    SignalObj,
    SignalROI,
    UniformDistributionParam,
    create_signal,
)
from sigima.objects.scalar import GeometryResult, TableResult

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


class SignalProcessor(BaseProcessor[SignalROI, ROI1DParam]):
    """Object handling signal processing: operations, processing, analysis"""

    # pylint: disable=duplicate-code

    def register_operations(self) -> None:
        """Register operations."""
        self.register_n_to_1(sips.addition, _("Sum"), icon_name="sum.svg")
        self.register_n_to_1(sips.average, _("Average"), icon_name="average.svg")
        self.register_n_to_1(
            sips.standard_deviation,
            _("Standard deviation"),
            icon_name="std.svg",
        )
        self.register_2_to_1(
            sips.difference,
            _("Difference"),
            icon_name="difference.svg",
            obj2_name=_("signal to subtract"),
        )
        self.register_2_to_1(
            sips.quadratic_difference,
            _("Quadratic Difference"),
            icon_name="quadratic_difference.svg",
            obj2_name=_("signal to subtract"),
        )
        self.register_n_to_1(sips.product, _("Product"), icon_name="product.svg")
        self.register_2_to_1(
            sips.division,
            _("Division"),
            icon_name="division.svg",
            obj2_name=_("divider"),
        )
        self.register_1_to_1(sips.inverse, _("Inverse"), icon_name="inverse.svg")
        self.register_2_to_1(
            sips.arithmetic,
            _("Arithmetic"),
            paramclass=sigima_base.ArithmeticParam,
            icon_name="arithmetic.svg",
            obj2_name=_("signal to operate with"),
        )
        self.register_1_to_1(
            sips.addition_constant,
            _("Add constant"),
            paramclass=sigima_base.ConstantParam,
            icon_name="constant_add.svg",
        )
        self.register_1_to_1(
            sips.difference_constant,
            _("Subtract constant"),
            paramclass=sigima_base.ConstantParam,
            icon_name="constant_subtract.svg",
        )
        self.register_1_to_1(
            sips.product_constant,
            _("Multiply by constant"),
            paramclass=sigima_base.ConstantParam,
            icon_name="constant_multiply.svg",
        )
        self.register_1_to_1(
            sips.division_constant,
            _("Divide by constant"),
            paramclass=sigima_base.ConstantParam,
            icon_name="constant_divide.svg",
        )
        self.register_1_to_1(sips.absolute, _("Absolute value"), icon_name="abs.svg")
        self.register_1_to_1(
            sips.phase,
            _("Phase"),
            paramclass=sigima.params.PhaseParam,
            icon_name="phase.svg",
        )
        self.register_2_to_1(
            sips.complex_from_magnitude_phase,
            _("Combine with phase"),
            paramclass=sigima_base.AngleUnitParam,
            icon_name="complex_from_magnitude_phase.svg",
            comment=_("Create a complex-valued signal from magnitude and phase"),
            obj2_name=_("phase"),
        )
        self.register_1_to_1(sips.real, _("Real part"), icon_name="re.svg")
        self.register_1_to_1(sips.imag, _("Imaginary part"), icon_name="im.svg")
        self.register_2_to_1(
            sips.complex_from_real_imag,
            _("Combine with imaginary part"),
            icon_name="complex_from_real_imag.svg",
            comment=_("Create a complex-valued signal from real and imaginary parts"),
            obj2_name=_("imaginary part"),
        )
        self.register_1_to_1(
            sips.astype,
            _("Convert data type"),
            paramclass=sigima.params.DataTypeSParam,
            icon_name="convert_dtype.svg",
        )
        self.register_1_to_1(sips.exp, _("Exponential"), icon_name="exp.svg")
        self.register_1_to_1(
            sips.log10, _("Logarithm (base 10)"), icon_name="log10.svg"
        )
        self.register_1_to_1(sips.sqrt, _("Square root"), icon_name="sqrt.svg")
        self.register_1_to_1(
            sips.derivative, _("Derivative"), icon_name="derivative.svg"
        )
        self.register_1_to_1(sips.integral, _("Integral"), icon_name="integral.svg")
        self.register_2_to_1(
            sips.convolution,
            _("Convolution"),
            icon_name="convolution.svg",
            obj2_name=_("signal to convolve with"),
        )
        self.register_2_to_1(
            sips.deconvolution,
            _("Deconvolution"),
            icon_name="deconvolution.svg",
            obj2_name=_("signal to deconvolve with"),
        )

    def register_processing(self) -> None:
        """Register processing functions."""
        # Axis transformation
        self.register_1_to_1(
            sips.calibration,
            _("Linear calibration"),
            sips.XYCalibrateParam,
            comment=_(
                "Apply linear calibration to the X or Y axis:\n"
                "  • x' = ax + b\n"
                "  • y' = ay + b"
            ),
        )
        self.register_1_to_1(
            sips.transpose, _("Swap X/Y axes"), icon_name="swap_x_y.svg"
        )
        self.register_1_to_1(
            sips.reverse_x,
            _("Reverse X-axis"),
            icon_name="reverse_signal_x.svg",
        )
        self.register_1_to_1(
            sips.to_polar,
            _("Convert to polar coordinates"),
            paramclass=sigima.params.AngleUnitParam,
        )
        self.register_1_to_1(
            sips.to_cartesian,
            _("Convert to cartesian coordinates"),
            paramclass=sigima.params.AngleUnitParam,
        )
        # Level adjustment
        self.register_1_to_1(
            sips.normalize,
            _("Normalize"),
            sigima_base.NormalizeParam,
            "normalize.svg",
        )
        self.register_1_to_1(
            sips.clip, _("Clipping"), sigima_base.ClipParam, "clip.svg"
        )
        self.register_1_to_1(
            sips.offset_correction,
            _("Offset correction"),
            icon_name="offset_correction.svg",
            comment=_("Evaluate and subtract the offset value from the data"),
        )
        # Noise addition
        self.register_1_to_1(
            sips.add_gaussian_noise,
            _("Add Gaussian noise"),
            NormalDistributionParam,
        )
        self.register_1_to_1(
            sips.add_poisson_noise,
            _("Add Poisson noise"),
            PoissonDistributionParam,
        )
        self.register_1_to_1(
            sips.add_uniform_noise,
            _("Add uniform noise"),
            UniformDistributionParam,
        )
        # Noise reduction
        self.register_1_to_1(
            sips.gaussian_filter,
            _("Gaussian filter"),
            sigima_base.GaussianParam,
        )
        self.register_1_to_1(
            sips.moving_average,
            _("Moving average"),
            sigima_base.MovingAverageParam,
        )
        self.register_1_to_1(
            sips.moving_median,
            _("Moving median"),
            sigima_base.MovingMedianParam,
        )
        self.register_1_to_1(sips.wiener, _("Wiener filter"))
        # Fourier analysis
        self.register_1_to_1(
            sips.zero_padding,
            _("Zero padding"),
            sips.ZeroPadding1DParam,
            comment=_(
                "Zero padding is used to increase the frequency resolution of the FFT"
            ),
        )
        self.register_1_to_1(
            sips.fft,
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
            sips.ifft,
            _("Inverse FFT"),
            comment=_(
                "Inverse Fast Fourier Transform (IFFT) is an estimation of the "
                "Inverse Discrete Fourier Transform (IDFT). "
                "Results are complex numbers, but only the real part is plotted."
            ),
            edit=False,
        )
        self.register_1_to_1(
            sips.magnitude_spectrum,
            _("Magnitude spectrum"),
            paramclass=sigima.params.SpectrumParam,
            comment=_(
                "Magnitude spectrum is the absolute value of the FFT result. "
                "It is a measure of the amplitude of the frequency components."
            ),
        )
        self.register_1_to_1(
            sips.phase_spectrum,
            _("Phase spectrum"),
            comment=_(
                "Phase spectrum is the angle of the FFT result. "
                "It is a measure of the phase of the frequency components."
            ),
        )
        self.register_1_to_1(
            sips.psd,
            _("Power spectral density"),
            paramclass=sigima.params.SpectrumParam,
            comment=_(
                "Power spectral density (PSD) is the square of the magnitude spectrum. "
                "It is a measure of the power of the frequency components."
            ),
        )

        self.register_1_to_1(
            sips.power,
            _("Power"),
            paramclass=sigima.params.PowerParam,
            icon_name="power.svg",
        )
        self.register_1_to_1(
            sips.peak_detection,
            _("Peak detection"),
            paramclass=sigima.params.PeakDetectionParam,
            icon_name="peak_detect.svg",
        )
        # Frequency filters
        self.register_1_to_1(
            sips.lowpass,
            _("Low-pass filter"),
            sigima.params.LowPassFilterParam,
            "lowpass.svg",
        )
        self.register_1_to_1(
            sips.highpass,
            _("High-pass filter"),
            sigima.params.HighPassFilterParam,
            "highpass.svg",
        )
        self.register_1_to_1(
            sips.bandpass,
            _("Band-pass filter"),
            sigima.params.BandPassFilterParam,
            "bandpass.svg",
        )
        self.register_1_to_1(
            sips.bandstop,
            _("Band-stop filter"),
            sigima.params.BandStopFilterParam,
            "bandstop.svg",
        )
        # Curve fitting
        for fit_name, fit_func in [
            (_("Linear fit"), sips.linear_fit),
            (_("Polynomial fit"), sips.polynomial_fit),
            (_("Gaussian fit"), sips.gaussian_fit),
            (_("Lorentzian fit"), sips.lorentzian_fit),
            (_("Voigt fit"), sips.voigt_fit),
            (_("Planckian fit"), sips.planckian_fit),
            (_("Two Half-Gaussians fit"), sips.twohalfgaussian_fit),
            (_("Piecewise exponential fit"), sips.piecewiseexponential_fit),
            (_("Exponential fit"), sips.exponential_fit),
            (_("Sinusoidal fit"), sips.sinusoidal_fit),
            (_("CDF fit"), sips.cdf_fit),
            (_("Sigmoid fit"), sips.sigmoid_fit),
        ]:
            icon_name = f"{fit_func.__name__}.svg"
            self.register_1_to_1(fit_func, fit_name, icon_name=icon_name)

        # Other processing
        self.register_1_to_1(
            sips.apply_window,
            _("Windowing"),
            paramclass=sigima.params.WindowingParam,
            icon_name="windowing.svg",
            comment=_("Apply a window (apodization) function: Hann, Hamming..."),
        )
        self.register_1_to_1(
            sips.detrending,
            _("Detrending"),
            sips.DetrendingParam,
            icon_name="detrending.svg",
        )
        self.register_2_to_1(
            sips.interpolate,
            _("Interpolation"),
            paramclass=sigima.params.InterpolationParam,
            obj2_name=_("signal for X values"),
            icon_name="interpolation.svg",
            skip_xarray_compat=True,
        )

        self.register_1_to_1(
            sips.resampling,
            _("Resampling"),
            sips.Resampling1DParam,
            icon_name="resampling1d.svg",
        )
        # Stability analysis
        self.register_1_to_1(
            sips.allan_variance,
            _("Allan variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sips.allan_deviation,
            _("Allan deviation"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sips.overlapping_allan_variance,
            _("Overlapping Allan variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sips.modified_allan_variance,
            _("Modified Allan variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sips.hadamard_variance,
            _("Hadamard variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sips.modified_allan_variance,
            _("Modified Allan variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sips.hadamard_variance,
            _("Hadamard variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sips.modified_allan_variance,
            _("Modified Allan variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sips.hadamard_variance,
            _("Hadamard variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sips.total_variance,
            _("Total variance"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        self.register_1_to_1(
            sips.time_deviation,
            _("Time deviation"),
            paramclass=sigima.params.AllanVarianceParam,
        )
        # Other processing
        self.register_2_to_1(
            sips.xy_mode,
            _("X-Y mode"),
            obj2_name=_("Y-signal of the X-Y mode"),
            comment=_("Plot one signal as a fonction of the other one"),
        )
        self.register_1_to_n(sips.extract_roi, "ROI", icon_name="roi.svg")

    def register_analysis(self) -> None:
        """Register analysis functions."""
        self.register_1_to_0(sips.stats, _("Statistics"), icon_name="stats.svg")
        self.register_1_to_1(
            sips.histogram,
            _("Histogram"),
            paramclass=sigima.params.HistogramParam,
            icon_name="histogram.svg",
        )
        self.register_1_to_0(
            sips.fwhm,
            _("Full width at half-maximum"),
            paramclass=sips.FWHMParam,
            icon_name="fwhm.svg",
        )
        self.register_1_to_0(
            sips.fw1e2,
            _("Full width at") + " 1/e²",
            icon_name="fw1e2.svg",
        )
        self.register_1_to_0(
            sips.full_width_at_y,
            _("Full width at y=..."),
            paramclass=sips.OrdinateParam,
            comment=_("Compute the full width at a given y value"),
        )
        self.register_1_to_0(
            sips.x_at_y,
            _("First abscissa at y=..."),
            paramclass=sips.OrdinateParam,
            comment=_(
                "Compute the first abscissa at a given y value (linear interpolation)"
            ),
        )
        self.register_1_to_0(
            sips.y_at_x,
            _("Ordinate at x=..."),
            paramclass=sips.AbscissaParam,
            comment=_("Compute the ordinate at a given x value (linear interpolation)"),
        )
        self.register_1_to_0(
            sips.extract_pulse_features,
            _("Extract pulse features"),
            paramclass=sips.PulseFeaturesParam,
            comment=_("Extract pulse features (amplitude, rise time, fall time...)"),
        )
        self.register_1_to_0(
            sips.x_at_minmax,
            _("Abscissa of the minimum and maximum"),
            comment=_(
                "Compute the smallest argument of the minima and the smallest "
                "argument of the maxima"
            ),
        )
        self.register_1_to_0(
            sips.sampling_rate_period,
            _("Sampling rate and period"),
            comment=_(
                "Compute sampling rate and period for a constant sampling signal"
            ),
        )
        self.register_1_to_0(
            sips.dynamic_parameters,
            _("Dynamic parameters"),
            paramclass=sips.DynamicParam,
            comment=_("Compute dynamic parameters: ENOB, SNR, SINAD, THD, ..."),
        )
        self.register_1_to_0(
            sips.bandwidth_3db,
            _("Bandwidth at -3dB"),
            comment=_(
                "Compute bandwidth at -3dB assuming a low-pass filter "
                "already expressed in dB"
            ),
        )
        self.register_1_to_0(
            sips.contrast,
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
            dlg = signalbaseline.SignalBaselineDialog(obj, parent=self.mainwindow)
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
            param = sips.AllanVarianceParam()
            if not param.edit(parent=self.mainwindow):
                return
        funcs = [
            sips.allan_variance,
            sips.allan_deviation,
            sips.overlapping_allan_variance,
            sips.modified_allan_variance,
            sips.hadamard_variance,
            sips.total_variance,
            sips.time_deviation,
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
            param, sips.PeakDetectionParam, _("Peak detection")
        )
        if edit:
            dlg = signalpeak.SignalPeakDetectionDialog(obj, parent=self.mainwindow)
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
        edit, param = self.init_param(param, sips.PolynomialFitParam, txt)
        if not edit or param.edit(self.mainwindow):
            dlgfunc = fitdialog.polynomial_fit

            def polynomialfit(x, y, parent=None):
                """Polynomial fit dialog function"""
                return dlgfunc(x, y, param.degree, parent=parent)

            self.compute_fit(txt, polynomialfit)

    def __row_compute_fit(
        self, obj: SignalObj, name: str, fitdlgfunc: Callable
    ) -> None:
        """Curve fitting computing sub-method"""
        output = fitdlgfunc(obj.x, obj.y, parent=self.mainwindow)
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
        fitdlgfunc = fitdialog.multigaussian_fit
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
    def compute_multilorentzianfit(self) -> None:
        """Compute Multi-Lorentzian fitting curve using an interactive dialog"""
        fitdlgfunc = fitdialog.multilorentzian_fit
        for obj in self.panel.objview.get_sel_objects():
            dlg = signalpeak.SignalPeakDetectionDialog(obj, parent=self.panel)
            if exec_dialog(dlg):
                # Computing x, y
                peaks = dlg.get_peak_indices()

                def multilorentzianfit(x, y, parent=None):
                    """Multi-Lorentzian fit dialog function"""
                    # pylint: disable=cell-var-from-loop
                    return fitdlgfunc(x, y, peaks, parent=parent)

                self.__row_compute_fit(
                    obj, _("Multi-Lorentzian fit"), multilorentzianfit
                )

    @qt_try_except()
    def _extract_multiple_roi_in_single_object(self, params: list[ROI1DParam]) -> None:
        """Extract multiple Regions Of Interest (ROIs) from data in a single object"""
        # TODO: This `compute_1_to_1` call is not ideal, as it passes a list of
        # parameter sets (`params` is a list of `DataSet` objects) instead of a single
        # parameter set as expected by the method. Currently, the method implementation
        # is compatible with this call, and it simply passes the second argument through
        # to the `extract_rois` function. However, this should be rectified in the
        # future to ensure that the method signature and its usage are consistent.
        self.compute_1_to_1(sips.extract_rois, params, title=_("Extract ROI"))

    # ------Signal Analysis

    @qt_try_except()
    def compute_full_width_at_y(
        self, param: sigima.params.OrdinateParam | None = None
    ) -> dict[str, GeometryResult] | None:
        """Compute full width at a given y
        with :py:func:`sigima.proc.signal.full_width_at_y`"""
        if param is None:
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg = signaldeltax.SignalDeltaXDialog(obj, parent=self.mainwindow)
            if exec_dialog(dlg):
                param = sips.OrdinateParam()
                param.y = dlg.get_y_value()
            else:
                return None
        return self.run_feature("full_width_at_y", param)

    @qt_try_except()
    def compute_x_at_y(
        self, param: sips.OrdinateParam | None = None
    ) -> dict[str, TableResult] | None:
        """Compute x at y with :py:func:`sigima.proc.signal.x_at_y`."""
        if param is None:
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg = signalcursor.SignalCursorDialog(
                obj, cursor_orientation="horizontal", parent=self.mainwindow
            )
            if exec_dialog(dlg):
                param = sips.OrdinateParam()
                param.y = dlg.get_y_value()
            else:
                return None
        return self.run_feature("x_at_y", param)

    @qt_try_except()
    def compute_y_at_x(
        self, param: sips.AbscissaParam | None = None
    ) -> dict[str, TableResult] | None:
        """Compute y at x with :py:func:`sigima.proc.signal.y_at_x`."""
        if param is None:
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg = signalcursor.SignalCursorDialog(
                obj, cursor_orientation="vertical", parent=self.mainwindow
            )
            if exec_dialog(dlg):
                param = sips.AbscissaParam()
                param.x = dlg.get_x_value()
            else:
                return None
        return self.run_feature("y_at_x", param)
