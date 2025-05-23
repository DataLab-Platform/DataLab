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
from cdl.config import Conf, _
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

    @qt_try_except()
    def compute_sum(self) -> None:
        """Compute sum with :py:func:`cdl.computation.signal.compute_addition`"""
        self.compute_n1("Σ", cps.compute_addition, title=_("Sum"))

    @qt_try_except()
    def compute_addition_constant(self, param: cpb.ConstantParam | None = None) -> None:
        """Compute sum with a constant
        with :py:func:`cdl.computation.signal.compute_addition_constant`"""
        self.compute_11(
            cps.compute_addition_constant,
            param,
            paramclass=cpb.ConstantParam,
            title=_("Sum with constant"),
        )

    @qt_try_except()
    def compute_average(self) -> None:
        """Compute average with :py:func:`cdl.computation.signal.compute_addition`
        and divide by the number of signals"""

        def func_objs(new_obj: SignalObj, old_objs: list[SignalObj]) -> None:
            """Finalize average computation"""
            new_obj.data = new_obj.data / float(len(old_objs))
            if new_obj.dy is not None:
                new_obj.dy = new_obj.dy / float(len(old_objs))

        self.compute_n1(
            "μ", cps.compute_addition, func_objs=func_objs, title=_("Average")
        )

    @qt_try_except()
    def compute_product(self) -> None:
        """Compute product with :py:func:`cdl.computation.signal.compute_product`"""
        self.compute_n1("Π", cps.compute_product, title=_("Product"))

    @qt_try_except()
    def compute_product_constant(self, param: cpb.ConstantParam | None = None) -> None:
        """Compute product with a constant
        with :py:func:`cdl.computation.signal.compute_product_constant`"""
        self.compute_11(
            cps.compute_product_constant,
            param,
            paramclass=cpb.ConstantParam,
            title=_("Product with constant"),
        )

    @qt_try_except()
    def compute_swap_axes(self) -> None:
        """Swap data axes with :py:func:`cdl.computation.signal.compute_swap_axes`"""
        self.compute_11(cps.compute_swap_axes, title=_("Swap axes"))

    @qt_try_except()
    def compute_inverse(self) -> None:
        """Compute inverse"""
        self.compute_11(cps.compute_inverse, title=_("Inverse"))

    @qt_try_except()
    def compute_abs(self) -> None:
        """Compute absolute value with :py:func:`cdl.computation.signal.compute_abs`"""
        self.compute_11(cps.compute_abs, title=_("Absolute value"))

    @qt_try_except()
    def compute_re(self) -> None:
        """Compute real part with :py:func:`cdl.computation.signal.compute_re`"""
        self.compute_11(cps.compute_re, title=_("Real part"))

    @qt_try_except()
    def compute_im(self) -> None:
        """Compute imaginary part with :py:func:`cdl.computation.signal.compute_im`"""
        self.compute_11(cps.compute_im, title=_("Imaginary part"))

    @qt_try_except()
    def compute_astype(self, param: cdl.param.DataTypeSParam | None = None) -> None:
        """Convert data type with :py:func:`cdl.computation.signal.compute_astype`"""
        self.compute_11(
            cps.compute_astype, param, cps.DataTypeSParam, title=_("Convert data type")
        )

    @qt_try_except()
    def compute_log10(self) -> None:
        """Compute Log10 with :py:func:`cdl.computation.signal.compute_log10`"""
        self.compute_11(cps.compute_log10, title="Log10")

    @qt_try_except()
    def compute_exp(self) -> None:
        """Compute Log10 with :py:func:`cdl.computation.signal.compute_exp`"""
        self.compute_11(cps.compute_exp, title=_("Exponential"))

    @qt_try_except()
    def compute_sqrt(self) -> None:
        """Compute square root with :py:func:`cdl.computation.signal.compute_sqrt`"""
        self.compute_11(cps.compute_sqrt, title=_("Square root"))

    @qt_try_except()
    def compute_power(self, param: cps.PowerParam | None = None) -> None:
        """Compute power with :py:func:`cdl.computation.signal.compute_power`"""
        self.compute_11(cps.compute_power, param, cps.PowerParam, title="Power")

    @qt_try_except()
    def compute_arithmetic(
        self, obj2: SignalObj | None = None, param: cpb.ArithmeticParam | None = None
    ) -> None:
        """Compute arithmetic operation between two signals
        with :py:func:`cdl.computation.signal.compute_arithmetic`"""
        self.compute_n1n(
            obj2,
            _("signal to operate with"),
            cps.compute_arithmetic,
            param=param,
            paramclass=cpb.ArithmeticParam,
            title=_("Arithmetic"),
        )

    @qt_try_except()
    def compute_difference(
        self, obj2: SignalObj | list[SignalObj] | None = None
    ) -> None:
        """Compute difference between two signals
        with :py:func:`cdl.computation.signal.compute_difference`"""
        self.compute_n1n(
            obj2,
            _("signal to subtract"),
            cps.compute_difference,
            title=_("Difference"),
        )

    @qt_try_except()
    def compute_difference_constant(
        self, param: cpb.ConstantParam | None = None
    ) -> None:
        """Compute difference with a constant
        with :py:func:`cdl.computation.signal.compute_difference_constant`"""
        self.compute_11(
            cps.compute_difference_constant,
            param,
            paramclass=cpb.ConstantParam,
            title=_("Difference with constant"),
            edit=True,
        )

    @qt_try_except()
    def compute_quadratic_difference(
        self, obj2: SignalObj | list[SignalObj] | None = None
    ) -> None:
        """Compute quadratic difference between two signals
        with :py:func:`cdl.computation.signal.compute_quadratic_difference`"""
        self.compute_n1n(
            obj2,
            _("signal to subtract"),
            cps.compute_quadratic_difference,
            title=_("Quadratic difference"),
        )

    @qt_try_except()
    def compute_division(self, obj2: SignalObj | list[SignalObj] | None = None) -> None:
        """Compute division between two signals
        with :py:func:`cdl.computation.signal.compute_division`"""
        self.compute_n1n(
            obj2,
            _("divider"),
            cps.compute_division,
            title=_("Division"),
        )

    @qt_try_except()
    def compute_division_constant(self, param: cpb.ConstantParam | None = None) -> None:
        """Compute division by a constant
        with :py:func:`cdl.computation.signal.compute_division_constant`"""
        self.compute_11(
            cps.compute_division_constant,
            param,
            paramclass=cpb.ConstantParam,
            title=_("Division by constant"),
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
        self.compute_11(cps.compute_peak_detection, param)

    @qt_try_except()
    def compute_reverse_x(self) -> None:
        """Reverse X axis with :py:func:`cdl.computation.signal.compute_reverse_x`"""
        self.compute_11(cps.compute_reverse_x, title=_("Reverse X axis"))

    @qt_try_except()
    def compute_cartesian2polar(
        self, param: cdl.param.AngleUnitParam | None = None
    ) -> None:
        """Convert cartesian to polar coordinates
        with :py:func:`cdl.computation.signal.compute_cartesian2polar`"""
        self.compute_11(
            cps.compute_cartesian2polar,
            param,
            cps.AngleUnitParam,
            title=_("Cartesian to Polar"),
        )

    @qt_try_except()
    def compute_polar2cartesian(
        self, param: cdl.param.AngleUnitParam | None = None
    ) -> None:
        """Convert polar to cartesian coordinates
        with :py:func:`cdl.computation.signal.compute_polar2cartesian`."""
        self.compute_11(
            cps.compute_polar2cartesian,
            param,
            cps.AngleUnitParam,
            title=_("Polar to Cartesian"),
        )

    # ------Signal Processing
    @qt_try_except()
    def compute_normalize(self, param: cdl.param.NormalizeParam | None = None) -> None:
        """Normalize data with :py:func:`cdl.computation.signal.compute_normalize`"""
        self.compute_11(
            cps.compute_normalize, param, cps.NormalizeParam, title=_("Normalize")
        )

    @qt_try_except()
    def compute_derivative(self) -> None:
        """Compute derivative
        with :py:func:`cdl.computation.signal.compute_derivative`"""
        self.compute_11(cps.compute_derivative, title=_("Derivative"))

    @qt_try_except()
    def compute_integral(self) -> None:
        """Compute integral with :py:func:`cdl.computation.signal.compute_integral`"""
        self.compute_11(cps.compute_integral, title=_("Integral"))

    @qt_try_except()
    def compute_calibration(
        self, param: cdl.param.XYCalibrateParam | None = None
    ) -> None:
        """Compute data linear calibration
        with :py:func:`cdl.computation.signal.compute_calibration`"""
        self.compute_11(
            cps.compute_calibration,
            param,
            cps.XYCalibrateParam,
            title=_("Linear calibration"),
            comment="y = a.x + b",
        )

    @qt_try_except()
    def compute_clip(self, param: cpb.ClipParam | None = None) -> None:
        """Compute maximum data clipping
        with :py:func:`cdl.computation.signal.compute_clip`"""
        self.compute_11(cps.compute_clip, param, cpb.ClipParam, title=_("Clipping"))

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
        self.compute_11(cps.compute_offset_correction, param)

    @qt_try_except()
    def compute_gaussian_filter(self, param: cpb.GaussianParam | None = None) -> None:
        """Compute gaussian filter
        with :py:func:`cdl.computation.signal.compute_gaussian_filter`"""
        self.compute_11(
            cps.compute_gaussian_filter,
            param,
            cpb.GaussianParam,
            title=_("Gaussian filter"),
        )

    @qt_try_except()
    def compute_moving_average(
        self, param: cpb.MovingAverageParam | None = None
    ) -> None:
        """Compute moving average
        with :py:func:`cdl.computation.signal.compute_moving_average`"""
        self.compute_11(
            cps.compute_moving_average,
            param,
            cpb.MovingAverageParam,
            title=_("Moving average"),
        )

    @qt_try_except()
    def compute_moving_median(self, param: cpb.MovingMedianParam | None = None) -> None:
        """Compute moving median
        with :py:func:`cdl.computation.signal.compute_moving_median`"""
        self.compute_11(
            cps.compute_moving_median,
            param,
            cpb.MovingMedianParam,
            title=_("Moving median"),
        )

    @qt_try_except()
    def compute_wiener(self) -> None:
        """Compute Wiener filter
        with :py:func:`cdl.computation.signal.compute_wiener`"""
        self.compute_11(cps.compute_wiener, title=_("Wiener filter"))

    def __freq_filter(
        self,
        param: cdl.param.LowPassFilterParam
        | cdl.param.HighPassFilterParam
        | cdl.param.BandPassFilterParam
        | cdl.param.BandStopFilterParam,
        paramclass: type[
            cdl.param.LowPassFilterParam
            | cdl.param.HighPassFilterParam
            | cdl.param.BandPassFilterParam
            | cdl.param.BandStopFilterParam
        ],
        title: str,
    ) -> None:
        """Compute frequency filter"""
        edit, param = self.init_param(param, paramclass, title)
        if edit:
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            param.update_from_signal(obj)
        self.compute_11(cps.compute_filter, param, title=title, edit=edit)

    @qt_try_except()
    def compute_lowpass(
        self, param: cdl.param.LowPassFilterParam | None = None
    ) -> None:
        """Compute high-pass filter
        with :py:func:`cdl.computation.signal.compute_filter`"""
        self.__freq_filter(param, cdl.param.LowPassFilterParam, _("Low-pass filter"))

    @qt_try_except()
    def compute_highpass(
        self, param: cdl.param.HighPassFilterParam | None = None
    ) -> None:
        """Compute high-pass filter
        with :py:func:`cdl.computation.signal.compute_filter`"""
        self.__freq_filter(param, cdl.param.HighPassFilterParam, _("High-pass filter"))

    @qt_try_except()
    def compute_bandpass(
        self, param: cdl.param.BandPassFilterParam | None = None
    ) -> None:
        """Compute band-pass filter
        with :py:func:`cdl.computation.signal.compute_filter`"""
        self.__freq_filter(param, cdl.param.BandPassFilterParam, _("Band-pass filter"))

    @qt_try_except()
    def compute_bandstop(
        self, param: cdl.param.BandStopFilterParam | None = None
    ) -> None:
        """Compute band-stop filter
        with :py:func:`cdl.computation.signal.compute_filter`"""
        self.__freq_filter(param, cdl.param.BandStopFilterParam, _("Band-stop filter"))

    @qt_try_except()
    def compute_zero_padding(
        self, param: cdl.param.ZeroPadding1DParam | None = None
    ) -> None:
        """Compute zero padding
        with :py:func:`cdl.computation.signal.compute_zero_padding`"""
        edit, param = self.init_param(param, cps.ZeroPadding1DParam, _("Zero padding"))
        if edit:
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            param.update_from_signal(obj)
        self.compute_11(
            cps.compute_zero_padding, param, title=_("Zero padding"), edit=edit
        )

    @qt_try_except()
    def compute_fft(self, param: cdl.param.FFTParam | None = None) -> None:
        """Compute FFT with :py:func:`cdl.computation.signal.compute_fft`"""
        if param is None:
            param = cpb.FFTParam.create(shift=Conf.proc.fft_shift_enabled.get())
        self.compute_11(cps.compute_fft, param, title=_("FFT"), edit=False)

    @qt_try_except()
    def compute_ifft(self, param: cdl.param.FFTParam | None = None) -> None:
        """Compute iFFT with :py:func:`cdl.computation.signal.compute_ifft`"""
        if param is None:
            param = cpb.FFTParam.create(shift=Conf.proc.fft_shift_enabled.get())
        self.compute_11(cps.compute_ifft, param, title=_("iFFT"), edit=False)

    @qt_try_except()
    def compute_magnitude_spectrum(
        self, param: cdl.param.SpectrumParam | None = None
    ) -> None:
        """Compute magnitude spectrum
        with :py:func:`cdl.computation.signal.compute_magnitude_spectrum`"""
        self.compute_11(
            cps.compute_magnitude_spectrum,
            param,
            cdl.param.SpectrumParam,
            title=_("Magnitude spectrum"),
        )

    @qt_try_except()
    def compute_phase_spectrum(self) -> None:
        """Compute phase spectrum
        with :py:func:`cdl.computation.signal.compute_phase_spectrum`"""
        self.compute_11(cps.compute_phase_spectrum, title=_("Phase spectrum"))

    @qt_try_except()
    def compute_psd(self, param: cdl.param.SpectrumParam | None = None) -> None:
        """Compute power spectral density
        with :py:func:`cdl.computation.signal.compute_psd`"""
        self.compute_11(cps.compute_psd, param, cdl.param.SpectrumParam, title=_("PSD"))

    @qt_try_except()
    def compute_interpolation(
        self,
        obj2: SignalObj | None = None,
        param: cdl.param.InterpolationParam | None = None,
    ):
        """Compute interpolation
        with :py:func:`cdl.computation.signal.compute_interpolation`"""
        self.compute_n1n(
            obj2,
            _("signal for X values"),
            cps.compute_interpolation,
            param,
            cps.InterpolationParam,
            title=_("Interpolation"),
        )

    @qt_try_except()
    def compute_resampling(self, param: cdl.param.ResamplingParam | None = None):
        """Compute resampling
        with :py:func:`cdl.computation.signal.compute_resampling`"""
        title = _("Resampling")
        edit, param = self.init_param(param, cps.ResamplingParam, title)
        if edit:
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            if param.xmin is None:
                param.xmin = obj.x[0]
            if param.xmax is None:
                param.xmax = obj.x[-1]
            if param.dx is None:
                param.dx = obj.x[1] - obj.x[0]
            if param.nbpts is None:
                param.nbpts = len(obj.x)
        self.compute_11(cps.compute_resampling, param, title=title, edit=edit)

    @qt_try_except()
    def compute_detrending(self, param: cdl.param.DetrendingParam | None = None):
        """Compute detrending
        with :py:func:`cdl.computation.signal.compute_detrending`"""
        self.compute_11(
            cps.compute_detrending,
            param,
            cps.DetrendingParam,
            title=_("Detrending"),
        )

    @qt_try_except()
    def compute_XY_mode(self, obj2: SignalObj | None = None) -> None:
        """Compute XY mode with :py:func:`cdl.computation.signal.compute_XY_mode.`"""
        self.compute_n1n(
            obj2,
            _("Y-signal of the X-Y mode"),
            cps.compute_XY_mode,
            title=_("X-Y Mode"),
        )

    @qt_try_except()
    def compute_convolution(self, obj2: SignalObj | None = None) -> None:
        """Compute convolution
        with :py:func:`cdl.computation.signal.compute_convolution`"""
        self.compute_n1n(
            obj2,
            _("signal to convolve with"),
            cps.compute_convolution,
            title=_("Convolution"),
        )

    @qt_try_except()
    def compute_windowing(self, param: cdl.param.WindowingParam | None = None) -> None:
        """Compute windowing
        with :py:func:`cdl.computation.signal.compute_windowing`"""
        self.compute_11(
            cps.compute_windowing,
            param,
            cdl.param.WindowingParam,
            title=_("Windowing"),
        )

    @qt_try_except()
    def compute_allan_variance(
        self, param: cdl.param.AllanVarianceParam | None = None
    ) -> None:
        """Compute Allan variance
        with :py:func:`cdl.computation.signal.compute_allan_variance`"""
        self.compute_11(
            cps.compute_allan_variance,
            param,
            cps.AllanVarianceParam,
            title=_("Allan variance"),
        )

    @qt_try_except()
    def compute_allan_deviation(
        self, param: cdl.param.AllanVarianceParam | None = None
    ) -> None:
        """Compute Allan deviation
        with :py:func:`cdl.computation.signal.compute_allan_deviation`"""
        self.compute_11(
            cps.compute_allan_deviation,
            param,
            cps.AllanVarianceParam,
            title=_("Allan deviation"),
        )

    @qt_try_except()
    def compute_overlapping_allan_variance(
        self, param: cdl.param.AllanVarianceParam | None = None
    ) -> None:
        """Compute overlapping Allan variance
        with :py:func:`cdl.computation.signal.compute_overlapping_allan_variance`"""
        self.compute_11(
            cps.compute_overlapping_allan_variance,
            param,
            cps.AllanVarianceParam,
            title=_("Overlapping Allan variance"),
        )

    @qt_try_except()
    def compute_modified_allan_variance(
        self, param: cdl.param.AllanVarianceParam | None = None
    ) -> None:
        """Compute modified Allan variance
        with :py:func:`cdl.computation.signal.compute_modified_allan_variance`"""
        self.compute_11(
            cps.compute_modified_allan_variance,
            param,
            cps.AllanVarianceParam,
            title=_("Modified Allan variance"),
        )

    @qt_try_except()
    def compute_hadamard_variance(
        self, param: cdl.param.AllanVarianceParam | None = None
    ) -> None:
        """Compute Hadamard variance
        with :py:func:`cdl.computation.signal.compute_hadamard_variance`"""
        self.compute_11(
            cps.compute_hadamard_variance,
            param,
            cps.AllanVarianceParam,
            title=_("Hadamard variance"),
        )

    @qt_try_except()
    def compute_total_variance(
        self, param: cdl.param.AllanVarianceParam | None = None
    ) -> None:
        """Compute total variance
        with :py:func:`cdl.computation.signal.compute_total_variance`"""
        self.compute_11(
            cps.compute_total_variance,
            param,
            cps.AllanVarianceParam,
            title=_("Total variance"),
        )

    @qt_try_except()
    def compute_time_deviation(
        self, param: cdl.param.AllanVarianceParam | None = None
    ) -> None:
        """Compute time deviation
        with :py:func:`cdl.computation.signal.compute_time_deviation`"""
        self.compute_11(
            cps.compute_time_deviation,
            param,
            cps.AllanVarianceParam,
            title=_("Time deviation"),
        )

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
        self.compute_1n(funcs, [param] * len(funcs), "Stability", edit=False)

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
        self.compute_11(cps.extract_multiple_roi, group, title=_("Extract ROI"))

    @qt_try_except()
    def _extract_each_roi_in_separate_object(self, group: gds.DataSetGroup) -> None:
        """Extract each single Region Of Interest (ROI) from data in a separate
        object (keep the ROI in the case of a circular ROI, for example)"""
        self.compute_1n(cps.extract_single_roi, group.datasets, "ROI", edit=False)

    # ------Signal Analysis
    @qt_try_except()
    def compute_fwhm(
        self, param: cdl.param.FWHMParam | None = None
    ) -> dict[str, ResultShape]:
        """Compute FWHM with :py:func:`cdl.computation.signal.compute_fwhm`"""
        return self.compute_10(cps.compute_fwhm, param, cps.FWHMParam, title=_("FWHM"))

    @qt_try_except()
    def compute_fw1e2(self) -> dict[str, ResultShape]:
        """Compute FW at 1/e² with :py:func:`cdl.computation.signal.compute_fw1e2`"""
        return self.compute_10(cps.compute_fw1e2, title=_("FW") + "1/e²")

    @qt_try_except()
    def compute_full_width_at_y(
        self, param: cdl.param.OrdinateParam | None = None
    ) -> dict[str, ResultShape]:
        """Compute full width at a given y
        with :py:func:`cdl.computation.signal.compute_full_width_at_y`"""
        if param is None:
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg = signaldeltax.SignalDeltaXDialog(obj, parent=self.panel.parent())
            if exec_dialog(dlg):
                param = cps.OrdinateParam()
                param.y = dlg.get_y_value()
            else:
                return
        return self.compute_10(cps.compute_full_width_at_y, param)

    @qt_try_except()
    def compute_stats(self) -> dict[str, ResultProperties]:
        """Compute data statistics
        with :py:func:`cdl.computation.signal.compute_stats`"""
        return self.compute_10(cps.compute_stats, title=_("Statistics"))

    @qt_try_except()
    def compute_histogram(
        self, param: cdl.param.HistogramParam | None = None
    ) -> dict[str, ResultShape]:
        """Compute histogram
        with :py:func:`cdl.computation.signal.compute_histogram`"""
        return self.compute_11(
            cps.compute_histogram, param, cps.HistogramParam, title=_("Histogram")
        )

    @qt_try_except()
    def compute_contrast(self) -> dict[str, ResultProperties]:
        """Compute contrast with :py:func:`cdl.computation.signal.compute_contrast`"""
        return self.compute_10(cps.compute_contrast, title=_("Contrast"))

    @qt_try_except()
    def compute_x_at_minmax(self) -> dict[str, ResultProperties]:
        """Compute x at min/max
        with :py:func:`cdl.computation.signal.compute_x_at_minmax`"""
        return self.compute_10(cps.compute_x_at_minmax, title="X @ min,max")

    @qt_try_except()
    def compute_x_at_y(
        self, param: cps.OrdinateParam | None = None
    ) -> dict[str, ResultProperties]:
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
                return
        return self.compute_10(cps.compute_x_at_y, param)

    @qt_try_except()
    def compute_y_at_x(
        self, param: cps.AbscissaParam | None = None
    ) -> dict[str, ResultProperties]:
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
                return
        return self.compute_10(cps.compute_y_at_x, param)

    @qt_try_except()
    def compute_sampling_rate_period(self) -> dict[str, ResultProperties]:
        """Compute sampling rate and period (mean and std)
        with :py:func:`cdl.computation.signal.compute_sampling_rate_period`"""
        return self.compute_10(
            cps.compute_sampling_rate_period, title=_("Sampling rate and period")
        )

    @qt_try_except()
    def compute_bandwidth_3db(self) -> None:
        """Compute bandwidth at -3dB
        with :py:func:`cdl.computation.signal.compute_bandwidth_3db`"""
        self.compute_10(cps.compute_bandwidth_3db, title=_("Bandwidth"))

    @qt_try_except()
    def compute_dynamic_parameters(
        self, param: cps.DynamicParam | None = None
    ) -> dict[str, ResultProperties]:
        """Compute Dynamic Parameters (ENOB, SINAD, THD, SFDR, SNR)
        with :py:func:`cdl.computation.signal.compute_dynamic_parameters`"""
        return self.compute_10(
            cps.compute_dynamic_parameters,
            param,
            cps.DynamicParam,
            title=_("Dynamic parameters"),
        )
