# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Signal Processor
----------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
from guidata.qthelpers import exec_dialog

import cdl.core.computation.base as cpb
import cdl.core.computation.signal as cps
import cdl.param
from cdl.config import Conf, _
from cdl.core.gui.processor.base import BaseProcessor
from cdl.core.model.base import ResultShape, ShapeTypes
from cdl.core.model.signal import SignalObj, create_signal
from cdl.utils.qthelpers import qt_try_except
from cdl.widgets import fitdialog, signalpeakdialog


class SignalProcessor(BaseProcessor):
    """Object handling signal processing: operations, processing, computing"""

    # pylint: disable=duplicate-code

    @qt_try_except()
    def compute_sum(self) -> None:
        """Compute sum"""
        self.compute_n1("Σ", cps.compute_add, title=_("Sum"))

    @qt_try_except()
    def compute_average(self) -> None:
        """Compute average"""

        def func_objs(new_obj: SignalObj, old_objs: list[SignalObj]) -> None:
            """Finalize average computation"""
            new_obj.data = new_obj.data / float(len(old_objs))
            if new_obj.dy is not None:
                new_obj.dy = new_obj.dy / float(len(old_objs))

        self.compute_n1("μ", cps.compute_add, func_objs=func_objs, title=_("Average"))

    @qt_try_except()
    def compute_product(self) -> None:
        """Compute product"""
        self.compute_n1("Π", cps.compute_product, title=_("Product"))

    @qt_try_except()
    def compute_roi_extraction(
        self, param: cdl.param.ROIDataParam | None = None
    ) -> None:
        """Extract Region Of Interest (ROI) from data"""
        param = self._get_roidataparam(param)
        if param is None or param.is_empty:
            return
        obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
        group = obj.roidata_to_params(param.roidata)
        if param.singleobj:
            self.compute_11(cps.extract_multiple_roi, group, title=_("Extract ROI"))
        else:
            self.compute_1n(cps.extract_single_roi, group.datasets, "ROI", edit=False)

    @qt_try_except()
    def compute_swap_axes(self) -> None:
        """Swap data axes"""
        self.compute_11(cps.compute_swap_axes, title=_("Swap axes"))

    @qt_try_except()
    def compute_abs(self) -> None:
        """Compute absolute value"""
        self.compute_11(cps.compute_abs, title=_("Absolute value"))

    @qt_try_except()
    def compute_re(self) -> None:
        """Compute real part"""
        self.compute_11(cps.compute_re, title=_("Real part"))

    @qt_try_except()
    def compute_im(self) -> None:
        """Compute imaginary part"""
        self.compute_11(cps.compute_im, title=_("Imaginary part"))

    @qt_try_except()
    def compute_astype(self, param: cdl.param.DataTypeSParam | None = None) -> None:
        """Convert data type"""
        self.compute_11(
            cps.compute_astype, param, cps.DataTypeSParam, title=_("Convert data type")
        )

    @qt_try_except()
    def compute_log10(self) -> None:
        """Compute Log10"""
        self.compute_11(cps.compute_log10, title="Log10")

    @qt_try_except()
    def compute_difference(self, obj2: SignalObj | None = None) -> None:
        """Compute difference between two signals"""
        self.compute_n1n(
            obj2,
            _("signal to subtract"),
            cps.compute_difference,
            title=_("Difference"),
        )

    @qt_try_except()
    def compute_quadratic_difference(self, obj2: SignalObj | None = None) -> None:
        """Compute quadratic difference between two signals"""
        self.compute_n1n(
            obj2,
            _("signal to subtract"),
            cps.compute_quadratic_difference,
            title=_("Quadratic difference"),
        )

    @qt_try_except()
    def compute_division(self, obj2: SignalObj | None = None) -> None:
        """Compute division between two signals"""
        self.compute_n1n(
            obj2,
            _("divider"),
            cps.compute_division,
            title=_("Division"),
        )

    @qt_try_except()
    def compute_peak_detection(
        self, param: cdl.param.PeakDetectionParam | None = None
    ) -> None:
        """Detect peaks from data"""
        obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
        edit, param = self.init_param(
            param, cps.PeakDetectionParam, _("Peak detection")
        )
        if edit:
            dlg = signalpeakdialog.SignalPeakDetectionDialog(self.panel)
            dlg.setup_data(obj.x, obj.y)
            if exec_dialog(dlg):
                param.threshold = int(dlg.get_threshold() * 100)
                param.min_dist = dlg.get_min_dist()
            else:
                return
        self.compute_11(cps.compute_peak_detection, param)

    # ------Signal Processing
    @qt_try_except()
    def compute_normalize(self, param: cdl.param.NormalizeYParam | None = None) -> None:
        """Normalize data"""
        self.compute_11(
            cps.compute_normalize, param, cps.NormalizeYParam, title=_("Normalize")
        )

    @qt_try_except()
    def compute_derivative(self) -> None:
        """Compute derivative"""
        self.compute_11(cps.compute_derivative, title=_("Derivative"))

    @qt_try_except()
    def compute_integral(self) -> None:
        """Compute integral"""
        self.compute_11(cps.compute_integral, title=_("Integral"))

    @qt_try_except()
    def compute_calibration(
        self, param: cdl.param.XYCalibrateParam | None = None
    ) -> None:
        """Compute data linear calibration"""
        self.compute_11(
            cps.compute_calibration,
            param,
            cps.XYCalibrateParam,
            title=_("Linear calibration"),
            comment="y = a.x + b",
        )

    @qt_try_except()
    def compute_threshold(self, param: cpb.ThresholdParam | None = None) -> None:
        """Compute threshold clipping"""
        self.compute_11(
            cps.compute_threshold, param, cpb.ThresholdParam, title=_("Thresholding")
        )

    @qt_try_except()
    def compute_clip(self, param: cpb.ClipParam | None = None) -> None:
        """Compute maximum data clipping"""
        self.compute_11(cps.compute_clip, param, cpb.ClipParam, title=_("Clipping"))

    @qt_try_except()
    def compute_gaussian_filter(self, param: cpb.GaussianParam | None = None) -> None:
        """Compute gaussian filter"""
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
        """Compute moving average"""
        self.compute_11(
            cps.compute_moving_average,
            param,
            cpb.MovingAverageParam,
            title=_("Moving average"),
        )

    @qt_try_except()
    def compute_moving_median(self, param: cpb.MovingMedianParam | None = None) -> None:
        """Compute moving median"""
        self.compute_11(
            cps.compute_moving_median,
            param,
            cpb.MovingMedianParam,
            title=_("Moving median"),
        )

    @qt_try_except()
    def compute_wiener(self) -> None:
        """Compute Wiener filter"""
        self.compute_11(cps.compute_wiener, title=_("Wiener filter"))

    @qt_try_except()
    def compute_fft(self, param: cdl.param.FFTParam | None = None) -> None:
        """Compute iFFT"""
        if param is None:
            param = cpb.FFTParam.create(shift=Conf.proc.fft_shift_enabled.get())
        self.compute_11(cps.compute_fft, param, title=_("FFT"), edit=False)

    @qt_try_except()
    def compute_ifft(self, param: cdl.param.FFTParam | None = None) -> None:
        """Compute FFT"""
        if param is None:
            param = cpb.FFTParam.create(shift=Conf.proc.fft_shift_enabled.get())
        self.compute_11(cps.compute_ifft, param, title=_("iFFT"), edit=False)

    @qt_try_except()
    def compute_interpolation(
        self,
        obj2: SignalObj | None = None,
        param: cdl.param.InterpolationParam | None = None,
    ):
        """Compute interpolation"""
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
        """Compute resampling"""
        edit, param = self.init_param(param, cps.ResamplingParam, _("Resampling"))
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
        self.compute_11(
            cps.compute_resampling,
            param,
            cps.ResamplingParam,
            title=_("Resampling"),
            edit=edit,
        )

    @qt_try_except()
    def compute_detrending(self, param: cdl.param.DetrendingParam | None = None):
        """Compute detrending"""
        self.compute_11(
            cps.compute_detrending,
            param,
            cps.DetrendingParam,
            title=_("Detrending"),
        )

    @qt_try_except()
    def compute_convolution(self, obj2: SignalObj | None = None) -> None:
        """Compute convolution"""
        self.compute_n1n(
            obj2,
            _("signal to convolve with"),
            cps.compute_convolution,
            title=_("Convolution"),
        )

    @qt_try_except()
    def compute_fit(self, name, fitdlgfunc):
        """Compute fitting curve"""
        for obj in self.panel.objview.get_sel_objects():
            self.__row_compute_fit(obj, name, fitdlgfunc)

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
    def compute_multigaussianfit(self) -> None:
        """Compute multi-Gaussian fitting curve"""
        fitdlgfunc = fitdialog.multigaussianfit
        for obj in self.panel.objview.get_sel_objects():
            dlg = signalpeakdialog.SignalPeakDetectionDialog(self.panel)
            dlg.setup_data(obj.x, obj.y)
            if exec_dialog(dlg):
                # Computing x, y
                peaks = dlg.get_peak_indexes()

                def multigaussianfit(x, y, parent=None):
                    """Multi-Gaussian fit dialog function"""
                    # pylint: disable=cell-var-from-loop
                    return fitdlgfunc(x, y, peaks, parent=parent)

                self.__row_compute_fit(obj, _("Multi-Gaussian fit"), multigaussianfit)

    # ------Signal Computing
    @qt_try_except()
    def compute_fwhm(
        self, param: cdl.param.FWHMParam | None = None
    ) -> dict[str, ResultShape]:
        """Compute FWHM"""
        return self.compute_10(
            cps.compute_fwhm, ShapeTypes.SEGMENT, param, cps.FWHMParam, title=_("FWHM")
        )

    @qt_try_except()
    def compute_fw1e2(self) -> dict[str, ResultShape]:
        """Compute FW at 1/e²"""
        return self.compute_10(
            cps.compute_fw1e2, ShapeTypes.SEGMENT, title=_("FW") + "1/e²"
        )

    def _get_stat_funcs(self) -> list[tuple[str, Callable[[np.ndarray], float]]]:
        """Return statistics functions list"""
        return [
            ("min(y)", lambda xy: xy[1].min()),
            ("max(y)", lambda xy: xy[1].max()),
            ("<y>", lambda xy: xy[1].mean()),
            ("median(y)", lambda xy: np.median(xy[1])),
            ("σ(y)", lambda xy: xy[1].std()),
            ("<y>/σ(y)", lambda xy: xy[1].mean() / xy[1].std()),
            ("peak-to-peak", lambda xy: xy[1].ptp()),
            ("Σ(y)", lambda xy: xy[1].sum()),
            ("∫ydx", lambda xy: np.trapz(xy[1], xy[0])),
        ]

    @qt_try_except()
    def compute_histogram(
        self, param: cdl.param.HistogramParam | None = None
    ) -> dict[str, ResultShape]:
        """Compute histogram"""
        return self.compute_11(
            cps.compute_histogram, param, cps.HistogramParam, title=_("Histogram")
        )
