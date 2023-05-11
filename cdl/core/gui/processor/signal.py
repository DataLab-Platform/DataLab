# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Signal Processor GUI module
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import scipy.integrate as spt
import scipy.ndimage as spi
import scipy.optimize as spo
import scipy.signal as sps
from guidata.dataset.dataitems import BoolItem, ChoiceItem, FloatItem, IntItem
from guidata.dataset.datatypes import DataSet, DataSetGroup

from cdl.config import Conf, _
from cdl.core.computation import fit
from cdl.core.computation.signal import (
    derivative,
    moving_average,
    normalize,
    peak_indexes,
    xpeak,
    xy_fft,
    xy_ifft,
)
from cdl.core.gui.processor.base import BaseProcessor, ClipParam, ThresholdParam
from cdl.core.model.base import ShapeTypes
from cdl.core.model.signal import SignalParam, create_signal
from cdl.utils.qthelpers import exec_dialog, qt_try_except
from cdl.widgets import fitdialog, signalpeakdialog

if TYPE_CHECKING:
    from cdl.core.gui.processor.base import (
        GaussianParam,
        MovingAverageParam,
        MovingMedianParam,
    )


class PeakDetectionParam(DataSet):
    """Peak detection parameters"""

    threshold = IntItem(
        _("Threshold"), default=30, min=0, max=100, slider=True, unit="%"
    )
    min_dist = IntItem(_("Minimum distance"), default=1, min=1, unit="points")


class NormalizeParam(DataSet):
    """Normalize parameters"""

    methods = (
        (_("maximum"), "maximum"),
        (_("amplitude"), "amplitude"),
        (_("sum"), "sum"),
        (_("energy"), "energy"),
    )
    method = ChoiceItem(_("Normalize with respect to"), methods)


class XYCalibrateParam(DataSet):
    """Signal calibration parameters"""

    axes = (("x", _("X-axis")), ("y", _("Y-axis")))
    axis = ChoiceItem(_("Calibrate"), axes, default="y")
    a = FloatItem("a", default=1.0)
    b = FloatItem("b", default=0.0)


class PolynomialFitParam(DataSet):
    """Polynomial fitting parameters"""

    degree = IntItem(_("Degree"), 3, min=1, max=10, slider=True)


class FWHMParam(DataSet):
    """FWHM parameters"""

    fittypes = (
        ("GaussianModel", _("Gaussian")),
        ("LorentzianModel", _("Lorentzian")),
        ("VoigtModel", "Voigt"),
    )

    fittype = ChoiceItem(_("Fit type"), fittypes, default="GaussianModel")


class FFTParam(DataSet):
    """FFT parameters"""

    shift = BoolItem(
        _("Shift"),
        default=Conf.proc.fft_shift_enabled.get(),
        help=_("Shift zero frequency to center"),
    )


class SignalProcessor(BaseProcessor):
    """Object handling signal processing: operations, processing, computing"""

    # pylint: disable=duplicate-code

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

            def suffix_func(group: DataSetGroup):
                if len(group.datasets) == 1:
                    p = group.datasets[0]
                    return f"indexes={p.col1:d}:{p.col2:d}"
                return ""

            def extract_roi_func(x, y, group: DataSetGroup):
                """Extract ROI function"""
                xout, yout = np.ones_like(x) * np.nan, np.ones_like(y) * np.nan
                for p in group.datasets:
                    slice0 = slice(p.col1, p.col2 + 1)
                    xout[slice0], yout[slice0] = x[slice0], y[slice0]
                nans = np.isnan(xout) | np.isnan(yout)
                return xout[~nans], yout[~nans]

            # TODO: [P2] Instead of removing geometric shapes, apply roi extract
            self.compute_11(
                "ROI",
                extract_roi_func,
                group,
                suffix=suffix_func,
                func_obj=lambda obj, _orig, _group: obj.remove_all_shapes(),
                edit=False,
            )
        else:
            # TODO: [P2] Instead of removing geometric shapes, apply roi extract
            self.compute_1n(
                [f"ROI{iroi}" for iroi in range(len(group.datasets))],
                lambda x, y, p: (x[p.col1 : p.col2 + 1], y[p.col1 : p.col2 + 1]),
                group.datasets,
                suffix=lambda p: f"indexes={p.col1:d}:{p.col2:d}",
                func_obj=lambda obj, _orig, _group: obj.remove_all_shapes(),
                edit=False,
            )

    def swap_axes(self) -> None:
        """Swap data axes"""
        self.compute_11(
            "SwapAxes",
            lambda x, y: (y, x),
            func_obj=lambda obj, _orig: obj.remove_all_shapes(),
        )

    def compute_abs(self) -> None:
        """Compute absolute value"""
        self.compute_11("Abs", lambda x, y: (x, np.abs(y)))

    def compute_log10(self) -> None:
        """Compute Log10"""
        self.compute_11("Log10", lambda x, y: (x, np.log10(y)))

    def detect_peaks(self, param: PeakDetectionParam | None = None) -> None:
        """Detect peaks from data"""
        obj = self.panel.objview.get_sel_objects()[0]
        edit, param = self.init_param(param, PeakDetectionParam, _("Peak detection"))
        if edit:
            dlg = signalpeakdialog.SignalPeakDetectionDialog(self.panel)
            dlg.setup_data(obj.x, obj.y)
            if exec_dialog(dlg):
                param.threshold = int(dlg.get_threshold() * 100)
                param.min_dist = dlg.get_min_dist()

        def func(x, y, p) -> tuple[np.ndarray, np.ndarray]:
            """Peak detection"""
            indexes = peak_indexes(y, thres=p.threshold * 0.01, min_dist=p.min_dist)
            return x[indexes], y[indexes]

        def func_obj(
            obj: SignalParam, orig: SignalParam, param: PeakDetectionParam
        ) -> None:  # pylint: disable=unused-argument
            """Customize signal object"""
            obj.metadata["curvestyle"] = "Sticks"

        self.compute_11(
            "Peaks",
            func,
            param,
            suffix=lambda p: f"threshold={p.threshold}%, min_dist={p.min_dist}pts",
            func_obj=func_obj,
            edit=edit,
        )

    # ------Signal Processing
    def apply_11_func(self, obj, orig, func, param, message) -> None:
        """Apply 11 function: 1 object in --> 1 object out"""

        # (self is used by @qt_try_except)
        # pylint: disable=unused-argument
        @qt_try_except(message)
        def apply_11_func_callback(self, obj, orig, func, param):
            """Apply 11 function callback: 1 object in --> 1 object out"""
            data = orig.xydata
            if len(data) == 2:  # x, y signal
                x, y = data
                if param is None:
                    obj.xydata = func(x, y)
                else:
                    obj.xydata = func(x, y, param)
            elif len(data) == 4:  # x, y, dx, dy error bar signal
                x, y, dx, dy = data
                if param is None:
                    x2, y2 = func(x, y)
                    _x3, dy2 = func(x, dy)
                else:
                    x2, y2 = func(x, y, param)
                    _x3, dy2 = func(x, dy, param)
                obj.xydata = x2, y2, dx, dy2

        return apply_11_func_callback(self, obj, orig, func, param)

    @qt_try_except()
    def normalize(self, param: NormalizeParam | None = None) -> None:
        """Normalize data"""
        edit, param = self.init_param(param, NormalizeParam, _("Normalize"))

        def func(x, y, p):
            return (x, normalize(y, p.method))

        self.compute_11(
            "Normalize", func, param, suffix=lambda p: f"ref={p.method}", edit=edit
        )

    @qt_try_except()
    def compute_derivative(self) -> None:
        """Compute derivative"""
        self.compute_11("Derivative", lambda x, y: (x, derivative(x, y)))

    @qt_try_except()
    def compute_integral(self) -> None:
        """Compute integral"""
        self.compute_11("Integral", lambda x, y: (x, spt.cumtrapz(y, x, initial=0.0)))

    @qt_try_except()
    def compute_calibration(self, param: XYCalibrateParam | None = None) -> None:
        """Compute data linear calibration"""
        edit, param = self.init_param(
            param, XYCalibrateParam, _("Linear calibration"), "y = a.x + b"
        )

        def func(
            x: np.ndarray, y: np.ndarray, p: XYCalibrateParam
        ) -> tuple[np.ndarray, np.ndarray]:
            """Compute linear calibration"""
            if p.axis == "x":
                return p.a * x + p.b, y
            return x, p.a * y + p.b

        self.compute_11(
            "LinearCal",
            func,
            param,
            suffix=lambda p: f"{p.axis}={p.a}*{p.axis}+{p.b}",
            edit=edit,
        )

    @qt_try_except()
    def compute_threshold(self, param: ThresholdParam | None = None) -> None:
        """Compute threshold clipping"""
        edit, param = self.init_param(param, ThresholdParam, _("Thresholding"))
        self.compute_11(
            "Threshold",
            lambda x, y, p: (x, np.clip(y, p.value, y.max())),
            param,
            suffix=lambda p: f"min={p.value} lsb",
            edit=edit,
        )

    @qt_try_except()
    def compute_clip(self, param: ClipParam | None = None) -> None:
        """Compute maximum data clipping"""
        edit, param = self.init_param(param, ClipParam, _("Clipping"))
        self.compute_11(
            "Clip",
            lambda x, y, p: (x, np.clip(y, y.min(), p.value)),
            param,
            suffix=lambda p: f"max={p.value} lsb",
            edit=edit,
        )

    @staticmethod
    def func_gaussian_filter(
        x: np.ndarray, y: np.ndarray, p: GaussianParam
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute gaussian filter"""
        return (x, spi.gaussian_filter1d(y, p.sigma))

    @staticmethod
    def func_moving_average(
        x: np.ndarray, y: np.ndarray, p: MovingAverageParam
    ) -> tuple[np.ndarray, np.ndarray]:
        """Moving average computing function"""
        return (x, moving_average(y, p.n))

    @staticmethod
    def func_moving_median(
        x: np.ndarray, y: np.ndarray, p: MovingMedianParam
    ) -> tuple[np.ndarray, np.ndarray]:
        """Moving median computing function"""
        return (x, sps.medfilt(y, kernel_size=p.n))

    @qt_try_except()
    def compute_wiener(self) -> None:
        """Compute Wiener filter"""
        self.compute_11("WienerFilter", lambda x, y: (x, sps.wiener(y)))

    @qt_try_except()
    def compute_fft(self, param: FFTParam | None = None) -> None:
        """Compute iFFT"""
        if param is None:
            param = FFTParam()
        self.compute_11(
            f"FFT",
            lambda x, y, p: xy_fft(x, y, shift=p.shift),
            param,
            edit=False,
        )

    @qt_try_except()
    def compute_ifft(self, param: FFTParam | None = None) -> None:
        """Compute FFT"""
        if param is None:
            param = FFTParam()
        self.compute_11(
            f"iFFT",
            lambda x, y, p: xy_ifft(x, y, shift=p.shift),
            param,
            edit=False,
        )

    @qt_try_except()
    def compute_fit(self, name, fitdlgfunc):
        """Compute fitting curve"""
        for obj in self.panel.objview.get_sel_objects():
            self.__row_compute_fit(obj, name, fitdlgfunc)

    @qt_try_except()
    def compute_polyfit(self, param: PolynomialFitParam | None = None) -> None:
        """Compute polynomial fitting curve"""
        txt = _("Polynomial fit")
        edit, param = self.init_param(param, PolynomialFitParam, txt)
        if not edit or param.edit(self):
            dlgfunc = fitdialog.polynomialfit
            self.compute_fit(
                txt,
                lambda x, y, degree=param.degree, parent=self.panel.parent(): dlgfunc(
                    x, y, degree, parent=parent
                ),
            )

    def __row_compute_fit(
        self, obj: SignalParam, name: str, fitdlgfunc: Callable
    ) -> None:
        """Curve fitting computing sub-method"""
        output = fitdlgfunc(obj.x, obj.y, parent=self.panel.parent())
        if output is not None:
            y, params = output
            results = {}
            for param in params:
                if re.match(r"[\S\_]*\d{2}$", param.name):
                    shname = param.name[:-2]
                    value = results.get(shname, np.array([]))
                    results[shname] = np.array(list(value) + [param.value])
                else:
                    results[param.name] = param.value
            # Creating new signal
            signal = create_signal(f"{name}({obj.title})", obj.x, y, metadata=results)
            # Creating new plot item
            self.panel.add_object(signal)

    @qt_try_except()
    def compute_multigaussianfit(self):
        """Compute multi-Gaussian fitting curve"""
        fitdlgfunc = fitdialog.multigaussianfit
        for obj in self.panel.objview.get_sel_objects():
            dlg = signalpeakdialog.SignalPeakDetectionDialog(self.panel)
            dlg.setup_data(obj.x, obj.y)
            if exec_dialog(dlg):
                # Computing x, y
                peaks = dlg.get_peak_indexes()
                self.__row_compute_fit(
                    obj,
                    _("Multi-Gaussian fit"),
                    lambda x, y, peaks=peaks, parent=self.panel.parent(): fitdlgfunc(
                        x, y, peaks, parent=parent
                    ),
                )

    # ------Signal Computing
    @qt_try_except()
    def compute_fwhm(self, param: FWHMParam | None = None) -> None:
        """Compute FWHM"""
        title = _("FWHM")

        def fwhm(signal: SignalParam, param: FWHMParam):
            """Compute FWHM"""
            res = []
            for i_roi in signal.iterate_roi_indexes():
                x, y = signal.get_data(i_roi)
                dx = np.max(x) - np.min(x)
                dy = np.max(y) - np.min(y)
                base = np.min(y)
                sigma, mu = dx * 0.1, xpeak(x, y)
                FitModel = getattr(fit, param.fittype)
                amp = FitModel.get_amp_from_amplitude(dy, sigma)

                def func(params):
                    """Fitting model function"""
                    # pylint: disable=cell-var-from-loop
                    return y - FitModel.func(x, *params)

                (amp, sigma, mu, base), _ier = spo.leastsq(
                    func, np.array([amp, sigma, mu, base])
                )
                x0, y0, x1, y1 = FitModel.half_max_segment(amp, sigma, mu, base)
                res.append([i_roi, x0, y0, x1, y1])
            return signal.add_resultshape(
                title, ShapeTypes.SEGMENT, np.array(res), param
            )

        edit, param = self.init_param(param, FWHMParam, title)
        self.compute_10(title, fwhm, param, edit=edit)

    @qt_try_except()
    def compute_fw1e2(self):
        """Compute FW at 1/e²"""
        title = _("FW") + "1/e²"

        def fw1e2(signal: SignalParam):
            """Compute FW at 1/e²"""
            res = []
            for i_roi in signal.iterate_roi_indexes():
                x, y = signal.get_data(i_roi)
                dx = np.max(x) - np.min(x)
                dy = np.max(y) - np.min(y)
                base = np.min(y)
                sigma, mu = dx * 0.1, xpeak(x, y)
                amp = fit.GaussianModel.get_amp_from_amplitude(dy, sigma)
                p_in = np.array([amp, sigma, mu, base])

                def func(params):
                    """Fitting model function"""
                    # pylint: disable=cell-var-from-loop
                    return y - fit.GaussianModel.func(x, *params)

                p_out, _ier = spo.leastsq(func, p_in)
                amp, sigma, mu, base = p_out
                hw = 2 * sigma
                amplitude = fit.GaussianModel.amplitude(amp, sigma)
                yhm = amplitude / np.e**2 + base
                res.append([i_roi, mu - hw, yhm, mu + hw, yhm])
            return signal.add_resultshape(title, ShapeTypes.SEGMENT, np.array(res))

        self.compute_10(title, fw1e2)

    def _get_stat_funcs(self) -> list[tuple[str, Callable[[np.ndarray], float]]]:
        """Return statistics functions list"""
        return [
            ("min(y)", lambda xy: xy[1].min()),
            ("max(y)", lambda xy: xy[1].max()),
            ("<y>", lambda xy: xy[1].mean()),
            ("Median(y)", lambda xy: np.median(xy[1])),
            ("σ(y)", lambda xy: xy[1].std()),
            ("Σ(y)", lambda xy: xy[1].sum()),
            ("∫ydx", lambda xy: np.trapz(xy[1], xy[0])),
        ]
