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
from typing import Any

import guidata.dataset.datatypes as gdt
import numpy as np

import cdl.core.computation.base as cpb
import cdl.core.computation.signal as cps
from cdl.config import _
from cdl.core.gui.processor.base import BaseProcessor
from cdl.core.model.base import ShapeTypes
from cdl.core.model.signal import SignalObj, create_signal
from cdl.utils.qthelpers import exec_dialog, qt_try_except
from cdl.widgets import fitdialog, signalpeakdialog


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

            def suffix_func(group: gdt.DataSetGroup) -> str:
                """Suffix function

                Args:
                    group (gdt.DataSetGroup): group of parameters

                Returns:
                    str: suffix
                """
                if len(group.datasets) == 1:
                    p = group.datasets[0]
                    return f"indexes={p.col1:d}:{p.col2:d}"
                return ""

            # TODO: [P2] Instead of removing geometric shapes, apply roi extract
            self.compute_11(
                "ROI",
                cps.extract_multiple_roi,
                group,
                suffix=suffix_func,
                func_obj=lambda obj, _orig, _group: obj.remove_all_shapes(),
                edit=False,
            )
        else:
            # TODO: [P2] Instead of removing geometric shapes, apply roi extract
            self.compute_1n(
                [f"ROI{iroi}" for iroi in range(len(group.datasets))],
                cps.extract_single_roi,
                group.datasets,
                suffix=lambda p: f"indexes={p.col1:d}:{p.col2:d}",
                func_obj=lambda obj, _orig, _group: obj.remove_all_shapes(),
                edit=False,
            )

    def compute_swap_axes(self) -> None:
        """Swap data axes"""
        self.compute_11(
            "SwapAxes",
            cps.compute_swap_axes,
            func_obj=lambda obj, _orig: obj.remove_all_shapes(),
        )

    def compute_abs(self) -> None:
        """Compute absolute value"""
        self.compute_11("Abs", cps.compute_abs)

    def compute_log10(self) -> None:
        """Compute Log10"""
        self.compute_11("Log10", cps.compute_log10)

    def compute_peak_detection(
        self, param: cps.PeakDetectionParam | None = None
    ) -> None:
        """Detect peaks from data"""
        obj = self.panel.objview.get_sel_objects()[0]
        edit, param = self.init_param(
            param, cps.PeakDetectionParam, _("Peak detection")
        )
        if edit:
            dlg = signalpeakdialog.SignalPeakDetectionDialog(self.panel)
            dlg.setup_data(obj.x, obj.y)
            if exec_dialog(dlg):
                param.threshold = int(dlg.get_threshold() * 100)
                param.min_dist = dlg.get_min_dist()

        # pylint: disable=unused-argument
        def func_obj(
            obj: SignalObj, orig: SignalObj, param: cps.PeakDetectionParam
        ) -> None:
            """Customize signal object"""
            obj.metadata["curvestyle"] = "Sticks"

        self.compute_11(
            "Peaks",
            cps.compute_peak_detection,
            param,
            suffix=lambda p: f"threshold={p.threshold}%, min_dist={p.min_dist}pts",
            func_obj=func_obj,
            edit=edit,
        )

    # ------Signal Processing
    def get_11_func_args(self, orig: SignalObj, param: gdt.DataSet) -> tuple[Any]:
        """Get 11 function args: 1 object in --> 1 object out"""
        data = orig.xydata
        if len(data) == 2:  # x, y signal
            x, y = data
            if param is None:
                return (x, y)
            return (x, y, param)
        if len(data) == 4:  # x, y, dx, dy error bar signal
            x, y, _dx, dy = data
            raise NotImplementedError("Error bar signal processing not implemented")
        raise ValueError("Invalid data")

    def set_11_func_result(self, new_obj: SignalObj, result: tuple[np.ndarray]) -> None:
        """Set 11 function result: 1 object in --> 1 object out"""
        x, y = result
        new_obj.set_xydata(x, y)

    @qt_try_except()
    def compute_normalize(self, param: cps.NormalizeParam | None = None) -> None:
        """Normalize data"""
        edit, param = self.init_param(param, cps.NormalizeParam, _("Normalize"))
        self.compute_11(
            "Normalize",
            cps.compute_normalize,
            param,
            suffix=lambda p: f"ref={p.method}",
            edit=edit,
        )

    @qt_try_except()
    def compute_derivative(self) -> None:
        """Compute derivative"""
        self.compute_11("Derivative", cps.compute_derivative)

    @qt_try_except()
    def compute_integral(self) -> None:
        """Compute integral"""
        self.compute_11("Integral", cps.compute_integral)

    @qt_try_except()
    def compute_calibration(self, param: cps.XYCalibrateParam | None = None) -> None:
        """Compute data linear calibration"""
        edit, param = self.init_param(
            param, cps.XYCalibrateParam, _("Linear calibration"), "y = a.x + b"
        )
        self.compute_11(
            "LinearCal",
            cps.compute_calibration,
            param,
            suffix=lambda p: f"{p.axis}={p.a}*{p.axis}+{p.b}",
            edit=edit,
        )

    @qt_try_except()
    def compute_threshold(self, param: cpb.ThresholdParam | None = None) -> None:
        """Compute threshold clipping"""
        edit, param = self.init_param(param, cpb.ThresholdParam, _("Thresholding"))
        self.compute_11(
            "Threshold",
            cps.compute_threshold,
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
            cps.compute_clip,
            param,
            suffix=lambda p: f"max={p.value} lsb",
            edit=edit,
        )

    @qt_try_except()
    def compute_gaussian_filter(self, param: cpb.GaussianParam | None = None) -> None:
        """Compute gaussian filter"""
        edit, param = self.init_param(param, cpb.GaussianParam, _("Gaussian filter"))
        self.compute_11(
            "GaussianFilter",
            cps.compute_gaussian_filter,
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
            cps.compute_moving_average,
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
            cps.compute_moving_median,
            param,
            suffix=lambda p: f"n={p.n}",
            edit=edit,
        )

    @qt_try_except()
    def compute_wiener(self) -> None:
        """Compute Wiener filter"""
        self.compute_11("WienerFilter", cps.compute_wiener)

    @qt_try_except()
    def compute_fft(self, param: cps.FFTParam | None = None) -> None:
        """Compute iFFT"""
        if param is None:
            param = cps.FFTParam()
        self.compute_11(
            "FFT",
            cps.compute_fft,
            param,
            edit=False,
        )

    @qt_try_except()
    def compute_ifft(self, param: cps.FFTParam | None = None) -> None:
        """Compute FFT"""
        if param is None:
            param = cps.FFTParam()
        self.compute_11(
            "iFFT",
            cps.compute_ifft,
            param,
            edit=False,
        )

    @qt_try_except()
    def compute_fit(self, name, fitdlgfunc):
        """Compute fitting curve"""
        for obj in self.panel.objview.get_sel_objects():
            self.__row_compute_fit(obj, name, fitdlgfunc)

    @qt_try_except()
    def compute_polyfit(self, param: cps.PolynomialFitParam | None = None) -> None:
        """Compute polynomial fitting curve"""
        txt = _("Polynomial fit")
        edit, param = self.init_param(param, cps.PolynomialFitParam, txt)
        if not edit or param.edit(self):
            dlgfunc = fitdialog.polynomialfit
            self.compute_fit(
                txt,
                lambda x, y, degree=param.degree, parent=self.panel.parent(): dlgfunc(
                    x, y, degree, parent=parent
                ),
            )

    def __row_compute_fit(
        self, obj: SignalObj, name: str, fitdlgfunc: Callable
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
    def compute_fwhm(self, param: cps.FWHMParam | None = None) -> None:
        """Compute FWHM"""
        title = _("FWHM")
        edit, param = self.init_param(param, cps.FWHMParam, title)
        self.compute_10(title, cps.compute_fwhm, ShapeTypes.SEGMENT, param, edit=edit)

    @qt_try_except()
    def compute_fw1e2(self):
        """Compute FW at 1/e²"""
        title = _("FW") + "1/e²"
        self.compute_10(title, cps.compute_fw1e2, ShapeTypes.SEGMENT)

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
