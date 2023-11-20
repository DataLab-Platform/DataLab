# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Signal processing test scenario
-------------------------------

Testing all the signal processing features.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

import cdl.obj as dlo
import cdl.param as dlp
from cdl.config import Conf, _
from cdl.core.gui.main import CDLMainWindow
from cdl.core.gui.panel.image import ImagePanel
from cdl.core.gui.panel.signal import SignalPanel
from cdl.env import execenv
from cdl.tests import test_cdl_app_context
from cdl.tests.data import create_paracetamol_signal
from cdl.tests.features.common.newobject_unit import iterate_signal_creation
from cdl.widgets import fitdialog


def test_compute_11_operations(panel: SignalPanel | ImagePanel, number: int) -> None:
    """Test compute_11 type operations on a signal or image

    Requires that one signal or image has been added at index."""
    assert panel.object_number >= number - 1
    panel.objview.select_objects((number,))
    panel.processor.compute_gaussian_filter(dlp.GaussianParam())
    panel.processor.compute_moving_average(dlp.MovingAverageParam())
    panel.processor.compute_moving_median(dlp.MovingMedianParam())
    panel.processor.compute_wiener()
    panel.processor.compute_fft()
    panel.processor.compute_ifft()
    panel.processor.compute_abs()
    panel.processor.compute_log10()
    panel.processor.compute_swap_axes()
    panel.processor.compute_swap_axes()


def test_common_operations(panel: SignalPanel | ImagePanel) -> None:
    """Test operations common to signal/image

    Requires that two (and only two) signals/images are created/added to panel

    First signal/image is supposed to be always the same (reference)
    Second signal/image is the tested object
    """
    assert panel.object_number == 2

    panel.objview.select_objects((2,))
    panel.processor.compute_difference()  # difference with itself
    panel.remove_object()
    panel.objview.select_objects((2,))
    panel.processor.compute_quadratic_difference()  # quadratic difference with itself
    panel.delete_metadata()
    panel.objview.select_objects((3,))
    panel.remove_object()

    panel.objview.select_objects((1, 2))
    panel.processor.compute_sum()
    panel.objview.select_objects((1, 2))
    panel.processor.compute_sum()
    panel.objview.select_objects((1, 2))
    panel.processor.compute_product()

    obj = panel.objmodel.get_groups()[0][-1]
    param = dlp.ThresholdParam()
    param.value = (obj.data.max() - obj.data.min()) * 0.2 + obj.data.min()
    panel.processor.compute_threshold(param)
    param = dlp.ClipParam()  # Clipping before division...
    param.value = (obj.data.max() - obj.data.min()) * 0.8 + obj.data.min()
    panel.processor.compute_clip(param)

    panel.objview.select_objects((3, 7))
    panel.processor.compute_division()
    panel.objview.select_objects((1, 2, 3))
    panel.processor.compute_average()

    panel.add_label_with_title()

    test_compute_11_operations(panel, 2)


def test_signal_features(
    win: CDLMainWindow, data_size: int = 500, all_types: bool = True
) -> None:
    """Testing signal features"""
    panel = win.signalpanel
    win.set_current_panel("signal")

    if all_types:
        for signal in iterate_signal_creation(data_size, non_zero=True):
            panel.add_object(create_paracetamol_signal(data_size))
            panel.add_object(signal)
            test_common_operations(panel)
            panel.remove_all_objects()

    sig1 = create_paracetamol_signal(data_size)
    win.add_object(sig1)

    # Add new signal based on s0
    panel.objview.set_current_object(sig1)
    newparam = dlo.new_signal_param(
        _("Random function"), stype=dlo.SignalTypes.UNIFORMRANDOM
    )
    addparam = dlo.UniformRandomParam.create(vmin=0, vmax=sig1.y.max() * 0.2)
    panel.new_object(newparam, addparam=addparam, edit=False)

    test_common_operations(panel)

    win.add_object(create_paracetamol_signal(data_size))

    param = dlp.NormalizeYParam()
    for _name, method in param.methods:
        param.method = method
        panel.processor.compute_normalize(param)

    param = dlp.XYCalibrateParam.create(a=1.2, b=0.1)
    panel.processor.compute_calibration(param)

    panel.processor.compute_derivative()
    panel.processor.compute_integral()

    param = dlp.PeakDetectionParam()
    panel.processor.compute_peak_detection(param)

    panel.processor.compute_multigaussianfit()

    panel.objview.select_objects([-3])
    sig = panel.objview.get_sel_objects()[0]
    i1 = data_size // 10
    i2 = len(sig.y) - i1
    panel.processor.compute_roi_extraction(dlp.ROIDataParam.create([[i1, i2]]))

    param = dlp.PolynomialFitParam()
    panel.processor.compute_polyfit(param)

    panel.processor.compute_fit(_("Gaussian fit"), fitdialog.gaussianfit)
    panel.processor.compute_fit(_("Lorentzian fit"), fitdialog.lorentzianfit)
    panel.processor.compute_fit(_("Voigt fit"), fitdialog.voigtfit)

    newparam = dlo.new_signal_param(_("Gaussian"), stype=dlo.SignalTypes.GAUSS)
    sig = dlo.create_signal_from_param(
        newparam, dlo.GaussLorentzVoigtParam(), edit=False
    )
    panel.add_object(sig)

    param = dlp.FWHMParam()
    for fittype, _name in param.fittypes:
        param.fittype = fittype
        panel.processor.compute_fwhm(param)
    panel.processor.compute_fw1e2()


def test_scenario_sig() -> None:
    """Run signal unit test scenario"""
    assert (
        Conf.main.process_isolation_enabled.get()
    ), "Process isolation must be enabled"
    with test_cdl_app_context(save=True) as win:
        execenv.print(f"Testing signal features (process isolation: off)...")
        win.set_process_isolation_enabled(False)
        test_signal_features(win, all_types=True)
        win.signalpanel.remove_all_objects()
        execenv.print("==> OK")
        execenv.print(f"Testing signal features (process isolation: on)...")
        win.set_process_isolation_enabled(True)
        test_signal_features(win, all_types=False)
        oids = win.signalpanel.objmodel.get_object_ids()
        win.signalpanel.open_separate_view(oids[:3])
        execenv.print("==> OK")


if __name__ == "__main__":
    test_scenario_sig()
