# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Unit test scenario Signal 01

Testing the following:
  - Add signal object at startup
  - Create signal (random type)
  - Sum two signals
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

from cdl.config import _
from cdl.core.gui.main import CDLMainWindow
from cdl.core.gui.panel.image import ImagePanel
from cdl.core.gui.panel.signal import SignalPanel
from cdl.core.gui.processor.base import (
    ClipParam,
    GaussianParam,
    MovingAverageParam,
    MovingMedianParam,
    ThresholdParam,
)
from cdl.core.gui.processor.signal import (
    FWHMParam,
    NormalizeParam,
    PeakDetectionParam,
    PolynomialFitParam,
    XYCalibrateParam,
)
from cdl.core.model.base import UniformRandomParam
from cdl.core.model.signal import (
    GaussLorentzVoigtParam,
    NewSignalParam,
    SignalTypes,
    create_signal_from_param,
    new_signal_param,
)
from cdl.tests import cdl_app_context
from cdl.tests.data import create_test_signal1
from cdl.tests.newobject_unit import iterate_signal_creation
from cdl.widgets import fitdialog

SHOW = True  # Show test in GUI-based test launcher


def test_compute_11_operations(panel: SignalPanel | ImagePanel, index: int) -> None:
    """Test compute_11 type operations on a signal or image

    Requires that one signal or image has been added at index."""
    assert len(panel.objmodel) >= index - 1
    panel.objview.select_nums((index,))
    panel.processor.compute_gaussian_filter(GaussianParam())
    panel.processor.compute_moving_average(MovingAverageParam())
    panel.processor.compute_moving_median(MovingMedianParam())
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
    assert len(panel.objmodel) == 2

    panel.duplicate_object()
    panel.objview.select_nums((1, 2))
    panel.processor.compute_difference(quadratic=False)
    panel.remove_object()
    panel.objview.select_nums((1, 2))
    panel.processor.compute_difference(quadratic=True)
    panel.delete_metadata()
    panel.objview.select_nums((2, 3))
    panel.remove_object()

    panel.objview.select_nums((0, 1))
    panel.processor.compute_sum()
    panel.objview.select_nums((0, 1))
    panel.processor.compute_sum()
    panel.objview.select_nums((0, 1))
    panel.processor.compute_product()

    obj = panel.objmodel.get_groups()[0][-1]
    param = ThresholdParam()
    param.value = (obj.data.max() - obj.data.min()) * 0.2 + obj.data.min()
    panel.processor.compute_threshold(param)
    param = ClipParam()  # Clipping before division...
    param.value = (obj.data.max() - obj.data.min()) * 0.8 + obj.data.min()
    panel.processor.compute_clip(param)

    panel.objview.select_nums((2, 6))
    panel.processor.compute_division()
    panel.objview.select_nums((0, 1, 2))
    panel.processor.compute_average()

    test_compute_11_operations(panel, 1)


def test_signal_features(win: CDLMainWindow, data_size: int = 500) -> None:
    """Testing signal features"""
    panel = win.signalpanel
    win.switch_to_panel("signal")

    for signal in iterate_signal_creation(data_size, non_zero=True):
        panel.add_object(create_test_signal1(data_size))
        panel.add_object(signal)
        test_common_operations(panel)
        panel.remove_all_objects()

    sig1 = create_test_signal1(data_size)
    win.add_object(sig1)

    # Add new signal based on s0
    panel.objview.set_current_object(sig1)
    newparam = new_signal_param(_("Random function"), stype=SignalTypes.UNIFORMRANDOM)
    addparam = UniformRandomParam()
    addparam.vmin = 0
    addparam.vmax = sig1.y.max() * 0.2
    panel.new_object(newparam, addparam=addparam, edit=False)

    test_common_operations(panel)

    win.add_object(create_test_signal1(data_size))

    param = NormalizeParam()
    for _name, method in param.methods:
        param.method = method
        panel.processor.compute_normalize(param)

    param = XYCalibrateParam()
    param.a, param.b = 1.2, 0.1
    panel.processor.compute_calibration(param)

    panel.processor.compute_derivative()
    panel.processor.compute_integral()

    param = PeakDetectionParam()
    panel.processor.compute_peak_detection(param)

    panel.processor.compute_multigaussianfit()

    panel.objview.select_nums([-3])
    sig = panel.objview.get_sel_objects()[0]
    i1 = data_size // 10
    i2 = len(sig.y) - i1
    panel.processor.extract_roi([[i1, i2]])

    param = PolynomialFitParam()
    panel.processor.compute_polyfit(param)

    panel.processor.compute_fit(_("Gaussian fit"), fitdialog.gaussianfit)
    panel.processor.compute_fit(_("Lorentzian fit"), fitdialog.lorentzianfit)
    panel.processor.compute_fit(_("Voigt fit"), fitdialog.voigtfit)

    newparam = NewSignalParam()
    newparam.title = _("Gaussian")
    newparam.type = SignalTypes.GAUSS
    sig = create_signal_from_param(newparam, GaussLorentzVoigtParam(), edit=False)
    panel.add_object(sig)

    param = FWHMParam()
    for fittype, _name in param.fittypes:
        param.fittype = fittype
        panel.processor.compute_fwhm(param)
    panel.processor.compute_fw1e2()


def test():
    """Run signal unit test scenario 1"""
    with cdl_app_context(save=True) as win:
        test_signal_features(win)
        oids = win.signalpanel.objmodel.get_object_ids()
        win.signalpanel.open_separate_view(oids[:3])


if __name__ == "__main__":
    test()
