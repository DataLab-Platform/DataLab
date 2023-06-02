# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Demo
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from qtpy import QtWidgets as QW

import cdl.param
from cdl.config import _, reset
from cdl.core.gui.main import CDLMainWindow
from cdl.env import execenv
from cdl.obj import (
    GaussLorentzVoigtParam,
    ImageTypes,
    NewSignalParam,
    SignalTypes,
    create_image,
    create_signal_from_param,
    new_image_param,
    new_signal_param,
)
from cdl.param import UniformRandomParam
from cdl.tests import cdl_app_context
from cdl.tests.data import (
    PeakDataParam,
    create_test_image1,
    create_test_image2,
    create_test_signal1,
    get_peak2d_data,
)
from cdl.tests.features.common.roi_app import create_test_image_with_roi
from cdl.tests.scenarios.scenario_sig_app import test_common_operations
from cdl.utils.qthelpers import qt_wait
from cdl.widgets import fitdialog

DELAY1, DELAY2, DELAY3 = 1, 2, 4
# DELAY1, DELAY2, DELAY3 = 0, 0, 0


def test_signal_features(win: CDLMainWindow, data_size: int = 500) -> None:
    """Testing signal features"""
    panel = win.signalpanel
    win.switch_to_panel("signal")

    qt_wait(DELAY2)

    sig1 = create_test_signal1(data_size)
    win.add_object(sig1)

    panel.objview.set_current_object(sig1)
    newparam = new_signal_param(_("Random function"), stype=SignalTypes.UNIFORMRANDOM)
    addparam = UniformRandomParam()
    addparam.vmin = 0
    addparam.vmax = sig1.y.max() * 0.2
    panel.new_object(newparam, addparam=addparam, edit=True)

    test_common_operations(panel)
    panel.processor.compute_normalize()
    panel.processor.compute_derivative()
    panel.processor.compute_integral()

    param = cdl.param.Peak2DDetectionParam()
    panel.processor.compute_peak_detection(param)

    qt_wait(DELAY2)

    param = cdl.param.PolynomialFitParam()
    panel.processor.compute_polyfit(param)

    panel.processor.compute_fit(_("Gaussian fit"), fitdialog.gaussianfit)

    newparam = NewSignalParam()
    newparam.title = _("Gaussian")
    newparam.type = SignalTypes.GAUSS
    sig = create_signal_from_param(newparam, GaussLorentzVoigtParam(), edit=False)
    panel.add_object(sig)

    panel.processor.compute_fwhm()
    panel.processor.compute_fw1e2()

    qt_wait(DELAY2)


def test_image_features(win: CDLMainWindow, data_size: int = 1000) -> None:
    """Testing signal features"""
    win.switch_to_panel("image")
    panel = win.imagepanel

    ima1 = create_test_image2(data_size)
    panel.add_object(ima1)

    qt_wait(DELAY2)

    panel.objview.set_current_object(ima1)
    newparam = new_image_param(itype=ImageTypes.UNIFORMRANDOM)
    addparam = UniformRandomParam()
    addparam.set_from_datatype(ima1.data.dtype)
    addparam.vmax = int(ima1.data.max() * 0.2)
    panel.new_object(newparam, addparam=addparam, edit=False)

    test_common_operations(panel)

    ima1 = create_test_image1(data_size)
    panel.add_object(ima1)

    qt_wait(DELAY3)

    panel.processor.compute_rotate90()
    panel.processor.compute_rotate270()
    panel.processor.compute_fliph()
    panel.processor.compute_flipv()

    param = cdl.param.RotateParam()
    param.angle = 5.0
    for boundary in param.boundaries[:-1]:
        param.mode = boundary
        panel.processor.compute_rotate(param)

    ima1 = create_test_image_with_roi(data_size)
    panel.add_object(ima1)

    qt_wait(DELAY2)

    panel.processor.compute_centroid()

    qt_wait(DELAY1)

    panel.processor.compute_enclosing_circle()

    qt_wait(DELAY2)

    data = get_peak2d_data(PeakDataParam(size=data_size))
    ima = create_image("Test image with peaks", data)
    panel.add_object(ima)
    param = cdl.param.Peak2DDetectionParam()
    param.create_rois = True
    panel.processor.compute_peak_detection(param)

    qt_wait(DELAY3)

    param = cdl.param.ContourShapeParam()
    panel.processor.compute_contour_shape(param)

    qt_wait(DELAY3)

    panel.processor.extract_roi()


def run():
    """Run demo"""
    reset()  # Reset configuration (remove configuration file and initialize it)
    execenv.enable_demo_mode(DELAY1)
    with cdl_app_context(console=False) as win:
        QW.QMessageBox.information(win, "Demo", "Click OK to start demo")
        test_signal_features(win)
        test_image_features(win)
        qt_wait(DELAY3)
        QW.QMessageBox.information(win, "Demo", "Click OK to end demo")


if __name__ == "__main__":
    run()
