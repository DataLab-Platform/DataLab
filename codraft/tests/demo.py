# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT Demo
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from qtpy import QtWidgets as QW

from codraft.config import _, reset
from codraft.core.gui.main import CodraFTMainWindow
from codraft.core.gui.processor.image import ContourShapeParam
from codraft.core.gui.processor.image import PeakDetectionParam as Peak2DDetectionParam
from codraft.core.gui.processor.image import RotateParam
from codraft.core.gui.processor.signal import PeakDetectionParam, PolynomialFitParam
from codraft.core.model.base import UniformRandomParam
from codraft.core.model.image import ImageTypes, create_image, new_image_param
from codraft.core.model.signal import (
    GaussLorentzVoigtParam,
    SignalParamNew,
    SignalTypes,
    create_signal_from_param,
    new_signal_param,
)
from codraft.env import execenv
from codraft.tests import codraft_app_context
from codraft.tests.data import (
    PeakDataParam,
    create_test_image1,
    create_test_image2,
    create_test_signal1,
    get_peak2d_data,
)
from codraft.tests.roi_app import create_test_image_with_roi
from codraft.tests.scenario_sig_app import test_common_operations
from codraft.utils.qthelpers import qt_wait
from codraft.widgets import fitdialog

DELAY1, DELAY2, DELAY3 = 1, 2, 4
# DELAY1, DELAY2, DELAY3 = 0, 0, 0


def test_signal_features(win: CodraFTMainWindow, data_size: int = 500) -> None:
    """Testing signal features"""
    panel = win.signalpanel
    win.switch_to_signal_panel()

    qt_wait(DELAY2)

    sig1 = create_test_signal1(data_size)
    win.add_object(sig1)

    panel.objlist.set_current_row(0)
    newparam = new_signal_param(_("Random function"), stype=SignalTypes.UNIFORMRANDOM)
    addparam = UniformRandomParam()
    addparam.vmin = 0
    addparam.vmax = sig1.y.max() * 0.2
    panel.new_object(newparam, addparam=addparam, edit=True)

    test_common_operations(panel)
    panel.processor.normalize()
    panel.processor.compute_derivative()
    panel.processor.compute_integral()

    param = PeakDetectionParam()
    panel.processor.detect_peaks(param)

    qt_wait(DELAY2)

    param = PolynomialFitParam()
    panel.processor.compute_polyfit(param)

    panel.processor.compute_fit(_("Gaussian fit"), fitdialog.gaussianfit)

    newparam = SignalParamNew()
    newparam.title = _("Gaussian")
    newparam.type = SignalTypes.GAUSS
    sig = create_signal_from_param(newparam, GaussLorentzVoigtParam(), edit=False)
    panel.add_object(sig)

    panel.processor.compute_fwhm()
    panel.processor.compute_fw1e2()

    qt_wait(DELAY2)


def test_image_features(win: CodraFTMainWindow, data_size: int = 1000) -> None:
    """Testing signal features"""
    win.switch_to_image_panel()
    panel = win.imagepanel

    ima1 = create_test_image2(data_size)
    panel.add_object(ima1)

    qt_wait(DELAY2)

    panel.objlist.set_current_row(0)
    newparam = new_image_param(itype=ImageTypes.UNIFORMRANDOM)
    addparam = UniformRandomParam()
    addparam.set_from_datatype(ima1.data.dtype)
    addparam.vmax = int(ima1.data.max() * 0.2)
    panel.new_object(newparam, addparam=addparam, edit=False)

    test_common_operations(panel)

    ima1 = create_test_image1(data_size)
    panel.add_object(ima1)

    qt_wait(DELAY3)

    panel.processor.rotate_90()
    panel.processor.rotate_270()
    panel.processor.flip_horizontally()
    panel.processor.flip_vertically()

    param = RotateParam()
    param.angle = 5.0
    for boundary in param.boundaries[:-1]:
        param.mode = boundary
        panel.processor.rotate_arbitrarily(param)

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
    param = Peak2DDetectionParam()
    param.create_rois = True
    panel.processor.compute_peak_detection(param)

    qt_wait(DELAY3)

    param = ContourShapeParam()
    panel.processor.compute_contour_shape(param)

    qt_wait(DELAY3)

    panel.processor.extract_roi()


def run():
    """Run demo"""
    reset()  # Reset configuration (remove configuration file and initialize it)
    execenv.enable_demo_mode(DELAY1)
    with codraft_app_context(console=False) as win:
        QW.QMessageBox.information(win, "Demo", "Click OK to start demo")
        test_signal_features(win)
        test_image_features(win)
        qt_wait(DELAY3)
        QW.QMessageBox.information(win, "Demo", "Click OK to end demo")


if __name__ == "__main__":
    run()
