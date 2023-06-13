# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Demo
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show,skip

from qtpy import QtWidgets as QW

import cdl.obj as dlo
import cdl.param as dlp
from cdl.config import _, reset
from cdl.core.gui.main import CDLMainWindow
from cdl.env import execenv
from cdl.tests import cdl_app_context
from cdl.tests.data import (
    create_noisygauss_image,
    create_paracetamol_signal,
    create_peak2d_image,
    create_sincos_image,
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

    sig1 = create_paracetamol_signal(data_size)
    win.add_object(sig1)

    panel.objview.set_current_object(sig1)
    newparam = dlo.new_signal_param(
        _("Random function"), stype=dlo.SignalTypes.UNIFORMRANDOM
    )
    addparam = dlo.UniformRandomParam.create(vmin=0, vmax=sig1.y.max() * 0.2)
    sig2 = dlo.create_signal_from_param(newparam, addparam=addparam, edit=False)
    win.add_object(sig2)

    test_common_operations(panel)
    panel.processor.compute_normalize()
    panel.processor.compute_derivative()
    panel.processor.compute_integral()

    param = dlp.Peak2DDetectionParam()
    panel.processor.compute_peak_detection(param)

    qt_wait(DELAY2)

    param = dlp.PolynomialFitParam()
    panel.processor.compute_polyfit(param)

    panel.processor.compute_fit(_("Gaussian fit"), fitdialog.gaussianfit)

    newparam = dlo.new_signal_param(_("Gaussian"), stype=dlo.SignalTypes.GAUSS)
    sig = dlo.create_signal_from_param(
        newparam, dlo.GaussLorentzVoigtParam(), edit=False
    )
    panel.add_object(sig)

    panel.processor.compute_fwhm()
    panel.processor.compute_fw1e2()

    qt_wait(DELAY2)


def test_image_features(win: CDLMainWindow, data_size: int = 1000) -> None:
    """Testing signal features"""
    win.switch_to_panel("image")
    panel = win.imagepanel

    newparam = dlo.new_image_param(height=data_size, width=data_size)

    ima1 = create_noisygauss_image(newparam)
    panel.add_object(ima1)

    qt_wait(DELAY2)

    panel.objview.set_current_object(ima1)

    newparam = dlo.new_image_param(itype=dlo.ImageTypes.UNIFORMRANDOM)
    addparam = dlo.UniformRandomParam()
    addparam.set_from_datatype(ima1.data.dtype)
    addparam.vmax = int(ima1.data.max() * 0.2)
    ima2 = dlo.create_image_from_param(newparam, addparam=addparam, edit=False)
    panel.add_object(ima2)

    test_common_operations(panel)

    ima1 = create_sincos_image(newparam)
    panel.add_object(ima1)

    qt_wait(DELAY3)

    panel.processor.compute_rotate90()
    panel.processor.compute_rotate270()
    panel.processor.compute_fliph()
    panel.processor.compute_flipv()

    param = dlp.RotateParam.create(angle=5.0)
    for boundary in param.boundaries[:-1]:
        param.mode = boundary
        panel.processor.compute_rotate(param)

    ima1 = create_test_image_with_roi(newparam)
    panel.add_object(ima1)

    qt_wait(DELAY2)

    panel.processor.compute_centroid()

    qt_wait(DELAY1)

    panel.processor.compute_enclosing_circle()

    qt_wait(DELAY2)

    ima = create_peak2d_image(newparam)
    panel.add_object(ima)
    param = dlp.Peak2DDetectionParam.create(create_rois=True)
    panel.processor.compute_peak_detection(param)

    qt_wait(DELAY3)

    param = dlp.ContourShapeParam()
    panel.processor.compute_contour_shape(param)

    qt_wait(DELAY3)

    panel.processor.compute_roi_extraction()


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
