# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Demo
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show,skip

from __future__ import annotations

from typing import TYPE_CHECKING

from guidata.qthelpers import qt_wait
from qtpy import QtWidgets as QW

import sigima_.image.geometry
import sigima_.param as sp
from cdl.config import _, reset
from cdl.env import execenv
from cdl.tests import cdltest_app_context
from cdl.tests.data import (
    create_multigauss_image,
    create_paracetamol_signal,
    create_peak2d_image,
    create_sincos_image,
    get_test_image,
)
from sigima_ import (
    GaussLorentzVoigtParam,
    ImageTypes,
    NewImageParam,
    NewSignalParam,
    SignalTypes,
    UniformRandomParam,
    create_image_from_param,
    create_image_roi,
    create_signal_from_param,
)

if TYPE_CHECKING:
    from cdl.gui.main import CDLMainWindow

DELAY1, DELAY2, DELAY3 = 1, 2, 3
# DELAY1, DELAY2, DELAY3 = 0, 0, 0


def test_signal_features(win: CDLMainWindow, data_size: int = 500) -> None:
    """Testing signal features"""
    panel = win.signalpanel
    win.set_current_panel("signal")

    qt_wait(DELAY2)

    sig1 = create_paracetamol_signal(data_size)
    win.add_object(sig1)

    qt_wait(DELAY1)

    panel.objview.set_current_object(sig1)
    base_param = NewSignalParam.create(
        title=_("Random function"), stype=SignalTypes.UNIFORMRANDOM
    )
    extra_param = UniformRandomParam.create(vmin=0, vmax=sig1.y.max() * 0.2)
    sig2 = create_signal_from_param(base_param, extra_param=extra_param)
    win.add_object(sig2)

    # compute_common_operations(panel)
    panel.objview.select_objects((1, 2))
    qt_wait(DELAY1)
    panel.processor.run_feature("addition")
    qt_wait(DELAY1)

    panel.processor.run_feature("normalize")
    panel.processor.run_feature("derivative")
    panel.processor.run_feature("integral")

    panel.objview.set_current_object(sig1)
    qt_wait(DELAY1)
    panel.processor.run_feature("detrending")
    sig3 = panel.objview.get_current_object()

    param = sp.PeakDetectionParam()
    panel.processor.run_feature("peak_detection", param)
    sig4 = panel.objview.get_current_object()
    panel.objview.select_objects([sig3, sig4])

    qt_wait(DELAY2)

    panel.objview.set_current_object(sig3)
    panel.processor.compute_multigaussianfit()

    base_param = NewSignalParam.create(title=_("Gaussian"), stype=SignalTypes.GAUSS)
    sig = create_signal_from_param(base_param, GaussLorentzVoigtParam())
    panel.add_object(sig)

    panel.processor.run_feature("fwhm")
    panel.processor.run_feature("fw1e2")

    qt_wait(DELAY2)


def test_image_features(win: CDLMainWindow, data_size: int = 512) -> None:
    """Testing signal features"""
    win.set_current_panel("image")
    panel = win.imagepanel

    base_param = NewImageParam.create(height=data_size, width=data_size)

    # ima1 = create_noisygauss_image(newparam)
    # panel.add_object(ima1)

    panel.add_object(get_test_image("flower.npy"))
    ima1 = panel.objview.get_current_object()

    qt_wait(DELAY2)

    panel.objview.set_current_object(ima1)

    base_param = NewImageParam.create(
        itype=ImageTypes.UNIFORMRANDOM,
        height=base_param.height,
        width=base_param.width,
    )
    base_param = UniformRandomParam()
    base_param.set_from_datatype(ima1.data.dtype)
    base_param.vmax = int(ima1.data.max() * 0.2)
    ima2 = create_image_from_param(base_param, extra_param=base_param)
    panel.add_object(ima2)

    panel.objview.select_objects((1, 2))
    panel.processor.run_feature("addition")
    qt_wait(DELAY2)
    # compute_common_operations(panel)

    panel.processor.run_feature("histogram")
    qt_wait(DELAY2)

    base_param.title = None
    ima1 = create_sincos_image(base_param)
    panel.add_object(ima1)

    qt_wait(DELAY3)

    panel.processor.run_feature("rotate90")
    panel.processor.run_feature("rotate270")
    panel.processor.run_feature("fliph")
    panel.processor.run_feature("flipv")

    param = sigima_.image.geometry.RotateParam.create(angle=5.0)
    for boundary in param.boundaries[:-1]:
        param.mode = boundary
        panel.processor.run_feature("rotate", param)

    base_param.title = None
    ima1 = create_multigauss_image(base_param)
    s = data_size
    roi = create_image_roi("rectangle", [s // 2, s // 2, s - 25 - s // 2, s - s // 2])
    roi.add_roi(create_image_roi("circle", [s // 3, s // 2, s // 4]))
    ima1.roi = roi
    panel.add_object(ima1)

    qt_wait(DELAY2)

    panel.processor.run_feature("centroid")

    qt_wait(DELAY1)

    panel.processor.run_feature("enclosing_circle")

    qt_wait(DELAY2)

    base_param.title = None
    ima = create_peak2d_image(base_param)
    panel.add_object(ima)
    param = sp.Peak2DDetectionParam.create(create_rois=True)
    panel.processor.run_feature("peak_detection", param)

    qt_wait(DELAY2)

    param = sp.ContourShapeParam()
    panel.processor.run_feature("contour_shape", param)

    qt_wait(DELAY2)

    n = data_size // 10
    roi = create_image_roi("rectangle", [n, n, data_size - 2 * n, data_size - 2 * n])
    panel.processor.compute_roi_extraction(roi)


def play_demo(win: CDLMainWindow) -> None:
    """Play demo

    Args:
        win: CDLMainWindow instance
    """
    ret = QW.QMessageBox.information(
        win,
        _("Demo"),
        _(
            "Click OK to start the demo.<br><br><u>Note:</u><br>"
            "- Demo will cover a <i>selection</i> of DataLab features "
            "(for a complete list of features, please refer to the documentation).<br>"
            "- It won't require any user interaction."
        ),
        QW.QMessageBox.Ok | QW.QMessageBox.Cancel,
    )
    if ret == QW.QMessageBox.Ok:
        execenv.enable_demo_mode(int(DELAY1 * 1000))
        test_signal_features(win)
        test_image_features(win)
        qt_wait(DELAY3)
        execenv.disable_demo_mode()
        QW.QMessageBox.information(win, _("Demo"), _("Click OK to end demo."))


def run():
    """Run demo"""
    reset()  # Reset configuration (remove configuration file and initialize it)
    with cdltest_app_context(console=False) as win:
        play_demo(win)


if __name__ == "__main__":
    run()
