# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Demo
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show,skip

from __future__ import annotations

from typing import TYPE_CHECKING

import sigima.objects
import sigima.params
import sigima.proc.image as sipi
from guidata.qthelpers import qt_wait
from qtpy import QtWidgets as QW
from sigima.enums import BorderMode
from sigima.tests.data import (
    create_multigaussian_image,
    create_paracetamol_signal,
    create_peak_image,
    create_sincos_image,
    get_test_image,
)

from datalab.config import _, reset
from datalab.env import execenv
from datalab.tests import datalab_test_app_context

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow

DELAY1, DELAY2, DELAY3 = 1, 2, 3
# DELAY1, DELAY2, DELAY3 = 0, 0, 0


def test_signal_features(win: DLMainWindow, data_size: int = 500) -> None:
    """Testing signal features"""
    panel = win.signalpanel
    win.set_current_panel("signal")

    qt_wait(DELAY2)

    sig1 = create_paracetamol_signal(data_size)
    win.add_object(sig1)

    qt_wait(DELAY1)

    panel.objview.set_current_object(sig1)
    param = sigima.objects.UniformDistribution1DParam.create(
        title=_("Random function"), vmin=0, vmax=sig1.y.max() * 0.2
    )
    sig2 = sigima.objects.create_signal_from_param(param)
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

    param = sigima.params.PeakDetectionParam()
    panel.processor.run_feature("peak_detection", param)
    sig4 = panel.objview.get_current_object()
    panel.objview.select_objects([sig3, sig4])

    qt_wait(DELAY2)

    panel.objview.set_current_object(sig3)
    panel.processor.compute_multigaussianfit()

    param = sigima.objects.GaussParam.create(title=_("Gaussian"))
    sig = sigima.objects.create_signal_from_param(param)
    panel.add_object(sig)

    panel.processor.run_feature("fwhm")
    panel.processor.run_feature("fw1e2")

    qt_wait(DELAY2)


def test_image_features(win: DLMainWindow, data_size: int = 512) -> None:
    """Testing signal features"""
    win.set_current_panel("image")
    panel = win.imagepanel

    panel.add_object(get_test_image("flower.npy"))
    ima1 = panel.objview.get_current_object()

    qt_wait(DELAY2)

    panel.objview.set_current_object(ima1)

    param = sigima.objects.UniformDistribution2DParam.create(
        width=data_size, height=data_size
    )
    param.set_from_datatype(ima1.data.dtype)
    param.vmax = int(ima1.data.max() * 0.2)
    ima2 = sigima.objects.create_image_from_param(param)
    panel.add_object(ima2)

    panel.objview.select_objects((1, 2))
    panel.processor.run_feature("addition")
    qt_wait(DELAY2)
    # compute_common_operations(panel)

    panel.processor.run_feature("histogram")
    qt_wait(DELAY2)

    param = sigima.objects.NewImageParam.create(width=data_size, height=data_size)
    ima1 = create_sincos_image(param)
    panel.add_object(ima1)

    qt_wait(DELAY3)

    panel.processor.run_feature("rotate90")
    panel.processor.run_feature("rotate270")
    panel.processor.run_feature("fliph")
    panel.processor.run_feature("flipv")

    param = sipi.RotateParam.create(angle=5.0)
    for boundary in BorderMode:
        if boundary is BorderMode.MIRROR:
            continue
        param.mode = boundary
        panel.processor.run_feature("rotate", param)

    param.title = None
    ima1 = create_multigaussian_image(param)
    s = data_size
    roi = sigima.objects.create_image_roi(
        "rectangle", [s // 2, s // 2, s - 25 - s // 2, s - s // 2]
    )
    roi.add_roi(sigima.objects.create_image_roi("circle", [s // 3, s // 2, s // 4]))
    ima1.roi = roi
    panel.add_object(ima1)

    qt_wait(DELAY2)

    panel.processor.run_feature("centroid")

    qt_wait(DELAY1)

    panel.processor.run_feature("enclosing_circle")

    qt_wait(DELAY2)

    param.title = None
    ima = create_peak_image(param)
    panel.add_object(ima)
    param = sigima.params.Peak2DDetectionParam.create(create_rois=True)
    panel.processor.run_feature("peak_detection", param)

    qt_wait(DELAY2)

    param = sigima.params.ContourShapeParam()
    panel.processor.run_feature("contour_shape", param)

    qt_wait(DELAY2)

    n = data_size // 10
    roi = sigima.objects.create_image_roi(
        "rectangle", [n, n, data_size - 2 * n, data_size - 2 * n]
    )
    panel.processor.compute_roi_extraction(roi)


def play_demo(win: DLMainWindow) -> None:
    """Play demo

    Args:
        win: DLMainWindow instance
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
    with datalab_test_app_context(console=False) as win:
        play_demo(win)


if __name__ == "__main__":
    run()
