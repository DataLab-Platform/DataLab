# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Demo
====

The :mod:`datalab.gui.demo` module implements DataLab's demo, which shows a
selection of DataLab features without user interaction.

The demo is started from the **?** menu, the welcome page or the last step of the
tour (:mod:`datalab.gui.tour`), or with the ``datalab-demo`` command.

.. note::

    This module is part of the application: it must not import the test suite
    (:mod:`datalab.tests`), whose dependencies (e.g., ``pytest``) are not available
    in the standalone application.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

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

from datalab.config import _, ensure_initialized
from datalab.env import execenv
from datalab.gui.main import DLMainWindow
from datalab.utils.qthelpers import datalab_app_context

#: Delays (in seconds) between the steps of the demo
DELAY1, DELAY2, DELAY3 = 1, 2, 3


def demo_signal_features(win: DLMainWindow, data_size: int = 500) -> None:
    """Demonstrate signal features

    Args:
        win: DataLab main window
        data_size: number of points of the signals
    """
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
    panel.processor.compute_peak_detection(param)
    panel.processor.compute_markers_to_signal()
    sig4 = panel.objview.get_current_object()
    panel.objview.select_objects([sig3, sig4])

    qt_wait(DELAY2)

    panel.objview.set_current_object(sig3)
    panel.processor.compute_multigaussianfit()

    sig = sigima.objects.create_signal_from_param(sigima.objects.StepPulseParam())
    panel.add_object(sig)

    panel.processor.run_feature("extract_pulse_features")

    qt_wait(DELAY2)


def demo_image_features(win: DLMainWindow, data_size: int = 512) -> None:
    """Demonstrate image features

    Args:
        win: DataLab main window
        data_size: width and height of the images
    """
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

    rotparam = sipi.RotateParam.create(angle=5.0)
    for boundary in BorderMode:
        if boundary is BorderMode.MIRROR:
            continue
        rotparam.mode = boundary
        panel.processor.run_feature("rotate", rotparam)

    param.title = _("Multi-Gaussian")
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

    param.title = _("2D Peaks with noise background")
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
    """Play the demo in a DataLab main window, once confirmed by the user

    Args:
        win: DataLab main window
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
        demo_signal_features(win)
        demo_image_features(win)
        qt_wait(DELAY3)
        execenv.disable_demo_mode()
        QW.QMessageBox.information(win, _("Demo"), _("Click OK to end demo."))


def run() -> None:
    """Start DataLab and play the demo (``datalab-demo`` command)

    The demo uses the default settings: the user configuration is neither loaded
    nor modified.
    """
    ensure_initialized(load_user_config=False)
    with datalab_app_context(exec_loop=True):
        win = DLMainWindow(console=False)
        win.resize(1200, 700)
        win.show()
        play_demo(win)


if __name__ == "__main__":
    run()
