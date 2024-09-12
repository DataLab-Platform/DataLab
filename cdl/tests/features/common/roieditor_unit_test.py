# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI editor unit test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import qtpy.QtWidgets as QW
from guidata.qthelpers import exec_dialog, qt_app_context
from plotpy.plot import PlotDialog

from cdl.core.gui.panel.image import ImagePanel
from cdl.core.gui.panel.signal import SignalPanel
from cdl.core.gui.roieditor import ImageROIEditor, SignalROIEditor
from cdl.env import execenv
from cdl.obj import create_image_roi, create_signal_roi
from cdl.tests.data import create_multigauss_image, create_paracetamol_signal


def test_signal_roi_editor():
    """Test signal ROI editor"""
    cls = SignalROIEditor
    title = f"Testing {cls.__name__}"
    options = SignalPanel.ROIDIALOGOPTIONS
    obj = create_paracetamol_signal()
    roi = create_signal_roi([50, 100], indices=True)
    obj.roi = roi
    with qt_app_context(exec_loop=False):
        execenv.print(title)
        toolbar = QW.QToolBar()
        dlg = PlotDialog(title=title, edit=True, options=options, toolbar=True)
        editor = cls(dlg, toolbar, obj, extract=True)
        dlg.button_layout.insertWidget(0, editor)
        exec_dialog(dlg)


def test_image_roi_editor():
    """Test image ROI editor"""
    cls = ImageROIEditor
    title = f"Testing {cls.__name__}"
    options = ImagePanel.ROIDIALOGOPTIONS
    obj = create_multigauss_image()
    roi = create_image_roi("rectangle", [500, 750, 1000, 1250])
    roi.add_roi(create_image_roi("circle", [1500, 1500, 500]))
    obj.roi = roi
    with qt_app_context(exec_loop=False):
        execenv.print(title)
        toolbar = QW.QToolBar()
        dlg = PlotDialog(title=title, edit=True, options=options, toolbar=True)
        editor = cls(dlg, toolbar, obj, extract=True)
        dlg.button_layout.insertWidget(0, editor)
        exec_dialog(dlg)


if __name__ == "__main__":
    test_signal_roi_editor()
    test_image_roi_editor()
