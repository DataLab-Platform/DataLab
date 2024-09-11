# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI editor unit test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

from guidata.qthelpers import exec_dialog, qt_app_context
from plotpy.plot import PlotDialog

import cdl.obj as dlo
from cdl.core.gui.panel.image import ImagePanel
from cdl.core.gui.panel.signal import SignalPanel
from cdl.core.gui.roieditor import ImageROIEditor, SignalROIEditor
from cdl.env import execenv
from cdl.tests.data import create_multigauss_image, create_paracetamol_signal


def __construct_roieditor_dialog(
    title: str, plot_options: dict, obj: dlo.SignalObj | dlo.ImageObj
) -> PlotDialog:
    """Construct ROI editor dialog"""
    dlg = PlotDialog(
        parent=None,
        title=title,
        edit=True,
        options=plot_options,
        toolbar=True,
    )
    plot = dlg.get_plot()
    plot.add_item(obj.make_item())
    return dlg


def test_signal_roi_editor():
    """Test signal ROI editor"""
    cls = SignalROIEditor
    title = f"Testing {cls.__name__}"
    obj = create_paracetamol_signal()
    with qt_app_context(exec_loop=False):
        execenv.print(title)
        dlg = __construct_roieditor_dialog(title, SignalPanel.ROIDIALOGOPTIONS, obj)
        editor = cls(dlg, obj, extract=True)
        dlg.button_layout.insertWidget(0, editor)
        editor.add_roi()
        exec_dialog(dlg)


def test_image_roi_editor():
    """Test image ROI editor"""
    cls = ImageROIEditor
    title = f"Testing {cls.__name__}"
    obj = create_multigauss_image()
    with qt_app_context(exec_loop=False):
        execenv.print(title)
        dlg = __construct_roieditor_dialog(title, ImagePanel.ROIDIALOGOPTIONS, obj)
        editor = cls(dlg, obj, extract=True)
        dlg.button_layout.insertWidget(0, editor)
        editor.add_roi("rectangle")
        editor.add_roi("circle")
        exec_dialog(dlg)


if __name__ == "__main__":
    test_signal_roi_editor()
    test_image_roi_editor()
