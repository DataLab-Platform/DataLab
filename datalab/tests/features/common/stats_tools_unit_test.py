# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Custom curve and image stats tool unit test.

PlotPy's statistics tools (`plotpy.tools.XCurveStatsTool`,
`plotpy.tools.YCurveStatsTool`, and `plotpy.tools.ImageStatsTool`
are customized in `datalab.gui.docks` module).
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import numpy as np
import plotpy
from guidata.qthelpers import exec_dialog, qt_app_context
from packaging.version import Version
from plotpy.constants import PlotType
from plotpy.tests.unit.utils import drag_mouse
from plotpy.tools import CurveStatsTool, ImageStatsTool

try:
    from plotpy.tools import YRangeCursorTool
except ImportError:
    # YRangeCursorTool is not available in PlotPy < 2.8
    YRangeCursorTool = None

from qtpy import QtWidgets as QW

from datalab.adapters_plotpy.factories import create_adapter_from_object
from datalab.gui.docks import DataLabPlotWidget
from sigima.objects import ImageObj, SignalObj
from sigima.tests.data import create_multigaussian_image, create_paracetamol_signal


def simulate_stats_tool(
    plot_type: PlotType,
    obj: SignalObj | ImageObj,
    x_path: np.ndarray,
    y_path: np.ndarray,
) -> None:
    """Simulate stats tool with a custom signal or image."""
    widget = DataLabPlotWidget(plot_type)
    plot = widget.get_plot()
    item = create_adapter_from_object(obj).make_item()
    plot.add_item(item)
    plot.set_active_item(item)
    item.unselect()
    if plot_type == PlotType.CURVE:
        classes = [CurveStatsTool]
        if YRangeCursorTool is not None:
            classes.append(YRangeCursorTool)
    else:
        classes = [
            ImageStatsTool,
        ]
    for klass in classes:
        stattool = widget.get_manager().get_tool(klass)
        stattool.activate()
        if Version(plotpy.__version__) < Version("2.4"):
            qapp = QW.QApplication.instance()
            drag_mouse(widget, qapp, x_path, y_path)
        else:
            drag_mouse(widget, x_path, y_path)
    return widget


def test_stats_tool() -> None:
    """Test `XCurveStatsTool` with a custom signal."""
    sig = create_paracetamol_signal()
    ima = create_multigaussian_image()
    with qt_app_context():
        dlg = QW.QDialog()
        dlg.setWindowTitle("Stats Tool Test")
        dlg.resize(1200, 600)
        swidget = simulate_stats_tool(PlotType.CURVE, sig, [0.01, 0.02], [0.7, 0.9])
        iwidget = simulate_stats_tool(PlotType.IMAGE, ima, [0.2, 0.8], [0.9, 0.95])
        hlayout = QW.QHBoxLayout()
        hlayout.addWidget(swidget)
        hlayout.addWidget(iwidget)
        dlg.setLayout(hlayout)
    exec_dialog(dlg)


if __name__ == "__main__":
    test_stats_tool()
