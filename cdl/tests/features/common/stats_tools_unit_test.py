# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Custom curve and image stats tool unit test.

PlotPy's statistics tools (`plotpy.tools.CurveStatsTool` and
`plotpy.tools.ImageStatsTool` are customized in `cdl.core.gui.docks` module).
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import numpy as np
import plotpy
import pytest
from guidata.qthelpers import exec_dialog, qt_app_context
from plotpy.constants import PlotType
from plotpy.tests.unit.utils import drag_mouse
from plotpy.tools import CurveStatsTool, ImageStatsTool
from qtpy import QtWidgets as QW

import cdl.obj
from cdl.core.gui.docks import DataLabPlotWidget
from cdl.tests.data import create_multigauss_image, create_paracetamol_signal
from cdl.utils.misc import compare_versions


def simulate_stats_tool(
    plot_type: PlotType,
    obj: cdl.obj.SignalObj | cdl.obj.ImageObj,
    x_path: np.ndarray,
    y_path: np.ndarray,
) -> None:
    """Simulate stats tool with a custom signal or image."""
    widget = DataLabPlotWidget(plot_type)
    plot = widget.get_plot()
    item = obj.make_item()
    plot.add_item(item)
    plot.set_active_item(item)
    item.unselect()
    klass = CurveStatsTool if plot_type == PlotType.CURVE else ImageStatsTool
    stattool = widget.get_manager().get_tool(klass)
    stattool.activate()
    if compare_versions(plotpy.__version__, "<", "2.4"):
        qapp = QW.QApplication.instance()
        drag_mouse(widget, qapp, x_path, y_path)
    else:
        drag_mouse(widget, x_path, y_path)
    return widget


# TODO: PlotPy 2.4 - Remove skipif decorator
@pytest.mark.skipif(
    compare_versions(plotpy.__version__, "<", "2.4"),
    reason="PlotPy version < 2.4",
)
def test_stats_tool() -> None:
    """Test CurveStatsTool with a custom signal."""
    sig = create_paracetamol_signal()
    ima = create_multigauss_image()
    with qt_app_context():
        dlg = QW.QDialog()
        dlg.setWindowTitle("Stats Tool Test")
        dlg.resize(1200, 600)
        swidget = simulate_stats_tool(PlotType.CURVE, sig, [0.01, 0.02], [0.5, 0.5])
        iwidget = simulate_stats_tool(PlotType.IMAGE, ima, [0.2, 0.8], [0.7, 0.9])
        hlayout = QW.QHBoxLayout()
        hlayout.addWidget(swidget)
        hlayout.addWidget(iwidget)
        dlg.setLayout(hlayout)
    exec_dialog(dlg)


if __name__ == "__main__":
    test_stats_tool()
