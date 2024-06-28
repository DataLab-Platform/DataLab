# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Custom curve stats tool unit test.
(PlotPy's CurveStatsTool is patched by `cdl.patch` module.)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import numpy as np
from guidata.qthelpers import exec_dialog, qt_app_context
from plotpy.builder import make
from plotpy.tests.unit.utils import drag_mouse
from plotpy.tools import CurveStatsTool

from cdl.tests.data import create_paracetamol_signal


def test_curve_stats_tool():
    """Test CurveStatsTool with a custom signal."""
    sig = create_paracetamol_signal()
    with qt_app_context() as qapp:
        dlg = make.dialog(title="Curve Stats Tool Test", toolbar=True)
        dlg.resize(640, 480)
        plot = dlg.get_plot()
        item = sig.make_item()
        plot.add_item(item)
        mgr = dlg.get_manager()
        plot.set_active_item(item)
        item.unselect()
        curvestattool = mgr.get_tool(CurveStatsTool)
        curvestattool.activate()
        x_path, y_path = np.array([0.1, 0.5]), np.array([0.1, 0.1])
        try:
            # PlotPy 2.3 and earlier
            drag_mouse(dlg, qapp, x_path, y_path)
        except TypeError:
            # PlotPy 2.4 and later
            drag_mouse(dlg, x_path, y_path)
        exec_dialog(dlg)


if __name__ == "__main__":
    test_curve_stats_tool()
