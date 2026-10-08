# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Docks
=====

The :mod:`datalab.gui.docks` module provides the dockable widgets for the
DataLab main window.

Plot widget
-----------

.. autoclass:: DataLabPlotWidget

Dockable plot widget
--------------------

.. autoclass:: DockablePlotWidget
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from guidata.configtools import get_icon
from guidata.qthelpers import create_action
from plotpy.items import CurveItem
from plotpy.panels import XCrossSection, YCrossSection
from qtpy.QtWidgets import QApplication, QMainWindow
from sigima.objects import create_signal
from sigimax.widgets import plotdock as sgmx_plotdock

from datalab.config import APP_NAME, _

if TYPE_CHECKING:
    from plotpy.plot import BasePlot


def profile_to_signal(plot: BasePlot, panel: XCrossSection | YCrossSection) -> None:
    """Send cross section curve to DataLab's signal list

    Args:
        panel: Cross section panel
    """
    win = None
    for win in QApplication.topLevelWidgets():
        if isinstance(win, QMainWindow):
            break
    if win is None or win.objectName() != APP_NAME:
        # pylint: disable=import-outside-toplevel
        # pylint: disable=cyclic-import
        from datalab.gui import main

        # Note : this is the only way to retrieve the DataLab main window instance
        # when the CrossSectionItem object is embedded into an image widget
        # parented to another main window.
        win = main.DLMainWindow.get_instance()
        assert win is not None  # Should never happen

    for item in panel.cs_plot.get_items():
        if not isinstance(item, CurveItem):
            continue
        x, y, _dx, _dy = item.get_data()
        if x is None or y is None or x.size == 0 or y.size == 0:
            continue
        signal = create_signal(item.param.label)
        if isinstance(panel, YCrossSection):
            signal.set_xydata(y, x)
            xaxis_name = "left"
            xunit = plot.get_axis_unit("bottom")
            if xunit:
                signal.title += " " + xunit
        else:
            signal.set_xydata(x, y)
            xaxis_name = "bottom"
            yunit = plot.get_axis_unit("left")
            if yunit:
                signal.title += " " + yunit
        signal.ylabel = plot.get_axis_title("right")
        signal.yunit = plot.get_axis_unit("right")
        signal.xlabel = plot.get_axis_title(xaxis_name)
        signal.xunit = plot.get_axis_unit(xaxis_name)
        win.signalpanel.add_object(signal)

    # Show DataLab main window on top, if not already visible
    win.show()
    win.raise_()


class DataLabPlotWidget(sgmx_plotdock.SigimaXPlotWidget):
    """DataLab PlotWidget

    Extends the SigimaX plot widget with the "Process signal" action, which
    sends a profile to the DataLab signal panel.

    Args:
        plot_type: Plot type
    """

    def _customize_image_panels(self) -> None:
        """Add the "Process signal" action to the cross section panel toolbars"""
        mgr = self.manager
        plot = mgr.get_plot()
        for panel in (mgr.get_xcs_panel(), mgr.get_ycs_panel()):
            to_signal_action = create_action(
                panel,
                _("Process signal"),
                icon=get_icon("to_signal.svg"),
                triggered=lambda panel=panel: profile_to_signal(plot, panel),
            )
            tb = panel.toolbar
            tb.insertSeparator(tb.actions()[0])
            tb.insertAction(tb.actions()[0], to_signal_action)


class DockablePlotWidget(sgmx_plotdock.DockablePlotWidget):
    """Docked plotting widget

    Args:
        parent: Parent widget
        plot_type: Plot type
    """

    PLOTWIDGET_CLASS = DataLabPlotWidget
