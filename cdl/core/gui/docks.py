# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Docks
=====

The :mod:`cdl.core.gui.docks` module provides the dockable widgets for the
DataLab main window.

Plot widget
-----------

.. autoclass:: DataLabPlotWidget

Dockable plot widget
--------------------

.. autoclass:: DockablePlotWidget
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import plotpy
import scipy.integrate as spt
from guidata.configtools import get_icon, get_image_file_path
from guidata.qthelpers import create_action, is_dark_mode
from guidata.widgets.dockable import DockableWidget
from plotpy.constants import PlotType
from plotpy.items import CurveItem
from plotpy.panels import XCrossSection, YCrossSection
from plotpy.plot import BasePlot, PlotOptions, PlotWidget
from plotpy.tools import (
    BasePlotMenuTool,
    CurveStatsTool,
    DeleteItemTool,
    DisplayCoordsTool,
    DoAutoscaleTool,
    EditItemDataTool,
    ExportItemDataTool,
    ImageStatsTool,
    ItemCenterTool,
    RectangularSelectionTool,
    RectZoomTool,
    SelectTool,
)
from plotpy.tools.image import get_stats as get_image_stats
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from qtpy.QtWidgets import QApplication, QMainWindow

from cdl.algorithms.image import get_centroid_fourier
from cdl.algorithms.signal import fwhm
from cdl.config import APP_NAME, Conf, _
from cdl.core.model.signal import create_signal
from cdl.utils.misc import compare_versions

if TYPE_CHECKING:
    from plotpy.items.image.base import BaseImageItem
    from plotpy.plot import BasePlot
    from plotpy.styles import BaseImageParam


def fwhm_info(x, y):
    """Return FWHM information string"""
    with warnings.catch_warnings(record=True) as w:
        x0, _y0, x1, _y1 = fwhm((x, y), "zero-crossing")
        wstr = " ⚠️" if w else ""
    return f"{x1 - x0:g}{wstr}"


CURVESTATSTOOL_LABELFUNCS = (
    ("%g &lt; x &lt; %g", lambda *args: (args[0].min(), args[0].max())),
    ("%g &lt; y &lt; %g", lambda *args: (args[1].min(), args[1].max())),
    ("&lt;y&gt;=%g", lambda *args: args[1].mean()),
    ("σ(y)=%g", lambda *args: args[1].std()),
    ("∑(y)=%g", lambda *args: spt.trapezoid(args[1])),
    ("∫ydx=%g<br>", lambda *args: spt.trapezoid(args[1], args[0])),
    ("FWHM = %s", fwhm_info),
)


def get_more_image_stats(
    item: BaseImageItem,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> str:
    """Return formatted string with stats on image rectangular area
    (output should be compatible with AnnotatedShape.get_infos)

    Args:
        item: image item
        x0: X0
        y0: Y0
        x1: X1
        y1: Y1
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        infos = get_image_stats(item, x0, y0, x1, y1)

    ix0, iy0, ix1, iy1 = item.get_closest_index_rect(x0, y0, x1, y1)
    data = item.data[iy0:iy1, ix0:ix1]
    p: BaseImageParam = item.param
    xunit, yunit, zunit = p.get_units()

    integral = data.sum()
    integral_fmt = r"%.3e " + zunit
    infos += "<br>∑ = %s" % (integral_fmt % integral)

    if xunit == yunit:
        surfacefmt = p.xformat.split()[0] + " " + xunit
        if xunit != "":
            surfacefmt = surfacefmt + "²"
        surface = abs((x1 - x0) * (y1 - y0))
        infos += "<br>A = %s" % (surfacefmt % surface)
        if xunit is not None and zunit is not None:
            if surface != 0:
                density = integral / surface
                densityfmt = r"%.3e"
                if xunit and zunit:
                    densityfmt += " " + zunit + "/" + xunit + "²"
                infos = infos + "<br>ρ = %s" % (densityfmt % density)

    c_i, c_j = get_centroid_fourier(data)
    c_x, c_y = item.get_plot_coordinates(c_j + ix0, c_i + iy0)
    infos += "<br>" + "<br>".join(
        [
            "C|x = " + p.xformat % c_x,
            "C|y = " + p.yformat % c_y,
        ]
    )

    return infos


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
        from cdl.core.gui import main

        # Note : this is the only way to retrieve the DataLab main window instance
        # when the CrossSectionItem object is embedded into an image widget
        # parented to another main window.
        win = main.CDLMainWindow.get_instance()
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


class DataLabPlotWidget(PlotWidget):
    """DataLab PlotWidget

    This class is a subclass of `plotpy.plot.PlotWidget` that provides a
    customized widget for DataLab, with a specific set of tools and a
    customized appearance.

    Args:
        plot_type: Plot type
    """

    def __init__(self, plot_type: PlotType) -> None:
        super().__init__(options=PlotOptions(type=plot_type), toolbar=True)

    def __register_standard_tools(self) -> None:
        """Register standard tools

        The only differences with the `manager.register_standard_tools` method are
        the following:

        1. We don't register the `BasePlotMenuTool, "axes"` tool, because it is not
        compatible with DataLab's approach to axes management.
        2. We don't register the `ItemListPanelTool` tool (this intends to prevent
        the user from accessing the item list panel, and thus, the parameters of all
        the items - some of them are read-only and should not be modified, like the
        annotations for example).
        """
        mgr = self.manager
        select_tool = mgr.add_tool(SelectTool)
        mgr.set_default_tool(select_tool)
        mgr.add_tool(RectangularSelectionTool, intersect=False)
        mgr.add_tool(RectZoomTool)
        mgr.add_tool(DoAutoscaleTool)
        mgr.add_tool(BasePlotMenuTool, "item")
        mgr.add_tool(ExportItemDataTool)
        mgr.add_tool(EditItemDataTool)
        mgr.add_tool(ItemCenterTool)
        mgr.add_tool(DeleteItemTool)
        mgr.add_separator_tool()
        mgr.add_tool(BasePlotMenuTool, "grid")
        mgr.add_tool(DisplayCoordsTool)

    def __register_other_tools(self) -> None:
        """Register other tools"""
        mgr = self.manager
        mgr.add_separator_tool()
        if self.options.type == PlotType.CURVE:
            mgr.register_curve_tools()
            statstool = mgr.get_tool(CurveStatsTool)
            statstool.set_labelfuncs(CURVESTATSTOOL_LABELFUNCS)
        else:
            mgr.register_image_tools()
            # Customizing the ImageStatsTool
            statstool = mgr.get_tool(ImageStatsTool)
            statstool.set_stats_func(get_more_image_stats, replace=True)
            # Customizing the X and Y cross section panels
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

        mgr.add_separator_tool()
        mgr.register_other_tools()
        mgr.add_separator_tool()
        mgr.update_tools_status()
        mgr.get_default_tool().activate()

    def register_tools(self) -> None:
        """Register the plotting tools according to the plot type"""
        self.__register_standard_tools()
        self.__register_other_tools()


class DockablePlotWidget(DockableWidget):
    """Docked plotting widget

    Args:
        parent: Parent widget
        plot_type: Plot type
    """

    LOCATION = QC.Qt.RightDockWidgetArea

    def __init__(
        self,
        parent: QW.QWidget,
        plot_type: PlotType,
    ) -> None:
        super().__init__(parent)
        self.plotwidget = DataLabPlotWidget(plot_type)
        self.toolbar = self.plotwidget.get_toolbar()
        self.watermark = QW.QLabel()
        original_image = QG.QPixmap(get_image_file_path("DataLab-watermark.png"))
        self.watermark.setPixmap(original_image)
        self.setup_layout()
        self.setup_plotwidget()

    def __get_toolbar_row_col(self) -> tuple[int, int]:
        """Return toolbar row and column"""
        tb_pos = Conf.view.plot_toolbar_position.get()
        tb_col, tb_row = 1, 1
        if tb_pos in ("left", "right"):
            self.toolbar.setOrientation(QC.Qt.Vertical)
            tb_col = 0 if tb_pos == "left" else 2
        else:
            self.toolbar.setOrientation(QC.Qt.Horizontal)
            tb_row = 0 if tb_pos == "top" else 2
        return tb_row, tb_col

    def setup_layout(self) -> None:
        """Setup layout"""
        tb_row, tb_col = self.__get_toolbar_row_col()
        layout = QW.QGridLayout()
        layout.addWidget(self.toolbar, tb_row, tb_col)
        layout.addWidget(self.plotwidget, 1, 1)
        layout.addWidget(self.watermark, 1, 1, QC.Qt.AlignCenter)
        self.setLayout(layout)

    def update_toolbar_position(self) -> None:
        """Update toolbar position"""
        tb_row, tb_col = self.__get_toolbar_row_col()
        layout = self.layout()
        layout.removeWidget(self.toolbar)
        layout.addWidget(self.toolbar, tb_row, tb_col)

    def setup_plotwidget(self) -> None:
        """Setup plotting widget"""
        title = self.toolbar.windowTitle()
        self.plotwidget.get_manager().add_toolbar(self.toolbar, title)
        #  Customizing widget appearances
        plot = self.plotwidget.get_plot()
        if not is_dark_mode():
            for widget in (self.plotwidget, plot, self):
                widget.setBackgroundRole(QG.QPalette.Window)
                widget.setAutoFillBackground(True)
                widget.setPalette(QG.QPalette(QC.Qt.white))
        canvas = plot.canvas()
        canvas.setFrameStyle(canvas.Plain | canvas.NoFrame)
        plot.SIG_ITEMS_CHANGED.connect(self.update_watermark)

    def get_plot(self) -> BasePlot:
        """Return plot instance"""
        return self.plotwidget.get_plot()

    def update_watermark(self, plot: BasePlot) -> None:
        """Update watermark visibility"""
        items = plot.get_items()
        if self.plotwidget.options.type == PlotType.IMAGE:
            enabled = len(items) <= 1
        else:
            enabled = len(items) <= 2
        self.watermark.setVisible(enabled)

    # ------DockableWidget API
    def visibility_changed(self, enable: bool) -> None:
        """DockWidget visibility has changed"""
        DockableWidget.visibility_changed(self, enable)
        self.toolbar.setVisible(enable)
