# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Dockable widgets
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from guidata.configtools import get_image_file_path
from guidata.qthelpers import is_dark_mode
from guidata.widgets.dockable import DockableWidget
from plotpy.constants import PlotType
from plotpy.plot import PlotOptions, PlotWidget
from plotpy.tools import (
    BasePlotMenuTool,
    DeleteItemTool,
    DisplayCoordsTool,
    DoAutoscaleTool,
    EditItemDataTool,
    ExportItemDataTool,
    ItemCenterTool,
    ItemListPanelTool,
    RectangularSelectionTool,
    RectZoomTool,
    SelectTool,
)
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from cdl.config import Conf

if TYPE_CHECKING:
    from plotpy.plot import BasePlot


class DataLabPlotWidget(PlotWidget):
    """DataLab PlotWidget"""

    def __init__(self, plot_type: PlotType) -> None:
        super().__init__(options=PlotOptions(type=plot_type), toolbar=True)

    def __register_standard_tools(self) -> None:
        """Register standard tools

        The only difference with the `manager.register_standard_tools` method
        is the fact that we don't register the `BasePlotMenuTool, "axes"` tool,
        because it is not compatible with DataLab's approach to axes management.
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
        if mgr.get_itemlist_panel():
            mgr.add_tool(ItemListPanelTool)

    def __register_other_tools(self) -> None:
        """Register other tools"""
        mgr = self.manager
        mgr.add_separator_tool()
        if self.options.type == PlotType.CURVE:
            mgr.register_curve_tools()
        else:
            mgr.register_image_tools()
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
    """Docked plotting widget"""

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
