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
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from cdl.config import Conf

if TYPE_CHECKING:  # pragma: no cover
    from plotpy.plot import BasePlot


class DockablePlotWidget(DockableWidget):
    """Docked plotting widget"""

    LOCATION = QC.Qt.LeftDockWidgetArea

    def __init__(
        self,
        parent: QW.QWidget,
        plot_type: PlotType,
    ) -> None:
        super().__init__(parent)
        self.plotwidget = PlotWidget(toolbar=True, options=PlotOptions(type=plot_type))
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

    def update_watermark(self) -> None:
        """Update watermark visibility"""
        items = self.get_plot().get_items()
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
