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
from guiqwt.plot import ImageWidget
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from cdl.config import Conf, _

if TYPE_CHECKING:  # pragma: no cover
    from guiqwt.plot import CurvePlot, CurveWidget, ImagePlot


class DockablePlotWidget(DockableWidget):
    """Docked plotting widget"""

    LOCATION = QC.Qt.LeftDockWidgetArea

    def __init__(
        self,
        parent: QW.QWidget,
        plotwidgetclass: CurveWidget | ImageWidget,
    ) -> None:
        super().__init__(parent)
        self.toolbar = QW.QToolBar(_("Plot Toolbar"), self)
        self.plotwidget = plotwidgetclass()
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
        pwidget = self.plotwidget
        pwidget.add_toolbar(self.toolbar, title)
        if isinstance(self.plotwidget, ImageWidget):
            pwidget.register_all_image_tools()
        else:
            pwidget.register_all_curve_tools()
        #  Customizing widget appearances
        plot = pwidget.get_plot()
        if not is_dark_mode():
            for widget in (pwidget, plot, self):
                widget.setBackgroundRole(QG.QPalette.Window)
                widget.setAutoFillBackground(True)
                widget.setPalette(QG.QPalette(QC.Qt.white))
        canvas = plot.canvas()
        canvas.setFrameStyle(canvas.Plain | canvas.NoFrame)
        plot.SIG_ITEMS_CHANGED.connect(self.update_watermark)

    def get_plot(self) -> CurvePlot | ImagePlot:
        """Return plot instance"""
        return self.plotwidget.plot

    def update_watermark(self) -> None:
        """Update watermark visibility"""
        items = self.get_plot().get_items()
        if isinstance(self.plotwidget, ImageWidget):
            enabled = len(items) <= 1
        else:
            enabled = len(items) <= 2
        self.watermark.setVisible(enabled)

    # ------DockableWidget API
    def visibility_changed(self, enable: bool) -> None:
        """DockWidget visibility has changed"""
        DockableWidget.visibility_changed(self, enable)
        self.toolbar.setVisible(enable)
