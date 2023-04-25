# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
DataLab Dockable widgets
"""

from guidata.configtools import get_image_file_path
from guidata.qthelpers import is_dark_mode
from guidata.qtwidgets import DockableWidget
from guiqwt.plot import ImageWidget
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW


class DockablePlotWidget(DockableWidget):
    """Docked plotting widget"""

    LOCATION = QC.Qt.LeftDockWidgetArea

    def __init__(self, parent, plotwidgetclass, toolbar):
        super().__init__(parent)
        self.toolbar = toolbar
        layout = QW.QGridLayout()
        self.plotwidget = plotwidgetclass()
        layout.addWidget(self.plotwidget, 0, 0)
        self.setLayout(layout)
        self.watermark = QW.QLabel()
        original_image = QG.QPixmap(get_image_file_path("DataLab-watermark.png"))
        self.watermark.setPixmap(original_image)
        layout.addWidget(self.watermark, 0, 0, QC.Qt.AlignCenter)
        self.setup()

    def get_plot(self):
        """Return plot instance"""
        return self.plotwidget.plot

    def setup(self):
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

    def update_watermark(self):
        """Update watermark visibility"""
        items = self.get_plot().get_items()
        if isinstance(self.plotwidget, ImageWidget):
            enabled = len(items) <= 1
        else:
            enabled = len(items) <= 2
        self.watermark.setVisible(enabled)

    # ------DockableWidget API
    def visibility_changed(self, enable):
        """DockWidget visibility has changed"""
        DockableWidget.visibility_changed(self, enable)
        self.toolbar.setVisible(enable)
