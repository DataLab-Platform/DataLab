# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT Dockable widgets
"""

from guidata.qthelpers import is_dark_mode
from guidata.qtwidgets import DockableWidget, DockableWidgetMixin
from guiqwt.plot import ImageWidget
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW


class DockablePlotWidget(DockableWidget):
    """Docked plotting widget"""

    LOCATION = QC.Qt.RightDockWidgetArea

    def __init__(self, parent, plotwidgetclass, toolbar):
        super().__init__(parent)
        self.toolbar = toolbar
        layout = QW.QVBoxLayout()
        self.plotwidget = plotwidgetclass()
        layout.addWidget(self.plotwidget)
        self.setLayout(layout)
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

    # ------DockableWidget API
    def visibility_changed(self, enable):
        """DockWidget visibility has changed"""
        DockableWidget.visibility_changed(self, enable)
        self.toolbar.setVisible(enable)


class DockableTabWidget(QW.QTabWidget, DockableWidgetMixin):
    """Docked tab widget"""

    LOCATION = QC.Qt.LeftDockWidgetArea
