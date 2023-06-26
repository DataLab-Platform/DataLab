# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Visualization tools (based on guiqwt)
"""

import guiqwt.image
import guiqwt.plot
import guiqwt.tools
from guidata.configtools import get_icon
from guidata.qthelpers import win32_fix_title_bar_background
from guiqwt.baseplot import BasePlot
from guiqwt.builder import make
from guiqwt.plot import ImagePlot, PlotManager, SubplotWidget
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from cdl.config import _
from cdl.utils.qthelpers import exec_dialog, qt_app_context
from cdl.utils.tests import get_default_test_name

TEST_NB = {}


def get_name_title(name, title):
    """Return (default) widget name and title"""
    if name is None:
        TEST_NB[name] = TEST_NB.setdefault(name, 0) + 1
        name = get_default_test_name(f"{TEST_NB[name]:02d}")
    if title is None:
        title = f'{_("Test dialog")} `{name}`'
    return name, title


def create_curve_dialog(name=None, title=None, xlabel=None, ylabel=None):
    """Create Curve Dialog"""
    name, title = get_name_title(name, title)
    win = guiqwt.plot.CurveDialog(
        edit=False,
        icon=get_icon("DataLab.svg"),
        toolbar=True,
        wintitle=title,
        options={"xlabel": xlabel, "ylabel": ylabel},
    )
    win.setObjectName(name)
    return win


def view_curve_items(items, name=None, title=None, xlabel=None, ylabel=None):
    """Create a curve dialog and plot items"""
    win = create_curve_dialog(name=name, title=title, xlabel=xlabel, ylabel=ylabel)
    plot = win.get_plot()
    for item in items:
        plot.add_item(item)
    exec_dialog(win)


def view_curves(data_or_datalist, name=None, title=None, xlabel=None, ylabel=None):
    """Create a curve dialog and plot curves"""
    if isinstance(data_or_datalist, (tuple, list)):
        datalist = data_or_datalist
    else:
        datalist = [data_or_datalist]
    items = []
    for data in datalist:
        if len(data) == 2:
            xdata, ydata = data
            item = make.mcurve(xdata, ydata)
        else:
            item = make.mcurve(data)
        items.append(item)
    view_curve_items(items, name=name, title=title, xlabel=xlabel, ylabel=ylabel)


def create_image_dialog(name=None, title=None, xlabel=None, ylabel=None):
    """Create Image Dialog"""
    name, title = get_name_title(name, title)
    win = guiqwt.plot.ImageDialog(
        edit=False,
        icon=get_icon("DataLab.svg"),
        toolbar=True,
        wintitle=title,
        options={"xlabel": xlabel, "ylabel": ylabel},
    )
    win.setObjectName(name)
    for toolklass in (
        guiqwt.tools.LabelTool,
        guiqwt.tools.VCursorTool,
        guiqwt.tools.HCursorTool,
        guiqwt.tools.XCursorTool,
        guiqwt.tools.AnnotatedRectangleTool,
        guiqwt.tools.AnnotatedCircleTool,
        guiqwt.tools.AnnotatedEllipseTool,
        guiqwt.tools.AnnotatedSegmentTool,
        guiqwt.tools.AnnotatedPointTool,
    ):
        win.add_tool(toolklass, switch_to_default_tool=True)
    return win


def view_image_items(items, name=None, title=None, xlabel=None, ylabel=None):
    """Create an image dialog and show items"""
    win = create_image_dialog(name=name, title=title, xlabel=xlabel, ylabel=ylabel)
    plot = win.get_plot()
    for item in items:
        plot.add_item(item)
    exec_dialog(win)


def view_images(data_or_datalist, name=None, title=None, xlabel=None, ylabel=None):
    """Create an image dialog and show images"""
    if isinstance(data_or_datalist, (tuple, list)):
        datalist = data_or_datalist
    else:
        datalist = [data_or_datalist]
    items = []
    for data in datalist:
        item = make.image(data, interpolation="nearest", eliminate_outliers=0.1)
        items.append(item)
    view_image_items(items, name=name, title=title, xlabel=xlabel, ylabel=ylabel)


class SyncPlotWindow(QW.QMainWindow):
    """Window for showing plots, optionally synchronized"""

    def __init__(self, parent=None, title=None, icon=None):
        super().__init__(parent)
        win32_fix_title_bar_background(self)
        self.setWindowTitle(self.__doc__ if title is None else title)
        if icon is not None:
            self.setWindowIcon(get_icon(icon) if isinstance(icon, str) else icon)
        self.manager = PlotManager(None)
        self.manager.set_main(self)
        self.subplotwidget = SubplotWidget(self.manager, parent=self)
        self.setCentralWidget(self.subplotwidget)
        toolbar = QW.QToolBar(_("Tools"))
        self.manager.add_toolbar(toolbar, "default")
        toolbar.setMovable(True)
        toolbar.setFloatable(True)
        self.addToolBar(toolbar)

    def add_panels(self):
        """Add standard panels"""
        self.subplotwidget.add_standard_panels()

    def rescale_plots(self):
        """Rescale all plots"""
        QW.QApplication.instance().processEvents()
        for plot in self.subplotwidget.plots:
            plot.do_autoscale()

    def showEvent(self, event):  # pylint: disable=C0103
        """Reimplement Qt method"""
        super().showEvent(event)
        QC.QTimer.singleShot(0, self.rescale_plots)

    def add_plot(self, row, col, plot, sync=False, plot_id=None):
        """Add plot to window"""
        if plot_id is None:
            plot_id = str(len(self.subplotwidget.plots) + 1)
        self.subplotwidget.add_subplot(plot, row, col, plot_id)
        if sync and len(self.subplotwidget.plots) > 1:
            syncaxis = self.manager.synchronize_axis
            for i_plot in range(len(self.subplotwidget.plots) - 1):
                syncaxis(BasePlot.X_BOTTOM, [plot_id, f"{i_plot + 1}"])
                syncaxis(BasePlot.Y_LEFT, [plot_id, f"{i_plot + 1}"])


def __compute_grid(num_objects, max_cols=4, fixed_num_rows=None):
    """Compute number of rows and columns for a grid of images"""
    num_cols = min(num_objects, max_cols)
    if fixed_num_rows is not None:
        num_rows = fixed_num_rows
        num_cols = (num_objects + num_rows - 1) // num_rows
    else:
        num_rows = (num_objects + num_cols - 1) // num_cols
    return num_rows, num_cols


def view_images_side_by_side(
    images,
    titles,
    share_axes=True,
    rows=None,
    maximized=False,
    title=None,
):
    """Show sequence of images"""
    rows, cols = __compute_grid(len(images), fixed_num_rows=rows, max_cols=4)
    with qt_app_context(exec_loop=True):
        win = SyncPlotWindow(title=title, icon="datalab.svg")
        for idx, (data, imtitle) in enumerate(zip(images, titles)):
            row = idx // cols
            col = idx % cols
            plot = ImagePlot(title=imtitle)
            item = make.image(data, interpolation="nearest", eliminate_outliers=0.1)
            plot.add_item(item)
            win.add_plot(row, col, plot, sync=share_axes)
        win.add_panels()
        if maximized:
            win.resize(1200, 800)
            win.showMaximized()
        else:
            win.show()
