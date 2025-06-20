# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Visualization tools (based on PlotPy)
"""

import plotpy.items
import plotpy.tools
from guidata.configtools import get_icon
from guidata.qthelpers import exec_dialog, qt_app_context
from plotpy.builder import make
from plotpy.items import ImageItem
from plotpy.plot import (
    BasePlot,
    BasePlotOptions,
    PlotDialog,
    PlotOptions,
    SyncPlotWindow,
)

from cdl.config import _
from cdl.utils.tests import get_default_test_name

TEST_NB = {}


def get_name_title(name, title):
    """Return (default) widget name and title"""
    if name is None:
        TEST_NB[name] = TEST_NB.setdefault(name, 0) + 1
        name = get_default_test_name(f"{TEST_NB[name]:02d}")
    if title is None:
        title = f"{_('Test dialog')} `{name}`"
    return name, title


def create_curve_dialog(name=None, title=None, xlabel=None, ylabel=None, size=None):
    """Create Curve Dialog"""
    name, title = get_name_title(name, title)
    win = PlotDialog(
        edit=False,
        icon=get_icon("DataLab.svg"),
        toolbar=True,
        title=title,
        options=PlotOptions(type="curve", xlabel=xlabel, ylabel=ylabel),
        size=(800, 600) if size is None else size,
    )
    win.setObjectName(name)
    return win


def view_curve_items(
    items, name=None, title=None, xlabel=None, ylabel=None, add_legend=True
):
    """Create a curve dialog and plot items"""
    win = create_curve_dialog(name=name, title=title, xlabel=xlabel, ylabel=ylabel)
    plot = win.get_plot()
    for item in items:
        plot.add_item(item)
    if add_legend:
        plot.add_item(make.legend())
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


def create_image_dialog(name=None, title=None, xlabel=None, ylabel=None, size=None):
    """Create Image Dialog"""
    name, title = get_name_title(name, title)
    win = PlotDialog(
        edit=False,
        icon=get_icon("DataLab.svg"),
        toolbar=True,
        title=title,
        options=PlotOptions(type="image", xlabel=xlabel, ylabel=ylabel),
        size=(800, 600) if size is None else size,
    )
    win.setObjectName(name)
    for toolklass in (
        plotpy.tools.LabelTool,
        plotpy.tools.VCursorTool,
        plotpy.tools.HCursorTool,
        plotpy.tools.XCursorTool,
        plotpy.tools.AnnotatedRectangleTool,
        plotpy.tools.AnnotatedCircleTool,
        plotpy.tools.AnnotatedEllipseTool,
        plotpy.tools.AnnotatedSegmentTool,
        plotpy.tools.AnnotatedPointTool,
    ):
        win.get_manager().add_tool(toolklass, switch_to_default_tool=True)
    return win


def view_image_items(
    items, name=None, title=None, xlabel=None, ylabel=None, show_itemlist=False
):
    """Create an image dialog and show items"""
    win = create_image_dialog(name=name, title=title, xlabel=xlabel, ylabel=ylabel)
    if show_itemlist:
        win.manager.get_itemlist_panel().show()
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
        for idx, (img, imtitle) in enumerate(zip(images, titles)):
            row = idx // cols
            col = idx % cols
            plot = BasePlot(options=BasePlotOptions(title=imtitle))
            if isinstance(img, ImageItem):
                item = img
            else:
                item = make.image(img, interpolation="nearest", eliminate_outliers=0.1)
            plot.add_item(item)
            win.add_plot(row, col, plot, sync=share_axes)
        win.finalize_configuration()
        if maximized:
            win.resize(1200, 800)
            win.showMaximized()
        else:
            win.show()
