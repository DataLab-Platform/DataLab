# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""
CodraFT Visualization tools (based on guiqwt)
"""

import guiqwt.image
import guiqwt.plot
import guiqwt.tools
from guidata.configtools import get_icon
from guiqwt.builder import make

from codraft.config import _
from codraft.utils.qthelpers import exec_dialog
from codraft.utils.tests import get_default_test_name

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
        icon=get_icon("codraft.svg"),
        toolbar=True,
        wintitle=title,
        options=dict(xlabel=xlabel, ylabel=ylabel),
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
        icon=get_icon("codraft.svg"),
        toolbar=True,
        wintitle=title,
        options=dict(xlabel=xlabel, ylabel=ylabel),
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
