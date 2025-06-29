# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Visualization tools for `sigima_` interactive tests (based on PlotPy)
"""

from __future__ import annotations

import numpy as np
import plotpy.tools
from guidata.qthelpers import exec_dialog, qt_app_context
from plotpy.builder import make
from plotpy.items import CurveItem, ImageItem
from plotpy.plot import (
    BasePlot,
    BasePlotOptions,
    PlotDialog,
    PlotOptions,
    SyncPlotWindow,
)

from sigima_.config import _
from sigima_.obj import ImageObj, SignalObj
from sigima_.tests.helpers import get_default_test_name

TEST_NB = {}


def get_name_title(name: str | None, title: str | None) -> tuple[str, str]:
    """Return (default) widget name and title

    Args:
        name: Name of the widget, or None to use a default name
        title: Title of the widget, or None to use a default title

    Returns:
        A tuple (name, title) where:
        - `name` is the widget name, which is either the provided name or a default
        - `title` is the widget title, which is either the provided title or a default
    """
    if name is None:
        TEST_NB[name] = TEST_NB.setdefault(name, 0) + 1
        name = get_default_test_name(f"{TEST_NB[name]:02d}")
    if title is None:
        title = f"{_('Test dialog')} `{name}`"
    return name, title


def create_curve_dialog(
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    size: tuple[int, int] | None = None,
) -> PlotDialog:
    """Create Curve Dialog

    Args:
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        size: Size of the dialog as a tuple (width, height), or None for default size

    Returns:
        A `PlotDialog` instance configured for curve plotting
    """
    name, title = get_name_title(name, title)
    win = PlotDialog(
        edit=False,
        toolbar=True,
        title=title,
        options=PlotOptions(type="curve", xlabel=xlabel, ylabel=ylabel),
        size=(800, 600) if size is None else size,
    )
    win.setObjectName(name)
    return win


def view_curve_items(
    items: list[CurveItem],
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    add_legend: bool = True,
) -> None:
    """Create a curve dialog and plot items

    Args:
        items: List of `CurveItem` objects to plot
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        add_legend: Whether to add a legend to the plot, default is True
    """
    win = create_curve_dialog(name=name, title=title, xlabel=xlabel, ylabel=ylabel)
    plot = win.get_plot()
    for item in items:
        plot.add_item(item)
    if add_legend:
        plot.add_item(make.legend())
    exec_dialog(win)


def view_curves(
    data_or_objs: list[SignalObj | np.ndarray] | SignalObj | np.ndarray,
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Create a curve dialog and plot curves

    Args:
        data_or_objs: Single `SignalObj` or `np.ndarray`, or a list/tuple of these
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
    """
    if isinstance(data_or_objs, (tuple, list)):
        datalist = data_or_objs
    else:
        datalist = [data_or_objs]
    items = []
    for data_or_obj in datalist:
        if isinstance(data_or_obj, SignalObj):
            data = data_or_obj.xydata
        elif isinstance(data_or_obj, np.ndarray):
            data = data_or_obj
        else:
            raise TypeError(f"Unsupported data type: {type(data_or_obj)}")
        if len(data) == 2:
            xdata, ydata = data
            item = make.mcurve(xdata, ydata)
        else:
            item = make.mcurve(data)
        items.append(item)
    view_curve_items(items, name=name, title=title, xlabel=xlabel, ylabel=ylabel)


def create_image_dialog(
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    size: tuple[int, int] | None = None,
) -> PlotDialog:
    """Create Image Dialog

    Args:
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        size: Size of the dialog as a tuple (width, height), or None for default size

    Returns:
        A `PlotDialog` instance configured for image plotting
    """
    name, title = get_name_title(name, title)
    win = PlotDialog(
        edit=False,
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
    items: list[ImageItem],
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_itemlist: bool = False,
) -> None:
    """Create an image dialog and show items

    Args:
        items: List of `ImageItem` objects to display
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
        show_itemlist: Whether to show the item list panel in the dialog,
         default is False
    """
    win = create_image_dialog(name=name, title=title, xlabel=xlabel, ylabel=ylabel)
    if show_itemlist:
        win.manager.get_itemlist_panel().show()
    plot = win.get_plot()
    for item in items:
        plot.add_item(item)
    exec_dialog(win)


def view_images(
    data_or_objs: list[ImageObj | np.ndarray] | ImageObj | np.ndarray,
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Create an image dialog and show images

    Args:
        data_or_objs: Single `ImageObj` or `np.ndarray`, or a list/tuple of these
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
    """
    if isinstance(data_or_objs, (tuple, list)):
        datalist = data_or_objs
    else:
        datalist = [data_or_objs]
    items = []
    for data_or_obj in datalist:
        if isinstance(data_or_obj, ImageObj):
            data = data_or_obj.data
        elif isinstance(data_or_obj, np.ndarray):
            data = data_or_obj
        else:
            raise TypeError(f"Unsupported data type: {type(data_or_obj)}")
        item = make.image(data, interpolation="nearest", eliminate_outliers=0.1)
        items.append(item)
    view_image_items(items, name=name, title=title, xlabel=xlabel, ylabel=ylabel)


def view_curves_and_images(
    data_or_objs: list[SignalObj | np.ndarray | ImageObj | np.ndarray],
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """View signals, then images in two successive dialogs

    Args:
        data_or_objs: List of `SignalObj`, `ImageObj`, `np.ndarray` or a mix of these
        name: Name of the dialog, or None to use a default name
        title: Title of the dialog, or None to use a default title
        xlabel: Label for the x-axis, or None for no label
        ylabel: Label for the y-axis, or None for no label
    """
    if isinstance(data_or_objs, (tuple, list)):
        objs = data_or_objs
    else:
        objs = [data_or_objs]
    sig_objs = [obj for obj in objs if isinstance(obj, (SignalObj, np.ndarray))]
    if sig_objs:
        view_curves(sig_objs, name=name, title=title, xlabel=xlabel, ylabel=ylabel)
    ima_objs = [obj for obj in objs if isinstance(obj, (ImageObj, np.ndarray))]
    if ima_objs:
        view_images(ima_objs, name=name, title=title, xlabel=xlabel, ylabel=ylabel)


def __compute_grid(
    num_objects: int, max_cols: int = 4, fixed_num_rows: int | None = None
) -> tuple[int, int]:
    """Compute number of rows and columns for a grid of images

    Args:
        num_objects: Total number of objects to display
        max_cols: Maximum number of columns in the grid
        fixed_num_rows: Fixed number of rows, if specified

    Returns:
        A tuple (num_rows, num_cols) representing the grid dimensions
    """
    num_cols = min(num_objects, max_cols)
    if fixed_num_rows is not None:
        num_rows = fixed_num_rows
        num_cols = (num_objects + num_rows - 1) // num_rows
    else:
        num_rows = (num_objects + num_cols - 1) // num_cols
    return num_rows, num_cols


def view_images_side_by_side(
    images: list[ImageItem | np.ndarray],
    titles: list[str],
    share_axes: bool = True,
    rows: int | None = None,
    maximized: bool = False,
    title: str | None = None,
) -> None:
    """Show sequence of images

    Args:
        images: List of `ImageItem` or `np.ndarray` objects to display
        titles: List of titles for each image
        share_axes: Whether to share axes across plots, default is True
        rows: Fixed number of rows in the grid, or None to compute automatically
        maximized: Whether to show the window maximized, default is False
        title: Title of the window, or None for a default title
    """
    rows, cols = __compute_grid(len(images), fixed_num_rows=rows, max_cols=4)
    with qt_app_context(exec_loop=True):
        win = SyncPlotWindow(title=title)
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
