# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
New object creation GUI
=======================

This module provides a graphical user interface (GUI) for creating new objects
in the DataLab environment. It allows users to create new signals and images
interactively.

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import guidata.dataset as gds
import numpy as np
from guidata.qthelpers import exec_dialog
from plotpy.builder import make
from plotpy.plot import PlotDialog
from plotpy.tools import EditPointTool
from qtpy import QtWidgets as QW
from sigima.objects import CustomSignalParam as OrigCustomSignalParam
from sigima.objects import (
    Gauss2DParam,
    ImageDatatypes,
    ImageObj,
    NewImageParam,
    NewSignalParam,
    SignalObj,
    create_signal,
)
from sigima.objects import create_image_from_param as create_image_headless
from sigima.objects import create_signal_from_param as create_signal_headless
from sigima.objects.base import BaseProcParam
from sigima.objects.signal import DEFAULT_TITLE as SIGNAL_DEFAULT_TITLE

from datalab.config import _


class CustomSignalParam(OrigCustomSignalParam):
    """Parameters for custom signal (e.g. manually defined experimental data)"""

    def edit_curve(self, *args) -> None:  # pylint: disable=unused-argument
        """Edit custom curve"""
        win: PlotDialog = make.dialog(
            wintitle=_("Select one point then press OK to accept"),
            edit=True,
            type="curve",
        )
        edit_tool = win.manager.add_tool(
            EditPointTool, title=_("Edit curve interactively")
        )
        edit_tool.activate()
        plot = win.manager.get_plot()
        x, y = self.xyarray[:, 0], self.xyarray[:, 1]
        curve = make.mcurve(x, y, "-+")
        plot.add_item(curve)
        plot.set_active_item(curve)

        insert_btn = QW.QPushButton(_("Insert point"), win)
        insert_btn.clicked.connect(edit_tool.trigger_insert_point_at_selection)
        win.button_layout.insertWidget(0, insert_btn)

        exec_dialog(win)

        new_x, new_y = curve.get_data()
        self.xmax = new_x.max()
        self.xmin = new_x.min()
        self.size = new_x.size
        self.xyarray = np.vstack((new_x, new_y)).T

    btn_curve_edit = gds.ButtonItem(
        "Edit curve", callback=edit_curve, icon="signal.svg"
    )


def create_signal_gui(
    param: NewSignalParam | None = None,
    edit: bool = False,
    parent: QW.QWidget | None = None,
) -> SignalObj | None:
    """Create a new Signal object from GUI

    Args:
        param: base signal parameters (NewSignalParam)
        edit: Open a dialog box to edit parameters (default: False)
        parent: parent widget

    Returns:
        Signal object or None if canceled

    Raises:
        ValueError: if base_param is None and edit is False
    """
    if param is None:
        param = NewSignalParam()
        edit = True  # Default to editing if no parameters provided

    if isinstance(param, OrigCustomSignalParam) and edit:
        p_init = NewSignalParam(_("Custom signal"))
        p_init.size = 10  # Set smaller default size for initial input
        if not p_init.edit(parent=parent):
            return None
        param.setup_array(size=p_init.size, xmin=p_init.xmin, xmax=p_init.xmax)

    if edit:
        if not param.edit(parent=parent):
            return None

    if isinstance(param, OrigCustomSignalParam):
        signal = create_signal(param.title)
        signal.xydata = param.xyarray.T
        if signal.title == SIGNAL_DEFAULT_TITLE:
            signal.title = f"custom(npts={param.size})"
        return signal

    try:
        signal = create_signal_headless(param)
    except Exception as exc:  # pylint: disable=broad-except
        if parent is not None:
            QW.QMessageBox.warning(parent, _("Error"), str(exc))
        else:
            raise ValueError(f"Error creating signal: {exc}") from exc
        signal = None

    return signal


def create_image_gui(
    param: NewImageParam | None = None,
    edit: bool = False,
    parent: QW.QWidget | None = None,
) -> ImageObj | None:
    """Create a new Image object from GUI

    Args:
        param: image parameters
        edit: Open a dialog box to edit parameters (default: False)
        parent: parent widget

    Returns:
        Image object or None if canceled

    Raises:
        ValueError: if base_param is None and edit is False
    """
    if param is None:
        param = NewImageParam()
        edit = True  # Default to editing if no parameters provided

    if param.height is None:
        param.height = 500
    if param.width is None:
        param.width = 500
    if param.dtype is None:
        param.dtype = ImageDatatypes.UINT16
    if isinstance(param, Gauss2DParam):
        if param.a is None:
            try:
                param.a = np.iinfo(param.dtype.value).max / 2.0
            except ValueError:
                param.a = 10.0
    elif isinstance(param, BaseProcParam):
        param.set_from_datatype(param.dtype.value)

    if edit:
        if not param.edit(parent=parent):
            return None

    try:
        image = create_image_headless(param)
    except Exception as exc:  # pylint: disable=broad-except
        if parent is not None:
            QW.QMessageBox.warning(parent, _("Error"), str(exc))
        else:
            raise ValueError(f"Error creating image: {exc}") from exc
        return None

    return image
