# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
New object creation GUI
=======================

This module provides a graphical user interface (GUI) for creating new objects
in the DataLab environment. It allows users to create new signals and images
interactively.

"""

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
    ImageObj,
    NewImageParam,
    NewSignalParam,
    SignalObj,
)

from datalab.config import _
from datalab.gui.creation import (
    CREATION_PARAMETERS_OPTION,
    create_image_from_param,
    create_signal_from_param,
    extract_creation_parameters,
    initialize_image_parameters,
    insert_creation_parameters,
    prepare_signal_parameters,
)

__all__ = [
    "CREATION_PARAMETERS_OPTION",
    "create_image_gui",
    "create_signal_gui",
    "extract_creation_parameters",
    "insert_creation_parameters",
]


class CustomSignalParam(OrigCustomSignalParam):
    """Parameters for custom signal (e.g. manually defined experimental data)"""

    def edit_curve(self, parent: QW.QWidget | None = None) -> None:
        """Edit custom curve"""
        win: PlotDialog = make.dialog(
            parent=parent,
            wintitle=_("Select one point then press OK to accept"),
            edit=True,
            type="curve",
        )
        edit_tool = win.manager.add_tool(
            EditPointTool, title=_("Edit curve interactively")
        )
        edit_tool.activate()
        plot = win.manager.get_plot()
        x_values, y_values = self.xyarray[:, 0], self.xyarray[:, 1]
        curve = make.mcurve(x_values, y_values, "-+")
        plot.add_item(curve)
        plot.set_active_item(curve)

        insert_btn = QW.QPushButton(_("Insert point"), win)
        insert_btn.clicked.connect(edit_tool.trigger_insert_point_at_selection)
        win.button_layout.insertWidget(0, insert_btn)

        exec_dialog(win)

        new_x_values, new_y_values = curve.get_data()
        self.xmax = new_x_values.max()
        self.xmin = new_x_values.min()
        self.size = new_x_values.size
        self.xyarray = np.vstack((new_x_values, new_y_values)).T

    def edit_curve_callback(
        self,
        button_item: gds.ButtonItem,
        current_value: object,
        parent: QW.QWidget,
    ) -> object:
        """Handle the curve edit button callback."""
        if button_item.get_name() != "btn_curve_edit":
            raise ValueError(f"Unexpected button item: {button_item.get_name()}")
        self.edit_curve(parent)
        return current_value

    btn_curve_edit = gds.ButtonItem(
        "Edit curve", callback=edit_curve_callback, icon="signal.svg"
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
    param = prepare_signal_parameters(param, edit, parent)
    if param is None:
        return None

    try:
        signal = create_signal_from_param(param)
    except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
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

    initialize_image_parameters(param)

    if edit:
        if not param.edit(parent=parent):
            return None

    try:
        image = create_image_from_param(param)
    except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
        if parent is not None:
            QW.QMessageBox.warning(parent, _("Error"), str(exc))
        else:
            raise ValueError(f"Error creating image: {exc}") from exc
        return None

    return image
