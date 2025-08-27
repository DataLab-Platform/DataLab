# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Signal horizontal or vertical cursor selection dialog."""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from guidata.configtools import get_icon
from plotpy.builder import make
from plotpy.plot import PlotDialog
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from sigima.tools.signal.features import find_first_x_at_given_y_value

from datalab.adapters_plotpy import CURVESTYLES, create_adapter_from_object
from datalab.config import _
from datalab.utils.qthelpers import block_signals

if TYPE_CHECKING:
    from plotpy.items import CurveItem, Marker
    from qtpy.QtWidgets import QWidget
    from sigima.objects import SignalObj


class SignalCursorDialog(PlotDialog):
    """Signal horizontal or vertical cursor selection dialog.

    Args:
        signal: signal object
        parent: parent widget. Defaults to None.
    """

    def __init__(
        self,
        signal: SignalObj,
        cursor_orientation: Literal["horizontal", "vertical"],
        parent: QWidget | None = None,
    ) -> None:
        assert cursor_orientation in (
            "horizontal",
            "vertical",
        ), "cursor_orientation must be 'horizontal' or 'vertical'"
        self.__curve_styles = CURVESTYLES.style_generator()
        self.__cursor_orientation = cursor_orientation
        self.__signal = signal
        self.__x_value: float | None = None
        self.__y_value: float | None = None
        self.curve: CurveItem | None = None
        self.hcursor: Marker | None = None
        self.vcursor: Marker | None = None
        self.xlineedit: QW.QLineEdit | None = None
        self.ylineedit: QW.QLineEdit | None = None
        if cursor_orientation == "horizontal":
            title = _("Select X value with cursor")
        else:
            title = _("Select Y value with cursor")
        super().__init__(title=title, edit=True, parent=parent)
        self.setObjectName("SignalCursorDialog")
        if parent is None:
            self.setWindowIcon(get_icon("DataLab.svg"))
        legend = make.legend("TR")
        self.get_plot().add_item(legend)
        self.__setup_dialog()

    def __setup_dialog(self) -> None:
        """Setup dialog box"""
        apply_button = QW.QPushButton(_("Apply"))
        apply_button.setIcon(get_icon("apply.svg"))
        apply_button.setToolTip(_("Apply cursor position"))
        xlabel = QW.QLabel("X=")
        ylabel = QW.QLabel("Y=")
        self.xlineedit = QW.QLineEdit()
        self.xlineedit.editingFinished.connect(self.xlineedit_editing_finished)
        self.xlineedit.setValidator(QG.QDoubleValidator())
        self.ylineedit = QW.QLineEdit()
        self.ylineedit.editingFinished.connect(self.ylineedit_editing_finished)
        self.ylineedit.setValidator(QG.QDoubleValidator())
        self.xlineedit.setReadOnly(self.__cursor_orientation == "horizontal")
        self.xlineedit.setDisabled(self.__cursor_orientation == "horizontal")
        self.ylineedit.setReadOnly(self.__cursor_orientation == "vertical")
        self.ylineedit.setDisabled(self.__cursor_orientation == "vertical")
        xygroup = QW.QGroupBox(_("Cursor position"))
        xylayout = QW.QHBoxLayout()
        xylayout.addWidget(xlabel)
        xylayout.addWidget(self.xlineedit)
        if self.__cursor_orientation == "vertical":
            xylayout.addWidget(apply_button)
            apply_button.clicked.connect(self.xlineedit_editing_finished)
            xylayout.addStretch()
            xylayout.addSpacing(10)
        xylayout.addWidget(ylabel)
        xylayout.addWidget(self.ylineedit)
        if self.__cursor_orientation == "horizontal":
            xylayout.addWidget(apply_button)
            apply_button.clicked.connect(self.ylineedit_editing_finished)
        xygroup.setLayout(xylayout)
        self.button_layout.insertWidget(0, xygroup)

        obj = self.__signal
        with CURVESTYLES.alternative(self.__curve_styles):
            self.curve = create_adapter_from_object(obj).make_item()
        plot = self.get_plot()
        plot.set_antialiasing(True)

        xcursor = make.xcursor(np.mean(obj.x), np.mean(obj.y), "X = %g, Y = %g")
        xcursor.set_selectable(False)
        param = xcursor.markerparam
        param.symbol.facecolor = "blue"
        param.symbol.edgecolor = "cyan"
        param.symbol.size = 9
        param.line.style = "DotLine"
        param.line.color = "blue"
        param.line.width = 2.0
        param.update_item(xcursor)

        plot.SIG_MARKER_CHANGED.connect(self.cursor_changed)
        if self.__cursor_orientation == "horizontal":
            self.hcursor = make.hcursor(np.mean(obj.y), "Y = %g")
            self.vcursor = xcursor
            self.vcursor.setVisible(False)
        else:
            self.vcursor = make.vcursor(np.mean(obj.x), "X = %g")
            self.hcursor = xcursor
            self.hcursor.setVisible(False)
        for item in (self.curve, self.vcursor, self.hcursor):
            plot.add_item(item)
        plot.replot()
        if self.__cursor_orientation == "horizontal":
            plot.set_active_item(self.hcursor)
            self.cursor_changed(self.hcursor)
        else:
            plot.set_active_item(self.vcursor)
            self.cursor_changed(self.vcursor)

    def cursor_changed(self, item: Marker) -> None:
        """Cursor changed"""
        sig = self.__signal
        plot = self.get_plot()
        if self.__cursor_orientation == "horizontal" and item is self.hcursor:
            _x, y = item.get_pos()
            x = find_first_x_at_given_y_value(sig.x, sig.y, y)
            x = None if np.isnan(x) else x
            if x is not None:
                with block_signals(plot):
                    self.vcursor.set_pos(x, y)
            self.vcursor.setVisible(x is not None)
            self.button_box.button(QW.QDialogButtonBox.Ok).setEnabled(x is not None)
        elif self.__cursor_orientation == "vertical" and item is self.vcursor:
            x, _y = item.get_pos()
            y_index = np.searchsorted(self.__signal.x, x)
            if x < self.__signal.x[0] or y_index >= len(self.__signal.y):
                y = None
            else:
                y = self.__signal.y[y_index]
                with block_signals(plot):
                    self.hcursor.set_pos(x, y)
                self.hcursor.setVisible(True)
            self.hcursor.setVisible(y is not None)
            self.button_box.button(QW.QDialogButtonBox.Ok).setEnabled(y is not None)
        self.xlineedit.setText(f"{x:g}" if x is not None else "")
        self.ylineedit.setText(f"{y:g}" if y is not None else "")
        self.__x_value, self.__y_value = x, y

    def xlineedit_editing_finished(self) -> None:
        """X line edit editing finished"""
        try:
            x = float(self.xlineedit.text())
            _x, y = self.vcursor.get_pos()
            if self.__cursor_orientation == "horizontal":
                self.hcursor.set_pos(x, y)
            else:
                self.vcursor.set_pos(x, y)
        except ValueError:
            pass
        plot = self.get_plot()
        plot.replot()

    def ylineedit_editing_finished(self) -> None:
        """Y line edit editing finished"""
        try:
            y = float(self.ylineedit.text())
            x, _y = self.hcursor.get_pos()
            if self.__cursor_orientation == "horizontal":
                self.hcursor.set_pos(x, y)
            else:
                self.vcursor.set_pos(x, y)
        except ValueError:
            pass
        plot = self.get_plot()
        plot.replot()

    def get_cursor_position(self) -> tuple[float, float]:
        """Get cursor position"""
        return self.__x_value, self.__y_value

    def get_x_value(self) -> float:
        """Get cursor x value"""
        return self.__x_value

    def get_y_value(self) -> float:
        """Get cursor y value"""
        return self.__y_value
