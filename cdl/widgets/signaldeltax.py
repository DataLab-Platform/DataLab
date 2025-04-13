# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
GUI dialog for analyzing signals and calculating full width at Y.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from guidata.configtools import get_icon
from plotpy.builder import make
from plotpy.plot import PlotDialog
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from cdl.algorithms.signal import full_width_at_y
from cdl.config import _
from cdl.core.model.signal import CURVESTYLES

if TYPE_CHECKING:
    from plotpy.items import CurveItem, Marker, XRangeSelection
    from qtpy.QtWidgets import QWidget

    from cdl.obj import SignalObj


class SignalDeltaXDialog(PlotDialog):
    """Signal Delta X dialog.

    Args:
        signal: signal object
        parent: parent widget. Defaults to None.
    """

    def __init__(
        self,
        signal: SignalObj,
        y: float | None = None,
        parent: QWidget | None = None,
    ) -> None:
        self.__curve_styles = CURVESTYLES.style_generator()
        self.__signal = signal
        self.__coords: list[float, float, float, float] | None = None
        self.curve: CurveItem | None = None
        self.hcursor: Marker | None = None
        self.delta_xrange: XRangeSelection | None = None
        self.deltaxlineedit: QW.QLineEdit | None = None
        self.ylineedit: QW.QLineEdit | None = None
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
        xlabel = QW.QLabel("∆X=")
        ylabel = QW.QLabel("Y=")
        self.deltaxlineedit = QW.QLineEdit()
        self.deltaxlineedit.setReadOnly(True)
        self.deltaxlineedit.setDisabled(True)
        self.ylineedit = QW.QLineEdit()
        self.ylineedit.editingFinished.connect(self.ylineedit_editing_finished)
        self.ylineedit.setValidator(QG.QDoubleValidator())
        xygroup = QW.QGroupBox(_("Cursor position"))
        xylayout = QW.QHBoxLayout()
        xylayout.addWidget(xlabel)
        xylayout.addWidget(self.deltaxlineedit)
        xylayout.addWidget(ylabel)
        xylayout.addWidget(self.ylineedit)
        xylayout.addWidget(apply_button)
        apply_button.clicked.connect(self.ylineedit_editing_finished)
        xygroup.setLayout(xylayout)
        self.button_layout.insertWidget(0, xygroup)

        obj = self.__signal
        with CURVESTYLES.alternative(self.__curve_styles):
            self.curve = obj.make_item()
        plot = self.get_plot()
        plot.set_antialiasing(True)

        self.delta_xrange = make.range(0.0, 1.0)
        self.delta_xrange.setVisible(False)
        self.delta_xrange.set_style("roi", "s/readonly")
        self.delta_xrange.set_selectable(False)

        plot.SIG_MARKER_CHANGED.connect(self.cursor_changed)
        self.hcursor = make.hcursor(np.mean(obj.y), "Y = %g")
        for item in (self.curve, self.delta_xrange, self.hcursor):
            plot.add_item(item)
        plot.replot()
        plot.set_active_item(self.hcursor)
        self.cursor_changed(self.hcursor)

    def cursor_changed(self, item: Marker) -> None:
        """Cursor changed"""
        sig = self.__signal
        _x, y = item.get_pos()

        try:
            with warnings.catch_warnings(record=True) as w:
                self.__coords = full_width_at_y((sig.x, sig.y), y)
                delta_str = f"{self.__coords[2] - self.__coords[0]:g}"
                warning_or_error = len(w) > 0
                self.delta_xrange.setVisible(True)
                self.delta_xrange.set_range(self.__coords[0], self.__coords[2])
        except ValueError:
            delta_str = ""
            warning_or_error = True
            self.delta_xrange.setVisible(False)

        self.button_box.button(QW.QDialogButtonBox.Ok).setEnabled(not warning_or_error)
        self.deltaxlineedit.setText(delta_str)
        self.ylineedit.setText(f"{y:g}" if y is not None else "")

    def ylineedit_editing_finished(self) -> None:
        """Y line edit editing finished"""
        try:
            y = float(self.ylineedit.text())
            x, _y = self.hcursor.get_pos()
            self.hcursor.set_pos(x, y)
        except ValueError:
            pass
        plot = self.get_plot()
        plot.replot()

    def get_coords(self) -> tuple[float, float, float, float]:
        """Return coordinates of segment associated to the width at Y"""
        return self.__coords

    def get_y_value(self) -> float:
        """Get cursor y value"""
        _x, y = self.hcursor.get_pos()
        return y
