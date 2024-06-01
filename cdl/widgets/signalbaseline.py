# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Signal base line selection dialog."""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from guidata.configtools import get_icon
from plotpy.builder import make
from plotpy.plot import PlotDialog

from cdl.config import _
from cdl.core.model.signal import CURVESTYLES

if TYPE_CHECKING:
    from plotpy.items import CurveItem, Marker, XRangeSelection
    from qtpy.QtWidgets import QWidget

    from cdl.obj import SignalObj


class SignalBaselineDialog(PlotDialog):
    """Signal baseline selection dialog.

    Args:
        signal: signal object
        parent: parent widget. Defaults to None.
    """

    def __init__(self, signal: SignalObj, parent: QWidget | None = None) -> None:
        self.__curve_styles = CURVESTYLES.style_generator()
        self.__baseline: float | None = None
        self.__indexrange: tuple[int, int] | None = None
        self.curve: CurveItem | None = None
        self.cursor: Marker | None = None
        self.xrange: XRangeSelection | None = None
        super().__init__(title=_("Signal baseline selection"), edit=True, parent=parent)
        self.setObjectName("baselineselection")
        if parent is None:
            self.setWindowIcon(get_icon("DataLab.svg"))
        legend = make.legend("TR")
        self.get_plot().add_item(legend)
        self.__signal = signal.copy()
        self.__setup_dialog()

    def __setup_dialog(self) -> None:
        """Setup dialog box"""
        obj = self.__signal
        with CURVESTYLES.alternative(self.__curve_styles):
            self.curve = obj.make_item()
        plot = self.get_plot()
        plot.set_antialiasing(True)
        plot.SIG_RANGE_CHANGED.connect(self.xrange_changed)
        plot.SIG_MARKER_CHANGED.connect(self.cursor_changed)
        self.cursor = make.hcursor(0.0, _("Base line") + " = %g")
        self.xrange = make.range(obj.x[0], obj.x[int(0.2 * len(obj.x))])
        for item in (self.curve, self.cursor, self.xrange):
            plot.add_item(item)
        plot.replot()
        plot.set_active_item(self.xrange)
        self.xrange_changed(self.xrange, *self.xrange.get_range())

    def xrange_changed(self, item: XRangeSelection, xmin: float, xmax: float) -> None:
        """X range changed"""
        imin, imax = np.searchsorted(self.__signal.x, sorted([xmin, xmax]))
        if imin == imax:
            return
        self.__indexrange = imin, imax
        self.cursor.set_pos(0, np.mean(self.__signal.y[imin:imax]))
        plot = self.get_plot()
        plot.replot()

    def cursor_changed(self, item: Marker) -> None:
        """Cursor changed"""
        _x, self.__baseline = item.get_pos()

    def get_baseline(self) -> float:
        """Get baseline"""
        return self.__baseline

    def get_index_range(self) -> tuple[int, int]:
        """Get index range"""
        return self.__indexrange

    def get_x_range(self) -> tuple[float, float]:
        """Get x range"""
        x = self.__signal.x
        return x[self.__indexrange[0]], x[self.__indexrange[1]]
