# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Signal peak detection feature"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from guidata.configtools import get_icon
from plotpy.builder import make
from plotpy.plot import PlotDialog
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from cdl.algorithms.signal import peak_indexes
from cdl.config import _
from cdl.core.model.signal import CURVESTYLES

if TYPE_CHECKING:
    from plotpy.items import Marker
    from qtpy.QtWidgets import QWidget

    from cdl.obj import SignalObj


class DistanceSlider(QW.QWidget):
    """Minimum distance slider

    Args:
        parent: parent widget. Defaults to None.
    """

    TITLE = _("Minimum distance:")
    SIG_VALUE_CHANGED = QC.Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.slider = QW.QSlider(QC.Qt.Horizontal)
        self.label = QW.QLabel()
        layout = QW.QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def value_changed(self, value: int) -> None:
        """Slider value has changed

        Args:
            value: slider value
        """
        plural = "s" if value > 1 else ""
        self.label.setText(f"{self.TITLE} {value} point{plural}")
        self.SIG_VALUE_CHANGED.emit(value)

    def setup_slider(self, value: int, maxval: int) -> None:
        """Setup slider

        Args:
            value: initial value
            maxval: maximum value
        """
        self.slider.setMinimum(1)
        self.slider.setMaximum(maxval)
        self.slider.setValue(value)
        self.slider.setTickPosition(QW.QSlider.TicksBothSides)
        self.value_changed(value)
        self.slider.valueChanged.connect(self.value_changed)


class SignalPeakDetectionDialog(PlotDialog):
    """Signal Peak detection dialog

    Args:
        signal: signal object
        parent: parent widget. Defaults to None.
    """

    def __init__(self, signal: SignalObj, parent: QWidget | None = None) -> None:
        self.__curve_styles = CURVESTYLES.style_generator()
        self.peaks = None
        self.peak_indexes = None
        self.in_curve = None
        self.in_threshold = None
        self.in_threshold_cursor = None
        self.co_results = None
        self.co_positions = None
        self.co_markers = None
        self.min_distance = None
        self.distance_slider: DistanceSlider | None = None
        super().__init__(title=_("Signal peak detection"), edit=True, parent=parent)
        self.setObjectName("peakdetection")
        if parent is None:
            self.setWindowIcon(get_icon("DataLab.svg"))
        legend = make.legend("TR")
        self.get_plot().add_item(legend)
        self.__signal = signal.copy()
        self.__setup_dialog()

    def populate_plot_layout(self) -> None:  # Reimplement PlotDialog method
        """Populate the plot layout"""
        super().populate_plot_layout()
        self.distance_slider = DistanceSlider(self)
        self.add_widget(self.distance_slider, 1, 0, 1, 1)

    def __setup_dialog(self) -> None:
        """Setup dialog box"""
        obj = self.__signal
        with CURVESTYLES.alternative(self.__curve_styles):
            self.in_curve = obj.make_item()
        plot = self.get_plot()
        plot.set_antialiasing(True)
        plot.add_item(self.in_curve)
        self.in_threshold = 0.5 * (np.max(obj.y) - np.min(obj.y)) + np.min(obj.y)
        cursor = make.hcursor(self.in_threshold)
        self.in_threshold_cursor = cursor
        plot.add_item(self.in_threshold_cursor)
        self.co_results = make.label("", "TL", (0, 0), "TL")
        plot.add_item(self.co_results)
        plot.SIG_MARKER_CHANGED.connect(self.hcursor_changed)
        self.min_distance = 1
        self.distance_slider.setup_slider(self.min_distance, len(obj.y) // 4)
        self.distance_slider.SIG_VALUE_CHANGED.connect(self.minimum_distance_changed)
        self.compute_peaks()
        # Replot, otherwise, it's not possible to set active item:
        plot.replot()
        plot.set_active_item(cursor)

    def get_peaks(self) -> list[tuple[float, float]]:
        """Return peaks coordinates"""
        return self.peaks

    def get_peak_indexes(self) -> list[int]:
        """Return peak indexes"""
        return self.peak_indexes

    def get_threshold(self) -> float:
        """Return relative threshold"""
        y = self.__signal.y
        return (self.in_threshold - np.min(y)) / (np.max(y) - np.min(y))

    def get_min_dist(self) -> int:
        """Return minimum distance"""
        return self.min_distance

    def compute_peaks(self) -> None:
        """Compute peak detection"""
        x, y = self.__signal.xydata
        plot = self.get_plot()
        self.peak_indexes = peak_indexes(
            y,
            thres=self.in_threshold,
            min_dist=self.min_distance,
            thres_abs=True,
        )
        self.peaks = [(x[index], y[index]) for index in self.peak_indexes]
        markers = [
            make.marker(
                pos,
                movable=False,
                color="orange",
                markerstyle="|",
                linewidth=1,
                marker="NoShape",
                linestyle="DashLine",
            )
            for pos in self.peaks
        ]
        if self.co_markers is not None:
            plot.del_items(self.co_markers)
        self.co_markers = markers
        for item in self.co_markers:
            plot.add_item(item)
        positions = [str(marker.get_pos()[0]) for marker in markers]
        prefix = f'<b>{_("Peaks:")}</b><br>'
        self.co_results.set_text(prefix + "<br>".join(positions))

    def hcursor_changed(self, marker: Marker) -> None:
        """Horizontal cursor position has changed

        Args:
            marker: marker item
        """
        _x, y = marker.get_pos()
        self.in_threshold = y
        self.compute_peaks()

    def minimum_distance_changed(self, value: int) -> None:
        """Minimum distance changed

        Args:
            value: minimum distance value
        """
        self.min_distance = value
        self.compute_peaks()
        self.get_plot().replot()
