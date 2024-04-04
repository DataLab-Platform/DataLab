# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Signal peak detection feature"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import numpy as np
from guidata.configtools import get_icon
from plotpy.builder import make
from plotpy.plot import PlotDialog
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from cdl.algorithms.signal import peak_indexes
from cdl.config import _


class DistanceSlider(QW.QWidget):
    """Minimum distance slider"""

    TITLE = _("Minimum distance:")
    SIG_VALUE_CHANGED = QC.Signal(int)

    def __init__(self, parent):
        super().__init__(parent)
        self.slider = QW.QSlider(QC.Qt.Horizontal)
        self.label = QW.QLabel()
        layout = QW.QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def value_changed(self, value):
        """Slider value has changed"""
        plural = "s" if value > 1 else ""
        self.label.setText(f"{self.TITLE} {value} point{plural}")
        self.SIG_VALUE_CHANGED.emit(value)

    def setup_slider(self, value, maxval):
        """Setup slider"""
        self.slider.setMinimum(1)
        self.slider.setMaximum(maxval)
        self.slider.setValue(value)
        self.slider.setTickPosition(QW.QSlider.TicksBothSides)
        self.value_changed(value)
        self.slider.valueChanged.connect(self.value_changed)


class SignalPeakDetectionDialog(PlotDialog):
    """Signal Peak detection dialog"""

    def __init__(self, parent=None):
        self.peaks = None
        self.peak_indexes = None
        self.in_x = None
        self.in_y = None
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

    def populate_plot_layout(self) -> None:  # Reimplement PlotDialog method
        """Populate the plot layout"""
        super().populate_plot_layout()
        self.distance_slider = DistanceSlider(self)
        self.add_widget(self.distance_slider, 1, 0, 1, 1)

    def get_peaks(self):
        """Return peaks coordinates"""
        return self.peaks

    def get_peak_indexes(self):
        """Return peak indexes"""
        return self.peak_indexes

    def get_threshold(self):
        """Return relative threshold"""
        y = self.in_y
        return (self.in_threshold - np.min(y)) / (np.max(y) - np.min(y))

    def get_min_dist(self):
        """Return minimum distance"""
        return self.min_distance

    def compute_peaks(self):
        """Compute peak detection"""
        x, y = self.in_x, self.in_y
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

    def hcursor_changed(self, marker):
        """Horizontal cursor position has changed"""
        _x, y = marker.get_pos()
        self.in_threshold = y
        self.compute_peaks()

    def minimum_distance_changed(self, value):
        """Minimum distance changed"""
        self.min_distance = value
        self.compute_peaks()
        self.get_plot().replot()

    def setup_data(self, x, y):
        """Setup dialog box"""
        self.in_curve = make.curve(x, y, "ab", "b")
        plot = self.get_plot()
        plot.add_item(self.in_curve)
        self.in_threshold = 0.5 * (np.max(y) - np.min(y)) + np.min(y)
        cursor = make.hcursor(self.in_threshold)
        self.in_threshold_cursor = cursor
        plot.add_item(self.in_threshold_cursor)
        self.co_results = make.label("", "TL", (0, 0), "TL")
        plot.add_item(self.co_results)
        plot.SIG_MARKER_CHANGED.connect(self.hcursor_changed)
        self.in_x, self.in_y = x, y
        self.min_distance = 1
        self.distance_slider.setup_slider(self.min_distance, len(y) // 4)
        self.distance_slider.SIG_VALUE_CHANGED.connect(self.minimum_distance_changed)
        self.compute_peaks()
        # Replot, otherwise, it's not possible to set active item:
        plot.replot()
        plot.set_active_item(cursor)
