# -*- coding: utf-8 -*-

"""
Isolevel map plugin
===================

This plugin generates isolevel (topographic contour) maps from images using
PlotPy's native contour items, with level value labels placed along the lines.

Contour lines can be exported as movable annotations directly into DataLab's
annotation system (fully editable, serialized with the image).

The image is smoothed with a Gaussian filter before computing contour lines
to avoid generating thousands of noise-induced micro-contours on large images.
A downsampling step is applied for images larger than 512x512 pixels.

Usage
-----

1. Copy this file to your DataLab plugins directory
2. Restart DataLab or use "Plugins > Reload plugins"
3. Select an image in the Image panel
4. Use "Plugins > Isolevel Map > Show isolevel map"
5. Adjust the number of levels and smoothing interactively
6. Use "Export to annotations" to save contours into DataLab's annotation layer

.. note::

    This plugin is not installed by default. To install it, copy this file to
    your DataLab plugins directory (see `DataLab documentation
    <https://datalab-platform.com/en/features/advanced/plugins.html>`_).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import scipy.ndimage as spi
from plotpy.builder import make
from plotpy.constants import PlotType
from plotpy.items.annotation import AnnotatedPolygon
from plotpy.items.contour import compute_contours
from plotpy.plot import PlotDialog, PlotOptions
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

import sigima.objects
import datalab.plugins
from datalab.adapters_plotpy.objects.image import get_obj_coords

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_LEVELS = 10
MIN_LEVELS = 2
MAX_LEVELS = 50
CONTOUR_MAX_SIZE = 512
DEFAULT_SIGMA = 3.0
MIN_SIGMA = 0
MAX_SIGMA = 20
SIGMA_SCALE = 10  # slider uses integers: value / SIGMA_SCALE = actual sigma
LABEL_FMT = "{:.3g}"
LABEL_OFFSET = (5, 5)   # canvas offset (pixels) for LabelItem anchor


def _prepare_contour_data(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = DEFAULT_SIGMA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth and optionally downsample image data for contour computation.

    Args:
        data: 2-D image array (float64)
        x: 1-D array of X coordinates (pixel centers)
        y: 1-D array of Y coordinates (pixel centers)
        sigma: Gaussian smoothing sigma (0 = no smoothing)

    Returns:
        Tuple of (prepared_data, x_coords, y_coords)
    """
    if sigma > 0:
        smoothed = spi.gaussian_filter(data, sigma=sigma)
    else:
        smoothed = data.copy()

    ny, nx = smoothed.shape
    max_dim = max(ny, nx)
    if max_dim > CONTOUR_MAX_SIZE:
        step = max(1, int(round(max_dim / CONTOUR_MAX_SIZE)))
        smoothed = smoothed[::step, ::step]
        x = x[::step]
        y = y[::step]

    return smoothed, x, y


def _compute_level_groups(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    n_levels: int,
    sigma: float = DEFAULT_SIGMA,
) -> dict[float, list]:
    """Compute contour line segments grouped by level value.

    Args:
        data: 2-D image array (float64)
        x: 1-D array of X coordinates
        y: 1-D array of Y coordinates
        n_levels: Number of isolevel bands
        sigma: Gaussian smoothing sigma

    Returns:
        Dict mapping level float value → list of ContourLine objects,
        or empty dict if image is flat
    """
    smoothed, xs, ys = _prepare_contour_data(data, x, y, sigma)

    vmin, vmax = np.nanmin(smoothed), np.nanmax(smoothed)
    if vmin == vmax:
        return {}

    levels = np.linspace(vmin, vmax, n_levels + 1)[1:-1]
    X, Y = np.meshgrid(xs, ys)

    all_clines = compute_contours(smoothed, levels, X, Y)

    groups: dict[float, list] = defaultdict(list)
    for cline in all_clines:
        groups[cline.level].append(cline)
    return groups


def compute_isolevel_items(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    n_levels: int,
    sigma: float = DEFAULT_SIGMA,
) -> list:
    """Compute PlotPy ContourItem + LabelItem pairs for each level.

    For each level threshold, ContourItems are created for all segments.
    A LabelItem is placed at the midpoint of the longest segment.

    Args:
        data: 2-D image array (float64)
        x: 1-D array of X coordinates (pixel centers)
        y: 1-D array of Y coordinates (pixel centers)
        n_levels: Number of isolevel bands
        sigma: Gaussian smoothing sigma

    Returns:
        List of PlotPy plot items (ContourItem and LabelItem objects)
    """
    groups = _compute_level_groups(data, x, y, n_levels, sigma)
    if not groups:
        return []

    smoothed, xs, ys = _prepare_contour_data(data, x, y, sigma)
    X, Y = np.meshgrid(xs, ys)

    items = []
    for level, clines in groups.items():
        contour_items = make.contours(smoothed, np.array([level]), X, Y)
        items.extend(contour_items)

        # Place label at midpoint of longest segment
        longest = max(clines, key=lambda c: len(c.vertices))
        mid = longest.vertices[len(longest.vertices) // 2]
        label_item = make.label(
            LABEL_FMT.format(level), (float(mid[0]), float(mid[1])), LABEL_OFFSET, "BL"
        )
        items.append(label_item)

    return items


def build_annotation_items(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    n_levels: int,
    sigma: float = DEFAULT_SIGMA,
) -> list[AnnotatedPolygon]:
    """Build movable AnnotatedPolygon items for each contour segment.

    Each segment is wrapped in an AnnotatedPolygon so it can be stored in
    DataLab's annotation layer, moved, and serialized with the image.
    The annotation label shows the level value.

    Args:
        data: 2-D image array (float64)
        x: 1-D array of X coordinates (pixel centers)
        y: 1-D array of Y coordinates (pixel centers)
        n_levels: Number of isolevel bands
        sigma: Gaussian smoothing sigma

    Returns:
        List of AnnotatedPolygon items
    """
    groups = _compute_level_groups(data, x, y, n_levels, sigma)
    if not groups:
        return []

    ann_items = []
    for level, clines in groups.items():
        level_text = LABEL_FMT.format(level)

        def _make_info_cb(lv_text: str):
            return lambda _ann: lv_text

        # Find the longest segment for this level to carry the label
        longest = max(clines, key=lambda c: len(c.vertices))
        mid = longest.vertices[len(longest.vertices) // 2]
        mid_x, mid_y = float(mid[0]), float(mid[1])

        for cline in clines:
            ann = AnnotatedPolygon(
                points=cline.vertices,
                closed=False,
                info_callback=_make_info_cb(level_text),
            )
            ann.setTitle(f"Isolevel {level_text}")
            # Move label to midpoint of longest segment instead of bounding box center
            if cline is longest:
                ann.label.set_pos(mid_x, mid_y)
            ann_items.append(ann)

    return ann_items


class IsolevelDialog(QW.QDialog):
    """Interactive isolevel map dialog with PlotPy plot and controls.

    Args:
        parent: Parent widget
        data: 2-D image array
        x: 1-D array of X coordinates
        y: 1-D array of Y coordinates
        title: Image title
        proxy: DataLab LocalProxy (for annotation export)
    """

    def __init__(
        self,
        parent: QW.QWidget,
        data: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        title: str,
        proxy: datalab.plugins.LocalProxy | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Isolevel Map \u2014 {title}")
        self.setWindowFlags(QC.Qt.Window)
        self.resize(900, 700)

        self._data = data
        self._x = x
        self._y = y
        self._title = title
        self._proxy = proxy
        self._contour_items: list = []

        # --- PlotPy plot (embedded as child widget) ---
        options = PlotOptions(type=PlotType.IMAGE)
        self._plot_dialog = PlotDialog(
            parent=self,
            toolbar=True,
            options=options,
            auto_tools=True,
            title=title,
        )
        self._plot_dialog.setWindowFlags(QC.Qt.Widget)

        # --- Controls (FormLayout) ---
        controls_widget = QW.QWidget()
        form = QW.QFormLayout(controls_widget)
        form.setContentsMargins(6, 4, 6, 4)

        self._levels_spin = QW.QSpinBox()
        self._levels_spin.setMinimum(MIN_LEVELS)
        self._levels_spin.setMaximum(MAX_LEVELS)
        self._levels_spin.setValue(DEFAULT_LEVELS)
        self._levels_spin.setFixedWidth(70)
        form.addRow("Levels:", self._levels_spin)

        self._sigma_slider = QW.QSlider(QC.Qt.Horizontal)
        self._sigma_slider.setMinimum(MIN_SIGMA)
        self._sigma_slider.setMaximum(MAX_SIGMA * SIGMA_SCALE)
        self._sigma_slider.setValue(int(DEFAULT_SIGMA * SIGMA_SCALE))
        self._sigma_slider.setTickPosition(QW.QSlider.TicksBelow)
        self._sigma_slider.setTickInterval(SIGMA_SCALE)
        self._sigma_label = QW.QLabel(f"{DEFAULT_SIGMA:.1f}")
        self._sigma_label.setMinimumWidth(35)
        sigma_row = QW.QWidget()
        sigma_hlayout = QW.QHBoxLayout(sigma_row)
        sigma_hlayout.setContentsMargins(0, 0, 0, 0)
        sigma_hlayout.addWidget(self._sigma_slider)
        sigma_hlayout.addWidget(self._sigma_label)
        form.addRow("Smoothing (\u03c3):", sigma_row)

        # --- Export button ---
        self._export_btn = QW.QPushButton("Export to annotations")
        self._export_btn.setEnabled(proxy is not None)

        # --- Main layout ---
        layout = QW.QVBoxLayout()
        layout.addWidget(self._plot_dialog, stretch=1)
        layout.addWidget(controls_widget)
        layout.addWidget(self._export_btn)
        self.setLayout(layout)

        # --- Initial plot ---
        self._setup_image()
        self._refresh()

        # --- Connections ---
        self._levels_spin.editingFinished.connect(self._refresh)
        self._sigma_slider.valueChanged.connect(
            lambda v: self._sigma_label.setText(f"{v / SIGMA_SCALE:.1f}")
        )
        self._sigma_slider.sliderReleased.connect(self._refresh)
        self._export_btn.clicked.connect(self._export_annotations)

    def _setup_image(self) -> None:
        """Add the base image to the plot."""
        plot = self._plot_dialog.get_plot()
        image_item = make.xyimage(
            self._x, self._y, self._data, title=self._title, colormap="viridis"
        )
        plot.add_item(image_item)
        plot.set_active_item(image_item)
        plot.do_autoscale()

    def _refresh(self) -> None:
        """Recompute and redisplay contour items and labels."""
        n_levels = self._levels_spin.value()
        sigma = self._sigma_slider.value() / SIGMA_SCALE

        plot = self._plot_dialog.get_plot()
        for item in self._contour_items:
            plot.del_item(item)
        self._contour_items.clear()

        QW.QApplication.setOverrideCursor(QC.Qt.WaitCursor)
        try:
            items = compute_isolevel_items(
                self._data, self._x, self._y, n_levels, sigma
            )
            for item in items:
                plot.add_item(item)
            self._contour_items = items
        finally:
            QW.QApplication.restoreOverrideCursor()

        plot.replot()

    def _export_annotations(self) -> None:
        """Export current contour lines as movable DataLab annotations."""
        if self._proxy is None:
            return

        n_levels = self._levels_spin.value()
        sigma = self._sigma_slider.value() / SIGMA_SCALE

        QW.QApplication.setOverrideCursor(QC.Qt.WaitCursor)
        try:
            ann_items = build_annotation_items(
                self._data, self._x, self._y, n_levels, sigma
            )
        finally:
            QW.QApplication.restoreOverrideCursor()

        if not ann_items:
            return

        self._proxy.add_annotations_from_items(ann_items, panel="image")


class IsolevelMapPlugin(datalab.plugins.PluginBase):
    """Isolevel (contour) map plugin for DataLab."""

    PLUGIN_INFO = datalab.plugins.PluginInfo(
        name="Isolevel Map",
        version="1.0.0",
        description="Generate isolevel contour maps from images",
    )

    def show_isolevel_map(self) -> None:
        """Show an interactive isolevel map for the selected image."""
        obj = self.proxy.get_object(panel="image")
        if obj is None:
            self.show_warning("No image selected.")
            return

        data = np.real(obj.data) if np.iscomplexobj(obj.data) else obj.data.copy()
        data = data.astype(np.float64)

        x, y = get_obj_coords(obj)
        title = obj.title or "Image"

        dlg = IsolevelDialog(self.main, data, x, y, title, proxy=self.proxy)
        dlg.exec()

    def generate_test_image(self) -> None:
        """Generate a synthetic topographic-like test image.

        Creates an image with a large off-center Gaussian peak combined with
        smaller secondary peaks and noise, simulating a topographic elevation
        map suitable for isolevel visualization.
        """
        newparam = self.edit_new_image_parameters(
            title="Topographic test", shape=(2048, 2048), hide_dtype=True
        )
        if newparam is None:
            return

        h, w = newparam.height, newparam.width
        y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float64)
        xn = x_grid / w
        yn = y_grid / h

        cx, cy, sx, sy = 0.45, 0.55, 0.25, 0.20
        z = 0.6 * np.exp(
            -((xn - cx) ** 2 / (2 * sx**2) + (yn - cy) ** 2 / (2 * sy**2))
        )
        z += 0.25 * np.exp(
            -((xn - 0.75) ** 2 / (2 * 0.10**2) + (yn - 0.35) ** 2 / (2 * 0.12**2))
        )
        z += 0.15 * np.exp(
            -((xn - 0.20) ** 2 / (2 * 0.08**2) + (yn - 0.80) ** 2 / (2 * 0.09**2))
        )

        rng = np.random.default_rng(42)
        z += rng.normal(0, 0.02, (h, w))
        z = np.clip(z, 0.0, 1.0)

        obj = sigima.objects.create_image(
            newparam.title, z, units=("pixel", "pixel", "a.u.")
        )
        self.proxy.add_object(obj)

    def create_actions(self) -> None:
        """Create plugin menu actions."""
        acth = self.imagepanel.acthandler
        with acth.new_menu(self.PLUGIN_INFO.name):
            acth.new_action(
                "Generate test image",
                triggered=self.generate_test_image,
                select_condition="always",
            )
            acth.new_action(
                "Show isolevel map",
                triggered=self.show_isolevel_map,
            )

