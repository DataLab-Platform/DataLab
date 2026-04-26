# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
PlotPy Adapter Signal ROI Module
--------------------------------
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np
from plotpy.items import AnnotatedXRange
from plotpy.items.shape.range import XRangeSelection
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from sigima.objects import SegmentROI, SignalObj, SignalROI
from sigima.objects.base import get_generic_roi_title

from datalab.adapters_plotpy.coordutils import round_signal_coords
from datalab.adapters_plotpy.roi.base import (
    BaseROIPlotPyAdapter,
    BaseSingleROIPlotPyAdapter,
    configure_roi_item,
)

if TYPE_CHECKING:
    import qwt.scale_map
    from qtpy.QtCore import QRectF
    from qtpy.QtGui import QPainter


# Color palette used to cycle through ROI fill colors when several ROIs are
# defined on the same signal. Inspired by the matplotlib ``tab10`` palette so
# that consecutive ROIs are easily distinguishable.
_ROI_FILL_COLORS: tuple[str, ...] = (
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
)

# Alpha (0..255) used for the translucent fill of ROI rectangles
_ROI_FILL_ALPHA: int = 90


def roi_color_for_index(index: int) -> QG.QColor:
    """Return the ROI fill color for the given ROI index (cycles through the
    predefined palette)."""
    name = _ROI_FILL_COLORS[index % len(_ROI_FILL_COLORS)]
    color = QG.QColor(name)
    color.setAlpha(_ROI_FILL_ALPHA)
    return color


class _CurveClippedXRangeSelection(XRangeSelection):
    """X-range selection whose translucent fill is clipped vertically to the
    underlying signal curve (instead of filling the whole canvas height).

    The fill color can additionally be overridden on a per-instance basis to
    enable color cycling between sibling ROIs displayed on the same signal.
    """

    def __init__(
        self,
        _min: float | None = None,
        _max: float | None = None,
        shapeparam=None,
    ) -> None:
        super().__init__(_min, _max, shapeparam)
        self._curve_x: np.ndarray | None = None
        self._curve_y: np.ndarray | None = None
        self._fill_color: QG.QColor | None = None

    def set_signal_curve(self, x: np.ndarray, y: np.ndarray) -> None:
        """Attach the signal curve coordinates used to clip the fill area."""
        self._curve_x = np.asarray(x, dtype=float)
        self._curve_y = np.asarray(y, dtype=float)

    def set_fill_color(self, color: QG.QColor) -> None:
        """Override the fill color for this ROI instance."""
        self._fill_color = QG.QColor(color)

    def _build_curve_polygon(
        self,
        xMap: qwt.scale_map.QwtScaleMap,
        yMap: qwt.scale_map.QwtScaleMap,
        rct: QRectF,
    ) -> QG.QPolygonF | None:
        """Build a polygon that follows the signal curve between ``self._min``
        and ``self._max``, with a flat baseline at y=0 (clamped to the visible
        canvas area when y=0 lies outside the current axis range).
        """
        if self._curve_x is None or self._curve_y is None:
            return None
        x_arr = self._curve_x
        y_arr = self._curve_y
        if x_arr.size < 2:
            return None
        xmin, xmax = self._min, self._max
        if xmin is None or xmax is None:
            return None
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        # Make sure x_arr is sorted (required by np.interp at the boundaries)
        if not np.all(np.diff(x_arr) >= 0):
            order = np.argsort(x_arr)
            x_arr = x_arr[order]
            y_arr = y_arr[order]
        # Restrict to the actual data extent to avoid drawing the polygon
        # outside the signal definition domain.
        x0 = max(xmin, float(x_arr[0]))
        x1 = min(xmax, float(x_arr[-1]))
        if x1 <= x0:
            return None
        mask = (x_arr >= x0) & (x_arr <= x1)
        xs_in = x_arr[mask]
        ys_in = y_arr[mask]
        y_left = float(np.interp(x0, x_arr, y_arr))
        y_right = float(np.interp(x1, x_arr, y_arr))
        xs = np.concatenate(([x0], xs_in, [x1]))
        ys = np.concatenate(([y_left], ys_in, [y_right]))
        # Remove duplicate boundary samples if interpolation hit an existing
        # data point exactly.
        keep = np.concatenate(([True], np.diff(xs) > 0))
        xs = xs[keep]
        ys = ys[keep]
        if xs.size < 2:
            return None
        # Baseline at y=0 in axis coordinates, clamped to the visible canvas
        # area so that the polygon stays renderable even when 0 is outside
        # the current y-axis range.
        baseline_y = yMap.transform(0.0)
        baseline_y = max(rct.top(), min(rct.bottom(), baseline_y))
        polygon = QG.QPolygonF()
        polygon.append(QC.QPointF(xMap.transform(xs[0]), baseline_y))
        for xv, yv in zip(xs, ys):
            polygon.append(QC.QPointF(xMap.transform(xv), yMap.transform(yv)))
        polygon.append(QC.QPointF(xMap.transform(xs[-1]), baseline_y))
        return polygon

    def draw(
        self,
        painter: QPainter,
        xMap: qwt.scale_map.QwtScaleMap,
        yMap: qwt.scale_map.QwtScaleMap,
        canvasRect: QRectF,
    ) -> None:
        """Draw the ROI: filled polygon clipped to the curve (or a fallback
        rectangle if no curve information is available), surrounded by the
        usual handle decoration (vertical edges and central dashed line)."""
        plot = self.plot()
        if not plot:
            return
        if self.selected:
            pen = self.sel_pen
            sym = self.sel_symbol
        else:
            pen = self.pen
            sym = self.symbol

        # Build the fallback rectangle covering the canvas height
        rct = QC.QRectF(plot.canvas().contentsRect())
        rct.setLeft(xMap.transform(self._min))
        rct.setRight(xMap.transform(self._max))

        # Choose the brush: per-instance color override has priority
        if self._fill_color is not None:
            brush = QG.QBrush(self._fill_color)
            # When using a per-instance color, also use it for the edges so
            # that the ROI keeps a consistent visual identity.
            edge_color = QG.QColor(self._fill_color)
            edge_color.setAlpha(255)
            pen = QG.QPen(pen)
            pen.setColor(edge_color)
        else:
            brush = self.brush

        polygon = self._build_curve_polygon(xMap, yMap, rct)
        painter.save()
        painter.setPen(QC.Qt.NoPen)
        painter.setBrush(brush)
        if polygon is not None:
            painter.drawPolygon(polygon)
        else:
            painter.fillRect(rct, brush)
        painter.restore()

        # Vertical edges at xmin and xmax (full canvas height)
        painter.setPen(pen)
        painter.drawLine(rct.topLeft(), rct.bottomLeft())
        painter.drawLine(rct.topRight(), rct.bottomRight())

        # Dashed central line
        dash = QG.QPen(pen)
        dash.setStyle(QC.Qt.DashLine)
        dash.setWidth(1)
        painter.setPen(dash)
        cx = rct.center().x()
        painter.drawLine(QC.QPointF(cx, rct.top()), QC.QPointF(cx, rct.bottom()))

        if self.can_resize() and not self.is_readonly():
            painter.setPen(pen)
            x0, x1, y = self.get_handles_pos()
            sym.drawSymbol(painter, QC.QPointF(x0, y))
            sym.drawSymbol(painter, QC.QPointF(x1, y))


class _CurveClippedAnnotatedXRange(AnnotatedXRange):
    """Annotated X-range selection whose underlying shape is a
    :class:`_CurveClippedXRangeSelection` (curve-clipped fill + per-instance
    color)."""

    SHAPE_CLASS = _CurveClippedXRangeSelection


class SegmentROIPlotPyAdapter(BaseSingleROIPlotPyAdapter[SegmentROI, AnnotatedXRange]):
    """Segment ROI plot item adapter

    Args:
        coords: ROI coordinates (xmin, xmax)
        title: ROI title
    """

    def to_plot_item(
        self,
        obj: SignalObj,
        fill_color: QG.QColor | None = None,
    ) -> AnnotatedXRange:
        """Make and return the annotated segment associated with the ROI

        Args:
            obj: object (signal), for physical-indices coordinates conversion
            fill_color: optional fill color override (used for color cycling
             between sibling ROIs)
        """
        xmin, xmax = self.single_roi.get_physical_coords(obj)
        item = _CurveClippedAnnotatedXRange(xmin, xmax)
        item.setTitle(self.single_roi.title)
        # Apply default range style so pen/brush/symbol attributes are set
        item.shape.set_style("plot", "range")
        # Provide curve coordinates for clipped rendering
        x, y = obj.xydata
        if x is not None and y is not None:
            item.shape.set_signal_curve(x, y)
        if fill_color is not None:
            item.shape.set_fill_color(fill_color)
        return item

    @classmethod
    def from_plot_item(
        cls, item: AnnotatedXRange, obj: SignalObj | None = None
    ) -> SegmentROI:
        """Create ROI from plot item

        Args:
            item: plot item
            obj: signal object for coordinate rounding (optional)

        Returns:
            ROI
        """
        if not isinstance(item, AnnotatedXRange):
            raise TypeError("Invalid plot item type")
        coords = sorted(item.get_range())
        # Round coordinates to appropriate precision
        if obj is not None:
            coords = round_signal_coords(obj, coords)
        title = str(item.title().text())
        return SegmentROI(coords, False, title)


class SignalROIPlotPyAdapter(BaseROIPlotPyAdapter[SignalROI]):
    """Signal ROI plot item adapter class

    Args:
        roi: ROI object
    """

    def to_plot_item(
        self,
        single_roi: SegmentROI,
        obj: SignalObj,
        fill_color: QG.QColor | None = None,
    ) -> AnnotatedXRange:
        """Make ROI plot item from single ROI

        Args:
            single_roi: single ROI object
            obj: object (signal/image), for physical-indices coordinates conversion
            fill_color: optional fill color override (used for color cycling)

        Returns:
            Plot item
        """
        return SegmentROIPlotPyAdapter(single_roi).to_plot_item(
            obj, fill_color=fill_color
        )

    def iterate_roi_items(
        self,
        obj: SignalObj,
        fmt: str,
        lbl: bool,
        editable: bool = True,
    ) -> Iterator[AnnotatedXRange]:
        """Iterate over ROI plot items, applying alternating fill colors so
        that several ROIs displayed on the same signal can be visually
        distinguished."""
        for index, single_roi in enumerate(self.roi.single_rois):
            color = roi_color_for_index(index)
            roi_item = self.to_plot_item(single_roi, obj, fill_color=color)
            item = configure_roi_item(
                roi_item, fmt, lbl, editable, option=self.roi.PREFIX
            )
            item.setTitle(single_roi.title or get_generic_roi_title(index))
            yield item
