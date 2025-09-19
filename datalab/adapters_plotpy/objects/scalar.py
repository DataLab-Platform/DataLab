# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
PlotPy Adapter Scalar Module
----------------------------

This module contains adapters for scalar results (GeometryResult, TableResult)
to avoid circular imports with the base and factories modules.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np
from guidata.configtools import get_font
from plotpy.builder import make
from plotpy.items import (
    AnnotatedCircle,
    AnnotatedEllipse,
    AnnotatedPoint,
    AnnotatedRectangle,
    AnnotatedSegment,
    AnnotatedShape,
    LabelItem,
    Marker,
    PolygonShape,
)
from sigima.objects.base import BaseObj
from sigima.objects.scalar import KindShape
from sigima.tools import coordinates

from datalab.adapters_metadata import (
    GeometryAdapter,
    TableAdapter,
    resultadapter_to_html,
)
from datalab.adapters_plotpy.base import (
    config_annotated_shape,
    items_to_json,
    json_to_items,
    set_plot_item_editable,
)
from datalab.config import PLOTPY_CONF

if TYPE_CHECKING:
    from plotpy.styles import ShapeParam


class ResultPlotPyAdapter:
    """Adapter for converting `sigima` table or geometry adapters to PlotPy

    Args:
        result: Table or geometry adapter
    """

    def __init__(self, result_adapter: TableAdapter | GeometryAdapter) -> None:
        self.result_adapter = result_adapter
        self.item_json = ""  # JSON representation of the item

    def update_obj_metadata_from_item(
        self, obj: BaseObj, item: LabelItem | None
    ) -> None:
        """Update object metadata with label item

        Args:
            obj: object (signal/image)
            item: label item
        """
        if item is not None:
            self.item_json = items_to_json([item])
        self.result_adapter.set_obj_metadata(obj)

    def create_label_item(self, obj: BaseObj) -> LabelItem | None:
        """Create label item

        Args:
            obj: object (signal/image)

        Returns:
            Label item

        .. note::

            The signal or image object is required as argument to create the label
            item because the label text may contain format strings that need to be
            filled with the object properties. For instance, the label text may contain
            the signal or image units.
        """
        text = resultadapter_to_html(self.result_adapter, obj)
        item = make.label(text, "TL", (0, 0), "TL", title=self.result_adapter.title)
        font = get_font(PLOTPY_CONF, "properties", "label/font")
        item.set_style("properties", "label")
        item.labelparam.font.update_param(font)
        item.labelparam.update_item(item)
        return item

    def get_label_item(self, obj: BaseObj) -> LabelItem | None:
        """Return label item associated to this result

        Args:
            obj: object (signal/image)

        Returns:
            Label item

        .. note::

            The signal or image object is required as argument to eventually create
            the label item if it has not been created yet.
            See :py:meth:`create_label_item`.
        """
        if not self.item_json:
            # Label item has not been created yet
            item = self.create_label_item(obj)
            if item is not None:
                self.update_obj_metadata_from_item(obj, item)
        if self.item_json:
            item = json_to_items(self.item_json)[0]
            assert isinstance(item, LabelItem)
            return item
        return None


class GeometryPlotPyAdapter(ResultPlotPyAdapter):
    """Adapter for converting `sigima` geometry adapters to PlotPy

    Args:
        result: Geometry adapter

    Raises:
        AssertionError: invalid argument
    """

    def __init__(self, result_adapter: GeometryAdapter) -> None:
        assert isinstance(result_adapter, GeometryAdapter)
        super().__init__(result_adapter)

    def create_label_item(self, obj: BaseObj) -> LabelItem | None:
        """Create label item

        Returns:
            Label item
        """
        if self.result_adapter.geometry.kind is KindShape.SEGMENT:
            # Add a label item for the segment shape
            return super().create_label_item(obj)
        return None

    def iterate_plot_items(
        self, fmt: str, lbl: bool, option: Literal["s", "i"]
    ) -> Iterable:
        """Iterate over metadata shape plot items.

        Args:
            fmt: numeric format (e.g. "%.3f")
            lbl: if True, show shape labels
            option: shape style option ("s" for signal, "i" for image)

        Yields:
            Plot item
        """
        for coords in self.result_adapter.raw_data:
            yield self.create_shape_item(coords, fmt, lbl, option)

    def create_shape_item(
        self, coords: np.ndarray, fmt: str, lbl: bool, option: Literal["s", "i"]
    ) -> (
        AnnotatedPoint
        | Marker
        | AnnotatedRectangle
        | AnnotatedCircle
        | AnnotatedSegment
        | AnnotatedEllipse
        | PolygonShape
        | None
    ):
        """Create individual shape item from coordinates

        Args:
            coords: coordinate array
            fmt: numeric format (e.g. "%.3f")
            lbl: if True, show shape labels
            option: shape style option ("s" for signal, "i" for image)

        Returns:
            Plot item
        """
        if self.result_adapter.geometry.kind is KindShape.POINT:
            assert len(coords) == 2, "Coordinates must be a 2-element array"
            x0, y0 = coords
            item = AnnotatedPoint(x0, y0)
            sparam: ShapeParam = item.shape.shapeparam
            sparam.symbol.marker = "Ellipse"
            sparam.symbol.size = 6
            sparam.sel_symbol.marker = "Ellipse"
            sparam.sel_symbol.size = 6
            aparam = item.annotationparam
            aparam.title = self.result_adapter.title
            sparam.update_item(item.shape)
            aparam.update_item(item)
        elif self.result_adapter.geometry.kind is KindShape.MARKER:
            assert len(coords) == 2, "Coordinates must be a 2-element array"
            x0, y0 = coords
            item = self.__make_marker_item(x0, y0, fmt)
        elif self.result_adapter.geometry.kind is KindShape.RECTANGLE:
            assert len(coords) == 4, "Coordinates must be a 4-element array"
            x0, y0, dx, dy = coords
            item = make.annotated_rectangle(
                x0, y0, x0 + dx, y0 + dy, title=self.result_adapter.title
            )
        elif self.result_adapter.geometry.kind is KindShape.CIRCLE:
            assert len(coords) == 3, "Coordinates must be a 3-element array"
            xc, yc, r = coords
            x0, y0, x1, y1 = coordinates.circle_to_diameter(xc, yc, r)
            item = make.annotated_circle(
                x0, y0, x1, y1, title=self.result_adapter.title
            )
        elif self.result_adapter.geometry.kind is KindShape.SEGMENT:
            assert len(coords) == 4, "Coordinates must be a 4-element array"
            x0, y0, x1, y1 = coords
            item = make.annotated_segment(
                x0, y0, x1, y1, title=self.result_adapter.title
            )
        elif self.result_adapter.geometry.kind is KindShape.ELLIPSE:
            assert len(coords) == 5, "Coordinates must be a 5-element array"
            xc, yc, a, b, t = coords
            coords = coordinates.ellipse_to_diameters(xc, yc, a, b, t)
            x0, y0, x1, y1, x2, y2, x3, y3 = coords
            item = make.annotated_ellipse(
                x0, y0, x1, y1, x2, y2, x3, y3, title=self.result_adapter.title
            )
        elif self.result_adapter.geometry.kind is KindShape.POLYGON:
            assert len(coords) >= 6, "Coordinates must be at least 6-element array"
            assert len(coords) % 2 == 0, "Coordinates must be even-length array"
            x, y = coords[::2], coords[1::2]
            item = make.polygon(x, y, title=self.result_adapter.title, closed=False)
        else:
            raise NotImplementedError(
                f"Unsupported shape kind: {self.result_adapter.geometry.kind}"
            )
        if isinstance(item, AnnotatedShape):
            config_annotated_shape(item, fmt, lbl, "results", option)
        set_plot_item_editable(item, False)
        return item

    def __make_marker_item(self, x0: float, y0: float, fmt: str) -> Marker:
        """Make marker item

        Args:
            x0: x coordinate
            y0: y coordinate
            fmt: numeric format (e.g. '%.3f')
        """
        if np.isnan(x0):
            mstyle = "-"

            def label(x, y):  # pylint: disable=unused-argument
                return (self.result_adapter.title + ": " + fmt) % y

        elif np.isnan(y0):
            mstyle = "|"

            def label(x, y):  # pylint: disable=unused-argument
                return (self.result_adapter.title + ": " + fmt) % x

        else:
            mstyle = "+"
            txt = self.result_adapter.title + ": (" + fmt + ", " + fmt + ")"

            def label(x, y):
                return txt % (x, y)

        return make.marker(
            position=(x0, y0),
            markerstyle=mstyle,
            label_cb=label,
            linestyle="DashLine",
            color="yellow",
        )


class TablePlotPyAdapter(ResultPlotPyAdapter):
    """Adapter for converting `sigima` table adapters to PlotPy

    Args:
        result: Table adapter

    Raises:
        AssertionError: invalid argument
    """

    def __init__(self, result_adapter: TableAdapter) -> None:
        assert isinstance(result_adapter, TableAdapter)
        super().__init__(result_adapter)
