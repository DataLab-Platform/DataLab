# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
PlotPy Adapter Converters
-------------------------
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import Any

from plotpy.builder import make
from plotpy.items import (
    AnnotatedCircle,
    AnnotatedEllipse,
    AnnotatedPoint,
    AnnotatedPolygon,
    AnnotatedRectangle,
    AnnotatedSegment,
    LabelItem,
    Marker,
    XRangeSelection,
)

from cdl.adapters_plotpy.factories import create_adapter_from_object
from sigima_ import (
    CircularROI,
    PolygonalROI,
    RectangularROI,
    SegmentROI,
)
from sigima_.model.annotation import (
    AnnotationRegistry,
    Circle,
    Ellipse,
    HCursor,
    Label,
    Line,
    Point,
    Polygon,
    Range,
    Rectangle,
    VCursor,
    XCursor,
)


def plotitem_to_singleroi(
    plot_item: XRangeSelection
    | AnnotatedRectangle
    | AnnotatedCircle
    | AnnotatedPolygon,
) -> SegmentROI | RectangularROI | CircularROI | PolygonalROI:
    """Create a single ROI from the given PlotPy item to integrate with DataLab

    Args:
        plot_item: The PlotPy item for which to create a single ROI

    Returns:
        A single ROI instance
    """
    # pylint: disable=import-outside-toplevel
    from cdl.adapters_plotpy.image import (
        CircularROIPlotPyAdapter,
        PolygonalROIPlotPyAdapter,
        RectangularROIPlotPyAdapter,
    )
    from cdl.adapters_plotpy.signal import (
        SegmentROIPlotPyAdapter,
    )

    if isinstance(plot_item, XRangeSelection):
        adapter = SegmentROIPlotPyAdapter
    elif isinstance(plot_item, AnnotatedRectangle):
        adapter = RectangularROIPlotPyAdapter
    elif isinstance(plot_item, AnnotatedCircle):
        adapter = CircularROIPlotPyAdapter
    elif isinstance(plot_item, AnnotatedPolygon):
        adapter = PolygonalROIPlotPyAdapter
    else:
        raise TypeError(f"Unsupported PlotPy item type: {type(plot_item)}")
    return adapter.from_plot_item(plot_item)


def singleroi_to_plotitem(
    roi: SegmentROI | RectangularROI | CircularROI | PolygonalROI,
) -> XRangeSelection | AnnotatedRectangle | AnnotatedCircle | AnnotatedPolygon:
    """Create a PlotPy item from the given single ROI to integrate with DataLab

    Args:
        roi: The single ROI for which to create a PlotPy item

    Returns:
        A PlotPy item instance
    """
    adapter = create_adapter_from_object(roi)
    return adapter.to_plot_item()


def items_to_annotation_json_dicts(items: list) -> list[dict[str, Any]]:
    """
    Convert PlotPy annotated items to sigima Annotation JSON dict representations.

    Args:
        items: List of PlotPy annotation items

    Returns:
        List of sigima Annotation JSON dict representations
    """
    annotations = []
    for item in items:
        try:
            if isinstance(item, LabelItem):
                x, y = item.G
                ann = Label(x, y, text=item.get_plain_text())

            elif isinstance(item, Marker):
                x, y = item.get_pos()
                mstyle = item.markerparam.markerstyle
                if mstyle == "|":
                    ann = VCursor(x)
                elif mstyle == "-":
                    ann = HCursor(y)
                elif mstyle == "+":
                    ann = XCursor(x, y)
                else:
                    continue  # Unsupported marker type

            elif isinstance(item, XRangeSelection):
                x1, x2 = item.get_range()
                ann = Range(x1, x2)

            elif isinstance(item, AnnotatedPoint):
                x, y = item.get_pos()
                ann = Point(x, y, text=item.get_text())

            elif isinstance(item, AnnotatedSegment):
                x1, y1, x2, y2 = item.get_rect()
                ann = Line(x1, y1, x2, y2, text=item.get_text())

            elif isinstance(item, AnnotatedRectangle):
                x1, y1, x2, y2 = item.get_rect()
                ann = Rectangle(x1, y1, x2, y2, text=item.get_text())

            elif isinstance(item, AnnotatedCircle):
                x1, y1, x2, y2 = item.get_rect()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                r = (x2 - x1) / 2
                ann = Circle(cx, cy, r, text=item.get_text())

            elif isinstance(item, AnnotatedEllipse):
                x1, y1, x2, y2 = item.get_rect()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                a = (x2 - x1) / 2
                b = (y2 - y1) / 2
                ann = Ellipse(cx, cy, a, b, text=item.get_text())

            elif isinstance(item, AnnotatedPolygon):
                pts = [tuple(p) for p in item.get_points()]
                ann = Polygon(points=pts, text=item.get_text())

            else:
                continue  # Unsupported item

            annotations.append(ann.to_json())

        except Exception as exc:
            raise TypeError(
                f"Failed to convert item {item} to annotation: {exc}"
            ) from exc

    return annotations


def annotation_json_dicts_to_items(
    annotations: list[dict[str, Any]],
) -> list[
    LabelItem
    | AnnotatedSegment
    | AnnotatedRectangle
    | AnnotatedCircle
    | AnnotatedPoint
    | AnnotatedEllipse
    | AnnotatedPolygon
]:
    """
    Convert sigima annotation JSON dict representations to PlotPy items.

    Args:
        annotations: List of sigima Annotation JSON dict representations

    Returns:
        List of PlotPy items
    """
    items = []
    for ann_dict in annotations:
        ann = AnnotationRegistry.create_from_json(ann_dict)
        pts = ann.get_points()
        text = ann.get_text()
        if ann.type_name == "Label":
            item = make.label(text, "L", tuple(pts[0]), "L")
        elif ann.type_name == "HCursor":
            item = make.hcursor(pts[0][0], title=text)
        elif ann.type_name == "VCursor":
            item = make.vcursor(pts[0][1], title=text)
        elif ann.type_name == "XCursor":
            item = make.xcursor(pts[0][0], pts[0][1], title=text)
        elif ann.type_name == "Range":
            item = make.xrange_selection(pts[0][0], pts[0][1], title=text)
        elif ann.type_name == "Point":
            item = make.annotated_point(*pts[0], title=text)
        elif ann.type_name == "Line":
            item = make.annotated_segment(*pts[0], *pts[1], title=text)
        elif ann.type_name == "Rectangle":
            item = make.annotated_rectangle(*pts[0], *pts[1], title=text)
        elif ann.type_name == "Circle":
            cx, cy, r = pts[0]
            item = make.annotated_circle(cx, cy, r, title=text)
        elif ann.type_name == "Ellipse":
            cx, cy, a, b = pts[0]
            item = make.annotated_ellipse(cx, cy, a, b, title=text)
        elif ann.type_name == "Polygon":
            # pts is a list of tuples, convert to separate x and y lists
            xs, ys = zip(*pts)
            item = make.polygon(xs, ys, title=text, closed=True)
        else:
            raise TypeError(f"Unsupported annotation type: {ann.type_name}")
        items.append(item)
    return items
