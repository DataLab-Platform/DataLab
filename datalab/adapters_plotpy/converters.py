# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
PlotPy Adapter Converters
-------------------------
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING

from plotpy.items import (
    AnnotatedCircle,
    AnnotatedPolygon,
    AnnotatedRectangle,
    AnnotatedXRange,
)
from sigima.objects import (
    CircularROI,
    PolygonalROI,
    RectangularROI,
    SegmentROI,
)

from datalab.adapters_plotpy.factories import create_adapter_from_object

if TYPE_CHECKING:
    from sigima.objects import ImageObj, SignalObj


def plotitem_to_singleroi(
    plot_item: AnnotatedXRange
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
    from datalab.adapters_plotpy.image import (
        CircularROIPlotPyAdapter,
        PolygonalROIPlotPyAdapter,
        RectangularROIPlotPyAdapter,
    )
    from datalab.adapters_plotpy.signal import (
        SegmentROIPlotPyAdapter,
    )

    if isinstance(plot_item, AnnotatedXRange):
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
    obj: SignalObj | ImageObj,
) -> AnnotatedXRange | AnnotatedRectangle | AnnotatedCircle | AnnotatedPolygon:
    """Create a PlotPy item from the given single ROI to integrate with DataLab

    Args:
        roi: The single ROI for which to create a PlotPy item
        obj: The object (signal or image) associated with the ROI

    Returns:
        A PlotPy item instance
    """
    adapter = create_adapter_from_object(roi)
    return adapter.to_plot_item(obj)
