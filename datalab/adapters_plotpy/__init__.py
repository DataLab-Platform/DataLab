# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Adapters for PlotPy
===================

The :mod:`datalab.adapters_plotpy` package provides adapters for
PlotPy to integrate with DataLab's data model and GUI.
"""

__all__ = [
    "GeometryPlotPyAdapter",
    "TablePlotPyAdapter",
    "TypePlotItem",
    "json_to_items",
    "items_to_json",
    "configure_roi_item",
    "TypeROIItem",
    "create_adapter_from_object",
    "plotitem_to_singleroi",
    "singleroi_to_plotitem",
    "CircularROIPlotPyAdapter",
    "ImageObjPlotPyAdapter",
    "PolygonalROIPlotPyAdapter",
    "RectangularROIPlotPyAdapter",
    "SegmentROIPlotPyAdapter",
    "SignalObjPlotPyAdapter",
    "CURVESTYLES",
    "SignalROIPlotPyAdapter",
]


from .base import items_to_json, json_to_items
from .converters import (
    create_adapter_from_object,
    plotitem_to_singleroi,
    singleroi_to_plotitem,
)
from .objects.base import TypePlotItem
from .objects.image import (
    ImageObjPlotPyAdapter,
)
from .objects.scalar import (
    GeometryPlotPyAdapter,
    TablePlotPyAdapter,
)
from .objects.signal import CURVESTYLES, SignalObjPlotPyAdapter
from .roi.base import TypeROIItem, configure_roi_item
from .roi.image import (
    CircularROIPlotPyAdapter,
    PolygonalROIPlotPyAdapter,
    RectangularROIPlotPyAdapter,
)
from .roi.signal import SegmentROIPlotPyAdapter, SignalROIPlotPyAdapter
