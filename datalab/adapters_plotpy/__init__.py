# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Adapters for PlotPy
===================

The :mod:`datalab.adapters_plotpy` package provides adapters for
PlotPy to integrate with DataLab's data model and GUI.
"""

__all__ = [
    "CURVESTYLES",
    "CircularROIPlotPyAdapter",
    "GeometryPlotPyAdapter",
    "ImageObjPlotPyAdapter",
    "PolygonalROIPlotPyAdapter",
    "RectangularROIPlotPyAdapter",
    "SegmentROIPlotPyAdapter",
    "SignalObjPlotPyAdapter",
    "SignalROIPlotPyAdapter",
    "TablePlotPyAdapter",
    "TypePlotItem",
    "TypeROIItem",
    "configure_roi_item",
    "create_adapter_from_object",
    "items_to_json",
    "json_to_items",
    "plotitem_to_singleroi",
    "singleroi_to_plotitem",
]


from sigimax.adapters_plotpy.base import items_to_json, json_to_items
from sigimax.adapters_plotpy.roi.base import TypeROIItem, configure_roi_item
from sigimax.adapters_plotpy.roi.image import (
    CircularROIPlotPyAdapter,
    PolygonalROIPlotPyAdapter,
    RectangularROIPlotPyAdapter,
)

from .converters import (
    create_adapter_from_object,
    plotitem_to_singleroi,
    singleroi_to_plotitem,
)
from .objects.adapters import (
    CURVESTYLES,
    ImageObjPlotPyAdapter,
    SignalObjPlotPyAdapter,
    TypePlotItem,
)
from .objects.scalar import (
    GeometryPlotPyAdapter,
    TablePlotPyAdapter,
)
from .roi.signal import SegmentROIPlotPyAdapter, SignalROIPlotPyAdapter
