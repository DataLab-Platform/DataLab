"""
Adapters for PlotPy
===================

The :mod:`datalab.adapters_plotpy` package provides adapters for
PlotPy to integrate with DataLab's data model and GUI.
"""

# pylint: disable=unused-import
# flake8: noqa

from __future__ import annotations
from datalab.adapters_plotpy.base import (
    GeometryPlotPyAdapter,
    TablePlotPyAdapter,
    TypePlotItem,
    json_to_items,
    items_to_json,
    configure_roi_item,
    TypeROIItem,
)
from datalab.adapters_plotpy.image import (
    CircularROIPlotPyAdapter,
    ImageObjPlotPyAdapter,
    PolygonalROIPlotPyAdapter,
    RectangularROIPlotPyAdapter,
)
from datalab.adapters_plotpy.signal import (
    SegmentROIPlotPyAdapter,
    SignalObjPlotPyAdapter,
    SignalROIPlotPyAdapter,
)
from datalab.adapters_plotpy.converters import (
    create_adapter_from_object,
    plotitem_to_singleroi,
    singleroi_to_plotitem,
)
