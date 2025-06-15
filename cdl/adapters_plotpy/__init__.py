"""
Adapters for PlotPy
===================

The :mod:`cdl.adapters_plotpy` package provides adapters for
PlotPy to integrate with DataLab's data model and GUI.
"""

# pylint: disable=unused-import
# flake8: noqa

from __future__ import annotations
from cdl.adapters_plotpy.base import (
    ResultPropertiesPlotPyAdapter,
    ResultShapePlotPyAdapter,
    TypePlotItem,
    json_to_items,
    items_to_json,
    configure_roi_item,
    TypeROIItem,
)
from cdl.adapters_plotpy.image import (
    CircularROIPlotPyAdapter,
    ImageObjPlotPyAdapter,
    PolygonalROIPlotPyAdapter,
    RectangularROIPlotPyAdapter,
)
from cdl.adapters_plotpy.signal import (
    SegmentROIPlotPyAdapter,
    SignalObjPlotPyAdapter,
    SignalROIPlotPyAdapter,
)
from cdl.adapters_plotpy.converters import (
    create_adapter_from_object,
    plotitem_to_singleroi,
    singleroi_to_plotitem,
)
