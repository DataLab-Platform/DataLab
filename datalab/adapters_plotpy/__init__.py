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
    ResultPropertiesPlotPyAdapter,
    ResultShapePlotPyAdapter,
    TypePlotItem,
    TypeROIItem,
    configure_roi_item,
    items_to_json,
    json_to_items,
)
from datalab.adapters_plotpy.converters import (
    create_adapter_from_object,
    plotitem_to_singleroi,
    singleroi_to_plotitem,
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
