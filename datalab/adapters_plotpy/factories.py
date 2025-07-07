# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
PlotPy Adapter Factories
------------------------
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING

from sigima.objects import (
    CircularROI,
    ImageObj,
    ImageROI,
    PolygonalROI,
    RectangularROI,
    ResultProperties,
    ResultShape,
    SegmentROI,
    SignalObj,
    SignalROI,
)

if TYPE_CHECKING:
    from datalab.adapters_plotpy.base import (
        ResultPropertiesPlotPyAdapter,
        ResultShapePlotPyAdapter,
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


def create_adapter_from_object(
    object_to_adapt: ResultProperties
    | ResultShape
    | SignalObj
    | SignalROI
    | SegmentROI
    | ImageObj
    | RectangularROI
    | CircularROI
    | PolygonalROI,
) -> (
    ResultPropertiesPlotPyAdapter
    | ResultShapePlotPyAdapter
    | SignalObjPlotPyAdapter
    | SignalROIPlotPyAdapter
    | SegmentROIPlotPyAdapter
    | ImageObjPlotPyAdapter
    | RectangularROIPlotPyAdapter
    | CircularROIPlotPyAdapter
    | PolygonalROIPlotPyAdapter
):
    """Create an adapter for the given object to integrate with PlotPy

    Args:
        object_to_adapt: The object to adapt (e.g., SignalObj, ImageObj)

    Returns:
        An adapter instance
    """
    # pylint: disable=import-outside-toplevel
    from datalab.adapters_plotpy.base import (
        ResultPropertiesPlotPyAdapter,
        ResultShapePlotPyAdapter,
    )
    from datalab.adapters_plotpy.image import (
        CircularROIPlotPyAdapter,
        ImageObjPlotPyAdapter,
        ImageROIPlotPyAdapter,
        PolygonalROIPlotPyAdapter,
        RectangularROIPlotPyAdapter,
    )
    from datalab.adapters_plotpy.signal import (
        SegmentROIPlotPyAdapter,
        SignalObjPlotPyAdapter,
        SignalROIPlotPyAdapter,
    )

    if isinstance(object_to_adapt, ResultProperties):
        adapter = ResultPropertiesPlotPyAdapter(object_to_adapt)
    elif isinstance(object_to_adapt, ResultShape):
        adapter = ResultShapePlotPyAdapter(object_to_adapt)
    elif isinstance(object_to_adapt, SignalObj):
        adapter = SignalObjPlotPyAdapter(object_to_adapt)
    elif isinstance(object_to_adapt, SignalROI):
        adapter = SignalROIPlotPyAdapter(object_to_adapt)
    elif isinstance(object_to_adapt, SegmentROI):
        adapter = SegmentROIPlotPyAdapter(object_to_adapt)
    elif isinstance(object_to_adapt, ImageObj):
        adapter = ImageObjPlotPyAdapter(object_to_adapt)
    elif isinstance(object_to_adapt, RectangularROI):
        adapter = RectangularROIPlotPyAdapter(object_to_adapt)
    elif isinstance(object_to_adapt, CircularROI):
        adapter = CircularROIPlotPyAdapter(object_to_adapt)
    elif isinstance(object_to_adapt, PolygonalROI):
        adapter = PolygonalROIPlotPyAdapter(object_to_adapt)
    elif isinstance(object_to_adapt, ImageROI):
        adapter = ImageROIPlotPyAdapter(object_to_adapt)
    else:
        raise TypeError(f"Unsupported object type: {type(object_to_adapt)}")
    return adapter
