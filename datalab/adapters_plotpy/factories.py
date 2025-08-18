# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
PlotPy Adapter Factories
------------------------
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations


def create_adapter_from_object(object_to_adapt):
    """Create an adapter for the given object to integrate with PlotPy

    Args:
        object_to_adapt: The object to adapt (signal, image, ROI, or scalar result)

    Returns:
        An adapter instance
    """
    # pylint: disable=import-outside-toplevel
    # Import adapters as needed to avoid circular imports
    from sigima.objects import (
        CircularROI,
        ImageObj,
        ImageROI,
        PolygonalROI,
        RectangularROI,
        SegmentROI,
        SignalObj,
        SignalROI,
    )

    from datalab.adapters_metadata import GeometryAdapter, TableAdapter

    if isinstance(object_to_adapt, GeometryAdapter):
        from datalab.adapters_plotpy.objects.scalar import GeometryPlotPyAdapter

        adapter = GeometryPlotPyAdapter(object_to_adapt)
    elif isinstance(object_to_adapt, TableAdapter):
        from datalab.adapters_plotpy.objects.scalar import TablePlotPyAdapter

        adapter = TablePlotPyAdapter(object_to_adapt)

    elif isinstance(object_to_adapt, SignalObj):
        from datalab.adapters_plotpy.objects.signal import SignalObjPlotPyAdapter

        adapter = SignalObjPlotPyAdapter(object_to_adapt)
    elif isinstance(object_to_adapt, SignalROI):
        from datalab.adapters_plotpy.roi.signal import SignalROIPlotPyAdapter

        adapter = SignalROIPlotPyAdapter(object_to_adapt)
    elif isinstance(object_to_adapt, SegmentROI):
        from datalab.adapters_plotpy.roi.signal import SegmentROIPlotPyAdapter

        adapter = SegmentROIPlotPyAdapter(object_to_adapt)
    elif isinstance(object_to_adapt, ImageObj):
        from datalab.adapters_plotpy.objects.image import ImageObjPlotPyAdapter

        adapter = ImageObjPlotPyAdapter(object_to_adapt)
    elif isinstance(object_to_adapt, RectangularROI):
        from datalab.adapters_plotpy.roi.image import RectangularROIPlotPyAdapter

        adapter = RectangularROIPlotPyAdapter(object_to_adapt)
    elif isinstance(object_to_adapt, CircularROI):
        from datalab.adapters_plotpy.roi.image import CircularROIPlotPyAdapter

        adapter = CircularROIPlotPyAdapter(object_to_adapt)
    elif isinstance(object_to_adapt, PolygonalROI):
        from datalab.adapters_plotpy.roi.image import PolygonalROIPlotPyAdapter

        adapter = PolygonalROIPlotPyAdapter(object_to_adapt)
    elif isinstance(object_to_adapt, ImageROI):
        from datalab.adapters_plotpy.roi.image import ImageROIPlotPyAdapter

        adapter = ImageROIPlotPyAdapter(object_to_adapt)
    else:
        raise TypeError(f"Unsupported object type: {type(object_to_adapt)}")
    return adapter
