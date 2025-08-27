# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI Adapter Factory
-------------------

Factory functions for creating ROI adapters without circular imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sigima.objects.base import TypeObj


def create_roi_adapter(roi):
    """Create ROI adapter from ROI object

    Args:
        roi: ROI object

    Returns:
        ROI adapter instance
    """
    # pylint: disable=import-outside-toplevel
    from sigima.objects import (
        CircularROI,
        ImageROI,
        PolygonalROI,
        RectangularROI,
        SegmentROI,
        SignalROI,
    )

    if isinstance(roi, SignalROI):
        from datalab.adapters_plotpy.roi.signal import SignalROIPlotPyAdapter

        return SignalROIPlotPyAdapter(roi)
    if isinstance(roi, RectangularROI):
        from datalab.adapters_plotpy.roi.image import RectangularROIPlotPyAdapter

        return RectangularROIPlotPyAdapter(roi)
    if isinstance(roi, CircularROI):
        from datalab.adapters_plotpy.roi.image import CircularROIPlotPyAdapter

        return CircularROIPlotPyAdapter(roi)
    if isinstance(roi, PolygonalROI):
        from datalab.adapters_plotpy.roi.image import PolygonalROIPlotPyAdapter

        return PolygonalROIPlotPyAdapter(roi)
    if isinstance(roi, ImageROI):
        from datalab.adapters_plotpy.roi.image import ImageROIPlotPyAdapter

        return ImageROIPlotPyAdapter(roi)
    if isinstance(roi, SegmentROI):
        from datalab.adapters_plotpy.roi.signal import SegmentROIPlotPyAdapter

        return SegmentROIPlotPyAdapter(roi)
    raise TypeError(f"Unsupported ROI type: {type(roi)}")


def create_single_roi_plot_item(single_roi, obj: TypeObj):
    """Create plot item from single ROI

    Args:
        single_roi: single ROI object
        obj: object (signal/image), for physical-indices coordinates conversion

    Returns:
        Plot item
    """
    # pylint: disable=import-outside-toplevel
    from sigima.objects import (
        CircularROI,
        PolygonalROI,
        RectangularROI,
    )

    if isinstance(single_roi, RectangularROI):
        from datalab.adapters_plotpy.roi.image import RectangularROIPlotPyAdapter

        return RectangularROIPlotPyAdapter(single_roi).to_plot_item(obj)
    if isinstance(single_roi, CircularROI):
        from datalab.adapters_plotpy.roi.image import CircularROIPlotPyAdapter

        return CircularROIPlotPyAdapter(single_roi).to_plot_item(obj)
    if isinstance(single_roi, PolygonalROI):
        from datalab.adapters_plotpy.roi.image import PolygonalROIPlotPyAdapter

        return PolygonalROIPlotPyAdapter(single_roi).to_plot_item(obj)
    raise TypeError(f"Unsupported ROI type: {type(single_roi)}")
