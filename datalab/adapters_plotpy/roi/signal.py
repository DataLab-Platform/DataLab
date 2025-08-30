# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
PlotPy Adapter Signal ROI Module
--------------------------------
"""

from __future__ import annotations

from plotpy.builder import make
from plotpy.items import AnnotatedXRange
from sigima.objects import SegmentROI, SignalObj, SignalROI

from datalab.adapters_plotpy.roi.base import (
    BaseROIPlotPyAdapter,
    BaseSingleROIPlotPyAdapter,
)


class SegmentROIPlotPyAdapter(BaseSingleROIPlotPyAdapter[SegmentROI, AnnotatedXRange]):
    """Segment ROI plot item adapter

    Args:
        coords: ROI coordinates (xmin, xmax)
        title: ROI title
    """

    def to_plot_item(self, obj: SignalObj) -> AnnotatedXRange:
        """Make and return the annnotated segment associated with the ROI

        Args:
            obj: object (signal), for physical-indices coordinates conversion
        """
        xmin, xmax = self.single_roi.get_physical_coords(obj)
        item = make.annotated_xrange(xmin, xmax, title=self.single_roi.title)
        return item

    @classmethod
    def from_plot_item(cls, item: AnnotatedXRange) -> SegmentROI:
        """Create ROI from plot item

        Args:
            item: plot item

        Returns:
            ROI
        """
        if not isinstance(item, AnnotatedXRange):
            raise TypeError("Invalid plot item type")
        coords = sorted(item.get_range())
        title = str(item.title().text())
        return SegmentROI(coords, False, title)


class SignalROIPlotPyAdapter(BaseROIPlotPyAdapter[SignalROI]):
    """Signal ROI plot item adapter class

    Args:
        roi: ROI object
    """

    def to_plot_item(self, single_roi: SegmentROI, obj: SignalObj) -> AnnotatedXRange:
        """Make ROI plot item from single ROI

        Args:
            single_roi: single ROI object
            obj: object (signal/image), for physical-indices coordinates conversion

        Returns:
            Plot item
        """
        return SegmentROIPlotPyAdapter(single_roi).to_plot_item(obj)
