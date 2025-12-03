# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
PlotPy Adapter Image ROI Module
-------------------------------
"""

from __future__ import annotations

import numpy as np
from plotpy.builder import make
from plotpy.items import AnnotatedCircle, AnnotatedPolygon, AnnotatedRectangle
from sigima.objects import CircularROI, ImageObj, ImageROI, PolygonalROI, RectangularROI
from sigima.tools import coordinates

from datalab.adapters_plotpy.coordutils import round_image_coords
from datalab.adapters_plotpy.roi.base import (
    BaseROIPlotPyAdapter,
    BaseSingleROIPlotPyAdapter,
)


def _vs(var: str, sub: str = "") -> str:
    """Return variable name with subscript"""
    txt = f"<var>{var}</var>"
    if sub:
        txt += f"<sub>{sub}</sub>"
    return txt


class PolygonalROIPlotPyAdapter(
    BaseSingleROIPlotPyAdapter[PolygonalROI, AnnotatedPolygon]
):
    """Polygonal ROI plot item adapter

    Args:
        single_roi: single ROI object
    """

    def to_plot_item(self, obj: ImageObj) -> AnnotatedPolygon:
        """Make and return the annnotated polygon associated to ROI

        Args:
            obj: object (image), for physical-indices coordinates conversion
        """

        def info_callback(item: AnnotatedPolygon) -> str:
            """Return info string for circular ROI"""
            xc, yc = item.get_center()
            if self.single_roi.indices:
                xc, yc = obj.physical_to_indices([xc, yc])
            return "<br>".join(
                [
                    f"({_vs('x', 'c')}, {_vs('y', 'c')}) = ({xc:g}, {yc:g})",
                ]
            )

        coords = np.array(self.single_roi.get_physical_coords(obj))
        points = coords.reshape(-1, 2)
        item = AnnotatedPolygon(points)
        item.set_info_callback(info_callback)
        item.annotationparam.title = self.single_roi.title
        item.annotationparam.update_item(item)
        item.set_style("plot", "shape/drag")
        return item

    @classmethod
    def from_plot_item(
        cls, item: AnnotatedPolygon, obj: ImageObj | None = None
    ) -> PolygonalROI:
        """Create ROI from plot item

        Args:
            item: plot item
            obj: image object for coordinate rounding (optional)
        """
        coords = item.get_points().flatten().tolist()
        # Round coordinates to appropriate precision
        if obj is not None:
            coords = round_image_coords(obj, coords)
        title = str(item.title().text())
        return PolygonalROI(coords, False, title)


class RectangularROIPlotPyAdapter(
    BaseSingleROIPlotPyAdapter[RectangularROI, AnnotatedRectangle]
):
    """Rectangular ROI plot item adapter

    Args:
        single_roi: single ROI object
    """

    def to_plot_item(self, obj: ImageObj) -> AnnotatedRectangle:
        """Make and return the annnotated rectangle associated to ROI

        Args:
            obj: object (image), for physical-indices coordinates conversion
        """

        def info_callback(item: AnnotatedRectangle) -> str:
            """Return info string for rectangular ROI"""
            x0, y0, x1, y1 = item.get_rect()
            if self.single_roi.indices:
                x0, y0, x1, y1 = obj.physical_to_indices([x0, y0, x1, y1])
            x0, y0, dx, dy = self.single_roi.rect_to_coords(x0, y0, x1, y1)
            return "<br>".join(
                [
                    f"({_vs('x', '0')}, {_vs('y', '0')}) = ({x0:g}, {y0:g})",
                    f"{_vs('Δx')} × {_vs('Δy')} = {dx:g} × {dy:g}",
                ]
            )

        x0, y0, dx, dy = self.single_roi.get_physical_coords(obj)
        x1, y1 = x0 + dx, y0 + dy
        item: AnnotatedRectangle = make.annotated_rectangle(
            x0, y0, x1, y1, title=self.single_roi.title
        )
        item.set_info_callback(info_callback)
        param = item.label.labelparam
        param.anchor = "BL"
        param.xc, param.yc = 5, -5
        param.update_item(item.label)
        return item

    @classmethod
    def from_plot_item(
        cls, item: AnnotatedRectangle, obj: ImageObj | None = None
    ) -> RectangularROI:
        """Create ROI from plot item

        Args:
            item: plot item
            obj: image object for coordinate rounding (optional)
        """
        rect = item.get_rect()
        coords = RectangularROI.rect_to_coords(*rect)
        # Round coordinates to appropriate precision
        if obj is not None:
            coords = round_image_coords(obj, coords)
        title = str(item.title().text())
        return RectangularROI(coords, False, title)


class CircularROIPlotPyAdapter(
    BaseSingleROIPlotPyAdapter[CircularROI, AnnotatedCircle]
):
    """Circular ROI plot item adapter

    Args:
        single_roi: single ROI object
    """

    def to_plot_item(self, obj: ImageObj) -> AnnotatedCircle:
        """Make and return the annnotated circle associated to ROI

        Args:
            obj: object (image), for physical-indices coordinates conversion
        """

        def info_callback(item: AnnotatedCircle) -> str:
            """Return info string for circular ROI"""
            x0, y0, x1, y1 = item.get_rect()
            if self.single_roi.indices:
                x0, y0, x1, y1 = obj.physical_to_indices([x0, y0, x1, y1])
            xc, yc, r = self.single_roi.rect_to_coords(x0, y0, x1, y1)
            return "<br>".join(
                [
                    f"({_vs('x', 'c')}, {_vs('y', 'c')}) = ({xc:g}, {yc:g})",
                    f"{_vs('r')} = {r:g}",
                ]
            )

        xc, yc, r = self.single_roi.get_physical_coords(obj)
        x0, y0, x1, y1 = coordinates.circle_to_diameter(xc, yc, r)
        item = AnnotatedCircle(x0, y0, x1, y1)
        item.set_info_callback(info_callback)
        item.annotationparam.title = self.single_roi.title
        item.annotationparam.update_item(item)
        item.set_style("plot", "shape/drag")
        return item

    @classmethod
    def from_plot_item(
        cls, item: AnnotatedCircle, obj: ImageObj | None = None
    ) -> CircularROI:
        """Create ROI from plot item

        Args:
            item: plot item
            obj: image object for coordinate rounding (optional)
        """
        rect = item.get_rect()
        coords = CircularROI.rect_to_coords(*rect)
        # Round coordinates to appropriate precision
        # For circular ROI: [xc, yc, r] - round center (xc, yc) as pair, then radius
        if obj is not None:
            xc, yc, r = coords
            # Round center coordinates
            xc_rounded, yc_rounded = round_image_coords(obj, [xc, yc])
            # Round radius using average of X and Y precision
            # For radius, we use the X precision (could also average X and Y)
            r_rounded = round_image_coords(obj, [r, 0])[0]
            coords = [xc_rounded, yc_rounded, r_rounded]
        title = str(item.title().text())
        return CircularROI(coords, False, title)


class ImageROIPlotPyAdapter(BaseROIPlotPyAdapter[ImageROI]):
    """Image ROI plot item adapter class

    Args:
        roi: ROI object
    """

    def to_plot_item(
        self,
        single_roi: PolygonalROI | RectangularROI | CircularROI,
        obj: ImageObj,
    ) -> AnnotatedCircle | AnnotatedRectangle | AnnotatedPolygon:
        """Make ROI plot item from single ROI

        Args:
            single_roi: single ROI object
            obj: object (signal/image), for physical-indices coordinates conversion

        Returns:
            Plot item
        """
        if isinstance(single_roi, PolygonalROI):
            return PolygonalROIPlotPyAdapter(single_roi).to_plot_item(obj)
        if isinstance(single_roi, RectangularROI):
            return RectangularROIPlotPyAdapter(single_roi).to_plot_item(obj)
        if isinstance(single_roi, CircularROI):
            return CircularROIPlotPyAdapter(single_roi).to_plot_item(obj)
        raise TypeError(f"Invalid ROI type {type(single_roi)}")
