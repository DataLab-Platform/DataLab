# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
PlotPy Adapter Image Module
---------------------------
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
from guidata.dataset import update_dataset
from plotpy.builder import make
from plotpy.items import (
    AnnotatedCircle,
    AnnotatedPolygon,
    AnnotatedRectangle,
    MaskedImageItem,
)
from sigima.obj import CircularROI, ImageObj, ImageROI, PolygonalROI, RectangularROI

from cdl.adapters_plotpy.base import (
    BaseObjPlotPyAdapter,
    BaseROIPlotPyAdapter,
    BaseSingleROIPlotPyAdapter,
)
from cdl.config import Conf


class PolygonalROIPlotPyAdapter(
    BaseSingleROIPlotPyAdapter[PolygonalROI, AnnotatedPolygon]
):
    """Polygonal ROI plot item adapter

    Args:
        single_roi: single ROI object
    """

    def to_plot_item(self, obj: ImageObj, title: str | None = None) -> AnnotatedPolygon:
        """Make and return the annnotated polygon associated to ROI

        Args:
            obj: object (image), for physical-indices coordinates conversion
            title: title
        """
        coords = np.array(self.single_roi.get_physical_coords(obj))
        points = coords.reshape(-1, 2)
        item = AnnotatedPolygon(points)
        item.annotationparam.title = self.single_roi.title if title is None else title
        item.annotationparam.update_item(item)
        item.set_style("plot", "shape/drag")
        return item

    @classmethod
    def from_plot_item(cls, item: AnnotatedPolygon) -> PolygonalROI:
        """Create ROI from plot item

        Args:
            item: plot item
        """
        return PolygonalROI(
            item.get_points().flatten(), False, item.annotationparam.title
        )


class RectangularROIPlotPyAdapter(
    BaseSingleROIPlotPyAdapter[RectangularROI, AnnotatedRectangle]
):
    """Rectangular ROI plot item adapter

    Args:
        single_roi: single ROI object
    """

    def to_plot_item(
        self, obj: ImageObj, title: str | None = None
    ) -> AnnotatedRectangle:
        """Make and return the annnotated rectangle associated to ROI

        Args:
            obj: object (image), for physical-indices coordinates conversion
            title: title
        """

        def info_callback(item: AnnotatedRectangle) -> str:
            """Return info string for rectangular ROI"""
            x0, y0, x1, y1 = item.get_rect()
            if self.single_roi.indices:
                x0, y0, x1, y1 = obj.physical_to_indices([x0, y0, x1, y1])
            x0, y0, dx, dy = self.single_roi.rect_to_coords(x0, y0, x1, y1)
            return "<br>".join(
                [
                    f"X0, Y0 = {x0:g}, {y0:g}",
                    f"ΔX x ΔY  = {dx:g} x {dy:g}",
                ]
            )

        x0, y0, dx, dy = self.single_roi.get_physical_coords(obj)
        x1, y1 = x0 + dx, y0 + dy
        title = self.single_roi.title if title is None else title
        roi_item: AnnotatedRectangle = make.annotated_rectangle(x0, y0, x1, y1, title)
        roi_item.set_info_callback(info_callback)
        param = roi_item.label.labelparam
        param.anchor = "BL"
        param.xc, param.yc = 5, -5
        param.update_item(roi_item.label)
        return roi_item

    @classmethod
    def from_plot_item(cls, item: AnnotatedRectangle) -> RectangularROI:
        """Create ROI from plot item

        Args:
            item: plot item
        """
        rect = item.get_rect()
        return RectangularROI(
            RectangularROI.rect_to_coords(*rect), False, item.annotationparam.title
        )


class CircularROIPlotPyAdapter(
    BaseSingleROIPlotPyAdapter[CircularROI, AnnotatedCircle]
):
    """Circular ROI plot item adapter

    Args:
        single_roi: single ROI object
    """

    def to_plot_item(self, obj: ImageObj, title: str | None = None) -> AnnotatedCircle:
        """Make and return the annnotated circle associated to ROI

        Args:
            obj: object (image), for physical-indices coordinates conversion
            title: title
        """

        def info_callback(item: AnnotatedCircle) -> str:
            """Return info string for circular ROI"""
            x0, y0, x1, y1 = item.get_rect()
            if self.single_roi.indices:
                x0, y0, x1, y1 = obj.physical_to_indices([x0, y0, x1, y1])
            xc, yc, r = self.single_roi.rect_to_coords(x0, y0, x1, y1)
            return "<br>".join(
                [
                    f"Center = {xc:g}, {yc:g}",
                    f"Radius = {r:g}",
                ]
            )

        xc, yc, r = self.single_roi.get_physical_coords(obj)
        item = AnnotatedCircle(xc - r, yc, xc + r, yc)
        item.set_info_callback(info_callback)
        item.annotationparam.title = self.single_roi.title if title is None else title
        item.annotationparam.update_item(item)
        item.set_style("plot", "shape/drag")
        return item

    @classmethod
    def from_plot_item(cls, item: AnnotatedCircle) -> CircularROI:
        """Create ROI from plot item

        Args:
            item: plot item
        """
        rect = item.get_rect()
        return CircularROI(
            CircularROI.rect_to_coords(*rect), False, item.annotationparam.title
        )


class ImageROIPlotPyAdapter(BaseROIPlotPyAdapter[ImageROI]):
    """Image ROI plot item adapter class

    Args:
        roi: ROI object
    """

    def to_plot_item(
        self,
        single_roi: PolygonalROI | RectangularROI | CircularROI,
        obj: ImageObj,
        title: str | None = None,
    ) -> AnnotatedCircle | AnnotatedRectangle | AnnotatedPolygon:
        """Make ROI plot item from single ROI

        Args:
            single_roi: single ROI object
            obj: object (signal/image), for physical-indices coordinates conversion
            title: ROI title

        Returns:
            Plot item
        """
        # pylint: disable=import-outside-toplevel
        from cdl.adapters_plotpy.factories import create_adapter_from_object

        return create_adapter_from_object(single_roi).to_plot_item(obj, title)


class ImageObjPlotPyAdapter(BaseObjPlotPyAdapter[ImageObj, MaskedImageItem]):
    """Image object plot item adapter class"""

    CONF_FMT = Conf.view.ima_format
    DEFAULT_FMT = ".1f"

    def update_plot_item_parameters(self, item: MaskedImageItem) -> None:
        """Update plot item parameters from object data/metadata

        Takes into account a subset of plot item parameters. Those parameters may
        have been overriden by object metadata entries or other object data. The goal
        is to update the plot item accordingly.

        This is *almost* the inverse operation of `update_metadata_from_plot_item`.

        Args:
            item: plot item
        """
        o = self.obj
        for axis in ("x", "y", "z"):
            unit = getattr(o, axis + "unit")
            fmt = r"%.1f"
            if unit:
                fmt = r"%.1f (" + unit + ")"
            setattr(item.param, axis + "format", fmt)
        # Updating origin and pixel spacing
        has_origin = o.x0 is not None and o.y0 is not None
        has_pixelspacing = o.dx is not None and o.dy is not None
        if has_origin or has_pixelspacing:
            x0, y0, dx, dy = 0.0, 0.0, 1.0, 1.0
            if has_origin:
                x0, y0 = o.x0, o.y0
            if has_pixelspacing:
                dx, dy = o.dx, o.dy
            shape = o.data.shape
            item.param.xmin, item.param.xmax = x0, x0 + dx * shape[1]
            item.param.ymin, item.param.ymax = y0, y0 + dy * shape[0]
        zmin, zmax = item.get_lut_range()
        if o.zscalemin is not None or o.zscalemax is not None:
            zmin = zmin if o.zscalemin is None else o.zscalemin
            zmax = zmax if o.zscalemax is None else o.zscalemax
            item.set_lut_range([zmin, zmax])
        super().update_plot_item_parameters(item)

    def update_metadata_from_plot_item(self, item: MaskedImageItem) -> None:
        """Update metadata from plot item.

        Takes into account a subset of plot item parameters. Those parameters may
        have been modified by the user through the plot item GUI. The goal is to
        update the metadata accordingly.

        This is *almost* the inverse operation of `update_plot_item_parameters`.

        Args:
            item: plot item
        """
        super().update_metadata_from_plot_item(item)
        o = self.obj
        # Updating the LUT range:
        o.zscalemin, o.zscalemax = item.get_lut_range()
        # Updating origin and pixel spacing:
        shape = o.data.shape
        param = item.param
        xmin, xmax, ymin, ymax = param.xmin, param.xmax, param.ymin, param.ymax
        if xmin == 0 and ymin == 0 and xmax == shape[1] and ymax == shape[0]:
            o.x0, o.y0, o.dx, o.dy = 0.0, 0.0, 1.0, 1.0
        else:
            o.x0, o.y0 = xmin, ymin
            o.dx, o.dy = (xmax - xmin) / shape[1], (ymax - ymin) / shape[0]

    def __viewable_data(self) -> np.ndarray:
        """Return viewable data"""
        data = self.obj.data.real
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data, posinf=0, neginf=0)
        return data

    def make_item(self, update_from: MaskedImageItem | None = None) -> MaskedImageItem:
        """Make plot item from data.

        Args:
            update_from: update from plot item

        Returns:
            Plot item
        """
        data = self.__viewable_data()
        item = make.maskedimage(
            data,
            self.obj.maskdata,
            title=self.obj.title,
            colormap="viridis",
            eliminate_outliers=Conf.view.ima_eliminate_outliers.get(),
            interpolation="nearest",
            show_mask=True,
        )
        if update_from is None:
            self.update_plot_item_parameters(item)
        else:
            update_dataset(item.param, update_from.param)
            item.param.update_item(item)
        return item

    def update_item(self, item: MaskedImageItem, data_changed: bool = True) -> None:
        """Update plot item from data.

        Args:
            item: plot item
            data_changed: if True, data has changed
        """
        if data_changed:
            item.set_data(self.__viewable_data(), lut_range=[item.min, item.max])
        item.set_mask(self.obj.maskdata)
        item.param.label = self.obj.title
        self.update_plot_item_parameters(item)
        item.plot().update_colormap_axis(item)

    def add_label_with_title(self, title: str | None = None) -> None:
        """Add label with title annotation

        Args:
            title: title (if None, use image title)
        """
        title = self.obj.title if title is None else title
        if title:
            label = make.label(title, (self.obj.x0, self.obj.y0), (10, 10), "TL")
            self.add_annotations_from_items([label])
