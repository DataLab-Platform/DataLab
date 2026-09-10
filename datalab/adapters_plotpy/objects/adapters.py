# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
PlotPy Object Adapters
----------------------

DataLab's PlotPy adapters for signal and image objects. They reuse the generic
SigimaX object adapters and add DataLab geometry-result rendering, which depends
on :mod:`datalab.adapters_metadata` (intentionally not part of SigimaX).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from plotpy.items import CurveItem, MaskedXYImageItem
from sigimax.adapters_plotpy.objects.base import BaseObjPlotPyAdapter, TypePlotItem
from sigimax.adapters_plotpy.objects.image import (
    ImageObjPlotPyAdapter as SGMXImageObjPlotPyAdapter,
)
from sigimax.adapters_plotpy.objects.image import get_obj_coords
from sigimax.adapters_plotpy.objects.signal import (
    CURVESTYLES,
    CurveStyles,
    apply_downsampling,
    apply_line_width,
)
from sigimax.adapters_plotpy.objects.signal import (
    SignalObjPlotPyAdapter as SGMXSignalObjPlotPyAdapter,
)

from datalab.objectmodel import shorten_uuids_in_title

__all__ = [
    "CURVESTYLES",
    "BaseObjPlotPyAdapter",
    "CurveStyles",
    "ImageObjPlotPyAdapter",
    "SignalObjPlotPyAdapter",
    "TypePlotItem",
    "apply_downsampling",
    "apply_line_width",
    "get_obj_coords",
]


def _iterate_geometry_result_items(
    obj: Any, key: str, value: Any, fmt: str, lbl: bool
) -> Iterator:
    """Yield plot items for a DataLab geometry-result metadata entry.

    DataLab stores geometry results (segments, circles, ...) as object metadata.
    This renders them as plot shape items via
    :class:`~datalab.adapters_plotpy.objects.scalar.GeometryPlotPyAdapter`.
    Nothing is yielded when the entry is not a geometry result.

    Args:
        obj: the signal/image object owning the metadata
        key: metadata key
        value: metadata value
        fmt: numeric format string
        lbl: whether to show labels

    Yields:
        Plot items for the geometry result
    """
    # pylint: disable=import-outside-toplevel
    from datalab.adapters_metadata import GeometryAdapter
    from datalab.adapters_plotpy.objects.scalar import GeometryPlotPyAdapter

    if GeometryAdapter.match(key, value):
        try:
            geomadapter = GeometryAdapter.from_metadata_entry(obj, key)
            yield from GeometryPlotPyAdapter(geomadapter).iterate_shape_items(
                fmt, lbl, obj.PREFIX
            )
        except (ValueError, TypeError):
            # Skip invalid entries
            pass


class SignalObjPlotPyAdapter(SGMXSignalObjPlotPyAdapter):
    """DataLab signal object plot item adapter.

    Extends the SigimaX signal adapter with DataLab geometry-result rendering.
    """

    def make_item(self, update_from: CurveItem | None = None) -> CurveItem:
        """Make a curve item with a display-safe title."""
        item = super().make_item(update_from)
        item.param.label = shorten_uuids_in_title(self.obj.title)
        item.param.update_item(item)
        return item

    def update_item(self, item: CurveItem, data_changed: bool = True) -> None:
        """Update a curve item and its displayed title."""
        super().update_item(item, data_changed)
        item.param.label = shorten_uuids_in_title(self.obj.title)
        item.param.update_item(item)

    def add_label_with_title(self, title: str | None = None) -> None:
        """Add a title annotation with shortened UUID references."""
        title = self.obj.title if title is None else title
        super().add_label_with_title(shorten_uuids_in_title(title))

    def iterate_metadata_shape_items(
        self, key: str, value: Any, fmt: str, lbl: bool
    ) -> Iterator:
        """Render DataLab geometry results attached to the signal metadata."""
        yield from _iterate_geometry_result_items(self.obj, key, value, fmt, lbl)


class ImageObjPlotPyAdapter(SGMXImageObjPlotPyAdapter):
    """DataLab image object plot item adapter.

    Extends the SigimaX image adapter with DataLab geometry-result rendering.
    """

    def make_item(
        self, update_from: MaskedXYImageItem | None = None
    ) -> MaskedXYImageItem:
        """Make an image item with a display-safe title."""
        item = super().make_item(update_from)
        item.param.label = shorten_uuids_in_title(self.obj.title)
        item.param.update_item(item)
        return item

    def update_item(self, item: MaskedXYImageItem, data_changed: bool = True) -> None:
        """Update an image item and its displayed title."""
        super().update_item(item, data_changed)
        item.param.label = shorten_uuids_in_title(self.obj.title)
        item.param.update_item(item)

    def add_label_with_title(self, title: str | None = None) -> None:
        """Add a title annotation with shortened UUID references."""
        title = self.obj.title if title is None else title
        super().add_label_with_title(shorten_uuids_in_title(title))

    def iterate_metadata_shape_items(
        self, key: str, value: Any, fmt: str, lbl: bool
    ) -> Iterator:
        """Render DataLab geometry results attached to the image metadata."""
        yield from _iterate_geometry_result_items(self.obj, key, value, fmt, lbl)
