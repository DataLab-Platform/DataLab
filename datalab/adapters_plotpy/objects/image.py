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
from plotpy.items import MaskedXYImageItem
from sigima.objects import ImageObj

from datalab.adapters_plotpy.objects.base import (
    BaseObjPlotPyAdapter,
)
from datalab.config import Conf


def get_obj_coords(obj: ImageObj) -> tuple[np.ndarray, np.ndarray]:
    """Get object coordinates

    Args:
        obj: image object

    Returns:
        x and y coordinates
    """
    if obj.is_uniform_coords:
        shape = obj.data.shape
        xcoords = np.linspace(obj.x0, obj.x0 + obj.dx * shape[1], shape[1] + 1)
        ycoords = np.linspace(obj.y0, obj.y0 + obj.dy * shape[0], shape[0] + 1)
    else:
        xcoords, ycoords = obj.xcoords, obj.ycoords
    return xcoords, ycoords


class ImageObjPlotPyAdapter(BaseObjPlotPyAdapter[ImageObj, MaskedXYImageItem]):
    """Image object plot item adapter class"""

    CONF_FMT = Conf.view.ima_format
    DEFAULT_FMT = ".1f"

    def update_plot_item_parameters(self, item: MaskedXYImageItem) -> None:
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
        item.set_xy(*get_obj_coords(o))
        zmin, zmax = item.get_lut_range()
        if o.zscalemin is not None or o.zscalemax is not None:
            zmin = zmin if o.zscalemin is None else o.zscalemin
            zmax = zmax if o.zscalemax is None else o.zscalemax
            item.set_lut_range([zmin, zmax])
        super().update_plot_item_parameters(item)

    def update_metadata_from_plot_item(self, item: MaskedXYImageItem) -> None:
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

    def __viewable_data(self) -> np.ndarray:
        """Return viewable data"""
        data = self.obj.data.real
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data, posinf=0, neginf=0)
        return data

    def make_item(
        self, update_from: MaskedXYImageItem | None = None
    ) -> MaskedXYImageItem:
        """Make plot item from data.

        Args:
            update_from: update from plot item

        Returns:
            Plot item
        """
        data = self.__viewable_data()
        item = make.maskedxyimage(
            *get_obj_coords(self.obj),
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

    def update_item(self, item: MaskedXYImageItem, data_changed: bool = True) -> None:
        """Update plot item from data.

        Args:
            item: plot item
            data_changed: if True, data has changed
        """
        if data_changed:
            # When data changes, let set_data() auto-calculate the LUT range from the
            # new data (by not passing lut_range parameter). The subsequent call to
            # update_plot_item_parameters() will override it if zscalemin/zscalemax
            # are explicitly set in the object's metadata.
            item.set_data(self.__viewable_data())
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
