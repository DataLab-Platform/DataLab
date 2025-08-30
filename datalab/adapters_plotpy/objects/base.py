# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
PlotPy Adapter Base Object Module
---------------------------------
"""

from __future__ import annotations

import abc
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
)

from guidata.dataset import update_dataset
from plotpy.items import (
    AnnotatedShape,
)
from sigima.objects.base import (
    ROI_KEY,
    TypeObj,
)

from datalab.adapters_plotpy.base import (
    config_annotated_shape,
    items_to_json,
    json_to_items,
    set_plot_item_editable,
)
from datalab.config import Conf

if TYPE_CHECKING:
    from plotpy.items import (
        CurveItem,
        MaskedImageItem,
    )

TypePlotItem = TypeVar("TypePlotItem", bound="CurveItem | MaskedImageItem")


class BaseObjPlotPyAdapter(Generic[TypeObj, TypePlotItem]):
    """Object (signal/image) plot item adapter class"""

    DEFAULT_FMT = "s"  # This is overriden in children classes
    CONF_FMT = Conf.view.sig_format  # This is overriden in children classes

    def __init__(self, obj: TypeObj) -> None:
        """Initialize the adapter with the object.

        Args:
            obj: object (signal/image)
        """
        self.obj = obj
        self.__default_options = {
            "format": "%" + self.CONF_FMT.get(self.DEFAULT_FMT),
            "showlabel": Conf.view.show_label.get(False),
        }

    def get_obj_option(self, name: str) -> Any:
        """Get object option value.
        Args:
            name: option name

        Returns:
            Option value
        """
        default = self.__default_options[name]
        return self.obj.get_metadata_option(name, default)

    @abc.abstractmethod
    def make_item(self, update_from: TypePlotItem | None = None) -> TypePlotItem:
        """Make plot item from data.

        Args:
            update_from: update

        Returns:
            Plot item
        """

    @abc.abstractmethod
    def update_item(self, item: TypePlotItem, data_changed: bool = True) -> None:
        """Update plot item from data.

        Args:
            item: plot item
            data_changed: if True, data has changed
        """

    def add_annotations_from_items(self, items: list) -> None:
        """Add object annotations (annotation plot items).

        Args:
            items: annotation plot items
        """
        ann_items = json_to_items(self.obj.annotations)
        ann_items.extend(items)
        if ann_items:
            self.obj.annotations = items_to_json(ann_items)

    @abc.abstractmethod
    def add_label_with_title(self, title: str | None = None) -> None:
        """Add label with title annotation

        Args:
            title: title (if None, use object title)
        """

    def iterate_shape_items(self, editable: bool = False):
        """Iterate over shape items encoded in metadata (if any).

        Args:
            editable: if True, annotations are editable

        Yields:
            Plot item
        """
        fmt = self.get_obj_option("format")
        lbl = self.get_obj_option("showlabel")
        for key, _value in self.obj.metadata.items():
            if key == ROI_KEY:
                roi = self.obj.roi
                if roi is not None:
                    # Delayed import to avoid circular dependency
                    # pylint: disable=import-outside-toplevel
                    from datalab.adapters_plotpy.roi.factory import create_roi_adapter

                    adapter = create_roi_adapter(roi)
                    yield from adapter.iterate_roi_items(
                        self.obj, fmt=fmt, lbl=lbl, editable=False
                    )
            # Process geometry results from metadata (using GeometryAdapter)
            elif key.startswith("Geometry_") and key.endswith("_array"):
                # pylint: disable=import-outside-toplevel
                from datalab.adapters_metadata import GeometryAdapter
                from datalab.adapters_plotpy.objects.scalar import GeometryPlotPyAdapter

                try:
                    geomadapter = GeometryAdapter.from_metadata_entry(self.obj, key)
                    plot_adapter = GeometryPlotPyAdapter(geomadapter)
                    yield from plot_adapter.iterate_plot_items(
                        fmt, lbl, self.obj.PREFIX
                    )
                except (ValueError, TypeError):
                    # Skip invalid entries
                    pass
        if self.obj.annotations:
            try:
                for item in json_to_items(self.obj.annotations):
                    if isinstance(item, AnnotatedShape):
                        config_annotated_shape(item, fmt, lbl)
                    set_plot_item_editable(item, editable)
                    yield item
            except json.decoder.JSONDecodeError:
                pass

    def update_plot_item_parameters(self, item: TypePlotItem) -> None:
        """Update plot item parameters from object data/metadata

        Takes into account a subset of plot item parameters. Those parameters may
        have been overriden by object metadata entries or other object data. The goal
        is to update the plot item accordingly.

        This is *almost* the inverse operation of `update_metadata_from_plot_item`.

        Args:
            item: plot item
        """
        def_dict = Conf.view.get_def_dict(self.__class__.__name__[:3].lower())
        self.obj.set_metadata_options_defaults(def_dict, overwrite=False)

        # Subclasses have to override this method to update plot item parameters,
        # then call this implementation of the method to update plot item.
        update_dataset(item.param, self.obj.get_metadata_options())
        item.param.update_item(item)
        if item.selected:
            item.select()

    def update_metadata_from_plot_item(self, item: TypePlotItem) -> None:
        """Update metadata from plot item.

        Takes into account a subset of plot item parameters. Those parameters may
        have been modified by the user through the plot item GUI. The goal is to
        update the metadata accordingly.

        This is *almost* the inverse operation of `update_plot_item_parameters`.

        Args:
            item: plot item
        """
        def_dict = Conf.view.get_def_dict(self.__class__.__name__[:3].lower())
        for key in def_dict:
            if hasattr(item.param, key):  # In case the PlotPy version is not up-to-date
                self.obj.set_metadata_option(key, getattr(item.param, key))
