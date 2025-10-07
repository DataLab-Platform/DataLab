# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Plot handler
============

The :mod:`datalab.gui.plothandler` module provides plot handlers for signal
and image panels, that is, classes handling `PlotPy` plot items for representing
signals and images.

Signal plot handler
-------------------

.. autoclass:: SignalPlotHandler
    :members:
    :inherited-members:

Image plot handler
------------------

.. autoclass:: ImagePlotHandler
    :members:
    :inherited-members:
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Callable, Generic, TypeVar
from weakref import WeakKeyDictionary

import numpy as np
from plotpy.constants import PlotType
from plotpy.items import CurveItem, GridItem, LegendBoxItem, MaskedImageItem
from plotpy.plot import PlotOptions
from qtpy import QtWidgets as QW
from sigima.objects import ImageObj, SignalObj, TypeObj

from datalab.adapters_metadata import GeometryAdapter, TableAdapter
from datalab.adapters_plotpy import TypePlotItem, create_adapter_from_object
from datalab.config import Conf, _
from datalab.objectmodel import get_uuid
from datalab.utils.qthelpers import block_signals, create_progress_bar

if TYPE_CHECKING:
    from plotpy.items import LabelItem
    from plotpy.plot import BasePlot, PlotWidget

    from datalab.gui.panel.base import BaseDataPanel


def calc_data_hash(obj: SignalObj | ImageObj) -> str:
    """Calculate a hash for a SignalObj | ImageObj object's data"""
    return hashlib.sha1(np.ascontiguousarray(obj.data)).hexdigest()


TypePlotHandler = TypeVar("TypePlotHandler", bound="BasePlotHandler")


class BasePlotHandler(Generic[TypeObj, TypePlotItem]):  # type: ignore
    """Object handling plot items associated to objects (signals/images)"""

    PLOT_TYPE: PlotType | None = None  # Replaced in subclasses

    def __init__(
        self,
        panel: BaseDataPanel,
        plotwidget: PlotWidget,
    ) -> None:
        self.panel = panel
        self.plotwidget = plotwidget
        self.plot = plotwidget.get_plot()

        # Plot items: key = object uuid, value = plot item
        self.__plotitems: dict[str, TypePlotItem] = {}

        self.__shapeitems = []
        self.__cached_hashes: WeakKeyDictionary[TypeObj, list[int]] = (
            WeakKeyDictionary()
        )
        self.__auto_refresh = False
        self.__show_first_only = False
        self.__result_items_mapping: WeakKeyDictionary[LabelItem, Callable] = (
            WeakKeyDictionary()
        )

    def __len__(self) -> int:
        """Return number of items"""
        return len(self.__plotitems)

    def __getitem__(self, oid: str) -> TypePlotItem:
        """Return item associated to object uuid"""
        try:
            return self.__plotitems[oid]
        except KeyError as exc:
            # Item does not exist: this may happen when "auto refresh" is disabled
            # (object has been added to model but the corresponding plot item has not
            # been created yet)
            if not self.__auto_refresh:
                self.refresh_plot(oid, True, force=True, only_visible=False)
                return self.__plotitems[oid]
            # Item does not exist and auto refresh is enabled: this should not happen
            raise exc

    def get(self, key: str, default: TypePlotItem | None = None) -> TypePlotItem | None:
        """Return item associated to object uuid.
        If the key is not found, default is returned if given,
        otherwise None is returned."""
        return self.__plotitems.get(key, default)

    def get_obj_from_item(self, item: TypePlotItem) -> TypeObj | None:
        """Return object associated to plot item

        Args:
            item: plot item

        Returns:
            Object associated to plot item
        """
        for obj in self.panel.objmodel:
            if self.get(get_uuid(obj)) is item:
                return obj
        return None

    def __setitem__(self, oid: str, item: TypePlotItem) -> None:
        """Set item associated to object uuid"""
        self.__plotitems[oid] = item

    def __iter__(self) -> Iterator[TypePlotItem]:
        """Return an iterator over plothandler values (plot items)"""
        return iter(self.__plotitems.values())

    def remove_item(self, oid: str) -> None:
        """Remove plot item associated to object uuid"""
        try:
            item = self.__plotitems.pop(oid)
        except KeyError:
            # Item does not exist: this may happen when "auto refresh" is disabled
            # (object has been added to model but the corresponding plot item has not
            # been created yet).
            # This may also happen after opening a project, then immediately selecting
            # a group containing more than one object: plot item would have been created
            # only for the first object in group, and this exception would be raised
            # for the second one (which does not have a plot item yet).
            return
        self.plot.del_item(item)

    def clear(self) -> None:
        """Clear plot items"""
        self.__plotitems = {}
        self.cleanup_dataview()

    def add_shapes(self, oid: str, do_autoscale: bool = False) -> None:
        """Add geometric shape items associated to computed results and annotations,
        for the object with the given uuid"""
        obj = self.panel.objmodel[oid]
        if obj.metadata:
            obj_adapter = create_adapter_from_object(obj)
            items = list(obj_adapter.iterate_shape_items(editable=False))
            results = list(TableAdapter.iterate_from_obj(obj)) + list(
                GeometryAdapter.iterate_from_obj(obj)
            )
            for result in results:
                result_adapter = create_adapter_from_object(result)
                item = result_adapter.get_label_item(obj)
                if item is not None:
                    items.append(item)
                    self.__result_items_mapping[item] = (
                        lambda item,
                        rprop=result_adapter: rprop.update_obj_metadata_from_item(
                            obj, item
                        )
                    )
                items.extend(result_adapter.get_other_items(obj))
            if items:
                if do_autoscale:
                    self.plot.do_autoscale()
                # Performance optimization: block `plotpy.plot.BasePlot` signals, add
                # all items except the last one, unblock signals, then add the last one
                # (this avoids some unnecessary refresh process by PlotPy)
                with block_signals(self.plot):
                    with create_progress_bar(
                        self.panel, _("Creating geometric shapes"), max_=len(items) - 1
                    ) as progress:
                        for i_item, item in enumerate(items[:-1]):
                            progress.setValue(i_item + 1)
                            if progress.wasCanceled():
                                break
                            self.plot.add_item(item)
                            self.__shapeitems.append(item)
                            QW.QApplication.processEvents()
                self.plot.add_item(items[-1])
                self.__shapeitems.append(items[-1])

    def update_resultproperty_from_plot_item(self, item: LabelItem) -> None:
        """Update result property from plot item"""
        callback = self.__result_items_mapping.get(item)
        if callback is not None:
            callback(item)

    def remove_all_shape_items(self) -> None:
        """Remove all geometric shapes associated to result items"""
        if set(self.__shapeitems).issubset(set(self.plot.items)):
            self.plot.del_items(self.__shapeitems)
        self.__shapeitems = []

    def __add_item_to_plot(self, oid: str) -> TypePlotItem:
        """Make plot item and add it to plot.

        Args:
            oid: object uuid

        Returns:
            Plot item
        """
        obj = self.panel.objmodel[oid]
        self.__cached_hashes[obj] = calc_data_hash(obj)
        item: TypePlotItem = create_adapter_from_object(obj).make_item()
        item.set_readonly(True)
        self[oid] = item
        self.plot.add_item(item)
        return item

    def __update_item_on_plot(
        self, oid: str, ref_item: TypePlotItem, just_show: bool = False
    ) -> None:
        """Update plot item.

        Args:
            oid: object uuid
            ref_item: reference item
            just_show: if True, only show the item (do not update it, except regarding
             the reference item). Defaults to False.
        """
        if not just_show:
            obj = self.panel.objmodel[oid]
            cached_hash = self.__cached_hashes.get(obj)
            new_hash = calc_data_hash(obj)
            data_changed = cached_hash is None or cached_hash != new_hash
            self.__cached_hashes[obj] = new_hash
            adapter = create_adapter_from_object(obj)
            adapter.update_item(self[oid], data_changed=data_changed)
        self.update_item_according_to_ref_item(self[oid], ref_item)

    @staticmethod
    def update_item_according_to_ref_item(
        item: TypePlotItem, ref_item: TypePlotItem
    ) -> None:  # pylint: disable=unused-argument
        """Update plot item according to reference item"""
        #  For now, nothing to do here: it's only used for images (contrast)

    def set_auto_refresh(self, auto_refresh: bool) -> None:
        """Set auto refresh mode.

        Args:
            auto_refresh: if True, refresh plot items automatically
        """
        self.__auto_refresh = auto_refresh
        if auto_refresh:
            self.refresh_plot("selected")

    def set_show_first_only(self, show_first_only: bool) -> None:
        """Set show first only mode.

        Args:
            show_first_only: if True, show only the first selected item
        """
        self.__show_first_only = show_first_only
        if self.__auto_refresh:
            self.refresh_plot("selected")

    def get_existing_oids(self) -> list[str]:
        """Get existing object uuids.

        Returns:
            List of object uuids that have a plot item associated to them.
        """
        return list(self.__plotitems.keys())

    def reduce_shown_oids(self, oids: list[str]) -> list[str]:
        """Reduce the number of shown objects to visible items only. The base
        implementation is to show only the first selected item if the option
        "Show first only" is enabled.

        Args:
            oids: list of object uuids

        Returns:
            Reduced list of object uuids
        """
        if self.__show_first_only:
            return oids[:1]
        return oids

    def refresh_plot(
        self,
        what: str,
        update_items: bool = True,
        force: bool = False,
        only_visible: bool = True,
        only_existing: bool = False,
    ) -> None:
        """Refresh plot.

        Args:
            what: string describing the objects to refresh.
             Valid values are "selected" (refresh the selected objects),
             "all" (refresh all objects), "existing" (refresh existing plot items),
             or an object uuid.
            update_items: if True, update the items.
             If False, only show the items (do not update them, except if the
             option "Use reference item LUT range" is enabled and more than one
             item is selected). Defaults to True.
            force: if True, force refresh even if auto refresh is disabled.
             Defaults to False.
            only_visible: if True, only refresh visible items. Defaults to True.
             Visible items are the ones that are not hidden by other items or the items
             except the first one if the option "Show first only" is enabled.
             This is useful for images, where the last image is the one that is shown.
             If False, all items are refreshed.
            only_existing: if True, only refresh existing items. Defaults to False.
             Existing items are the ones that have already been created and are
             associated to the object uuid. If False, create new items for the
             objects that do not have an item yet.

        Raises:
            ValueError: if `what` is not a valid value
        """
        if not self.__auto_refresh and not force:
            return
        if what == "selected":
            # Refresh selected objects
            oids = self.panel.objview.get_sel_object_uuids(include_groups=True)
            if len(oids) == 1:
                self.cleanup_dataview()
            self.remove_all_shape_items()
            for item in self:
                if item is not None:
                    item.hide()
        elif what == "existing":
            # Refresh existing objects
            oids = self.get_existing_oids()
        elif what == "all":
            # Refresh all objects
            oids = self.panel.objmodel.get_object_ids()
        else:
            # Refresh a single object defined by its uuid
            oids = [what]
            try:
                # Check if this is a valid object uuid
                self.panel.objmodel.get_objects(oids)
            except KeyError as exc:
                raise ValueError(f"Invalid value for `what`: {what}") from exc

        # Initialize titles and scales dictionaries
        title_keys = ("title", "xlabel", "ylabel", "zlabel", "xunit", "yunit", "zunit")
        titles_dict = {}
        autoscale = False
        scale_keys = (
            "xscalelog",
            "xscalemin",
            "xscalemax",
            "yscalelog",
            "yscalemin",
            "yscalemax",
        )
        scales_dict = {}

        if oids:
            if what != "existing" and only_visible:
                # Remove hidden items from the list of objects to refresh
                oids = self.reduce_shown_oids(oids)
            ref_item = None
            with create_progress_bar(
                self.panel, _("Creating plot items"), max_=len(oids)
            ) as progress:
                # Iterate over objects
                for i_obj, oid in enumerate(oids):
                    progress.setValue(i_obj + 1)
                    if progress.wasCanceled():
                        break
                    obj = self.panel.objmodel[oid]

                    # Collecting titles information
                    for key in title_keys:
                        title = getattr(obj, key, "")
                        value = titles_dict.get(key)
                        if value is None:
                            titles_dict[key] = title
                        elif value != title:
                            titles_dict[key] = ""

                    # Collecting scales information
                    autoscale = autoscale or obj.autoscale
                    for key in scale_keys:
                        scale = getattr(obj, key, None)
                        if scale is not None:
                            cmp = min if "min" in key else max
                            scales_dict[key] = cmp(scales_dict.get(key, scale), scale)

                    # Update or add item to plot
                    item = self.get(oid)
                    if item is None:
                        if only_existing:
                            continue
                        item = self.__add_item_to_plot(oid)
                    else:
                        self.__update_item_on_plot(
                            oid, ref_item=ref_item, just_show=not update_items
                        )
                        if ref_item is None:
                            ref_item = item
                    if what != "existing" or item.isVisible():
                        self.plot.set_item_visible(item, True, replot=False)
                        self.plot.set_active_item(item)
                        item.unselect()

                    # Add geometric shapes
                    self.add_shapes(oid, do_autoscale=autoscale)

            self.plot.replot()

        else:
            # No object to refresh: clean up titles
            for key in title_keys:
                titles_dict[key] = ""

        # Set titles
        tdict = titles_dict
        tdict["ylabel"] = (tdict["ylabel"], tdict.pop("zlabel"))
        tdict["yunit"] = (tdict["yunit"], tdict.pop("zunit"))
        self.plot.set_titles(**titles_dict)

        # Set scales
        replot = False
        for axis_name, axis in (("bottom", "x"), ("left", "y")):
            axis_id = self.plot.get_axis_id(axis_name)
            scalelog = scales_dict.get(f"{axis}scalelog")
            if scalelog is not None:
                new_scale = "log" if scalelog else "lin"
                self.plot.set_axis_scale(axis_id, new_scale, autoscale=False)
                replot = True
        if autoscale:
            self.plot.do_autoscale()
        else:
            for axis_name, axis in (("bottom", "x"), ("left", "y")):
                axis_id = self.plot.get_axis_id(axis_name)
                new_vmin = scales_dict.get(f"{axis}scalemin")
                new_vmax = scales_dict.get(f"{axis}scalemax")
                if new_vmin is not None or new_vmax is not None:
                    self.plot.do_autoscale(replot=False, axis_id=axis_id)
                    vmin, vmax = self.plot.get_axis_limits(axis_id)
                    new_vmin = new_vmin if new_vmin is not None else vmin
                    new_vmax = new_vmax if new_vmax is not None else vmax
                    self.plot.set_axis_limits(axis_id, new_vmin, new_vmax)
                    replot = True
            if replot:
                self.plot.replot()

    def cleanup_dataview(self) -> None:
        """Clean up data view"""
        # Performance optimization: using `baseplot.BasePlot.del_items` instead of
        # `baseplot.BasePlot.del_item` (avoid emitting unnecessary signals)
        self.plot.del_items(
            [
                item
                for item in self.plot.items[:]
                if item not in self and not isinstance(item, (LegendBoxItem, GridItem))
            ]
        )

    def get_plot_options(self) -> PlotOptions:
        """Return standard signal/image plot options"""
        return PlotOptions(
            type=self.PLOT_TYPE,
            xlabel=self.plot.get_axis_title("bottom"),
            ylabel=self.plot.get_axis_title("left"),
            xunit=self.plot.get_axis_unit("bottom"),
            yunit=self.plot.get_axis_unit("left"),
            show_axes_tab=False,
        )


class SignalPlotHandler(BasePlotHandler[SignalObj, CurveItem]):
    """Object handling signal plot items, plot dialogs, plot options"""

    PLOT_TYPE = PlotType.CURVE

    def toggle_anti_aliasing(self, state: bool) -> None:
        """Toggle anti-aliasing

        Args:
            state: if True, enable anti-aliasing
        """
        self.plot.set_antialiasing(state)
        self.plot.replot()

    def get_plot_options(self) -> PlotOptions:
        """Return standard signal/image plot options"""
        options = super().get_plot_options()
        options.curve_antialiasing = self.plot.antialiased
        return options

    def refresh_plot(
        self,
        what: str,
        update_items: bool = True,
        force: bool = False,
        only_visible: bool = True,
        only_existing: bool = False,
    ) -> None:
        """Refresh plot and configure datetime axis if needed.

        This override adds automatic datetime axis configuration when at least one
        of the displayed signals has a datetime X-axis.

        Args:
            what: string describing the objects to refresh
            update_items: if True, update the items
            force: if True, force refresh even if auto refresh is disabled
            only_visible: if True, only refresh visible items
            only_existing: if True, only refresh existing items
        """
        # Call parent implementation
        super().refresh_plot(what, update_items, force, only_visible, only_existing)

        # Check if any visible signal has datetime X-axis
        has_datetime = False
        datetime_format = None
        for item in self:
            if item is not None and item.isVisible():
                obj = self.get_obj_from_item(item)
                if obj is not None and obj.is_x_datetime():
                    has_datetime = True
                    # Get format from signal metadata, or use configured default
                    if datetime_format is None:
                        datetime_format = obj.metadata.get("x_datetime_format")
                        if datetime_format is None:
                            # Use configured format based on time unit
                            unit = obj.xunit if obj.xunit else "s"
                            if unit in ("ns", "us", "ms"):
                                datetime_format = Conf.view.sig_datetime_format_ms.get(
                                    "%H:%M:%S.%f"
                                )
                            else:
                                datetime_format = Conf.view.sig_datetime_format_s.get(
                                    "%H:%M:%S"
                                )
                    break

        # Configure X-axis for datetime or restore default
        if has_datetime and datetime_format is not None:
            self.plot.set_axis_datetime("bottom", format=datetime_format)
        else:
            # Restore default scale draw (remove datetime formatting)
            from qwt import QwtScaleDraw

            self.plot.setAxisScaleDraw(self.plot.get_axis_id("bottom"), QwtScaleDraw())


class ImagePlotHandler(BasePlotHandler[ImageObj, MaskedImageItem]):
    """Object handling image plot items, plot dialogs, plot options"""

    PLOT_TYPE = PlotType.IMAGE

    @staticmethod
    def update_item_according_to_ref_item(
        item: MaskedImageItem, ref_item: MaskedImageItem
    ) -> None:
        """Update plot item according to reference item"""
        if ref_item is not None and Conf.view.ima_ref_lut_range.get():
            item.set_lut_range(ref_item.get_lut_range())
            plot: BasePlot = item.plot()
            plot.update_colormap_axis(item)

    def reduce_shown_oids(self, oids: list[str]) -> list[str]:
        """Reduce the number of shown objects to visible items only. The base
        implementation is to show only the first selected item if the option
        "Show first only" is enabled.

        Args:
            oids: list of object uuids

        Returns:
            Reduced list of object uuids
        """
        oids = super().reduce_shown_oids(oids)

        # For Image View, we show only the last image (which is the highest z-order
        # plot item) if more than one image is selected, if last image has no
        # transparency and if the other images are all completely covered by the last
        # image.
        # TODO: [P4] Enhance this algorithm to handle more complex cases
        # (not sure it's worth it)
        if len(oids) > 1:
            # Get objects associated to the oids
            objs = self.panel.objmodel.get_objects(oids)
            # First condition is about the image transparency
            last_obj = objs[-1]
            alpha_cond = (
                last_obj.get_metadata_option("alpha", 1.0) == 1.0
                and last_obj.get_metadata_option("alpha_function", 0) == 0
            )
            if alpha_cond:
                # Second condition is about the image size and position
                geom_cond = True
                for obj in objs[:-1]:
                    geom_cond = (
                        geom_cond
                        and last_obj.x0 <= obj.x0
                        and last_obj.y0 <= obj.y0
                        and last_obj.x0 + last_obj.width >= obj.x0 + obj.width
                        and last_obj.y0 + last_obj.height >= obj.y0 + obj.height
                    )
                    if not geom_cond:
                        break
                if geom_cond:
                    oids = oids[-1:]
        return oids

    def refresh_plot(
        self,
        what: str,
        update_items: bool = True,
        force: bool = False,
        only_visible: bool = True,
        only_existing: bool = False,
    ) -> None:
        """Refresh plot.

        Args:
            what: string describing the objects to refresh.
             Valid values are "selected" (refresh the selected objects),
             "all" (refresh all objects), "existing" (refresh existing plot items),
             or an object uuid.
            update_items: if True, update the items.
             If False, only show the items (do not update them, except if the
             option "Use reference item LUT range" is enabled and more than one
             item is selected). Defaults to True.
            force: if True, force refresh even if auto refresh is disabled.
             Defaults to False.
            only_visible: if True, only refresh visible items. Defaults to True.
             Visible items are the ones that are not hidden by other items or the items
             except the first one if the option "Show first only" is enabled.
             This is useful for images, where the last image is the one that is shown.
             If False, all items are refreshed.
            only_existing: if True, only refresh existing items. Defaults to False.
             Existing items are the ones that have already been created and are
             associated to the object uuid. If False, create new items for the
             objects that do not have an item yet.

        Raises:
            ValueError: if `what` is not a valid value
        """
        super().refresh_plot(
            what=what,
            update_items=update_items,
            force=force,
            only_visible=only_visible,
            only_existing=only_existing,
        )
        self.plotwidget.contrast.setVisible(Conf.view.show_contrast.get(True))
        plot = self.plotwidget.get_plot()
        new_aspect_ratio = current_aspect_ratio = plot.get_aspect_ratio()
        if Conf.view.ima_aspect_ratio_1_1.get():
            # Lock aspect ratio to 1:1
            new_aspect_ratio = 1.0
        else:
            # Use physical pixel size to set aspect ratio
            for oid in reversed(self.reduce_shown_oids(self.get_existing_oids())):
                if self.get(oid).isVisible():
                    obj: ImageObj = self.panel.objmodel[oid]
                    new_aspect_ratio = obj.dx / obj.dy
                    break
        if new_aspect_ratio != current_aspect_ratio:
            # Update aspect ratio only if it has changed
            plot.set_aspect_ratio(new_aspect_ratio)
            plot.do_autoscale()

    def cleanup_dataview(self) -> None:
        """Clean up data view"""
        for widget in (self.plotwidget.xcsw, self.plotwidget.ycsw):
            widget.hide()
        super().cleanup_dataview()

    def get_plot_options(self) -> PlotOptions:
        """Return standard signal/image plot options"""
        options = super().get_plot_options()
        options.zlabel = self.plot.get_axis_title("right")
        options.zunit = self.plot.get_axis_unit("right")
        return options
