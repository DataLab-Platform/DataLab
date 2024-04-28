# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Plot handler
============

The :mod:`cdl.core.gui.plothandler` module provides plot handlers for signal
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
from typing import TYPE_CHECKING, Callable
from weakref import WeakKeyDictionary

import numpy as np
from plotpy.constants import PlotType
from plotpy.items import GridItem, LegendBoxItem
from plotpy.plot import PlotOptions
from qtpy import QtWidgets as QW

from cdl.config import Conf, _
from cdl.utils.qthelpers import block_signals, create_progress_bar

if TYPE_CHECKING:
    from plotpy.items import CurveItem, LabelItem, MaskedImageItem
    from plotpy.plot import BasePlot, PlotWidget

    from cdl.core.gui.panel.base import BaseDataPanel
    from cdl.core.model.image import ImageObj
    from cdl.core.model.signal import SignalObj


def calc_data_hash(obj: SignalObj | ImageObj) -> str:
    """Calculate a hash for a SignalObj | ImageObj object's data"""
    return hashlib.sha1(np.ascontiguousarray(obj.data)).hexdigest()


class BasePlotHandler:
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
        self.__plotitems: dict[str, CurveItem | MaskedImageItem] = {}

        self.__shapeitems = []
        self.__cached_hashes: WeakKeyDictionary[SignalObj | ImageObj, list[int]] = (
            WeakKeyDictionary()
        )
        self.__auto_refresh = False
        self.__result_items_mapping: WeakKeyDictionary[LabelItem, Callable] = (
            WeakKeyDictionary()
        )

    def __len__(self) -> int:
        """Return number of items"""
        return len(self.__plotitems)

    def __getitem__(self, oid: str) -> CurveItem | MaskedImageItem:
        """Return item associated to object uuid"""
        try:
            return self.__plotitems[oid]
        except KeyError as exc:
            # Item does not exist: this may happen when "auto refresh" is disabled
            # (object has been added to model but the corresponding plot item has not
            # been created yet)
            if not self.__auto_refresh:
                self.refresh_plot("selected", True, force=True)
                return self.__plotitems[oid]
            # Item does not exist and auto refresh is enabled: this should not happen
            raise exc

    def get(
        self, key: str, default: CurveItem | MaskedImageItem | None = None
    ) -> CurveItem | MaskedImageItem | None:
        """Return item associated to object uuid.
        If the key is not found, default is returned if given,
        otherwise None is returned."""
        return self.__plotitems.get(key, default)

    def get_obj_from_item(
        self, item: CurveItem | MaskedImageItem
    ) -> SignalObj | ImageObj | None:
        """Return object associated to plot item

        Args:
            item: plot item

        Returns:
            Object associated to plot item
        """
        for obj in self.panel.objmodel:
            if self.get(obj.uuid) is item:
                return obj
        return None

    def __setitem__(self, oid: str, item: CurveItem | MaskedImageItem) -> None:
        """Set item associated to object uuid"""
        self.__plotitems[oid] = item

    def __iter__(self) -> Iterator[CurveItem | MaskedImageItem]:
        """Return an iterator over plothandler values (plot items)"""
        return iter(self.__plotitems.values())

    def remove_item(self, oid: str) -> None:
        """Remove plot item associated to object uuid"""
        try:
            item = self.__plotitems.pop(oid)
        except KeyError as exc:
            # Item does not exist: this may happen when "auto refresh" is disabled
            # (object has been added to model but the corresponding plot item has not
            # been created yet)
            if not self.__auto_refresh:
                return
            # Item does not exist and auto refresh is enabled: this should not happen
            raise exc
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
            # Performance optimization: block `plotpy.plot.BasePlot` signals,
            # add all items except the last one, unblock signals, then add the last one
            # (this avoids some unnecessary refresh process by PlotPy)
            items = list(obj.iterate_shape_items(editable=False))
            resultproperties = list(obj.iterate_resultproperties())
            if resultproperties:
                for resultprop in resultproperties:
                    item = resultprop.get_plot_item()
                    items.append(item)
                    self.__result_items_mapping[item] = (
                        lambda item: resultprop.update_obj_metadata(obj, item)
                    )
            if items:
                if do_autoscale:
                    self.plot.do_autoscale()
                with block_signals(self.plot, True):
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

    def __add_item_to_plot(self, oid: str) -> CurveItem | MaskedImageItem:
        """Make plot item and add it to plot.

        Args:
            oid (str): object uuid

        Returns:
            CurveItem | MaskedImageItem: plot item
        """
        obj = self.panel.objmodel[oid]
        self.__cached_hashes[obj] = calc_data_hash(obj)
        item: CurveItem | MaskedImageItem = obj.make_item()
        item.set_readonly(True)
        self[oid] = item
        self.plot.add_item(item)
        return item

    def __update_item_on_plot(
        self, oid: str, ref_item: CurveItem | MaskedImageItem, just_show: bool = False
    ) -> None:
        """Update plot item.

        Args:
            oid (str): object uuid
            ref_item (CurveItem | MaskedImageItem): reference item
            just_show (bool | None): if True, only show the item (do not update it,
                except regarding the reference item). Defaults to False.
        """
        if not just_show:
            obj = self.panel.objmodel[oid]
            cached_hash = self.__cached_hashes.get(obj)
            new_hash = calc_data_hash(obj)
            data_changed = cached_hash is None or cached_hash != new_hash
            self.__cached_hashes[obj] = new_hash
            obj.update_item(self[oid], data_changed=data_changed)
        self.update_item_according_to_ref_item(self[oid], ref_item)

    @staticmethod
    def update_item_according_to_ref_item(
        item: MaskedImageItem, ref_item: MaskedImageItem
    ) -> None:  # pylint: disable=unused-argument
        """Update plot item according to reference item"""
        #  For now, nothing to do here: it's only used for images (contrast)

    def set_auto_refresh(self, auto_refresh: bool) -> None:
        """Set auto refresh mode.

        Args:
            auto_refresh (bool): if True, refresh plot items automatically
        """
        self.__auto_refresh = auto_refresh
        if auto_refresh:
            self.refresh_plot("selected")

    def refresh_plot(
        self, what: str, update_items: bool = True, force: bool = False
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
            oids = self.__plotitems.keys()
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
        title_keys = ("title", "xlabel", "ylabel", "zlabel", "xunit", "yunit", "zunit")
        titles_dict = {}
        if oids:
            ref_item = None
            with create_progress_bar(
                self.panel, _("Creating plot items"), max_=len(oids)
            ) as progress:
                for i_obj, oid in enumerate(oids):
                    progress.setValue(i_obj + 1)
                    if progress.wasCanceled():
                        break
                    obj = self.panel.objmodel[oid]
                    for key in title_keys:
                        title = getattr(obj, key, "")
                        value = titles_dict.get(key)
                        if value is None:
                            titles_dict[key] = title
                        elif value != title:
                            titles_dict[key] = ""
                    item = self.get(oid)
                    if item is None:
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
                    self.add_shapes(oid, do_autoscale=True)
            self.plot.replot()
        else:
            for key in title_keys:
                titles_dict[key] = ""
        tdict = titles_dict
        tdict["ylabel"] = (tdict["ylabel"], tdict.pop("zlabel"))
        tdict["yunit"] = (tdict["yunit"], tdict.pop("zunit"))
        self.plot.set_titles(**titles_dict)
        self.plot.do_autoscale()

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

    def get_current_plot_options(self) -> PlotOptions:
        """Return standard signal/image plot options"""
        return PlotOptions(
            type=self.PLOT_TYPE,
            xlabel=self.plot.get_axis_title("bottom"),
            ylabel=self.plot.get_axis_title("left"),
            xunit=self.plot.get_axis_unit("bottom"),
            yunit=self.plot.get_axis_unit("left"),
        )


class SignalPlotHandler(BasePlotHandler):
    """Object handling signal plot items, plot dialogs, plot options"""

    PLOT_TYPE = PlotType.CURVE

    def toggle_anti_aliasing(self, state: bool) -> None:
        """Toggle anti-aliasing

        Args:
            state: if True, enable anti-aliasing
        """
        self.plot.set_antialiasing(state)
        self.plot.replot()


class ImagePlotHandler(BasePlotHandler):
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

    def refresh_plot(
        self, what: str, update_items: bool = True, force: bool = False
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

        Raises:
            ValueError: if `what` is not a valid value
        """
        super().refresh_plot(what=what, update_items=update_items, force=force)
        self.plotwidget.contrast.setVisible(Conf.view.show_contrast.get(True))

    def cleanup_dataview(self) -> None:
        """Clean up data view"""
        for widget in (self.plotwidget.xcsw, self.plotwidget.ycsw):
            widget.hide()
        super().cleanup_dataview()

    def get_current_plot_options(self) -> PlotOptions:
        """Return standard signal/image plot options"""
        options = super().get_current_plot_options()
        options.zlabel = self.plot.get_axis_title("right")
        options.zunit = self.plot.get_axis_unit("right")
        return options
