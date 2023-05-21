# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Plot item list classes
------------------------------

These classes handle guiqwt plot items for signal and image panels.

.. autosummary::

    SignalPlotHandler
    ImagePlotHandler

.. autoclass:: SignalPlotHandler
    :members:
    :inherited-members:

.. autoclass:: ImagePlotHandler
    :members:
    :inherited-members:
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import numpy as np
from guiqwt.curve import GridItem
from guiqwt.label import LegendBoxItem

from cdl.config import Conf, _
from cdl.utils.qthelpers import create_progress_bar

if TYPE_CHECKING:  # pragma: no cover
    from guiqwt.curve import CurveItem
    from guiqwt.image import MaskedImageItem
    from guiqwt.plot import CurveWidget, ImagePlot, ImageWidget

    from cdl.core.gui.panel.base import BaseDataPanel
    from cdl.core.model.image import ImageParam
    from cdl.core.model.signal import SignalParam


def calc_data_hash(obj: SignalParam | ImageParam) -> str:
    """Calculate a hash for a SignalParam | ImageParam object's data"""
    return hashlib.sha1(np.ascontiguousarray(obj.data)).hexdigest()


class BasePlotHandler:
    """Object handling plot items associated to objects (signals/images)"""

    def __init__(
        self,
        panel: BaseDataPanel,
        plotwidget: CurveWidget | ImageWidget,
    ) -> None:
        self.panel = panel
        self.plotwidget = plotwidget
        self.plot = plotwidget.get_plot()

        # Plot items: key = object uuid, value = plot item
        self.__plotitems: dict[str, CurveItem | MaskedImageItem] = {}

        self.__shapeitems = []
        self.__cached_hashes: WeakKeyDictionary[
            SignalParam | ImageParam, list[int]
        ] = WeakKeyDictionary()

    def __len__(self) -> int:
        """Return number of items"""
        return len(self.__plotitems)

    def __getitem__(self, oid: str) -> CurveItem | MaskedImageItem:
        """Return item associated to object uuid"""
        return self.__plotitems[oid]

    def get(
        self, key: str, default: CurveItem | MaskedImageItem | None = None
    ) -> CurveItem | MaskedImageItem | None:
        """Return item associated to object uuid.
        If the key is not found, default is returned if given,
        otherwise None is returned."""
        return self.__plotitems.get(key, default)

    def __setitem__(self, oid: str, item: CurveItem | MaskedImageItem) -> None:
        """Set item associated to object uuid"""
        self.__plotitems[oid] = item

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over plothandler values (plot items)"""
        return iter(self.__plotitems.values())

    def remove_item(self, oid: str) -> None:
        """Remove plot item associated to object uuid"""
        item = self.__plotitems.pop(oid)
        self.plot.del_item(item)

    def clear(self) -> None:
        """Clear plot items"""
        self.__plotitems = {}
        self.cleanup_dataview()

    def add_shapes(self, oid: str) -> None:
        """Add geometric shape items associated to computed results and annotations,
        for the object with the given uuid"""
        obj = self.panel.objmodel[oid]
        if obj.metadata:
            # Performance optimization: block `guiqwt.baseplot.BasePlot` signals,
            # add all items except the last one, unblock signals, then add the last one
            # (this avoids some unnecessary refresh process by guiqwt)
            items = list(obj.iterate_shape_items(editable=False))
            if items:
                block = self.plot.blockSignals(True)
                for item in items[:-1]:
                    self.plot.add_item(item)
                    self.__shapeitems.append(item)
                self.plot.blockSignals(block)
                self.plot.add_item(items[-1])
                self.__shapeitems.append(items[-1])

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
            just_show (bool, optional): if True, only show the item (do not update it,
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

    def refresh_plot(
        self, only_oid: str | None = None, just_show: bool = False
    ) -> None:
        """Refresh plot.

        Args:
            only_oid (str, optional): if not None, only refresh the item associated
                to this object uuid. Defaults to None.
            just_show (bool, optional): if True, only show the item (do not update it,
                except regarding the reference item). Defaults to False.
        """
        if only_oid is None:
            oids = self.panel.objview.get_sel_object_uuids(include_groups=True)
            if len(oids) == 1:
                self.cleanup_dataview()
            self.remove_all_shape_items()
            for item in self:
                if item is not None:
                    item.hide()
        else:
            oids = [only_oid]
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
                            oid, ref_item=ref_item, just_show=just_show
                        )
                        if ref_item is None:
                            ref_item = item
                    self.plot.set_item_visible(item, True, replot=False)
                    self.plot.set_active_item(item)
                    item.unselect()
                    self.add_shapes(oid)
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

    def get_current_plot_options(self) -> dict:
        """
        Return standard signal/image plot options

        :return: Dictionary containing plot arguments for CurveDialog/ImageDialog
        """
        return dict(
            xlabel=self.plot.get_axis_title("bottom"),
            ylabel=self.plot.get_axis_title("left"),
            xunit=self.plot.get_axis_unit("bottom"),
            yunit=self.plot.get_axis_unit("left"),
        )


class SignalPlotHandler(BasePlotHandler):
    """Object handling signal plot items, plot dialogs, plot options"""

    # Nothing specific to signals, as of today


class ImagePlotHandler(BasePlotHandler):
    """Object handling image plot items, plot dialogs, plot options"""

    @staticmethod
    def update_item_according_to_ref_item(
        item: MaskedImageItem, ref_item: MaskedImageItem
    ) -> None:
        """Update plot item according to reference item"""
        if ref_item is not None and Conf.view.ima_ref_lut_range.get(True):
            item.set_lut_range(ref_item.get_lut_range())
            plot: ImagePlot = item.plot()
            plot.update_colormap_axis(item)

    def refresh_plot(
        self, only_oid: str | None = None, just_show: bool = False
    ) -> None:
        """Refresh plot.

        Args:
            only_oid (str, optional): if not None, only refresh the item associated
                to this object uuid. Defaults to None.
            just_show (bool, optional): if True, only show the item (do not update it,
                except regarding the reference item). Defaults to False.
        """
        super().refresh_plot(only_oid=only_oid, just_show=just_show)
        self.plotwidget.contrast.setVisible(Conf.view.show_contrast.get(True))

    def cleanup_dataview(self) -> None:
        """Clean up data view"""
        for widget in (self.plotwidget.xcsw, self.plotwidget.ycsw):
            widget.hide()
        super().cleanup_dataview()

    def get_current_plot_options(self) -> dict:
        """
        Return standard signal/image plot options

        :return: Dictionary containing plot arguments for CurveDialog/ImageDialog
        """
        options = super().get_current_plot_options()
        options.update(
            dict(
                zlabel=self.plot.get_axis_title("right"),
                zunit=self.plot.get_axis_unit("right"),
                show_contrast=True,
            )
        )
        return options
