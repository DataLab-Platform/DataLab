# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
CobraDataLab Plot item list classes

These classes handle guiqwt plot items for signal and image panels.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations  # To be removed when dropping Python <=3.9 support

import hashlib
from typing import TYPE_CHECKING, List
from weakref import WeakKeyDictionary

import numpy as np
from guiqwt.builder import make
from guiqwt.curve import GridItem
from guiqwt.label import LegendBoxItem
from guiqwt.styles import style_generator

from cdl.config import Conf

if TYPE_CHECKING:
    from guiqwt.curve import CurveItem
    from guiqwt.image import MaskedImageItem
    from guiqwt.plot import CurveWidget, ImagePlot, ImageWidget

    from cdl.core.gui.objectlist import ObjectList
    from cdl.core.gui.panel.base import BaseDataPanel
    from cdl.core.model.image import ImageParam
    from cdl.core.model.signal import SignalParam


def calc_data_hash(obj: SignalParam | ImageParam) -> str:
    """Calculate a hash for a SignalParam | ImageParam object's data"""
    return hashlib.sha1(np.ascontiguousarray(obj.data)).hexdigest()


class BaseItemList:
    """Object handling plot items associated to objects (signals/images)"""

    def __init__(
        self,
        panel: BaseDataPanel,
        objlist: ObjectList,
        plotwidget: CurveWidget | ImageWidget,
    ):
        self.panel = panel
        self.objlist = objlist
        self.plotwidget = plotwidget
        self.plot = plotwidget.get_plot()
        self.__plotitems = []  # plot items associated to objects (sig/ima)
        self.__shapeitems = []
        self.__cached_hashes: WeakKeyDictionary[
            SignalParam | ImageParam, List[int]
        ] = WeakKeyDictionary()

    def __len__(self) -> int:
        """Return number of items"""
        return len(self.__plotitems)

    def __getitem__(self, row: int):
        """Return item at row"""
        return self.__plotitems[row]

    def __setitem__(self, row, item):
        """Set item at row"""
        self.__plotitems[row] = item

    def __delitem__(self, row):
        """Del item at row"""
        item = self.__plotitems.pop(row)
        self.plot.del_item(item)

    def __iter__(self):
        """Return an iterator over items"""
        yield from self.__plotitems

    def append(self, item):
        """Append item"""
        self.__plotitems.append(item)

    def insert(self, row):
        """Insert object at row index"""
        self.__plotitems.insert(row, None)

    def add_shapes(self, row):
        """Add geometric shape items associated to computed results and annotations"""
        obj = self.objlist[row]
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

    def remove_all(self):
        """Remove all plot items"""
        self.__plotitems = []
        self.plot.del_all_items()

    def remove_all_shape_items(self):
        """Remove all geometric shapes associated to result items"""
        if set(self.__shapeitems).issubset(set(self.plot.items)):
            self.plot.del_items(self.__shapeitems)
        self.__shapeitems = []

    def __add_item_to_plot(self, row):
        """Make plot item and add it to plot"""
        obj = self.objlist[row]
        self.__cached_hashes[obj] = calc_data_hash(obj)
        item = obj.make_item()
        item.set_readonly(True)
        if row < len(self):
            self[row] = item
        else:
            self.append(item)
        self.plot.add_item(item)
        return item

    def __update_item_on_plot(
        self, row: int, ref_item: CurveItem | MaskedImageItem, just_show: bool = False
    ) -> None:
        """Update plot item.

        param int row: row index
        param ref_item: reference item
        param bool just_show: if True, only show the item (do not update it,
        except regarding the reference item)"""
        if not just_show:
            obj = self.objlist[row]
            cached_hash = self.__cached_hashes.get(obj)
            new_hash = calc_data_hash(obj)
            data_changed = cached_hash is None or cached_hash != new_hash
            self.__cached_hashes[obj] = new_hash
            obj.update_item(self[row], data_changed=data_changed)
        self.update_item_according_to_ref_item(self[row], ref_item)

    @staticmethod
    def update_item_according_to_ref_item(
        item: MaskedImageItem, ref_item: MaskedImageItem
    ) -> None:  # pylint: disable=unused-argument
        """Update plot item according to reference item"""
        #  For now, nothing to do here: it's only used for images (contrast)

    def refresh_plot(self, only_row: int = None, just_show: bool = False) -> None:
        """Refresh plot.

        param int only_row: if not None, only refresh the item at this row
        param bool just_show: if True, only show the item(s) (do not update it/them)
        """
        if only_row is None:
            rows = self.objlist.get_selected_rows()
            if len(rows) == 1:
                self.cleanup_dataview()
            self.remove_all_shape_items()
            for item in self:
                if item is not None:
                    item.hide()
        else:
            rows = [only_row]
        title_keys = ("title", "xlabel", "ylabel", "zlabel", "xunit", "yunit", "zunit")
        titles_dict = {}
        if rows:
            ref_item = None
            for i_row, row in enumerate(rows):
                obj = self.objlist[row]
                for key in title_keys:
                    title = getattr(obj, key, "")
                    value = titles_dict.get(key)
                    if value is None:
                        titles_dict[key] = title
                    elif value != title:
                        titles_dict[key] = ""
                item = self[row]
                if item is None:
                    item = self.__add_item_to_plot(row)
                else:
                    if i_row == 0:
                        make.style = style_generator()
                    self.__update_item_on_plot(
                        row, ref_item=ref_item, just_show=just_show
                    )
                    if ref_item is None:
                        ref_item = item
                self.plot.set_item_visible(item, True, replot=False)
                self.plot.set_active_item(item)
                item.unselect()
                self.add_shapes(row)
            self.plot.replot()
        else:
            for key in title_keys:
                titles_dict[key] = ""
        tdict = titles_dict
        tdict["ylabel"] = (tdict["ylabel"], tdict.pop("zlabel"))
        tdict["yunit"] = (tdict["yunit"], tdict.pop("zunit"))
        self.plot.set_titles(**titles_dict)
        self.plot.do_autoscale()

    def cleanup_dataview(self):
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

    def get_current_plot_options(self):
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


class SignalItemList(BaseItemList):
    """Object handling signal plot items, plot dialogs, plot options"""

    # Nothing specific to signals, as of today


class ImageItemList(BaseItemList):
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

    def refresh_plot(self, only_row: int = None, just_show: bool = False) -> None:
        """Refresh plot.

        param int only_row: if not None, only refresh the item at this row
        param bool just_show: if True, only show the item(s) (do not update it/them)
        """
        super().refresh_plot(only_row=only_row, just_show=just_show)
        self.plotwidget.contrast.setVisible(Conf.view.show_contrast.get(True))

    def cleanup_dataview(self):
        """Clean up data view"""
        for widget in (self.plotwidget.xcsw, self.plotwidget.ycsw):
            widget.hide()
        super().cleanup_dataview()

    def get_current_plot_options(self):
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
