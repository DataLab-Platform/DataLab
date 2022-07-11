# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
ROI editor widgets
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import abc
from typing import Callable

from guidata.configtools import get_icon
from guiqwt.builder import make
from guiqwt.label import ObjectInfo
from qtpy import QtWidgets as QW

from codraft.config import _
from codraft.core.model.base import ObjectItf


class BaseROIEditorMeta(type(QW.QWidget), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class BaseROIEditor(QW.QWidget, metaclass=BaseROIEditorMeta):
    """ROI Editor"""

    ICON_NAME = None

    def __init__(self, parent: QW.QDialog, obj: ObjectItf):
        super().__init__(parent)
        self.plot = parent.get_plot()
        self.obj = obj

        fmt = obj.metadata.get(obj.METADATA_FMT, "%s")
        self.roi_items = list(obj.iterate_roi_items(fmt, True))
        self.new_roi_func = lambda: obj.new_roi_item(fmt, True, editable=True)

        for roi_item in self.roi_items:
            self.plot.add_item(roi_item)
            self.plot.set_active_item(roi_item)

        self.setup_widget()

        self.update_roi_titles()
        self.plot.SIG_ITEMS_CHANGED.connect(lambda _plot: self.update_roi_titles())
        self.plot.SIG_ITEM_REMOVED.connect(self.item_removed)

    def setup_widget(self):
        """Setup ROI editor widget"""
        add_btn = QW.QPushButton(
            get_icon(self.ICON_NAME), _("Add region of interest"), self
        )
        add_btn.clicked.connect(self.add_roi)
        layout = QW.QHBoxLayout()
        layout.addWidget(add_btn)
        layout.addStretch()
        self.setLayout(layout)

    def add_roi(self):
        """Add ROI"""
        self.plot.unselect_all()
        roi_item = self.new_roi_func()
        self.roi_items.append(roi_item)
        self.update_roi_titles()
        self.plot.add_item(roi_item)
        self.plot.set_active_item(roi_item)

    @abc.abstractmethod
    def update_roi_titles(self):
        """Update ROI annotation titles"""

    def item_removed(self, item):
        """Item was removed. Since all items are read-only except ROIs...
        this must be an ROI."""
        assert item in self.roi_items
        self.roi_items.remove(item)
        self.update_roi_titles()

    @staticmethod
    @abc.abstractmethod
    def get_roi_item_coords(roi_item):
        """Return ROI item coords"""

    def get_roi_coords(self) -> list:
        """Return list of ROI plot coordinates"""
        coords = []
        for roi_item in self.roi_items:
            coords.append(list(self.get_roi_item_coords(roi_item)))
        return coords


class ROIRangeInfo(ObjectInfo):
    """ObjectInfo for ROI selection"""

    def __init__(self, roi_items):
        self.roi_items = roi_items

    def get_text(self):
        textlist = []
        for index, roi_item in enumerate(self.roi_items):
            x0, x1 = roi_item.get_range()
            textlist.append(f"ROI{index:02d}: {x0} ≤ x ≤ {x1}")
        return "<br>".join(textlist)


class SignalROIEditor(BaseROIEditor):
    """Signal ROI Editor"""

    ICON_NAME = "signal_roi_new.svg"

    def setup_widget(self):
        """Setup ROI editor widget"""
        super().setup_widget()
        info = ROIRangeInfo(self.roi_items)
        info_label = make.info_label("BL", info, title=_("Regions of interest"))
        self.plot.add_item(info_label)
        self.info_label = info_label

    def update_roi_titles(self):
        """Update ROI annotation titles"""
        self.info_label.update_text()

    @staticmethod
    def get_roi_item_coords(roi_item):
        """Return ROI item coords"""
        return roi_item.get_range()


class ImageROIEditor(BaseROIEditor):
    """Image ROI Editor"""

    ICON_NAME = "image_roi_new.svg"

    def update_roi_titles(self):
        """Update ROI annotation titles"""
        for index, roi_item in enumerate(self.roi_items):
            roi_item.annotationparam.title = f"ROI{index:02d}"
            roi_item.annotationparam.update_annotation(roi_item)

    @staticmethod
    def get_roi_item_coords(roi_item):
        """Return ROI item coords"""
        return roi_item.get_rect()
