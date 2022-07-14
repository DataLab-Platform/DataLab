# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
ROI editor widgets
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import abc

from guidata.configtools import get_icon
from guidata.qthelpers import add_actions, create_action
from guiqwt.annotations import AnnotatedCircle
from guiqwt.builder import make
from guiqwt.interfaces import IImageItemType
from guiqwt.label import ObjectInfo
from qtpy import QtWidgets as QW

from codraft.config import _
from codraft.core.model.base import ObjectItf
from codraft.core.model.image import RoiDataGeometries


class BaseROIEditorMeta(type(QW.QWidget), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class BaseROIEditor(QW.QWidget, metaclass=BaseROIEditorMeta):
    """ROI Editor"""

    ICON_NAME = None
    OBJ_NAME = None

    def __init__(self, parent: QW.QDialog, obj: ObjectItf, extract: bool):
        super().__init__(parent)
        self.plot = parent.get_plot()
        self.obj = obj
        self.extract = extract

        self.fmt = obj.metadata.get(obj.METADATA_FMT, "%s")
        self.roi_items = list(obj.iterate_roi_items(self.fmt, True))

        for roi_item in self.roi_items:
            self.plot.add_item(roi_item)
            self.plot.set_active_item(roi_item)

        self.add_btn = None
        self.singleobj_btn = None
        self.setup_widget()

        self.update_roi_titles()
        self.plot.SIG_ITEMS_CHANGED.connect(lambda _plot: self.update_roi_titles())
        self.plot.SIG_ITEM_REMOVED.connect(self.item_removed)

    def setup_widget(self):
        """Setup ROI editor widget"""
        self.add_btn = QW.QPushButton(
            get_icon(self.ICON_NAME), _("Add region of interest"), self
        )
        layout = QW.QHBoxLayout()
        layout.addWidget(self.add_btn)
        if self.extract:
            self.singleobj_btn = QW.QCheckBox(
                _("Extract all regions of interest into a single %s object")
                % self.OBJ_NAME,
                self,
            )
            layout.addWidget(self.singleobj_btn)
        layout.addStretch()
        self.setLayout(layout)

    @property
    def singleobj_extraction(self):
        """Return True if a single object extraction has been chosen"""
        if self.singleobj_btn is None:
            return None
        return self.singleobj_btn.isChecked()

    def add_roi_item(self, roi_item):
        """Add ROI item to plot and refresh titles"""
        self.plot.unselect_all()
        self.roi_items.append(roi_item)
        self.update_roi_titles()
        self.plot.add_item(roi_item)
        self.plot.set_active_item(roi_item)

    @abc.abstractmethod
    def update_roi_titles(self):
        """Update ROI annotation titles"""
        dlg = self.parent()
        dlg.button_box.button(QW.QDialogButtonBox.Ok).setEnabled(
            len(self.roi_items) > 0
        )

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
    OBJ_NAME = _("signal")

    def setup_widget(self):
        """Setup ROI editor widget"""
        super().setup_widget()
        info = ROIRangeInfo(self.roi_items)
        info_label = make.info_label("BL", info, title=_("Regions of interest"))
        self.plot.add_item(info_label)
        self.info_label = info_label
        self.add_btn.clicked.connect(self.add_roi)

    def add_roi(self):
        """Simply add an ROI"""
        roi_item = self.obj.new_roi_item(self.fmt, True, editable=True)
        self.add_roi_item(roi_item)

    def update_roi_titles(self):
        """Update ROI annotation titles"""
        super().update_roi_titles()
        self.info_label.update_text()

    @staticmethod
    def get_roi_item_coords(roi_item):
        """Return ROI item coords"""
        return roi_item.get_range()


class ImageROIEditor(BaseROIEditor):
    """Image ROI Editor"""

    ICON_NAME = "image_roi_new.svg"
    OBJ_NAME = _("image")

    def setup_widget(self):
        """Setup ROI editor widget"""
        super().setup_widget()
        item = self.plot.get_items(item_type=IImageItemType)[0]
        item.set_mask_visible(False)
        menu = QW.QMenu()
        rectact = create_action(
            self,
            _("Rectangular ROI"),
            lambda: self.add_roi(RoiDataGeometries.RECTANGLE),
            icon=get_icon("rectangle.png"),
        )
        circact = create_action(
            self,
            _("Circular ROI"),
            lambda: self.add_roi(RoiDataGeometries.CIRCLE),
            icon=get_icon("circle.png"),
        )
        add_actions(menu, (rectact, circact))
        self.add_btn.setMenu(menu)

    def add_roi(self, geometry: RoiDataGeometries):
        """Add new ROI"""
        item = self.obj.new_roi_item(self.fmt, True, editable=True, geometry=geometry)
        self.add_roi_item(item)

    def update_roi_titles(self):
        """Update ROI annotation titles"""
        super().update_roi_titles()
        for index, roi_item in enumerate(self.roi_items):
            roi_item.annotationparam.title = f"ROI{index:02d}"
            roi_item.annotationparam.update_annotation(roi_item)

    @staticmethod
    def get_roi_item_coords(roi_item):
        """Return ROI item coords"""
        x0, y0, x1, y1 = roi_item.get_rect()
        if isinstance(roi_item, AnnotatedCircle):
            y0 = y1 = 0.5 * (y0 + y1)
        return x0, y0, x1, y1
