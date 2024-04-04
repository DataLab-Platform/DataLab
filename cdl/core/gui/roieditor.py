# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI editor
==========

The :mod:`cdl.core.gui.roieditor` module provides the ROI editor widgets
for signals and images.

Signal ROI editor
-----------------

.. autoclass:: SignalROIEditor
    :members:

Image ROI editor
----------------

.. autoclass:: ImageROIEditor
    :members:
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Union

from guidata.configtools import get_icon
from guidata.qthelpers import add_actions, create_action
from plotpy.builder import make
from plotpy.interfaces import IImageItemType
from plotpy.items import AnnotatedCircle, ObjectInfo
from qtpy import QtWidgets as QW

from cdl.config import Conf, _
from cdl.core.computation.base import ROIDataParam
from cdl.core.model.image import RoiDataGeometries

if TYPE_CHECKING:
    from plotpy.plot import PlotDialog

    from cdl.obj import ImageObj, SignalObj

    AnyObj = Union[SignalObj, ImageObj]


class BaseROIEditorMeta(type(QW.QWidget), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class BaseROIEditor(QW.QWidget, metaclass=BaseROIEditorMeta):
    """ROI Editor"""

    ICON_NAME = None
    OBJ_NAME = None

    def __init__(
        self,
        parent: PlotDialog,
        obj: AnyObj,
        extract: bool,
        singleobj: bool | None = None,
    ):
        super().__init__(parent)
        parent.accepted.connect(self.dialog_accepted)
        self.plot = parent.get_plot()
        self.obj = obj
        self.extract = extract
        self.__modified = None

        self.__data = ROIDataParam.create(singleobj=singleobj)

        self.fmt = obj.get_metadata_option("format")
        self.roi_items = list(obj.iterate_roi_items(self.fmt, True))

        for roi_item in self.roi_items:
            self.plot.add_item(roi_item)
            self.plot.set_active_item(roi_item)

        self.add_btn = None
        self.singleobj_btn = None
        self.setup_widget()

        self.update_roi_titles()
        self.plot.SIG_ITEMS_CHANGED.connect(lambda _plt: self.update_roi_titles())
        self.plot.SIG_ITEM_REMOVED.connect(self.item_removed)
        self.plot.SIG_RANGE_CHANGED.connect(lambda _rng, _min, _max: self.item_moved())
        self.plot.SIG_ANNOTATION_CHANGED.connect(lambda _plt: self.item_moved())

        #  In "extract mode", the dialog box OK button should always been enabled
        #  when at least one ROI is defined,
        #  whereas in non-extract mode (when editing ROIs) the OK button is by default
        #  disabled (until ROI data is modified)
        self.modified = extract

    @property
    def modified(self) -> bool:
        """Return dialog modified state"""
        return self.__modified

    @modified.setter
    def modified(self, value: bool):
        """Set dialog modified state"""
        self.__modified = value
        dlg = self.parent()
        if self.extract:
            #  In "extract mode", OK button is enabled when at least one ROI is defined
            value = value and len(self.roi_items) > 0
        dlg.button_box.button(QW.QDialogButtonBox.Ok).setEnabled(value)

    def dialog_accepted(self):
        """Parent dialog was accepted: updating ROI Editor data"""
        coords = []
        for roi_item in self.roi_items:
            coords.append(list(self.get_roi_item_coords(roi_item)))
        self.__data.roidata = self.obj.roi_coords_to_indexes(coords)
        if self.singleobj_btn is not None:
            singleobj = self.singleobj_btn.isChecked()
            self.__data.singleobj = singleobj
            Conf.proc.extract_roi_singleobj.set(singleobj)
        self.__data.modified = self.modified

    def get_data(self) -> ROIDataParam:
        """Get ROI Editor data (results of the dialog box)"""
        return self.__data

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
            singleobj = self.__data.singleobj
            if singleobj is None:
                singleobj = Conf.proc.extract_roi_singleobj.get()
            self.singleobj_btn.setChecked(singleobj)
        layout.addStretch()
        self.setLayout(layout)

    def add_roi_item(self, roi_item):
        """Add ROI item to plot and refresh titles"""
        self.plot.unselect_all()
        self.roi_items.append(roi_item)
        self.update_roi_titles()
        self.modified = True
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
        self.modified = True
        self.update_roi_titles()

    def item_moved(self):
        """ROI plot item has just been moved"""
        self.modified = True

    @staticmethod
    @abc.abstractmethod
    def get_roi_item_coords(roi_item):
        """Return ROI item coords"""


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
            self.add_roi,
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

    def add_roi(self, geometry: RoiDataGeometries = RoiDataGeometries.RECTANGLE):
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
