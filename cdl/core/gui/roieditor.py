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
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from guidata.configtools import get_icon
from guidata.qthelpers import add_actions, create_action
from plotpy.builder import make
from plotpy.interfaces import IImageItemType
from plotpy.items import (
    AnnotatedCircle,
    AnnotatedPolygon,
    AnnotatedRectangle,
    ObjectInfo,
    XRangeSelection,
)
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from cdl.config import Conf, _
from cdl.core.model.base import TypeObj, TypeROI
from cdl.core.model.image import CircularROI, PolygonalROI, RectangularROI
from cdl.core.model.signal import SegmentROI
from cdl.obj import ImageObj, ImageROI, SignalObj, SignalROI

if TYPE_CHECKING:
    from plotpy.items import MaskedImageItem
    from plotpy.plot import BasePlot, PlotDialog


def plot_item_to_single_roi(
    item: XRangeSelection | AnnotatedRectangle | AnnotatedCircle | AnnotatedPolygon,
) -> SegmentROI | CircularROI | RectangularROI | PolygonalROI:
    """Factory function to create a single ROI object from plot item

    Args:
        item: Plot item

    Returns:
        Single ROI object
    """
    if isinstance(item, XRangeSelection):
        cls = SegmentROI
    elif isinstance(item, AnnotatedRectangle):
        cls = RectangularROI
    elif isinstance(item, AnnotatedCircle):
        cls = CircularROI
    elif isinstance(item, AnnotatedPolygon):
        cls = PolygonalROI
    else:
        raise TypeError(f"Unsupported ROI item type: {type(item)}")
    return cls.from_plot_item(item)


TypeROIEditor = TypeVar("TypeROIEditor", bound="BaseROIEditor")


class BaseROIEditorMeta(type(QW.QWidget), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class BaseROIEditor(QW.QWidget, Generic[TypeObj, TypeROI], metaclass=BaseROIEditorMeta):
    """ROI Editor"""

    ICON_NAME = None
    OBJ_NAME = None

    def __init__(
        self,
        parent: PlotDialog,
        obj: TypeObj,
        extract: bool,
    ) -> None:
        super().__init__(parent)
        self.plot_dialog = parent
        parent.accepted.connect(self.dialog_accepted)
        self.plot = parent.get_plot()
        self.obj = obj
        self.extract = extract
        self.__modified: bool | None = None

        self.__roi: TypeROI | None = obj.roi
        if self.__roi is None:
            self.__roi = ImageROI()

        self.fmt = obj.get_metadata_option("format")
        self.roi_items = list(self.__roi.iterate_roi_items(obj, self.fmt, True))

        for roi_item in self.roi_items:
            self.plot.add_item(roi_item)
            self.plot.set_active_item(roi_item)

        self.toolbar: QW.QToolBar | None = None
        self.remove_all_action: QW.QAction | None = None
        self.singleobj_btn: QW.QToolButton | None = None
        self.setup_widget()

        # force update of ROI titles and remove_all_btn state
        self.items_changed(self.plot)

        self.plot.SIG_ITEMS_CHANGED.connect(self.items_changed)
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
    def modified(self, value: bool) -> None:
        """Set dialog modified state"""
        self.__modified = value
        if self.extract:
            #  In "extract mode", OK button is enabled when at least one ROI is defined
            value = value and len(self.roi_items) > 0
        self.plot_dialog.button_box.button(QW.QDialogButtonBox.Ok).setEnabled(value)

    def dialog_accepted(self) -> None:
        """Parent dialog was accepted: updating ROI Editor data"""
        self.__roi.empty()
        for roi_item in self.roi_items:
            self.__roi.add_roi(plot_item_to_single_roi(roi_item))
        if self.singleobj_btn is not None:
            singleobj = self.singleobj_btn.isChecked()
            self.__roi.singleobj = singleobj
            Conf.proc.extract_roi_singleobj.set(singleobj)

    def get_roieditor_results(self) -> tuple[TypeROI, bool]:
        """Get ROI editor results

        Returns:
            A tuple containing the ROI data parameters and ROI modified state.
            ROI modified state is True if the ROI data has been modified within
            the dialog box.
        """
        return self.__roi, self.modified

    def create_actions(self) -> list[QW.QAction]:
        """Create actions"""
        self.remove_all_action = create_action(
            self,
            _("Remove all ROIs"),
            icon=get_icon("roi_delete.svg"),
            triggered=self.remove_all_rois,
        )
        return [None, self.remove_all_action]

    def setup_widget(self) -> None:
        """Setup ROI editor widget"""
        layout = QW.QHBoxLayout()
        self.toolbar = QW.QToolBar(self)
        self.toolbar.setToolButtonStyle(QC.Qt.ToolButtonTextUnderIcon)
        add_actions(self.toolbar, self.create_actions())
        layout.addWidget(self.toolbar)
        if self.extract:
            self.singleobj_btn = QW.QCheckBox(
                _("Extract all ROIs into a single %s object") % self.OBJ_NAME,
                self,
            )
            layout.addWidget(self.singleobj_btn)
            self.singleobj_btn.setChecked(self.__roi.singleobj)
        layout.addStretch()
        self.setLayout(layout)

    def add_roi_item(self, roi_item) -> None:
        """Add ROI item to plot and refresh titles"""
        self.plot.unselect_all()
        self.roi_items.append(roi_item)
        self.update_roi_titles()
        self.modified = True
        self.plot.add_item(roi_item)
        self.plot.set_active_item(roi_item)

    def remove_all_rois(self) -> None:
        """Remove all ROIs"""
        if QW.QMessageBox.question(
            self,
            _("Remove all ROIs"),
            _("Are you sure you want to remove all ROIs?"),
        ):
            self.plot.del_items(self.roi_items)

    @abc.abstractmethod
    def update_roi_titles(self) -> None:
        """Update ROI annotation titles"""

    def items_changed(self, _plot: BasePlot) -> None:
        """Items have changed"""
        self.update_roi_titles()
        self.remove_all_action.setEnabled(len(self.roi_items) > 0)

    def item_removed(self, item) -> None:
        """Item was removed. Since all items are read-only except ROIs...
        this must be an ROI."""
        assert item in self.roi_items
        self.roi_items.remove(item)
        self.modified = True
        self.update_roi_titles()

    def item_moved(self) -> None:
        """ROI plot item has just been moved"""
        self.modified = True


class ROIRangeInfo(ObjectInfo):
    """ObjectInfo for ROI selection"""

    def __init__(self, roi_items) -> None:
        self.roi_items = roi_items

    def get_text(self) -> str:
        textlist = []
        for index, roi_item in enumerate(self.roi_items):
            x0, x1 = roi_item.get_range()
            textlist.append(f"ROI{index:02d}: {x0} ≤ x ≤ {x1}")
        return "<br>".join(textlist)


class SignalROIEditor(BaseROIEditor[SignalObj, SignalROI]):
    """Signal ROI Editor"""

    ICON_NAME = "signal_roi.svg"
    OBJ_NAME = _("signal")

    def create_actions(self) -> list[QW.QAction]:
        """Create actions"""
        add_action = create_action(
            self,
            _("Add ROI"),
            icon=get_icon(self.ICON_NAME),
            triggered=self.add_roi,
        )
        return [add_action] + super().create_actions()

    def setup_widget(self) -> None:
        """Setup ROI editor widget"""
        super().setup_widget()
        info = ROIRangeInfo(self.roi_items)
        info_label = make.info_label("BL", info, title=_("Regions of interest"))
        self.plot.add_item(info_label)
        self.info_label = info_label

    def add_roi(self) -> None:
        """Simply add an ROI"""
        roi_item = self.obj.new_roi_item(self.fmt, True, editable=True)
        self.add_roi_item(roi_item)

    def update_roi_titles(self):
        """Update ROI annotation titles"""
        super().update_roi_titles()
        self.info_label.update_text()


class ImageROIEditor(BaseROIEditor[ImageObj, ImageROI]):
    """Image ROI Editor"""

    ICON_NAME = "image_roi.svg"
    OBJ_NAME = _("image")

    def create_actions(self) -> list[QW.QAction]:
        """Create actions"""
        rect_action = create_action(
            self,
            _("Add rectangular ROI"),
            icon=get_icon("roi_new_rectangle.svg"),
            triggered=lambda: self.add_roi("rectangle"),
        )
        circ_action = create_action(
            self,
            _("Add circular ROI"),
            icon=get_icon("roi_new_circle.svg"),
            triggered=lambda: self.add_roi("circle"),
        )
        return [rect_action, circ_action] + super().create_actions()

    def setup_widget(self) -> None:
        """Setup ROI editor widget"""
        super().setup_widget()
        item: MaskedImageItem = self.plot.get_items(item_type=IImageItemType)[0]
        item.set_mask_visible(False)

    def add_roi(self, geometry: Literal["rectangle", "circle"]) -> None:
        """Add new ROI"""
        item = self.obj.new_roi_item(self.fmt, True, editable=True, geometry=geometry)
        self.add_roi_item(item)

    def update_roi_titles(self) -> None:
        """Update ROI annotation titles"""
        super().update_roi_titles()
        for index, roi_item in enumerate(self.roi_items):
            roi_item: AnnotatedRectangle | AnnotatedCircle | AnnotatedPolygon
            roi_item.annotationparam.title = f"ROI{index:02d}"
            roi_item.annotationparam.update_item(roi_item)
