# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Object (signal/image) view widgets
----------------------------------

This module provides widgets to display object (signal/image) trees.

.. autosummary::
    :toctree:

    SimpleObjectTree
    GetObjectDialog
    ObjectView

.. autoclass:: SimpleObjectTree
    :members:

.. autoclass:: GetObjectDialog
    :members:

.. autoclass:: ObjectView
    :members:

.. note:: This module provides tree widgets to display signals, images and groups. It
    is important to note that, by design, the user can only select either individual
    signals/images or groups, but not both at the same time. This is an important
    design choice, as it allows to simplify the user experience, and to avoid
    potential confusion between the two types of selection.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

from guidata.configtools import get_icon
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from cdl.config import _
from cdl.core.model.image import ImageObj
from cdl.core.model.signal import SignalObj
from cdl.utils.qthelpers import block_signals

if TYPE_CHECKING:  # pragma: no cover
    from cdl.core.gui.objectmodel import ObjectGroup, ObjectModel
    from cdl.core.gui.panel.base import BaseDataPanel


class SimpleObjectTree(QW.QTreeWidget):
    """Base object handling panel list widget, object (sig/ima) lists"""

    SIG_ITEM_DOUBLECLICKED = QC.Signal(str)
    SIG_CONTEXT_MENU = QC.Signal(QC.QPoint)

    def __init__(self, parent: QW.QWidget, objmodel: ObjectModel) -> None:
        self.objmodel: ObjectModel = objmodel
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setColumnCount(1)
        self.setAlternatingRowColors(True)
        self.itemDoubleClicked.connect(self.item_double_clicked)

    def __str__(self) -> str:
        """Return string representation"""
        textlist = []
        for tl_index in range(self.topLevelItemCount()):
            tl_item = self.topLevelItem(tl_index)
            textlist.append(tl_item.text(0))
            for index in range(tl_item.childCount()):
                textlist.append("    " + tl_item.child(index).text(0))
        return os.linesep.join(textlist)

    def init_from(self, sobjlist: SimpleObjectTree) -> None:
        """Init from another SimpleObjectList, without making copies of objects"""
        self.objmodel = sobjlist.objmodel
        self.populate_tree()
        self.set_current_item_id(sobjlist.get_current_item_id())

    def iter_items(
        self, item: QW.QTreeWidgetItem | None = None
    ) -> Iterator[QW.QTreeWidgetItem]:
        """Recursively iterate over all items"""
        if item is None:
            for index in range(self.topLevelItemCount()):
                yield from self.iter_items(self.topLevelItem(index))
        else:
            yield item
            for index in range(item.childCount()):
                yield from self.iter_items(item.child(index))

    def get_item_from_id(self, item_id) -> QW.QTreeWidgetItem:
        """Return QTreeWidgetItem from id (stored in item's data)"""
        for item in self.iter_items():
            if item.data(0, QC.Qt.UserRole) == item_id:
                return item
        return None

    def get_actions_from_items(self, items):  # pylint: disable=W0613,R0201
        """Get actions from item"""
        return []

    def get_current_object(self) -> SignalObj | ImageObj | None:
        """Return current object"""
        oid = self.get_current_item_id(object_only=True)
        if oid is not None:
            return self.objmodel[oid]
        return None

    def set_current_object(self, obj: SignalObj | ImageObj) -> None:
        """Set current object"""
        self.set_current_item_id(obj.uuid)

    def get_current_item_id(self, object_only: bool = False) -> str | None:
        """Return current item id"""
        item = self.currentItem()
        if item is not None and (not object_only or item.parent() is not None):
            return item.data(0, QC.Qt.UserRole)
        return None

    def set_current_item_id(self, uuid: str, extend: bool = False) -> None:
        """Set current item by id"""
        item = self.get_item_from_id(uuid)
        if extend:
            self.setCurrentItem(item, 0, QC.QItemSelectionModel.Select)
        else:
            self.setCurrentItem(item)

    def get_current_group_id(self) -> str:
        """Return current group ID"""
        selected_item = self.currentItem()
        if selected_item is None:
            return None
        if selected_item.parent() is None:
            return selected_item.data(0, QC.Qt.UserRole)
        return selected_item.parent().data(0, QC.Qt.UserRole)

    @staticmethod
    def __update_item(
        item: QW.QTreeWidgetItem, obj: SignalObj | ImageObj | ObjectGroup
    ) -> None:
        """Update item"""
        item.setText(0, f"{obj.short_id}: {obj.title}")
        if isinstance(obj, (SignalObj, ImageObj)):
            item.setToolTip(0, obj.metadata_to_html())
        item.setData(0, QC.Qt.UserRole, obj.uuid)

    def populate_tree(self) -> None:
        """Populate tree with objects"""
        uuid = self.get_current_item_id()
        with block_signals(widget=self, enable=True):
            self.clear()
        for group in self.objmodel.get_groups():
            self.add_group_item(group)
        if uuid is not None:
            self.set_current_item_id(uuid)

    def update_tree(self) -> None:
        """Update tree"""
        self.objmodel.refresh_short_ids()
        for group in self.objmodel.get_groups():
            self.__update_item(self.get_item_from_id(group.uuid), group)
            for obj in group:
                self.__update_item(self.get_item_from_id(obj.uuid), obj)

    def __add_to_group_item(
        self, obj: SignalObj | ImageObj, group_item: QW.QTreeWidgetItem
    ) -> None:
        """Add object to group item"""
        item = QW.QTreeWidgetItem()
        self.__update_item(item, obj)
        group_item.addChild(item)

    def add_group_item(self, group: ObjectGroup) -> None:
        """Add group item"""
        self.objmodel.refresh_short_ids()
        group_item = QW.QTreeWidgetItem()
        group_item.setIcon(0, get_icon("group.svg"))
        self.__update_item(group_item, group)
        self.addTopLevelItem(group_item)
        group_item.setExpanded(True)
        for obj in group:
            self.__add_to_group_item(obj, group_item)

    def add_object_item(
        self, obj: SignalObj | ImageObj, group_id: str, set_current: bool = True
    ) -> None:
        """Add item"""
        self.objmodel.refresh_short_ids()
        group_item = self.get_item_from_id(group_id)
        self.__add_to_group_item(obj, group_item)
        if set_current:
            self.set_current_item_id(obj.uuid)

    def update_item(self, uuid: str) -> None:
        """Update item"""
        obj_or_group = self.objmodel.get_object_or_group(uuid)
        item = self.get_item_from_id(uuid)
        self.__update_item(item, obj_or_group)

    def remove_item(self, oid: str, refresh: bool = True) -> None:
        """Remove item"""
        item = self.get_item_from_id(oid)
        if item is not None:
            with block_signals(widget=self, enable=not refresh):
                if item.parent() is None:
                    #  Group item: remove from tree
                    self.takeTopLevelItem(self.indexOfTopLevelItem(item))
                else:
                    #  Object item: remove from parent
                    item.parent().removeChild(item)

    def item_double_clicked(self, item: QW.QTreeWidgetItem) -> None:
        """Item was double-clicked: open a pop-up plot dialog"""
        if item.parent() is not None:
            oid = item.data(0, QC.Qt.UserRole)
            self.SIG_ITEM_DOUBLECLICKED.emit(oid)

    def contextMenuEvent(
        self, event: QG.QContextMenuEvent
    ) -> None:  # pylint: disable=C0103
        """Override Qt method"""
        self.SIG_CONTEXT_MENU.emit(event.globalPos())


class GetObjectDialog(QW.QDialog):
    """Get object dialog box"""

    def __init__(self, parent: QW.QWidget, panel: BaseDataPanel, title: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        vlayout = QW.QVBoxLayout()
        self.setLayout(vlayout)

        self.tree = SimpleObjectTree(parent, panel.objmodel)
        self.tree.init_from(panel.objview)
        self.tree.SIG_ITEM_DOUBLECLICKED.connect(lambda oid: self.accept())
        self.tree.itemSelectionChanged.connect(self.current_object_changed)
        vlayout.addWidget(self.tree)

        bbox = QW.QDialogButtonBox(QW.QDialogButtonBox.Ok | QW.QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        self.ok_btn = bbox.button(QW.QDialogButtonBox.Ok)
        vlayout.addSpacing(10)
        vlayout.addWidget(bbox)
        # Update OK button state:
        self.current_object_changed()

    def get_current_object(self) -> SignalObj | ImageObj:
        """Return current object"""
        return self.tree.get_current_object()

    def current_object_changed(self) -> None:
        """Item selection has changed"""
        self.ok_btn.setEnabled(
            isinstance(self.get_current_object(), (SignalObj, ImageObj))
        )


class ObjectView(SimpleObjectTree):
    """Object handling panel list widget, object (sig/ima) lists"""

    SIG_SELECTION_CHANGED = QC.Signal()
    SIG_IMPORT_FILES = QC.Signal(list)

    def __init__(self, parent: QW.QWidget, objmodel: ObjectModel) -> None:
        super().__init__(parent, objmodel)
        self.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.itemSelectionChanged.connect(self.item_selection_changed)

    def paintEvent(self, event):  # pylint: disable=C0103
        """Reimplement Qt method"""
        super().paintEvent(event)
        if len(self.objmodel) > 0:
            return
        painter = QG.QPainter(self.viewport())
        painter.drawText(self.rect(), QC.Qt.AlignCenter, _("Drag files here to open"))

    def dropEvent(self, event):  # pylint: disable=C0103
        """Reimplement Qt method"""
        if event.mimeData().hasUrls():
            fnames = [url.toLocalFile() for url in event.mimeData().urls()]
            self.SIG_IMPORT_FILES.emit(fnames)
            event.setDropAction(QC.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dragEnterEvent(self, event):  # pylint: disable=C0103,R0201
        """Reimplement Qt method"""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):  # pylint: disable=C0103,R0201
        """Reimplement Qt method"""
        if event.mimeData().hasUrls():
            event.setDropAction(QC.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def get_sel_group_items(self) -> list[QW.QTreeWidgetItem]:
        """Return selected group items"""
        return [item for item in self.selectedItems() if item.parent() is None]

    def get_sel_group_uuids(self) -> list[str]:
        """Return selected group uuids"""
        return [item.data(0, QC.Qt.UserRole) for item in self.get_sel_group_items()]

    def get_sel_object_items(self) -> list[QW.QTreeWidgetItem]:
        """Return selected object items"""
        return [item for item in self.selectedItems() if item.parent() is not None]

    def get_sel_object_uuids(self, include_groups: bool = False) -> list[str]:
        """Return selected objects uuids.

        If include_groups is True, also return objects from selected groups."""
        sel_items = self.get_sel_object_items()
        if not sel_items:
            cur_item = self.currentItem()
            if cur_item is not None and cur_item.parent() is not None:
                sel_items = [cur_item]
        uuids = [item.data(0, QC.Qt.UserRole) for item in sel_items]
        if include_groups:
            for group_id in self.get_sel_group_uuids():
                uuids.extend(self.objmodel.get_group_object_ids(group_id))
        return uuids

    def get_sel_objects(
        self, include_groups: bool = False
    ) -> list[SignalObj | ImageObj]:
        """Return selected objects.

        If include_groups is True, also return objects from selected groups."""
        return [self.objmodel[oid] for oid in self.get_sel_object_uuids(include_groups)]

    def get_sel_groups(self) -> list[ObjectGroup]:
        """Return selected groups"""
        return self.objmodel.get_groups(self.get_sel_group_uuids())

    def item_selection_changed(self) -> None:
        """Refreshing the selection of objects and groups, emitting the
        SIG_SELECTION_CHANGED signal which triggers the update of the
        object properties panel, the plot items and the actions of the
        toolbar and menu bar.

        This method is called when the user selects or deselects items in the tree.
        It is also called when the user clicks on an item that was already selected.

        This method emits the SIG_SELECTION_CHANGED signal.
        """
        # ==> This is a very important design choice <==
        # When a group is selected, all individual objects are deselected, even if
        # they belong to other groups. This is intended to simplify the user experience.
        # In other words, the user may either select groups or individual objects, but
        # not both at the same time.
        sel_groups = self.get_sel_group_items()
        if sel_groups:
            for item in self.get_sel_object_items():
                item.setSelected(False)
            if self.currentItem().parent() is not None:
                self.setCurrentItem(sel_groups[0])

        self.SIG_SELECTION_CHANGED.emit()

    def select_nums(self, obj_nums: list[int], group_num: int = 0) -> None:
        """Select multiple objects by their numbers"""
        uuids = [self.objmodel.get_groups()[group_num][num].uuid for num in obj_nums]
        self.clearSelection()
        for uuid in uuids:
            self.set_current_item_id(uuid, extend=True)

    def select_objects(self, objs: list[SignalObj | ImageObj]) -> None:
        """Select multiple objects"""
        self.clearSelection()
        for obj in objs:
            self.set_current_item_id(obj.uuid, extend=True)

    def select_groups(self, groups: list[ObjectGroup]) -> None:
        """Select multiple groups"""
        self.clearSelection()
        for group in groups:
            self.set_current_item_id(group.uuid, extend=True)
