# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Object view
===========

The :mod:`cdl.core.gui.objectview` module provides widgets to display object
(signal/image) trees.

.. note::

    This module provides tree widgets to display signals, images and groups. It
    is important to note that, by design, the user can only select either individual
    signals/images or groups, but not both at the same time. This is an important
    design choice, as it allows to simplify the user experience, and to avoid
    potential confusion between the two types of selection.

Simple object tree
------------------

.. autoclass:: SimpleObjectTree

Get object dialog
-----------------

.. autoclass:: GetObjectsDialog

Object view
-----------

.. autoclass:: ObjectView
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
from cdl.core.gui.objectmodel import ObjectGroup
from cdl.core.model.image import ImageObj
from cdl.core.model.signal import SignalObj
from cdl.utils.qthelpers import block_signals

if TYPE_CHECKING:
    from typing import Any

    from cdl.core.gui.objectmodel import ObjectModel
    from cdl.core.gui.panel.base import BaseDataPanel


def metadata_to_html(metadata: dict[str, Any]) -> str:
    """Convert metadata to human-readable string.

    Returns:
        HTML string
    """
    textlines = []
    for key, value in metadata.items():
        if len(textlines) > 5:
            textlines.append("[...]")
            break
        if not key.startswith("_"):
            vlines = str(value).splitlines()
            if vlines:
                text = f"<b>{key}:</b> {vlines[0]}"
                if len(vlines) > 1:
                    text += " [...]"
                textlines.append(text)
    if textlines:
        ptit = _("Object metadata")
        psub = _("(click on Metadata button for more details)")
        prefix = f"<i><u>{ptit}:</u> {psub}</i><br>"
        return f"<p style='white-space:pre'>{prefix}{'<br>'.join(textlines)}</p>"
    return ""


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
        self.header().setSectionResizeMode(QW.QHeaderView.Interactive)
        self.itemChanged.connect(lambda item: self.resizeColumnToContents(0))

    def __str__(self) -> str:
        """Return string representation"""
        textlist = []
        for tl_index in range(self.topLevelItemCount()):
            tl_item = self.topLevelItem(tl_index)
            textlist.append(tl_item.text(0))
            for index in range(tl_item.childCount()):
                textlist.append("    " + tl_item.child(index).text(0))
        return os.linesep.join(textlist)

    def initialize_from(self, sobjlist: SimpleObjectTree) -> None:
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

        Args:
            include_groups: If True, also return objects from selected groups.

        Returns:
            List of selected objects uuids.
        """
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

    @staticmethod
    def __update_item(
        item: QW.QTreeWidgetItem, obj: SignalObj | ImageObj | ObjectGroup
    ) -> None:
        """Update item"""
        item.setText(0, f"{obj.short_id}: {obj.title}")
        if isinstance(obj, (SignalObj, ImageObj)):
            item.setToolTip(0, metadata_to_html(obj.metadata))
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
        for group in self.objmodel.get_groups():
            self.__update_item(self.get_item_from_id(group.uuid), group)
            for obj in group:
                self.__update_item(self.get_item_from_id(obj.uuid), obj)

    def __add_to_group_item(
        self, obj: SignalObj | ImageObj, group_item: QW.QTreeWidgetItem
    ) -> None:
        """Add object to group item"""
        item = QW.QTreeWidgetItem()
        icon = "signal.svg" if isinstance(obj, SignalObj) else "image.svg"
        item.setIcon(0, get_icon(icon))
        self.__update_item(item, obj)
        group_item.addChild(item)

    def add_group_item(self, group: ObjectGroup) -> None:
        """Add group item"""
        group_item = QW.QTreeWidgetItem()
        group_item.setIcon(0, get_icon("group.svg"))
        self.__update_item(group_item, group)
        self.addTopLevelItem(group_item)
        group_item.setExpanded(True)
        for obj in group:
            self.__add_to_group_item(obj, group_item)
        self.resizeColumnToContents(0)

    def add_object_item(
        self, obj: SignalObj | ImageObj, group_id: str, set_current: bool = True
    ) -> None:
        """Add item"""
        group_item = self.get_item_from_id(group_id)
        self.__add_to_group_item(obj, group_item)
        if set_current:
            self.set_current_item_id(obj.uuid)
        self.resizeColumnToContents(0)

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

    def contextMenuEvent(self, event: QG.QContextMenuEvent) -> None:  # pylint: disable=C0103
        """Override Qt method"""
        self.SIG_CONTEXT_MENU.emit(event.globalPos())


class GetObjectsDialog(QW.QDialog):
    """Dialog box showing groups and objects (signals or images) to select one, or more.

    Args:
        parent: parent widget
        panel: data panel
        title: dialog title
        comment: optional dialog comment
        nb_objects: number of objects to select (default: 1)
        minimum_size: minimum size (width, height)
    """

    def __init__(
        self,
        parent: QW.QWidget,
        panel: BaseDataPanel,
        title: str,
        comment: str = "",
        nb_objects: int = 1,
        minimum_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(parent)
        self.__nb_objects = nb_objects
        self.__selected_objects: list[SignalObj | ImageObj] = []
        self.setWindowTitle(title)
        vlayout = QW.QVBoxLayout()
        self.setLayout(vlayout)

        self.tree = SimpleObjectTree(parent, panel.objmodel)
        self.tree.initialize_from(panel.objview)
        self.tree.SIG_ITEM_DOUBLECLICKED.connect(lambda oid: self.accept())
        self.tree.itemSelectionChanged.connect(self.__item_selection_changed)
        if nb_objects > 1:
            self.tree.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)
        vlayout.addWidget(self.tree)

        if comment:
            lbl = QW.QLabel(comment)
            lbl.setWordWrap(True)
            vlayout.addSpacing(10)
            vlayout.addWidget(lbl)

        bbox = QW.QDialogButtonBox(QW.QDialogButtonBox.Ok | QW.QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        self.ok_btn = bbox.button(QW.QDialogButtonBox.Ok)
        vlayout.addSpacing(10)
        vlayout.addWidget(bbox)
        # Update OK button state:
        self.__item_selection_changed()

        if minimum_size is not None:
            self.setMinimumSize(*minimum_size)
        else:
            self.setMinimumWidth(400)

    def __item_selection_changed(self) -> None:
        """Item selection has changed"""
        nobj = self.__nb_objects
        self.__selected_objects = self.tree.get_sel_objects(include_groups=nobj > 1)
        self.ok_btn.setEnabled(len(self.__selected_objects) == nobj)

    def get_selected_objects(self) -> list[SignalObj | ImageObj]:
        """Return selected objects"""
        return self.__selected_objects


class ObjectView(SimpleObjectTree):
    """Object handling panel list widget, object (sig/ima) lists"""

    SIG_SELECTION_CHANGED = QC.Signal()
    SIG_IMPORT_FILES = QC.Signal(list)

    def __init__(self, parent: QW.QWidget, objmodel: ObjectModel) -> None:
        super().__init__(parent, objmodel)
        self.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QW.QAbstractItemView.InternalMove)
        self.itemSelectionChanged.connect(self.item_selection_changed)
        self.__dragged_objects: list[QW.QListWidgetItem] = []
        self.__dragged_groups: list[QW.QListWidgetItem] = []
        self.__dragged_expanded_states: dict[QW.QListWidgetItem, bool] = {}

    def paintEvent(self, event):  # pylint: disable=C0103
        """Reimplement Qt method"""
        super().paintEvent(event)
        if len(self.objmodel) > 0:
            return
        painter = QG.QPainter(self.viewport())
        painter.drawText(self.rect(), QC.Qt.AlignCenter, _("Drag files here to open"))

    # pylint: disable=unused-argument
    def dragEnterEvent(self, event: QG.QDragEnterEvent) -> None:
        """Reimplement Qt method"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)
            self.__dragged_groups = self.get_sel_group_items()
            self.__dragged_objects = self.get_sel_object_items()
            self.__dragged_expanded_states = {
                item.data(0, QC.Qt.UserRole): item.isExpanded()
                for item in self.__dragged_groups
            }

    def dragLeaveEvent(self, event: QG.QDragLeaveEvent) -> None:
        """Reimplement Qt method"""
        super().dragLeaveEvent(event)
        self.__dragged_groups = []
        self.__dragged_objects = []
        self.__dragged_expanded_states = {}

    # pylint: disable=unused-argument
    def dragMoveEvent(self, event: QG.QDragMoveEvent) -> None:
        """Reimplement Qt method"""
        self.setDropIndicatorShown(True)
        if event.mimeData().hasUrls():
            event.setDropAction(QC.Qt.CopyAction)
            event.accept()
        else:
            super().dragMoveEvent(event)
            self.setDropIndicatorShown(self.__is_drop_allowed(event))

    def __is_drop_allowed(self, event: QG.QDropEvent | QG.QDragMoveEvent) -> bool:
        """Return True if drop is allowed"""

        # Yes, this method has too many return statements.
        # But it's still quite readable, so let's focus on other things and just disable
        # the pylint warning.
        #
        # pylint: disable=too-many-return-statements

        if event.mimeData().hasUrls():
            return True
        drop_pos = self.dropIndicatorPosition()
        on_item = drop_pos == QW.QAbstractItemView.OnItem
        above_item = drop_pos == QW.QAbstractItemView.AboveItem
        below_item = drop_pos == QW.QAbstractItemView.BelowItem
        on_viewport = drop_pos == QW.QAbstractItemView.OnViewport
        target_item = self.itemAt(event.pos())
        # If moved items are objects, refuse the drop on the viewport
        if self.__dragged_objects and on_viewport:
            return False
        # If drop indicator is on an item, refuse the drop if the target item
        # is anything but a group
        if on_item and (target_item is None or target_item.parent() is not None):
            return False
        # If drop indicator is on an item, refuse the drop if the moved items
        # are groups
        if on_item and self.__dragged_groups:
            return False
        # If target item is None, it means that the drop position is
        # outside of the tree. In this case, we accept the drop and move
        # the objects to the end of the list.
        if target_item is None or on_viewport:
            return True
        # If moved items are groups, refuse the drop if the target item is
        # not a group
        if self.__dragged_groups and target_item.parent() is not None:
            return False
        # If moved items are groups, refuse the drop if the target item is
        # a group but the target position is below the target instead of above
        if self.__dragged_groups and below_item:
            return False
        # If moved items are objects, refuse the drop if the target item is
        # a group and the drop indicator is anything but on the target item
        if self.__dragged_objects and target_item.parent() is None and not on_item:
            return False
        # If moved items are objects, refuse the drop if the target item is
        # the first group item and the drop position is above the target item
        if (
            self.__dragged_objects
            and target_item.parent() is None
            and self.indexFromItem(target_item).row() == 0
            and above_item
        ):
            return False
        return True

    def get_all_group_uuids(self) -> list[str]:
        """Return all group uuids, in a list ordered by group position in the tree"""
        return [
            self.topLevelItem(index).data(0, QC.Qt.UserRole)
            for index in range(self.topLevelItemCount())
        ]

    def get_all_object_uuids(self) -> dict[str, list[str]]:
        """Return all object uuids, in a dictionary that maps group uuids to the
        list of object uuids in each group, in the correct order"""
        return {
            group_id: [
                self.topLevelItem(index).child(idx).data(0, QC.Qt.UserRole)
                for idx in range(self.topLevelItem(index).childCount())
            ]
            for index, group_id in enumerate(self.get_all_group_uuids())
        }

    def dropEvent(self, event: QG.QDropEvent) -> None:  # pylint: disable=C0103
        """Reimplement Qt method"""
        if event.mimeData().hasUrls():
            fnames = [url.toLocalFile() for url in event.mimeData().urls()]
            self.SIG_IMPORT_FILES.emit(fnames)
            event.setDropAction(QC.Qt.CopyAction)
            event.accept()
        else:
            is_allowed = self.__is_drop_allowed(event)
            if not is_allowed:
                event.ignore()
            else:
                drop_pos = self.dropIndicatorPosition()
                on_viewport = drop_pos == QW.QAbstractItemView.OnViewport
                target_item = self.itemAt(event.pos())
                # If target item is None, it means that the drop position is
                # outside of the tree. In this case, we accept the drop and move
                # the objects to the end of the list.
                if target_item is None or on_viewport:
                    # If moved items are groups, move them to the end of the list
                    if self.__dragged_groups:
                        for item in self.__dragged_groups:
                            self.takeTopLevelItem(self.indexOfTopLevelItem(item))
                            self.addTopLevelItem(item)
                    # If moved items are objects, move them to the last group
                    if self.__dragged_objects:
                        lastgrp_item = self.topLevelItem(self.topLevelItemCount() - 1)
                        for item in self.__dragged_objects:
                            item.parent().removeChild(item)
                            lastgrp_item.addChild(item)

                    event.accept()
                else:
                    super().dropEvent(event)

            if event.isAccepted():
                # Ok, the drop was accepted, so we need to update the model accordingly
                # (at this stage, the model has not been updated yet but the tree has
                # been updated already, e.g. by the super().dropEvent(event) calls).
                # Thus, we have to loop over all tree items and reproduce the tree
                # structure in the model, by reordering the groups and objects.
                # We have two cases to consider (that mutually exclude each other):
                # 1. Groups are moved: we need to reorder the groups in the model
                # 2. Objects are moved: we need to reorder the objects in all groups
                #    in the model
                # Let's start with case 1:
                if self.__dragged_groups:
                    # First, we need to get the list of all groups in the model
                    # (in the correct order)
                    gids = self.get_all_group_uuids()
                    # Then, we need to reorder the groups in the model
                    self.objmodel.reorder_groups(gids)
                # Now, let's consider case 2:
                if self.__dragged_objects:
                    # First, we need to get a dictionary that maps group ids to
                    # the list of objects in each group (in the correct order)
                    oids = self.get_all_object_uuids()
                    # Then, we need to reorder the objects in all groups in the model
                    self.objmodel.reorder_objects(oids)
                # Finally, we need to update tree
                self.update_tree()
                # Restore expanded states of moved groups
                for item in self.__dragged_groups:
                    item.setExpanded(
                        self.__dragged_expanded_states[item.data(0, QC.Qt.UserRole)]
                    )
                # Restore selection, either of groups or objects
                sel_items = self.__dragged_groups or self.__dragged_objects
                extend = len(sel_items) > 1
                for item in sel_items:
                    if extend:
                        self.setCurrentItem(item, 0, QC.QItemSelectionModel.Select)
                    else:
                        self.setCurrentItem(item)

    def get_current_object(self) -> SignalObj | ImageObj | None:
        """Return current object"""
        oid = self.get_current_item_id(object_only=True)
        if oid is not None:
            return self.objmodel[oid]
        return None

    def set_current_object(self, obj: SignalObj | ImageObj) -> None:
        """Set current object"""
        self.set_current_item_id(obj.uuid)

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

    def select_objects(
        self,
        selection: list[SignalObj | ImageObj | int | str],
    ) -> None:
        """Select multiple objects

        Args:
            selection: list of objects, object numbers (1 to N) or object uuids
        """
        if all(isinstance(obj, int) for obj in selection):
            all_uuids = self.objmodel.get_object_ids()
            uuids = [all_uuids[num - 1] for num in selection]
        elif all(isinstance(obj, str) for obj in selection):
            uuids = selection
        else:
            assert all(isinstance(obj, (SignalObj, ImageObj)) for obj in selection)
            uuids = [obj.uuid for obj in selection]
        for idx, uuid in enumerate(uuids):
            self.set_current_item_id(uuid, extend=idx > 0)

    def select_groups(
        self, groups: list[ObjectGroup | int | str] | None = None
    ) -> None:
        """Select multiple groups

        Args:
            groups: list of groups, group numbers (1 to N), group names or None
             (select all groups). Defaults to None.
        """
        if groups is None:
            groups = self.objmodel.get_groups()
        elif all(isinstance(group, int) for group in groups):
            groups = [self.objmodel.get_groups()[grp_num - 1] for grp_num in groups]
        elif all(isinstance(group, str) for group in groups):
            groups = self.objmodel.get_groups(groups)
        assert all(isinstance(group, ObjectGroup) for group in groups)
        for idx, group in enumerate(groups):
            self.set_current_item_id(group.uuid, extend=idx > 0)

    def __reorder_model(self) -> None:
        """Reorder model"""
        self.objmodel.reorder_groups(self.get_all_group_uuids())
        self.objmodel.reorder_objects(self.get_all_object_uuids())
        self.update_tree()

    def move_up(self):
        """Move selected objects/groups up"""
        sel_objs = self.get_sel_object_items()
        sel_groups = self.get_sel_group_items()
        # Sort selected objects/groups by their position in the tree
        sel_objs.sort(key=lambda item: self.indexFromItem(item).row())
        sel_groups.sort(key=lambda item: self.indexFromItem(item).row())
        if not sel_objs and not sel_groups:
            return
        if sel_objs:
            for item in sel_objs:
                parent = item.parent()
                idx_item = parent.indexOfChild(item)
                idx_parent = self.indexOfTopLevelItem(parent)
                if idx_item > 0:
                    parent.takeChild(idx_item)
                    parent.insertChild(idx_item - 1, item)
                elif idx_parent > 0:
                    # If the object is the first child of its parent, we check if
                    # there is a group above the parent. If so, we move the object
                    # to the end of the group above.
                    parent.takeChild(idx_item)
                    self.topLevelItem(idx_parent - 1).addChild(item)
                else:
                    return
        else:
            # Store groups expanded state
            expstates = {
                item.data(0, QC.Qt.UserRole): item.isExpanded() for item in sel_groups
            }
            for item in sel_groups:
                idx_item = self.indexOfTopLevelItem(item)
                if idx_item > 0:
                    self.takeTopLevelItem(idx_item)
                    self.insertTopLevelItem(idx_item - 1, item)
                else:
                    return
            # Restore groups expanded state
            for item in sel_groups:
                item.setExpanded(expstates[item.data(0, QC.Qt.UserRole)])
        self.__reorder_model()
        # Restore selection
        for item in sel_objs + sel_groups:
            item.setSelected(True)

    def move_down(self):
        """Move selected objects/groups down"""
        sel_objs = self.get_sel_object_items()
        sel_groups = self.get_sel_group_items()
        # Sort selected objects/groups by their position in the tree
        sel_objs.sort(key=lambda item: self.indexFromItem(item).row(), reverse=True)
        sel_groups.sort(key=lambda item: self.indexFromItem(item).row(), reverse=True)
        if not sel_objs and not sel_groups:
            return
        if sel_objs:
            for item in sel_objs:
                parent = item.parent()
                idx_item = parent.indexOfChild(item)
                idx_parent = self.indexOfTopLevelItem(parent)
                if idx_item < parent.childCount() - 1:
                    parent.takeChild(idx_item)
                    parent.insertChild(idx_item + 1, item)
                elif idx_parent < self.topLevelItemCount() - 1:
                    # If the object is the last child of its parent, we check if
                    # there is a group below the parent. If so, we move the object
                    # to the beginning of the group below.
                    parent.takeChild(idx_item)
                    self.topLevelItem(idx_parent + 1).insertChild(0, item)
                else:
                    return
        else:
            # Store groups expanded state
            expstates = {
                item.data(0, QC.Qt.UserRole): item.isExpanded() for item in sel_groups
            }
            for item in sel_groups:
                idx_item = self.indexOfTopLevelItem(item)
                if idx_item < self.topLevelItemCount() - 1:
                    self.takeTopLevelItem(idx_item)
                    self.insertTopLevelItem(idx_item + 1, item)
                else:
                    return
            # Restore groups expanded state
            for item in sel_groups:
                item.setExpanded(expstates[item.data(0, QC.Qt.UserRole)])
        self.__reorder_model()
        # Restore selection
        for item in sel_objs + sel_groups:
            item.setSelected(True)
