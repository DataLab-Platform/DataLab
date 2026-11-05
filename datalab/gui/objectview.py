# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Object view
===========

The :mod:`datalab.gui.objectview` module provides widgets to display object
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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

from guidata.configtools import get_icon
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from sigima.objects import ImageObj, SignalObj

from datalab.config import Conf, _
from datalab.objectmodel import (
    ObjectGroup,
    find_title_references,
    get_title_reference,
    get_uuid,
    resolve_title_references,
)
from datalab.utils.qthelpers import block_signals, event_pos, view_item_option
from datalab.widgets.titledelegate import (
    TITLE_REFERENCE_ROLE,
    ClickableTitleDelegate,
)

if TYPE_CHECKING:
    from typing import Any

    from datalab.gui.panel.base import BaseDataPanel
    from datalab.objectmodel import ObjectModel


#: Side of the delete button shown at the right of the hovered item (pixels)
DELETE_BUTTON_SIZE = 20
#: Side of the icon of the delete button (pixels)
DELETE_BUTTON_ICON_SIZE = 14
#: Space between the delete button and the right edge of the viewport (pixels)
DELETE_BUTTON_MARGIN = 6
#: Space between the item title and the delete button (pixels)
DELETE_BUTTON_SPACING = 2
#: Corner radius of the delete button background (pixels)
DELETE_BUTTON_RADIUS = 3
#: Background colors of the delete button under the mouse cursor, and pressed
#: (the former is the "danger" color of DataLab-Web)
DELETE_BUTTON_HOVER_COLOR = "#c4302b"
DELETE_BUTTON_PRESSED_COLOR = "#9e2420"


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
                text = f"<b>{key}:</b> {vlines[0][:100]}"
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
        self.itemChanged.connect(lambda item: self.resize_title_column())
        self.title_delegate = ClickableTitleDelegate(
            self, self.get_title_reference_display
        )
        self.setItemDelegateForColumn(0, self.title_delegate)
        self.viewport().setMouseTracking(True)
        # Hover events let the view flag the hovered item, whatever the Qt style
        self.viewport().setAttribute(QC.Qt.WA_Hover, True)
        #: Hovered item, tracked from the same events as the view's own hover state
        self.__hover_index = QC.QPersistentModelIndex()
        #: Title reference indexes of the models (by model id), while a batch of
        #: items is updated (see :meth:`_title_reference_indexes`); None otherwise.
        self.__title_reference_indexes: dict[int, dict] | None = None

    @contextmanager
    def _title_reference_indexes(self) -> Iterator[None]:
        """Resolve title references from indexes built once while updating many
        items.

        Each item tooltip resolves the references of its title: without indexes,
        each resolution scans the whole model. The indexes are built on first
        use, for the objects present when the batch starts.
        """
        if self.__title_reference_indexes is not None:
            yield
            return
        self.__title_reference_indexes = {}
        try:
            yield
        finally:
            self.__title_reference_indexes = None

    def _find_title_reference(
        self, model: ObjectModel, reference: str
    ) -> list[SignalObj | ImageObj | ObjectGroup]:
        """Return the items of ``model`` matching ``reference``."""
        if self.__title_reference_indexes is None:
            return model.find_by_title_reference(reference)
        index = self.__title_reference_indexes.get(id(model))
        if index is None:
            index = model.get_title_reference_index()
            self.__title_reference_indexes[id(model)] = index
        return index.get(reference, [])

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

    def resolve_title_reference(
        self, reference: str
    ) -> tuple[str, SignalObj | ImageObj | ObjectGroup] | None:
        """Resolve a title reference to ``(panel_str, obj)``.

        Default implementation only looks up the tree's own model and returns
        an empty ``panel_str``. Subclasses with access to several panels
        should override this method.
        """
        matches = self._find_title_reference(self.objmodel, reference)
        return ("", matches[0]) if len(matches) == 1 else None

    def get_title_reference_display(self, reference: str) -> str | None:
        """Return the configured display text for a resolvable title reference."""
        resolved = self.resolve_title_reference(reference)
        if resolved is None:
            return None
        _panel_str, obj = resolved
        if Conf.result_title_mode.get() == "title":
            return resolve_title_references(obj.title, self.get_source_title)
        return reference

    def get_source_title(self, reference: str) -> str | None:
        """Return the title of the object or group a reference resolves to."""
        resolved = self.resolve_title_reference(reference)
        return None if resolved is None else resolved[1].title

    def build_title_reference_tooltip(self, text: str) -> str:
        """Return an HTML tooltip fragment listing the source objects
        referenced in ``text``, or an empty string when no such reference is
        found."""
        rows: list[str] = []
        seen: set[str] = set()
        for _start, _end, reference in find_title_references(text):
            if reference in seen:
                continue
            seen.add(reference)
            resolved = self.resolve_title_reference(reference)
            if resolved is None:
                continue
            panel_str, obj = resolved
            kind = (
                _("group")
                if isinstance(obj, ObjectGroup)
                else _("signal")
                if isinstance(obj, SignalObj)
                else _("image")
            )
            suffix = f" &middot; {panel_str}" if panel_str else ""
            rows.append(
                f"<b>{reference}</b> &rarr; {obj.title} <i>({kind}{suffix})</i>"
            )
        if not rows:
            return ""
        title = _("Source objects")
        return (
            f"<p style='white-space:pre'><i><u>{title}:</u></i><br>"
            f"{'<br>'.join(rows)}</p>"
        )

    def __update_item(
        self, item: QW.QTreeWidgetItem, obj: SignalObj | ImageObj | ObjectGroup
    ) -> None:
        """Update item"""
        text = obj.title
        item.setText(0, text)
        tooltip_parts: list[str] = []
        reference_tooltip = self.build_title_reference_tooltip(text)
        if reference_tooltip:
            tooltip_parts.append(reference_tooltip)
        if isinstance(obj, (SignalObj, ImageObj)):
            meta_tooltip = metadata_to_html(obj.metadata)
            if meta_tooltip:
                tooltip_parts.append(meta_tooltip)
        item.setToolTip(0, "".join(tooltip_parts))
        item.setData(0, QC.Qt.UserRole, get_uuid(obj))
        item.setData(0, TITLE_REFERENCE_ROLE, get_title_reference(obj))

    def handle_title_reference_click(self, reference: str) -> None:
        """Handle a click on a title-reference hyperlink. Default implementation
        selects the matching object within the tree's own model."""
        resolved = self.resolve_title_reference(reference)
        if resolved is None:
            return
        _panel_str, obj = resolved
        self.set_current_item_id(get_uuid(obj))

    def title_reference_at(self, pos: QC.QPoint) -> str | None:
        """Return the title reference under viewport position ``pos``, or ``None``
        when the cursor is not over an anchor."""
        index = self.indexAt(pos)
        if not index.isValid() or index.column() != 0:
            return None
        rect = self.visualRect(index)
        option = view_item_option(self)
        option.rect = rect
        return self.title_delegate.anchor_at(index, rect, pos, option)

    # pylint: disable=invalid-name
    def mousePressEvent(self, event: QG.QMouseEvent) -> None:
        """Reimplement Qt method to handle short-ID hyperlink clicks."""
        if event.button() == QC.Qt.LeftButton:
            reference = self.title_reference_at(event_pos(event))
            if reference is not None:
                event.accept()
                self.handle_title_reference_click(reference)
                return
        super().mousePressEvent(event)

    # pylint: disable=invalid-name
    def mouseMoveEvent(self, event: QG.QMouseEvent) -> None:
        """Reimplement Qt method to update the cursor over short-ID anchors."""
        reference = self.title_reference_at(event_pos(event))
        viewport = self.viewport()
        if reference is not None:
            viewport.setCursor(QC.Qt.PointingHandCursor)
        else:
            viewport.unsetCursor()
        super().mouseMoveEvent(event)

    def hovered_index(self) -> QC.QModelIndex:
        """Return the index of the item under the mouse cursor (invalid if none)"""
        return QC.QModelIndex(self.__hover_index)

    # pylint: disable=invalid-name
    def viewportEvent(self, event: QC.QEvent) -> bool:
        """Reimplement Qt method to track the hovered item"""
        etype = event.type()
        if etype in (QC.QEvent.HoverEnter, QC.QEvent.HoverMove):
            index = self.indexAt(event_pos(event))
            self.__hover_index = QC.QPersistentModelIndex(index)
        elif etype == QC.QEvent.HoverLeave:
            self.__hover_index = QC.QPersistentModelIndex()
        return super().viewportEvent(event)

    # pylint: disable=invalid-name
    def changeEvent(self, event: QC.QEvent) -> None:
        """Reimplement Qt method to keep hover events after a style change (some
        styles disable them when unpolishing item views)"""
        if event.type() == QC.QEvent.StyleChange:
            self.viewport().setAttribute(QC.Qt.WA_Hover, True)
        super().changeEvent(event)

    def populate_tree(self) -> None:
        """Populate tree with objects"""
        uuid = self.get_current_item_id()
        with block_signals(widget=self):
            self.clear()
        with self._title_reference_indexes():
            for group in self.objmodel.get_groups():
                self.add_group_item(group, resize=False)
        self.resize_title_column()
        if uuid is not None:
            self.set_current_item_id(uuid)

    def update_tree(self) -> None:
        """Update tree

        Note: If an item is not found in the tree, the tree is repopulated to ensure
        consistency between the tree and the model. This can happen in rare cases when
        objects are added to the model but the tree was not properly updated.
        """
        items = {}
        for item in self.iter_items():
            items.setdefault(item.data(0, QC.Qt.UserRole), item)
        # Items are updated without fitting the column once per changed item
        with self._title_reference_indexes(), block_signals(widget=self):
            complete = self.__update_items(items)
        if complete:
            self.resize_title_column()
        else:
            # Group or object item not found: repopulate tree to fix inconsistency
            self.populate_tree()

    def __update_items(self, items: dict[str, QW.QTreeWidgetItem]) -> bool:
        """Update the items of all groups and objects from the model.

        Returns:
            False if an item is missing from the tree (nothing more is updated).
        """
        for group in self.objmodel.get_groups():
            group_item = items.get(get_uuid(group))
            if group_item is None:
                return False
            self.__update_item(group_item, group)
            for obj in group:
                obj_item = items.get(get_uuid(obj))
                if obj_item is None:
                    return False
                self.__update_item(obj_item, obj)
        return True

    def resize_title_column(
        self, items: Sequence[QW.QTreeWidgetItem] | None = None
    ) -> None:
        """Fit the title column to the titles.

        Args:
            items: Items just added to a tree whose column was already fitted to
             the other items: the column is only widened if one of them needs it,
             without measuring every title again. All items when None.
        """
        with self._title_reference_indexes():
            if items is None:
                self.resizeColumnToContents(0)
                return
            width = max((self.__title_width(item) for item in items), default=0)
        if width > self.columnWidth(0):
            self.setColumnWidth(0, width)

    def __title_width(self, item: QW.QTreeWidgetItem) -> int:
        """Return the width needed by *item* in the title column (indentation
        included, as computed by :meth:`resizeColumnToContents`)."""
        depth = 0
        parent = item.parent()
        while parent is not None:
            depth += 1
            parent = parent.parent()
        if self.rootIsDecorated():
            depth += 1
        hint = self.sizeHintForIndex(self.indexFromItem(item, 0))
        return hint.width() + depth * self.indentation()

    def __add_to_group_item(
        self, obj: SignalObj | ImageObj, group_item: QW.QTreeWidgetItem
    ) -> QW.QTreeWidgetItem:
        """Add object to group item and return its item"""
        item = QW.QTreeWidgetItem()
        icon = (
            "sigimax:signal.svg" if isinstance(obj, SignalObj) else "sigimax:image.svg"
        )
        item.setIcon(0, get_icon(icon))
        self.__update_item(item, obj)
        group_item.addChild(item)
        return item

    def add_group_item(self, group: ObjectGroup, resize: bool = True) -> None:
        """Add group item

        Args:
            group: Group to add, with its objects
            resize: If True, fit the title column to the new items
        """
        group_item = QW.QTreeWidgetItem()
        group_item.setIcon(0, get_icon("group.svg"))
        self.__update_item(group_item, group)
        self.addTopLevelItem(group_item)
        group_item.setExpanded(True)
        items = [group_item]
        for obj in group:
            items.append(self.__add_to_group_item(obj, group_item))
        if resize:
            self.resize_title_column(items)

    def add_object_item(
        self, obj: SignalObj | ImageObj, group_id: str, set_current: bool = True
    ) -> None:
        """Add item"""
        self.add_object_items((obj,), group_id, set_current)

    def add_object_items(
        self,
        objects: Sequence[SignalObj | ImageObj],
        group_id: str,
        set_current: bool = True,
    ) -> None:
        """Add multiple object items with a single tree update cycle.

        Args:
            objects: Objects to add, in insertion order
            group_id: UUID of the parent group
            set_current: If True, set the last object as current
        """
        objects = tuple(objects)
        if not objects:
            return
        group_item = self.get_item_from_id(group_id)
        if group_item is None:
            raise KeyError(f"Group with uuid '{group_id}' not found in tree")
        signals_blocked = self.blockSignals(True)
        updates_enabled = self.updatesEnabled()
        self.setUpdatesEnabled(False)
        try:
            with self._title_reference_indexes():
                items = [self.__add_to_group_item(obj, group_item) for obj in objects]
            if set_current:
                self.set_current_item_id(get_uuid(objects[-1]))
        finally:
            self.setUpdatesEnabled(updates_enabled)
            self.blockSignals(signals_blocked)
        # The column fits the existing items: only the new ones are measured
        self.resize_title_column(items)

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
    """Object handling panel list widget, object (sig/ima) lists

    A delete button is shown at the right of the hovered item (signal, image or
    group), unless disabled (see :meth:`set_delete_button_enabled`): clicking it
    emits :attr:`SIG_DELETE_REQUESTED` with the item UUID.
    """

    SIG_SELECTION_CHANGED = QC.Signal()
    SIG_IMPORT_FILES = QC.Signal(list)
    SIG_DELETE_REQUESTED = QC.Signal(str)

    def __init__(self, parent: BaseDataPanel, objmodel: ObjectModel) -> None:
        super().__init__(parent, objmodel)
        self.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QW.QAbstractItemView.InternalMove)
        self.itemSelectionChanged.connect(self.item_selection_changed)
        self.__dragged_objects: list[QW.QListWidgetItem] = []
        self.__dragged_groups: list[QW.QListWidgetItem] = []
        self.__dragged_expanded_states: dict[QW.QListWidgetItem, bool] = {}
        self.__delete_button_enabled = Conf.show_delete_button_on_hover.get()
        self.__delete_icon = get_icon("datalab:delete.svg")
        #: True when the mouse cursor is over the delete button of the hovered item
        self.__delete_button_hovered = False
        #: Item whose delete button is pressed, None when no button is pressed
        self.__delete_button_pressed: QC.QPersistentModelIndex | None = None
        self.title_delegate.text_clip_provider = self.__title_clip_right
        # The button sticks to the right edge of the viewport: scrolling
        # horizontally must not move its pixels with the items
        self.horizontalScrollBar().valueChanged.connect(self.__repaint_viewport)

    def set_delete_button_enabled(self, enabled: bool) -> None:
        """Show or hide the delete button of the hovered item

        Args:
            enabled: True to show the button
        """
        self.__delete_button_enabled = enabled
        self.__delete_button_hovered = False
        self.__delete_button_pressed = None
        self.viewport().update()

    def is_delete_button_enabled(self) -> bool:
        """Return True if the delete button of the hovered item is shown"""
        return self.__delete_button_enabled

    def delete_button_rect(self, index: QC.QModelIndex) -> QC.QRect | None:
        """Return the rectangle of the delete button of an item

        The button is shown on the hovered item only, at the right edge of the
        visible part of the viewport (the title column may be wider than the
        viewport).

        Args:
            index: model index of the item

        Returns:
            Button rectangle in viewport coordinates, or None if the item shows
            no delete button.
        """
        if (
            not self.__delete_button_enabled
            or not index.isValid()
            or index.sibling(index.row(), 0) != self.hovered_index()
            or self.state()
            not in (QW.QAbstractItemView.NoState, QW.QAbstractItemView.AnimatingState)
        ):
            return None
        item_rect = self.visualRect(index)
        size = DELETE_BUTTON_SIZE
        return QC.QRect(
            self.viewport().width() - DELETE_BUTTON_MARGIN - size,
            item_rect.top() + (item_rect.height() - size) // 2,
            size,
            size,
        )

    def get_delete_targets(self, uuid: str) -> list[str]:
        """Return the UUIDs of the items deleted by the delete button of an item

        Args:
            uuid: UUID of the item whose delete button is clicked

        Returns:
            UUIDs of the selected objects or groups if the item is part of the
            selection, else the item UUID only.
        """
        selection = self.get_sel_object_uuids() + self.get_sel_group_uuids()
        return selection if uuid in selection else [uuid]

    def delete_button_tooltip(self, index: QC.QModelIndex) -> str:
        """Return the tooltip of the delete button of an item

        Args:
            index: model index of the item

        Returns:
            Tooltip text
        """
        uuid = index.data(QC.Qt.UserRole)
        targets = self.get_delete_targets(uuid)
        if len(targets) > 1:
            if isinstance(self.objmodel.get_object_or_group(targets[0]), ObjectGroup):
                return _("Delete the %d selected groups") % len(targets)
            return _("Delete the %d selected objects") % len(targets)
        obj = self.objmodel.get_object_or_group(uuid)
        if isinstance(obj, ObjectGroup):
            return _("Delete this group")
        if isinstance(obj, SignalObj):
            return _("Delete this signal")
        return _("Delete this image")

    def __delete_button_index_at(self, pos: QC.QPoint) -> QC.QModelIndex | None:
        """Return the index of the item whose delete button is at viewport
        position ``pos``, or None"""
        index = self.indexAt(pos)
        rect = self.delete_button_rect(index)
        if rect is not None and rect.contains(pos):
            return index
        return None

    def __title_clip_right(self, index: QC.QModelIndex) -> int | None:
        """Return the viewport X coordinate beyond which the title of an item must
        not be drawn, to leave room for its delete button (None if no button)"""
        rect = self.delete_button_rect(index)
        return None if rect is None else rect.left() - DELETE_BUTTON_SPACING

    def __update_row(self, index: QC.QModelIndex) -> None:
        """Repaint the row of an item, over the whole viewport width"""
        if index.isValid():
            rect = self.visualRect(index)
            width = self.viewport().width()
            self.viewport().update(QC.QRect(0, rect.top(), width, rect.height()))

    def __repaint_viewport(self) -> None:
        """Repaint the whole viewport"""
        self.viewport().update()

    def __set_delete_button_hovered(self, hovered: bool) -> None:
        """Set whether the mouse cursor is over the delete button"""
        if hovered != self.__delete_button_hovered:
            self.__delete_button_hovered = hovered
            self.__update_row(self.hovered_index())

    def __paint_delete_button(self, painter: QG.QPainter, rect: QC.QRect) -> None:
        """Paint the delete button in ``rect``"""
        painter.save()
        if self.__delete_button_hovered:
            pressed = self.__delete_button_pressed is not None
            color = (
                DELETE_BUTTON_PRESSED_COLOR if pressed else DELETE_BUTTON_HOVER_COLOR
            )
            painter.setRenderHint(QG.QPainter.Antialiasing)
            painter.setPen(QC.Qt.NoPen)
            painter.setBrush(QG.QColor(color))
            radius = DELETE_BUTTON_RADIUS
            painter.drawRoundedRect(QC.QRectF(rect), radius, radius)
        icon_rect = QC.QRect(0, 0, DELETE_BUTTON_ICON_SIZE, DELETE_BUTTON_ICON_SIZE)
        icon_rect.moveCenter(rect.center())
        self.__delete_icon.paint(painter, icon_rect)
        painter.restore()

    # pylint: disable=invalid-name
    def drawRow(
        self,
        painter: QG.QPainter,
        option: QW.QStyleOptionViewItem,
        index: QC.QModelIndex,
    ) -> None:
        """Reimplement Qt method to draw the delete button of the hovered item"""
        super().drawRow(painter, option, index)
        rect = self.delete_button_rect(index)
        if rect is not None:
            self.__paint_delete_button(painter, rect)

    # pylint: disable=invalid-name
    def viewportEvent(self, event: QC.QEvent) -> bool:
        """Reimplement Qt method to show the tooltip of the delete button and
        follow the mouse cursor over it"""
        etype = event.type()
        if etype == QC.QEvent.ToolTip:
            index = self.__delete_button_index_at(event.pos())
            if index is not None:
                QW.QToolTip.showText(
                    event.globalPos(),
                    self.delete_button_tooltip(index),
                    self.viewport(),
                    self.delete_button_rect(index),
                )
                return True
        result = super().viewportEvent(event)
        if etype in (QC.QEvent.HoverEnter, QC.QEvent.HoverMove):
            index = self.__delete_button_index_at(event_pos(event))
            self.__set_delete_button_hovered(index is not None)
        elif etype == QC.QEvent.HoverLeave:
            self.__set_delete_button_hovered(False)
        return result

    # pylint: disable=invalid-name
    def mousePressEvent(self, event: QG.QMouseEvent) -> None:
        """Reimplement Qt method to press the delete button, without selecting
        or dragging the item"""
        if event.button() == QC.Qt.LeftButton:
            index = self.__delete_button_index_at(event_pos(event))
            if index is not None:
                self.__delete_button_pressed = QC.QPersistentModelIndex(index)
                self.__update_row(index)
                event.accept()
                return
        super().mousePressEvent(event)

    # pylint: disable=invalid-name
    def mouseMoveEvent(self, event: QG.QMouseEvent) -> None:
        """Reimplement Qt method to follow the mouse cursor over the delete
        button"""
        pos = event_pos(event)
        if self.__delete_button_pressed is not None:
            # The press started on a delete button: neither select nor drag items
            rect = self.delete_button_rect(QC.QModelIndex(self.__delete_button_pressed))
            self.__set_delete_button_hovered(rect is not None and rect.contains(pos))
            event.accept()
            return
        super().mouseMoveEvent(event)
        if self.__delete_button_index_at(pos) is not None:
            self.viewport().setCursor(QC.Qt.PointingHandCursor)

    # pylint: disable=invalid-name
    def mouseReleaseEvent(self, event: QG.QMouseEvent) -> None:
        """Reimplement Qt method to click the delete button"""
        if self.__delete_button_pressed is None or event.button() != QC.Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        pressed = QC.QModelIndex(self.__delete_button_pressed)
        self.__delete_button_pressed = None
        event.accept()
        self.__update_row(pressed)
        if pressed.isValid() and self.__delete_button_index_at(event_pos(event)) == (
            pressed
        ):
            self.SIG_DELETE_REQUESTED.emit(pressed.data(QC.Qt.UserRole))

    # pylint: disable=invalid-name
    def mouseDoubleClickEvent(self, event: QG.QMouseEvent) -> None:
        """Reimplement Qt method to ignore double-clicks on the delete button
        (which would open the item in a separate view)"""
        if self.__delete_button_index_at(event_pos(event)) is not None:
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resolve_title_reference(
        self, reference: str
    ) -> tuple[str, SignalObj | ImageObj | ObjectGroup] | None:
        """Resolve a title reference across the signal and image panels of the main
        window, so titles can reference objects living in either panel.
        """
        panel: BaseDataPanel = self.parent()
        mainwindow = getattr(panel, "mainwindow", None)
        if mainwindow is None:
            return super().resolve_title_reference(reference)
        matches = [
            (other.PANEL_STR_ID, obj)
            for other in (mainwindow.signalpanel, mainwindow.imagepanel)
            for obj in self._find_title_reference(other.objmodel, reference)
        ]
        return matches[0] if len(matches) == 1 else None

    def handle_title_reference_click(self, reference: str) -> None:
        """Select the referenced object, switching active panel if needed."""
        resolved = self.resolve_title_reference(reference)
        if resolved is None:
            return
        panel_str, obj = resolved
        panel: BaseDataPanel = self.parent()
        mainwindow = getattr(panel, "mainwindow", None)
        if mainwindow is not None and panel_str and panel_str != panel.PANEL_STR_ID:
            mainwindow.set_current_panel(panel_str)
            mainwindow.select_objects([get_uuid(obj)], panel=panel_str)
        else:
            self.set_current_item_id(get_uuid(obj))

    def paintEvent(self, event):  # pylint: disable=C0103
        """Reimplement Qt method"""
        super().paintEvent(event)
        if len(self.objmodel) > 0:
            return

        # Draw empty state message with contextual hints
        painter = QG.QPainter(self.viewport())
        rect = self.rect()

        # Set font for main message
        font = painter.font()
        font.setPointSize(11)
        painter.setFont(font)

        # Main message
        main_msg = _("Drag files here to open")
        main_rect = QC.QRect(rect)
        main_rect.setHeight(rect.height() // 2)
        painter.drawText(main_rect, QC.Qt.AlignCenter, main_msg)

        # Contextual hint
        hint_rect = QC.QRect(rect)
        hint_rect.setTop(rect.height() // 2)
        font.setPointSize(9)
        font.setItalic(True)
        painter.setFont(font)
        painter.setPen(QG.QColor(128, 128, 128))  # Gray color

        panel: BaseDataPanel = self.parent()
        if panel.PANEL_STR_ID == "signal":
            hint = _(
                "Working with 2D images?\n"
                "Switch to the Image Panel using the tab above."
            )
        else:
            hint = _(
                "Working with 1D signals?\n"
                "Switch to the Signal Panel using the tab above."
            )
        painter.drawText(hint_rect, QC.Qt.AlignCenter, hint)

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

    def get_current_object_uuid(self) -> str | None:
        """Return current object uuid, or None if current item is a group
        or if no item is current"""
        return self.get_current_item_id(object_only=True)

    def set_current_object(self, obj: SignalObj | ImageObj) -> None:
        """Set current object"""
        self.set_current_item_id(get_uuid(obj))

    def select_uuids(self, uuids: Sequence[str], notify: bool = True) -> None:
        """Select objects or groups atomically, the last one becoming current

        Args:
            uuids: UUIDs of the objects or groups to select
            notify: if True, publish the complete final state once (see
             :meth:`item_selection_changed`)
        """
        if not uuids:
            return
        signals_blocked = self.blockSignals(True)
        try:
            for idx, uuid in enumerate(uuids):
                self.set_current_item_id(uuid, extend=idx > 0)
        finally:
            self.blockSignals(signals_blocked)
        if notify:
            self.item_selection_changed()

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
            uuids = [get_uuid(obj) for obj in selection]
        self.select_uuids(uuids)

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
        self.select_uuids([get_uuid(group) for group in groups])

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
