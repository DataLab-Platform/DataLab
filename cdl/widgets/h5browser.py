# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab HDF5 browser module
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import os
import os.path as osp
from typing import TYPE_CHECKING, Any

from guidata.qthelpers import (
    add_actions,
    create_action,
    create_toolbutton,
    get_icon,
    win32_fix_title_bar_background,
)
from plotpy.plot import PlotOptions, PlotWidget
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from cdl.config import _
from cdl.core.io.h5 import H5Importer
from cdl.core.model.signal import CurveStyles
from cdl.env import execenv
from cdl.obj import ImageObj, SignalObj
from cdl.utils.qthelpers import qt_handle_error_message
from cdl.utils.strings import to_string

if TYPE_CHECKING:
    from cdl.core.io.h5.common import BaseNode


class AbstractTreeWidgetMeta(type(QW.QTreeWidget), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class AbstractTreeWidget(QW.QTreeWidget, metaclass=AbstractTreeWidgetMeta):
    """One-column tree widget with context menu, ..."""

    def __init__(self, parent: QW.QWidget) -> None:
        super().__init__(parent)
        self.setItemsExpandable(True)
        self.itemActivated.connect(self.activated)
        self.itemClicked.connect(self.clicked)
        # Setup context menu
        self.menu = QW.QMenu(self)
        self.collapse_all_action = None
        self.collapse_selection_action = None
        self.expand_all_action = None
        self.expand_selection_action = None
        self.common_actions = self.setup_common_actions()

        self.itemSelectionChanged.connect(self.item_selection_changed)
        self.item_selection_changed()

    @abc.abstractmethod
    def activated(self, item: QW.QTreeWidgetItem) -> None:
        """Double-click event"""

    @abc.abstractmethod
    def clicked(self, item: QW.QTreeWidgetItem) -> None:
        """Item was clicked"""

    @abc.abstractmethod
    def get_actions_from_items(
        self, items: list[QW.QTreeWidgetItem]
    ) -> list[QW.QAction]:
        """Get actions from item"""
        # Right here: add other actions if necessary (reimplement this method)
        return []

    def setup_common_actions(self) -> list[QW.QAction]:
        """Setup context menu common actions"""
        self.collapse_all_action = create_action(
            self,
            _("Collapse all"),
            icon=get_icon("collapse.svg"),
            triggered=self.collapseAll,
        )
        self.expand_all_action = create_action(
            self, _("Expand all"), icon=get_icon("expand.svg"), triggered=self.expandAll
        )
        self.restore_action = create_action(
            self,
            _("Restore"),
            tip=_("Restore original tree layout"),
            icon=get_icon("restore.svg"),
            triggered=self.restore,
        )
        self.collapse_selection_action = create_action(
            self,
            _("Collapse selection"),
            icon=get_icon("collapse_selection.svg"),
            triggered=self.collapse_selection,
        )
        self.expand_selection_action = create_action(
            self,
            _("Expand selection"),
            icon=get_icon("expand_selection.svg"),
            triggered=self.expand_selection,
        )
        return [
            self.collapse_all_action,
            self.expand_all_action,
            self.restore_action,
            None,
            self.collapse_selection_action,
            self.expand_selection_action,
        ]

    def update_menu(self) -> None:
        """Update context menu"""
        self.menu.clear()
        items = self.selectedItems()
        actions = self.get_actions_from_items(items)
        if actions:
            actions.append(None)
        actions += self.common_actions
        add_actions(self.menu, actions)

    def restore(self) -> None:
        """Restore tree state"""
        self.collapseAll()
        for item in self.get_top_level_items():
            self.expandItem(item)

    def __expand_item(self, item: QW.QTreeWidgetItem) -> None:  # pragma: no cover
        """Expand item tree branch"""
        self.expandItem(item)
        for index in range(item.childCount()):
            child = item.child(index)
            self.__expand_item(child)

    def expand_selection(self) -> None:  # pragma: no cover
        """Expand selection"""
        items = self.selectedItems()
        if not items:
            items = self.get_top_level_items()
        for item in items:
            self.__expand_item(item)
        if items:
            self.scrollToItem(items[0])

    def __collapse_item(self, item: QW.QTreeWidgetItem) -> None:  # pragma: no cover
        """Collapse item tree branch"""
        self.collapseItem(item)
        for index in range(item.childCount()):
            child = item.child(index)
            self.__collapse_item(child)

    def collapse_selection(self) -> None:  # pragma: no cover
        """Collapse selection"""
        items = self.selectedItems()
        if not items:
            items = self.get_top_level_items()
        for item in items:
            self.__collapse_item(item)
        if items:
            self.scrollToItem(items[0])

    def item_selection_changed(self) -> None:
        """Item selection has changed"""
        is_selection = len(self.selectedItems()) > 0
        self.expand_selection_action.setEnabled(is_selection)
        self.collapse_selection_action.setEnabled(is_selection)

    def get_top_level_items(self) -> list[QW.QTreeWidgetItem]:
        """Iterate over top level items"""
        return [self.topLevelItem(_i) for _i in range(self.topLevelItemCount())]

    def get_items(self) -> list[QW.QTreeWidgetItem]:
        """Return items (excluding top level items)"""
        itemlist = []

        def add_to_itemlist(item: QW.QTreeWidgetItem):
            for index in range(item.childCount()):
                citem = item.child(index)
                itemlist.append(citem)
                add_to_itemlist(citem)

        for tlitem in self.get_top_level_items():
            add_to_itemlist(tlitem)
        return itemlist

    def find_all_items(self):
        """Find all items"""
        return self.findItems("", QC.Qt.MatchContains | QC.Qt.MatchRecursive)

    def contextMenuEvent(self, event: QG.QContextMenuEvent) -> None:
        """Override Qt method"""
        self.update_menu()
        self.menu.popup(event.globalPos())


class H5TreeWidget(AbstractTreeWidget):
    """HDF5 Browser Tree Widget

    Args:
        parent: Parent widget
    """

    SIG_SELECTED = QC.Signal(QW.QTreeWidgetItem)

    def __init__(self, parent: QW.QWidget) -> None:
        super().__init__(parent)
        title = _("HDF5 Browser")
        self.setColumnCount(4)
        self.setWindowTitle(title)
        self.setHeaderLabels([_("Name"), _("Size"), _("Type"), _("Value")])
        self.header().setSectionResizeMode(0, QW.QHeaderView.Stretch)
        self.header().setStretchLastSection(False)
        self.fname = None
        self.h5importer = None

    def setup(self, fname: str) -> None:
        """Setup H5TreeWidget

        Args:
            fname: HDF5 file name
        """
        self.fname = osp.abspath(fname)
        self.h5importer = H5Importer(self.fname)
        self.clear()
        self.populate_tree()
        self.expandAll()
        for col in range(1, 4):
            self.resizeColumnToContents(col)

    def cleanup(self) -> None:
        """Clean up widget"""
        self.h5importer.close()
        self.h5importer = None

    def get_node(self, item: QW.QTreeWidgetItem) -> BaseNode:
        """Get HDF5 dataset associated to item

        Args:
            item: Tree item

        Returns:
            HDF5 node
        """
        node_id = item.data(0, QC.Qt.UserRole)
        if node_id:
            return self.h5importer.get(node_id)
        return None

    def get_nodes(self, only_checked_items: bool = True) -> list[BaseNode]:
        """Get all nodes associated to checked items

        Args:
            only_checked_items: If True, only checked items are returned

        Returns:
            List of HDF5 nodes
        """
        datasets = []
        for item in self.find_all_items():
            if item.flags() & QC.Qt.ItemIsUserCheckable:
                if only_checked_items and item.checkState(0) == 0:
                    continue
                if item is not self.topLevelItem(0):
                    node_id = item.data(0, QC.Qt.UserRole)
                    datasets.append(self.h5importer.get(node_id))
        return datasets

    def activated(self, item: QW.QTreeWidgetItem) -> None:
        """Double-click event"""
        if item is not self.topLevelItem(0):
            self.SIG_SELECTED.emit(item)

    def clicked(self, item: QW.QTreeWidgetItem) -> None:
        """Click event"""
        self.activated(item)

    def get_actions_from_items(self, items):  # pylint: disable=W0613
        """Get actions from item"""
        return []

    def is_empty(self) -> bool:
        """Return True if tree is empty"""
        return len(self.find_all_items()) == 1

    def is_any_item_checked(self) -> bool:
        """Return True if any item is checked"""
        for item in self.find_all_items():
            if item.checkState(0) > 0:
                return True
        return False

    def select_all(self, state: bool) -> None:
        """Select all items

        Args:
            state: If True, all items are selected
        """
        for item in self.findItems("", QC.Qt.MatchContains | QC.Qt.MatchRecursive):
            if item.flags() & QC.Qt.ItemIsUserCheckable:
                item.setSelected(state)
                if state:
                    self.clicked(item)

    def toggle_all(self, state: bool) -> None:
        """Toggle all item state from 'unchecked' to 'checked'
        (or vice-versa)

        Args:
            state: If True, all items are checked
        """
        for item in self.findItems("", QC.Qt.MatchContains | QC.Qt.MatchRecursive):
            if item.flags() & QC.Qt.ItemIsUserCheckable:
                item.setCheckState(0, QC.Qt.Checked if state else QC.Qt.Unchecked)

    @staticmethod
    def __create_node(node: BaseNode) -> QW.QTreeWidgetItem:
        """Create tree node from HDF5 node

        Args:
            node: HDF5 node

        Returns:
            Tree widget node
        """
        text = to_string(node.text)
        if len(text) > 10:
            text = text[:10] + "..."
        treeitem = QW.QTreeWidgetItem([node.name, node.shape_str, node.dtype_str, text])
        treeitem.setData(0, QC.Qt.UserRole, node.id)
        if node.description:
            for col in range(treeitem.columnCount()):
                treeitem.setToolTip(col, node.description)
        return treeitem

    @staticmethod
    def __recursive_popfunc(parent_item: QW.QTreeWidgetItem, node: BaseNode) -> None:
        """Recursive HDF5 analysis

        Args:
            parent_item: Parent tree item
            node: HDF5 node
        """
        tree_item = H5TreeWidget.__create_node(node)
        if node.IS_ARRAY:
            tree_item.setCheckState(0, QC.Qt.Unchecked)
        else:
            tree_item.setFlags(QC.Qt.ItemIsEnabled)
        tree_item.setIcon(0, get_icon(node.icon_name))
        parent_item.addChild(tree_item)
        for child in node.children:
            H5TreeWidget.__recursive_popfunc(tree_item, child)

    def populate_tree(self) -> None:
        """Populate tree"""
        root = self.h5importer.root
        rootitem = QW.QTreeWidgetItem([root.name])
        rootitem.setToolTip(0, root.description)
        rootitem.setData(0, QC.Qt.UserRole, root.id)
        rootitem.setFlags(QC.Qt.ItemIsEnabled)
        rootitem.setIcon(0, get_icon(root.icon_name))
        self.addTopLevelItem(rootitem)
        for node in root.children:
            self.__recursive_popfunc(rootitem, node)


class VisualPreview(QW.QStackedWidget):
    """Visual preview"""

    def __init__(self, parent: QW.QWidget) -> None:
        super().__init__(parent)
        self.curvewidget = PlotWidget(self, options=PlotOptions(type="curve"))
        self.addWidget(self.curvewidget)
        self.imagewidget = PlotWidget(
            self, options=PlotOptions(type="image", show_contrast=True)
        )
        self.addWidget(self.imagewidget)

    def cleanup(self) -> None:
        """Clean up widget"""
        for widget in (self.imagewidget, self.curvewidget):
            widget.get_plot().del_all_items()

    def update_visual_preview(self, node: BaseNode) -> None:
        """Update visual preview widget"""
        try:
            obj = node.get_native_object()
        except Exception as msg:  # pylint: disable=broad-except
            qt_handle_error_message(self, msg)
            return
        if obj is None:
            # An error occurred while creating the object (invalid data, ...)
            return
        if isinstance(obj, SignalObj):
            obj: SignalObj
            widget = self.curvewidget
            CurveStyles.reset_styles()
        else:
            obj: ImageObj
            widget = self.imagewidget
        item = obj.make_item()
        plot = widget.get_plot()
        plot.del_all_items()
        plot.add_item(item)
        plot.set_active_item(item)
        item.unselect()
        plot.do_autoscale()

        # FIXME: This is strange: why do we need to update the item here?
        #        It reveals a design flaw in the way we handle items: we should
        #        not have to update the item just after adding it to the plot.
        #        The `make_item` method should return an item that is ready to
        #        be added to the plot.
        obj.update_item(item)

        self.setCurrentWidget(widget)


class TablePreview(QW.QWidget):
    """Table preview

    Args:
        title: Group title
        parent: Parent widget
    """

    def __init__(self, parent: QW.QWidget) -> None:
        super().__init__(parent)
        self.setLayout(QW.QVBoxLayout())
        self.table = QW.QTableWidget(self)
        self.table.setEditTriggers(QW.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.layout().addWidget(self.table)

    def clear(self) -> None:
        """Clear table"""
        self.table.clear()

    def update_table_preview(self, data: dict[str, Any]) -> None:
        """Update table preview widget

        Args:
            node: HDF5 node
        """
        self.clear()
        self.table.setRowCount(len(data))
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels([_("Value")])
        self.table.setVerticalHeaderLabels(list(data.keys()))
        for row, value in enumerate(data.values()):
            self.table.setItem(row, 0, QW.QTableWidgetItem(str(value)))
        self.table.resizeRowsToContents()


class GroupAndAttributes(QW.QTabWidget):
    """Group and attributes"""

    def __init__(self, parent: QW.QWidget) -> None:
        super().__init__(parent)
        self.group = TablePreview(self)
        self.addTab(self.group, get_icon("h5group.svg"), _("Group"))
        self.attrs = TablePreview(self)
        self.addTab(self.attrs, get_icon("h5attrs.svg"), _("Attributes"))

    def cleanup(self) -> None:
        """Clean up widget"""
        self.group.clear()
        self.attrs.clear()

    def update_group(self, node: BaseNode) -> None:
        """Update group widget

        Args:
            node: HDF5 node
        """
        text = to_string(node.text)
        if text:
            lines = text.splitlines()[:5]
            if len(lines) == 5:
                lines += ["[...]"]
            text = os.linesep.join(lines)
        data = {
            _("Path"): node.id,
            _("Name"): node.name,
            _("Description"): node.description,
            _("Textual preview"): text,
            # "Raw": repr(node.data),
        }
        self.group.update_table_preview(data)

    def update_attrs(self, node: BaseNode) -> None:
        """Update attributes widget

        Args:
            node: HDF5 node
        """
        self.attrs.update_table_preview(node.metadata)


class H5Browser(QW.QSplitter):
    """HDF5 Browser Widget

    Args:
        parent: Parent widget
    """

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(parent)
        self.tree = H5TreeWidget(self)
        self.tree.SIG_SELECTED.connect(self.view_selected_item)
        self.addWidget(self.tree)
        preview = QW.QSplitter(self)
        preview.setOrientation(QC.Qt.Vertical)
        self.addWidget(preview)
        self.visualpreview = VisualPreview(self)
        preview.addWidget(self.visualpreview)
        self.groupandattrs = GroupAndAttributes(self)
        preview.addWidget(self.groupandattrs)
        preview.setSizes([int(self.size().height() / 2)] * 2)

    def setup(self, fname: str) -> None:
        """Setup widget

        Args:
            fname: HDF5 file name
        """
        self.tree.setup(fname)

    def cleanup(self) -> None:
        """Clean up widget"""
        self.tree.cleanup()
        self.visualpreview.cleanup()

    def get_node(self, item: QW.QTreeWidgetItem | None = None) -> BaseNode:
        """Return (selected) dataset

        Args:
            item: Tree item

        Returns:
            HDF5 node
        """
        if item is None:
            item = self.tree.currentItem()
        return self.tree.get_node(item)

    def view_selected_item(self, item: QW.QTreeWidgetItem) -> None:
        """View selected item

        Args:
            item: Tree item
        """
        node = self.get_node(item)
        if node.IS_ARRAY:
            self.visualpreview.update_visual_preview(node)
        self.groupandattrs.update_group(node)
        self.groupandattrs.update_attrs(node)


class H5BrowserDialog(QW.QDialog):
    """HDF5 Browser Dialog

    Args:
        parent: Parent widget
        size: Dialog size
    """

    def __init__(
        self, parent: QW.QWidget | None = None, size: tuple[int, int] = (1280, 720)
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(QC.Qt.Window)
        self.setObjectName("h5browser")
        self.setWindowTitle(_("HDF5 Browser"))
        self.setWindowIcon(get_icon("h5browser.svg"))
        win32_fix_title_bar_background(self)
        vlayout = QW.QVBoxLayout()
        self.setLayout(vlayout)
        self.button_layout = None
        self.bbox = None
        self.nodes = None

        self.browser = H5Browser(self)
        vlayout.addWidget(self.browser)

        self.browser.tree.itemChanged.connect(lambda item: self.refresh_buttons())

        self.install_button_layout()

        self.setMinimumSize(QC.QSize(900, 500))
        self.resize(QC.QSize(*size))
        self.browser.setSizes([int(self.size().height() / 2)] * 2)
        self.refresh_buttons()

    def accept(self) -> None:
        """Accept changes"""
        self.nodes = self.browser.tree.get_nodes()
        QW.QDialog.accept(self)

    def cleanup(self) -> None:
        """Cleanup dialog"""
        self.browser.cleanup()

    def refresh_buttons(self) -> None:
        """Refresh buttons"""
        state = self.browser.tree.is_any_item_checked()
        self.bbox.button(QW.QDialogButtonBox.Ok).setEnabled(state)

    def setup(self, fname: str) -> None:
        """Setup dialog

        Args:
            fname: HDF5 file name
        """
        self.browser.setup(fname)
        if self.browser.tree.is_empty():
            if not execenv.unattended:
                QW.QMessageBox.warning(
                    self.parent(),
                    self.windowTitle(),
                    _("Warning:")
                    + "\n"
                    + _("No supported data available in HDF5 file."),
                )
            QC.QTimer.singleShot(0, self.reject)

    def get_all_nodes(self) -> list[BaseNode]:
        """Return all supported datasets

        Returns:
            List of HDF5 nodes
        """
        return self.browser.tree.get_nodes(only_checked_items=False)

    def get_nodes(self) -> list[BaseNode]:
        """Return datasets

        Returns:
            List of HDF5 nodes
        """
        return self.nodes

    def install_button_layout(self) -> None:
        """Install button layout"""
        bbox = QW.QDialogButtonBox(QW.QDialogButtonBox.Ok | QW.QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)

        btn_check_all = create_toolbutton(
            self,
            text=_("Check all"),
            autoraise=False,
            shortcut=QG.QKeySequence.SelectAll,
            triggered=lambda checked=True: self.browser.tree.toggle_all(checked),
        )
        btn_uncheck_all = create_toolbutton(
            self,
            text=_("Uncheck all"),
            autoraise=False,
            triggered=lambda checked=False: self.browser.tree.toggle_all(checked),
        )

        self.button_layout = QW.QHBoxLayout()
        self.button_layout.addWidget(btn_check_all)
        self.button_layout.addWidget(btn_uncheck_all)
        self.button_layout.addStretch()
        self.button_layout.addWidget(bbox)
        self.bbox = bbox

        vlayout: QW.QVBoxLayout = self.layout()
        vlayout.addSpacing(10)
        vlayout.addLayout(self.button_layout)
