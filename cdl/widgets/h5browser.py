# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab HDF5 browser module
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import os
import os.path as osp
from typing import TYPE_CHECKING, Any, Callable

from guidata.qthelpers import (
    add_actions,
    create_action,
    create_toolbutton,
    exec_dialog,
    get_icon,
    get_std_icon,
    win32_fix_title_bar_background,
)
from guidata.widgets.arrayeditor import ArrayEditor
from plotpy.builder import make
from plotpy.plot import PlotOptions, PlotWidget
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from qtpy.compat import getopenfilename

from cdl.adapters_plotpy.factories import create_adapter_from_object
from cdl.adapters_plotpy.signal import CURVESTYLES
from cdl.config import _
from cdl.h5 import H5Importer
from cdl.utils.qthelpers import qt_handle_error_message
from sigima_ import ImageObj, SignalObj
from sigima_.io.common.converters import to_string

if TYPE_CHECKING:
    from plotpy.plot import BasePlot

    from cdl.h5.common import BaseNode


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
        self.fnames: list[str] = []
        self.h5importers: list[H5Importer] = []

    def add_root(self, fname: str) -> None:
        """Add HDF5 root (new file)

        Args:
            fname: HDF5 file name
        """
        self.fnames.append(osp.abspath(fname))
        importer = H5Importer(fname)
        self.h5importers.append(importer)
        self.add_root_to_tree(importer)
        for col in range(1, 4):
            self.resizeColumnToContents(col)

    def remove_root(self, fname: str) -> None:
        """Remove HDF5 root

        Args:
            fname: HDF5 file name
        """
        index = self.fnames.index(osp.abspath(fname))
        self.fnames.pop(index)
        importer = self.h5importers.pop(index)
        importer.close()
        # Remove root item associated with file
        item = self.topLevelItem(index)
        self.takeTopLevelItem(index)
        del item

    def cleanup(self) -> None:
        """Clean up widget"""
        for importer in self.h5importers:
            importer.close()
        self.fnames: list[str] = []
        self.h5importers: list[H5Importer] = []
        self.clear()

    def __get_top_level_item(self, item: QW.QTreeWidgetItem) -> QW.QTreeWidgetItem:
        """Get top level item associated to item

        Args:
            item: Tree item

        Returns:
            Top level item
        """
        while item.parent():
            item = item.parent()
        return item

    def get_node(self, item: QW.QTreeWidgetItem) -> BaseNode:
        """Get HDF5 dataset associated to item

        Args:
            item: Tree item

        Returns:
            HDF5 node
        """
        toplevel_item = self.__get_top_level_item(item)
        toplevel_index = self.indexOfTopLevelItem(toplevel_item)
        node_id = item.data(0, QC.Qt.UserRole)
        if node_id:
            importer = self.h5importers[toplevel_index]
            return importer.get(node_id)
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
                    node = self.get_node(item)
                    datasets.append(node)
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
        for item in self.find_all_items():
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
        for item in self.find_all_items():
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
        if len(text) > 30:
            text = text[:30] + "..."
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
        if node.is_supported():
            tree_item.setCheckState(0, QC.Qt.Unchecked)
        else:
            tree_item.setFlags(QC.Qt.ItemIsEnabled)
        tree_item.setIcon(0, get_icon(node.icon_name))
        parent_item.addChild(tree_item)
        for child in node.children:
            H5TreeWidget.__recursive_popfunc(tree_item, child)

    def expand_all_children(self, item: QW.QTreeWidgetItem) -> None:
        """Expand all children (recursively)

        Args:
            item: Tree item
        """
        self.expandItem(item)
        for index in range(item.childCount()):
            child = item.child(index)
            self.expand_all_children(child)

    def add_root_to_tree(self, importer: H5Importer) -> None:
        """Add root to tree

        Args:
            importer: HDF5 importer
        """
        root = importer.root
        rootitem = QW.QTreeWidgetItem([root.name])
        rootitem.setToolTip(0, root.description)
        rootitem.setData(0, QC.Qt.UserRole, root.id)
        rootitem.setFlags(QC.Qt.ItemIsEnabled)
        rootitem.setIcon(0, get_icon(root.icon_name))
        self.addTopLevelItem(rootitem)
        for node in root.children:
            self.__recursive_popfunc(rootitem, node)
        self.expand_all_children(rootitem)

    def toggle_show_only_checkable_items(self, state: bool) -> None:
        """Show only checkable items

        Args:
            state: If True, only checkable items are shown
        """
        for item in self.find_all_items():
            item.setHidden(state)
        if state:
            # Iterate over checkable items and show them (and their parents)
            for item in self.find_all_items():
                if item.flags() & QC.Qt.ItemIsUserCheckable:
                    item.setHidden(False)
                    parent = item.parent()
                    while parent:
                        parent.setHidden(False)
                        parent = parent.parent()

    def toggle_show_values(self, state: bool) -> None:
        """Show values

        Args:
            state: If True, values are shown
        """
        # Hide or show the "Value" column
        self.setColumnHidden(3, not state)

    def set_current_file(self, fname: str) -> None:
        """Set current file

        Args:
            fname: HDF5 file name
        """
        index = self.fnames.index(osp.abspath(fname))
        item = self.topLevelItem(index)
        self.setCurrentItem(item)
        self.scrollToItem(item, QW.QAbstractItemView.PositionAtTop)


class PlotPreview(QW.QStackedWidget):
    """Plot preview"""

    def __init__(self, parent: QW.QWidget) -> None:
        super().__init__(parent)
        self.curvewidget = PlotWidget(
            self, options=PlotOptions(type="curve", curve_antialiasing=True)
        )
        self.addWidget(self.curvewidget)
        self.imagewidget = PlotWidget(
            self, options=PlotOptions(type="image", show_contrast=True)
        )
        self.addWidget(self.imagewidget)

    def cleanup(self) -> None:
        """Clean up widget"""
        for widget in (self.imagewidget, self.curvewidget):
            widget.get_plot().del_all_items()

    def update_plot_preview(self, node: BaseNode) -> None:
        """Update plot preview widget"""
        try:
            obj = node.get_native_object()
        except Exception as msg:  # pylint: disable=broad-except
            qt_handle_error_message(self, msg)
            return
        if obj is None:
            # An error occurred while creating the object (invalid data, ...)
            label = make.label(_("Unsupported data"), "C", (0, 0), "C")
            plot: BasePlot = self.currentWidget().get_plot()
            plot.del_all_items()
            plot.add_item(label)
            plot.replot()
            return
        if isinstance(obj, SignalObj):
            obj: SignalObj
            widget = self.curvewidget
        else:
            obj: ImageObj
            widget = self.imagewidget
        with CURVESTYLES.suspend():
            item = create_adapter_from_object(obj).make_item()
        plot = widget.get_plot()
        plot.del_all_items()
        plot.add_item(item)
        plot.set_active_item(item)
        item.unselect()
        plot.do_autoscale()
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
    """Group and attributes

    Args:
        parent: Parent widget
        show_array_callback: Callback to show array
    """

    def __init__(self, parent: QW.QWidget, show_array_callback: Callable) -> None:
        super().__init__(parent)
        self.group = TablePreview(self)
        self.addTab(self.group, get_icon("h5group.svg"), _("Group"))
        self.attrs = TablePreview(self)
        self.addTab(self.attrs, get_icon("h5attrs.svg"), _("Attributes"))
        # Add a button as corner widget to show the array (if any):
        self.__show_array_btn = create_toolbutton(
            self,
            icon=get_icon("show_results.svg"),
            text=_("Show array"),
            autoraise=False,
            triggered=show_array_callback,
        )
        self.__show_array_btn.setEnabled(False)
        self.setCornerWidget(self.__show_array_btn, QC.Qt.TopRightCorner)

    def cleanup(self) -> None:
        """Clean up widget"""
        self.group.clear()
        self.attrs.clear()

    def update_from_node(self, node: BaseNode) -> None:
        """Update widget from node

        Args:
            node: HDF5 node
        """
        # Update group =================================================================
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

        # Update attributes ============================================================
        self.attrs.update_table_preview(node.metadata)

        # Update show array button =====================================================
        self.__show_array_btn.setEnabled(node.IS_ARRAY)


class H5FileSelector(QW.QWidget):
    """HDF5 file selector

    Args:
        parent: Parent widget
    """

    SIG_ADD_FILENAME = QC.Signal(str)
    SIG_REMOVE_FILENAME = QC.Signal(str)
    SIG_CURRENT_CHANGED = QC.Signal(str)

    def __init__(self, parent: QW.QWidget) -> None:
        super().__init__(parent)
        self.setLayout(QW.QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.combo = QW.QComboBox(self)
        self.combo.currentTextChanged.connect(self.current_file_changed)
        self.layout().addWidget(self.combo)
        self.btn_add = create_toolbutton(
            self,
            icon=get_std_icon("DirOpenIcon"),
            text=_("Open") + " ...",
            autoraise=False,
            triggered=lambda _checked=False: self.add_file(),
        )
        self.btn_add.setSizePolicy(QW.QSizePolicy.Fixed, QW.QSizePolicy.Fixed)
        self.layout().addWidget(self.btn_add)
        self.btn_rmv = create_toolbutton(
            self,
            icon=get_std_icon("DialogCloseButton"),
            text=_("Close"),
            autoraise=False,
            triggered=self.remove_file,
        )
        self.btn_rmv.setSizePolicy(QW.QSizePolicy.Fixed, QW.QSizePolicy.Fixed)
        self.layout().addWidget(self.btn_rmv)
        self.btn_rmv.setEnabled(False)

    def set_current_fname(self, fname: str) -> None:
        """Set current file name

        Args:
            fname: HDF5 file name
        """
        index = self.combo.findText(fname)
        if index >= 0:
            self.combo.setCurrentIndex(index)

    def get_current_fname(self) -> str:
        """Return current file name

        Returns:
            HDF5 file name
        """
        return self.combo.currentText()

    def current_file_changed(self, fname: str) -> None:
        """Current file changed

        Args:
            fname: HDF5 file name
        """
        self.SIG_CURRENT_CHANGED.emit(fname)

    def add_fname(self, fname: str) -> None:
        """Add file name

        Args:
            fname: HDF5 file name
        """
        self.combo.addItem(get_icon("h5file.svg"), fname)
        self.btn_rmv.setEnabled(True)

    def remove_fname(self, fname: str) -> None:
        """Remove file name

        Args:
            fname: HDF5 file name
        """
        index = self.combo.findText(fname)
        if index >= 0:
            self.combo.removeItem(index)
        if self.combo.count() == 0:
            self.btn_rmv.setEnabled(False)

    def add_file(self, fname: str | None = None) -> None:
        """Browse file

        Args:
            fname: HDF5 file name. Default is None.
             (this is used for testing only)
        """
        if fname is None:
            fname = getopenfilename(
                self, _("Select HDF5 file"), "", _("HDF5 files (*.h5 *.hdf5)")
            )[0]
        if fname:
            self.SIG_ADD_FILENAME.emit(osp.abspath(fname))

    def remove_file(self, fname: str | None = None) -> None:
        """Remove file name

        Args:
            fname: HDF5 file name
        """
        if fname is None:
            fname = self.combo.currentText()
        self.SIG_REMOVE_FILENAME.emit(fname)


class H5Browser(QW.QSplitter):
    """HDF5 Browser Widget

    Args:
        parent: Parent widget
    """

    SIG_SELECT_NEW_FILE = QC.Signal(str)
    SIG_REMOVE_FILE = QC.Signal(str)

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(parent)
        self.selector = H5FileSelector(self)
        self.selector.SIG_ADD_FILENAME.connect(self.__add_new_file)
        self.selector.SIG_REMOVE_FILENAME.connect(self.__remove_file)
        self.selector.SIG_CURRENT_CHANGED.connect(self.__selector_current_file_changed)
        self.tree = H5TreeWidget(self)
        self.tree.SIG_SELECTED.connect(self.__item_selected_on_tree)
        selectorandtree = QW.QFrame(self)
        selectorandtree.setLayout(QW.QVBoxLayout())
        selectorandtree.layout().addWidget(self.selector)
        selectorandtree.layout().addWidget(self.tree)
        selectorandtree.layout().setContentsMargins(0, 0, 0, 0)
        self.addWidget(selectorandtree)
        preview = QW.QSplitter(self)
        preview.setOrientation(QC.Qt.Vertical)
        self.addWidget(preview)
        self.plotpreview = PlotPreview(self)
        preview.addWidget(self.plotpreview)
        self.groupandattrs = GroupAndAttributes(self, self.show_array)
        preview.addWidget(self.groupandattrs)
        preview.setSizes([int(self.size().height() / 2)] * 2)

    def open_file(self, fname: str) -> None:
        """Open HDF5 file

        Args:
            fname: HDF5 file name
        """
        self.tree.add_root(fname)
        self.selector.add_fname(fname)

    def close_file(self, fname: str) -> None:
        """Close HDF5 file

        Args:
            fname: HDF5 file name
        """
        self.tree.remove_root(fname)
        self.selector.remove_fname(fname)

    def __add_new_file(self, fname: str) -> None:
        """Add new file

        Args:
            fname: HDF5 file name
        """
        self.open_file(fname)
        self.selector.set_current_fname(fname)
        self.SIG_SELECT_NEW_FILE.emit(fname)

    def __remove_file(self, fname: str) -> None:
        """Remove file

        Args:
            fname: HDF5 file name
        """
        self.close_file(fname)
        self.SIG_REMOVE_FILE.emit(fname)

    def cleanup(self) -> None:
        """Clean up widget"""
        self.tree.cleanup()
        self.plotpreview.cleanup()

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

    def __item_selected_on_tree(self, item: QW.QTreeWidgetItem) -> None:
        """Item selected on tree

        Args:
            item: Tree item
        """
        # View the selected item
        node = self.get_node(item)
        if node.is_supported():
            self.plotpreview.update_plot_preview(node)
        self.groupandattrs.update_from_node(node)
        # Update the file selector combo box
        self.selector.set_current_fname(node.h5file.filename)

    def __selector_current_file_changed(self, fname: str) -> None:
        """Selector current file changed

        Args:
            fname: HDF5 file name
        """
        if fname:
            self.tree.set_current_file(fname)

    def show_array(self) -> None:
        """Show array"""
        node = self.get_node()
        assert node.IS_ARRAY
        arrayeditor = ArrayEditor(self)
        arrayeditor.setup_and_check(node.data, title=node.name, readonly=True)
        exec_dialog(arrayeditor)


class H5BrowserDialog(QW.QDialog):
    """HDF5 Browser Dialog

    Args:
        parent: Parent widget
        size: Dialog size
    """

    def __init__(
        self, parent: QW.QWidget | None = None, size: tuple[int, int] = (1150, 700)
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(QC.Qt.Window)
        self.setObjectName("h5browser")
        self.setWindowTitle(_("HDF5 Browser"))
        self.setWindowIcon(get_icon("h5browser.svg"))
        win32_fix_title_bar_background(self)
        vlayout = QW.QVBoxLayout()
        self.setLayout(vlayout)
        self.button_layout: QW.QHBoxLayout | None = None
        self.bbox: QW.QDialogButtonBox | None = None
        self.nodes: list[BaseNode] = []
        self.checkbox_show_only: QW.QCheckBox | None = None
        self.checkbox_show_values: QW.QCheckBox | None = None

        self.browser = H5Browser(self)
        self.browser.SIG_SELECT_NEW_FILE.connect(self.select_new_file)
        self.browser.SIG_REMOVE_FILE.connect(self.remove_file)
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

    def is_empty(self) -> bool:
        """Return True if tree is empty"""
        return self.browser.tree.is_empty()

    def cleanup(self) -> None:
        """Cleanup dialog"""
        self.browser.cleanup()

    def refresh_buttons(self) -> None:
        """Refresh buttons"""
        state = self.browser.tree.is_any_item_checked()
        self.bbox.button(QW.QDialogButtonBox.Ok).setEnabled(state)

    def show_only_checkable_items(self, state: int) -> None:
        """Show only checkable items

        Args:
            state: If True, only checkable items are shown
        """
        self.browser.tree.toggle_show_only_checkable_items(state)
        fname = self.browser.selector.get_current_fname()
        if fname:
            self.browser.tree.set_current_file(fname)

    def __finalize_setup(self) -> None:
        """Finalize setup"""
        tree = self.browser.tree
        tree.toggle_show_only_checkable_items(self.checkbox_show_only.isChecked())
        tree.toggle_show_values(self.checkbox_show_values.isChecked())

    def open_file(self, fname: str) -> None:
        """Open file

        Args:
            fname: HDF5 file name
        """
        self.browser.open_file(fname)
        self.__finalize_setup()

    def open_files(self, fnames: list[str]) -> None:
        """Open files

        Args:
            fnames: HDF5 file names
        """
        for fname in fnames:
            self.browser.open_file(fname)
        self.__finalize_setup()

    def select_new_file(self, fname: str) -> None:  # pylint:disable=unused-argument
        """Select new file

        Args:
            fname: HDF5 file name
        """
        self.__finalize_setup()
        self.refresh_buttons()

    def remove_file(self, fname: str) -> None:  # pylint:disable=unused-argument
        """Remove file

        Args:
            fname: HDF5 file name
        """
        self.refresh_buttons()

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
            icon=get_icon("check_all.svg"),
            text=_("Check all"),
            autoraise=False,
            shortcut=QG.QKeySequence.SelectAll,
            triggered=lambda checked=True: self.browser.tree.toggle_all(checked),
        )
        btn_uncheck_all = create_toolbutton(
            self,
            icon=get_icon("uncheck_all.svg"),
            text=_("Uncheck all"),
            autoraise=False,
            triggered=lambda checked=False: self.browser.tree.toggle_all(checked),
        )
        self.checkbox_show_only = QW.QCheckBox(_("Show only supported data"))
        self.checkbox_show_only.stateChanged.connect(self.show_only_checkable_items)
        self.checkbox_show_values = QW.QCheckBox(_("Show values"))
        self.checkbox_show_values.stateChanged.connect(
            self.browser.tree.toggle_show_values
        )

        self.button_layout = QW.QHBoxLayout()
        self.button_layout.addWidget(self.checkbox_show_only)
        self.button_layout.addWidget(self.checkbox_show_values)
        self.button_layout.addSpacing(10)
        self.button_layout.addWidget(btn_check_all)
        self.button_layout.addWidget(btn_uncheck_all)
        self.button_layout.addStretch()
        self.button_layout.addWidget(bbox)
        self.bbox = bbox

        vlayout: QW.QVBoxLayout = self.layout()
        vlayout.addSpacing(10)
        vlayout.addLayout(self.button_layout)
