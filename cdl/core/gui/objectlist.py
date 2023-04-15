# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
Object (signal/image) list widgets
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations  # To be removed when dropping Python <=3.9 support

import re
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple

from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from cdl.config import _
from cdl.utils.qthelpers import block_signals

if TYPE_CHECKING:
    from cdl.core.gui.panel.base import BaseDataPanel
    from cdl.core.model.base import ObjectItf


class SimpleObjectList(QW.QListWidget):
    """Base object handling panel list widget, object (sig/ima) lists"""

    SIG_ITEM_DOUBLECLICKED = QC.Signal(int)
    SIG_CONTEXT_MENU = QC.Signal(QC.QPoint)

    def __init__(self, panel: BaseDataPanel, parent: QW.QWidget = None) -> None:
        parent = panel if parent is None else parent
        super().__init__(parent)
        self.panel = panel
        self.prefix = panel.PREFIX
        self.setAlternatingRowColors(True)
        self._objects: List[ObjectItf] = []  # signals or images
        self.itemDoubleClicked.connect(self.item_double_clicked)

    def init_from(self, objlist: SimpleObjectList) -> None:
        """Init from another SimpleObjectList, without making copies of objects"""
        self._objects = objlist.get_objects()
        self.refresh_list()
        self.setCurrentRow(objlist.currentRow())

    def get_objects(self) -> List[ObjectItf]:
        """Get all objects"""
        return self._objects

    def get_titles(self) -> List[str]:
        """Get object titles as diplayed in QListWidget"""
        return [self.item(row).text() for row in range(self.count())]

    def set_current_row(
        self, row: int, extend: bool = False, refresh: bool = True
    ) -> None:
        """Set list widget current row"""
        if row < 0:
            row += self.count()
        if extend:
            command = QC.QItemSelectionModel.Select
        else:
            command = QC.QItemSelectionModel.ClearAndSelect
        with block_signals(widget=self, enable=not refresh):
            self.setCurrentRow(row, command)

    def refresh_list(self, new_current_row: Optional[int] = None) -> None:
        """
        Refresh object list

        :param new_current_row: New row (if None, new current row is unchanged)
        """
        row = self.currentRow()
        if new_current_row is not None:
            row = new_current_row
        self.clear()
        for idx, obj in enumerate(self._objects):
            item = QW.QListWidgetItem(f"{self.prefix}{idx:03d}: {obj.title}", self)
            item.setToolTip(obj.metadata_to_html())
            self.addItem(item)
        if row < self.count():
            self.set_current_row(row)

    def item_double_clicked(self, listwidgetitem):
        """Item was double-clicked: open a pop-up plot dialog"""
        self.SIG_ITEM_DOUBLECLICKED.emit(self.row(listwidgetitem))

    def contextMenuEvent(self, event):  # pylint: disable=C0103
        """Override Qt method"""
        self.SIG_CONTEXT_MENU.emit(event.globalPos())


class GetObjectDialog(QW.QDialog):
    """Get object dialog box"""

    def __init__(self, parent: QW.QWidget, panel: BaseDataPanel, title: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setLayout(QW.QVBoxLayout())

        self.objlist = SimpleObjectList(panel, parent=parent)
        self.objlist.init_from(panel.objlist)
        self.objlist.SIG_ITEM_DOUBLECLICKED.connect(lambda row: self.accept())
        self.layout().addWidget(self.objlist)

        bbox = QW.QDialogButtonBox(QW.QDialogButtonBox.Ok | QW.QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        bbox.button(QW.QDialogButtonBox.Ok).setEnabled(self.objlist.count() > 0)
        self.layout().addSpacing(10)
        self.layout().addWidget(bbox)

    def get_object(self) -> ObjectItf:
        """Return current object"""
        return self.objlist.get_objects()[self.objlist.currentRow()]


class ObjectList(SimpleObjectList):
    """Object handling panel list widget, object (sig/ima) lists"""

    SIG_IMPORT_FILES = QC.Signal(list)

    def __init__(self, panel: BaseDataPanel) -> None:
        super().__init__(panel)
        self.setSelectionMode(QW.QListWidget.ExtendedSelection)
        self.setAcceptDrops(True)

    def paintEvent(self, event):  # pylint: disable=C0103
        """Reimplement Qt method"""
        super().paintEvent(event)
        if self.model() and self.model().rowCount(self.rootIndex()) > 0:
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

    def __len__(self) -> int:
        """Return number of objects"""
        return len(self._objects)

    def __getitem__(self, row: int) -> ObjectItf:
        """Return object at row"""
        return self._objects[row]

    def __setitem__(self, row: int, obj: ObjectItf) -> None:
        """Set object at row"""
        self._objects[row] = obj

    def __contains__(self, obj: ObjectItf) -> bool:
        """Return True if list contain obj"""
        return obj in self._objects

    def get_row(self, obj: ObjectItf) -> int:
        """Return row associated to object obj"""
        return self._objects.index(obj)

    def __fix_obj_titles(self, row: int, sign: int) -> None:
        """Fix all object titles before adding (sign==1) or removing (sign==-1)
        an object at row index"""
        pfx = self.prefix
        oname = f"{pfx}%03d"
        for obj in self:
            for match in re.finditer(pfx + "[0-9]{3}", obj.title):
                before = match.group()
                i_match = int(before[1:])
                if sign == -1 and i_match == row:
                    after = f"{pfx}xxx"
                elif (sign == -1 and i_match > row) or (sign == 1 and i_match >= row):
                    after = oname % (i_match + sign)
                else:
                    continue
                obj.title = obj.title.replace(before, after)

    def __delitem__(self, row: int) -> None:
        """Del object at row"""
        self.__fix_obj_titles(row, -1)
        self._objects.pop(row)

    def __iter__(self) -> Iterator[ObjectItf]:
        """Return an iterator over objects"""
        yield from self._objects

    def get_sel_object(self, position: int = 0) -> ObjectItf:
        """
        Return currently selected object

        :param int position: Position in selection list (0 means first, -1 means last)
        :return: Current object or None if there is no selection
        """
        rows = self.get_selected_rows()
        if rows:
            return self[rows[position]]
        return None

    def get_sel_objects(self) -> List[ObjectItf]:
        """Return selected objects"""
        return [self[row] for row in self.get_selected_rows()]

    def append(self, obj: ObjectItf) -> None:
        """Append object"""
        self._objects.append(obj)

    def insert(self, row: int, obj: ObjectItf) -> None:
        """Insert object at row index"""
        self.__fix_obj_titles(row, 1)
        self._objects.insert(row, obj)

    def remove_all(self) -> None:
        """Remove all objects"""
        self._objects = []

    def select_rows(self, rows: Tuple) -> None:
        """Select multiple list widget rows"""
        for index, row in enumerate(sorted(rows)):
            self.set_current_row(row, extend=index != 0, refresh=row == len(rows) - 1)

    def select_all_rows(self) -> None:
        """Select all widget rows"""
        self.selectAll()

    def get_selected_rows(self) -> List[int]:
        """Return selected rows"""
        return [index.row() for index in self.selectionModel().selectedRows()]
