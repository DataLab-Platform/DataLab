# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Object (signal/image) list widgets
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import re
from typing import Tuple

from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from codraft.utils.qthelpers import block_signals


class SimpleObjectList(QW.QListWidget):
    """Base object handling panel list widget, object (sig/ima) lists"""

    SIG_ITEM_DOUBLECLICKED = QC.Signal(int)
    SIG_CONTEXT_MENU = QC.Signal(QC.QPoint)

    def __init__(self, panel, parent=None):
        parent = panel if parent is None else parent
        super().__init__(parent)
        self.panel = panel
        self.prefix = panel.PREFIX
        self.setAlternatingRowColors(True)
        self._objects = []  # signals or images
        self.itemDoubleClicked.connect(self.item_double_clicked)

    def init_from(self, objlist):
        """Init from another SimpleObjectList, without making copies of objects"""
        self._objects = objlist.get_objects()
        self.refresh_list()
        self.setCurrentRow(objlist.currentRow())

    def get_objects(self):
        """Get all objects"""
        return self._objects

    def set_current_row(self, row, extend=False, refresh=True):
        """Set list widget current row"""
        if row < 0:
            row += self.count()
        if extend:
            command = QC.QItemSelectionModel.Select
        else:
            command = QC.QItemSelectionModel.ClearAndSelect
        with block_signals(widget=self, enable=not refresh):
            self.setCurrentRow(row, command)

    def refresh_list(self, new_current_row=None):
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

    def __init__(self, parent, panel, title):
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

    def get_object(self):
        """Return current object"""
        return self.objlist.get_objects()[self.objlist.currentRow()]


class ObjectList(SimpleObjectList):
    """Object handling panel list widget, object (sig/ima) lists"""

    def __init__(self, panel):
        super().__init__(panel)
        self.setSelectionMode(QW.QListWidget.ExtendedSelection)

    def __len__(self):
        """Return number of objects"""
        return len(self._objects)

    def __getitem__(self, row):
        """Return object at row"""
        return self._objects[row]

    def __setitem__(self, row, obj):
        """Set object at row"""
        self._objects[row] = obj

    def __contains__(self, obj):
        """Return True if list contain obj"""
        return obj in self._objects

    def get_row(self, obj):
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

    def __delitem__(self, row):
        """Del object at row"""
        self.__fix_obj_titles(row, -1)
        self._objects.pop(row)

    def __iter__(self):
        """Return an iterator over objects"""
        yield from self._objects

    def get_sel_object(self, position=0):
        """
        Return currently selected object

        :param int position: Position in selection list (0 means first, -1 means last)
        :return: Current object or None if there is no selection
        """
        rows = self.get_selected_rows()
        if rows:
            return self[rows[position]]
        return None

    def get_sel_objects(self):
        """Return selected objects"""
        return [self[row] for row in self.get_selected_rows()]

    def append(self, obj):
        """Append object"""
        self._objects.append(obj)

    def insert(self, row, obj):
        """Insert object at row index"""
        self.__fix_obj_titles(row, 1)
        self._objects.insert(row, obj)

    def remove_all(self):
        """Remove all objects"""
        self._objects = []

    def select_rows(self, rows: Tuple):
        """Select multiple list widget rows"""
        for index, row in enumerate(sorted(rows)):
            self.set_current_row(row, extend=index != 0, refresh=row == len(rows) - 1)

    def select_all_rows(self):
        """Select all widget rows"""
        self.selectAll()

    def get_selected_rows(self):
        """Return selected rows"""
        return [index.row() for index in self.selectionModel().selectedRows()]
