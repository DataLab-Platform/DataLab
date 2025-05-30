# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Module providing a file dialog widget based on Qt's QFileDialog.getOpenFileNames
but supporting multiple file preselection (Qt original dialog only supports single file
selection).
"""

from __future__ import annotations

import os
import os.path as osp

from guidata.qthelpers import qt_app_context
from qtpy.QtCore import QItemSelectionModel
from qtpy.QtWidgets import QAbstractItemView, QFileDialog, QListView, QWidget


def get_open_file_names(
    parent: QWidget | None = None,
    caption: str = "",
    basedir: str | list[str] = "",
    filters: str = "",
    selectedfilter: str = "",
    options: QFileDialog.Options = None,
) -> tuple[list[str], str]:
    """Wrapper around QtGui.QFileDialog.getOpenFileNames static method
    Returns a tuple (filenames, selectedfilter) -- when dialog box is canceled,
    returns a tuple (empty list, empty string)

    Args:
        parent: Parent widget for the dialog.
        caption: Dialog title.
        basedir: Initial directory to open the dialog in, or preselected files
         (single string or list of strings).
        filters: File filters for the dialog.
        selectedfilter: Default filter to be selected.
        options: Additional options for the dialog.

    Returns:
        A tuple containing a list of selected filenames and the selected filter.
    """
    if isinstance(basedir, str):
        if osp.isfile(basedir):
            sel_files = [basedir]
            basedir = osp.dirname(basedir)
        else:
            sel_files = []
    else:
        assert isinstance(basedir, list)
        sel_files = basedir
        basedir = osp.dirname(sel_files[0]) if sel_files else ""
    dlg = QFileDialog(
        parent, caption, basedir, filters, options=QFileDialog.DontUseNativeDialog
    )
    if options is not None:
        dlg.setOptions(options | QFileDialog.DontUseNativeDialog)
    file_view = dlg.findChild(QListView, "listView")
    sel_model = file_view.selectionModel()
    for fname in sel_files:
        idx = sel_model.model().index(fname)
        sel_model.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
    file_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
    file_view.setSelectionBehavior(QAbstractItemView.SelectRows)
    if dlg.exec():
        filenames = dlg.selectedFiles()
        selectedfilter = dlg.selectedNameFilter()
    else:
        filenames = []
        selectedfilter = ""
    return filenames, selectedfilter


def test_get_open_file_names():
    """Test get_open_file_names function"""
    cdl_widgets_path = osp.dirname(__file__)
    sel_files = [
        osp.join(cdl_widgets_path, fname) for fname in os.listdir(cdl_widgets_path)[:2]
    ]
    with qt_app_context():
        filenames, selectedfilter = get_open_file_names(
            filters="Python files (*.py);;All files (*)",
            caption="Select Python files",
            basedir=sel_files,
        )
        print(filenames, selectedfilter)


if __name__ == "__main__":
    test_get_open_file_names()
