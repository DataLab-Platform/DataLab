# -*- coding: utf-8 -*-
#
# Copyright © 2022 Codra
# Pierre Raybaut

"""
Module providing a log viewer widget, a log viewer window and CodraFT's log viewer
"""

import os.path as osp
from pathlib import Path

from guidata.configtools import get_icon
from guidata.widgets.codeeditor import CodeEditor
from qtpy import QtWidgets as QW

from codraft.config import APP_NAME, Conf, _, get_old_log_fname
from codraft.utils.qthelpers import qt_app_context


class LogViewerWidget(QW.QWidget):
    """Log viewer widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.editor = CodeEditor(language="Python")
        self.editor.setReadOnly(True)
        layout = QW.QVBoxLayout()
        self.label = QW.QLabel("")
        layout.addWidget(self.label)
        layout.addWidget(self.editor)
        self.setLayout(layout)

    def set_path(self, path):
        """Set log path"""
        with open(path, "r", encoding="utf-8") as fdesc:
            contents = fdesc.read()
        pathobj = Path(path)
        uri_path = pathobj.absolute().as_uri()
        text = f'Contents of file <a href="{uri_path}">{path}</a>:'
        self.label.setText(text)
        self.label.setOpenExternalLinks(True)
        self.editor.setPlainText(contents)


class LogViewerWindow(QW.QMainWindow):
    """Log viewer window"""

    def __init__(self, fnames, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_("CodraFT log files"))
        self.setWindowIcon(get_icon("codraft.svg"))
        tabs = QW.QTabWidget()
        for fname in fnames:
            if osp.isfile(fname):
                viewer = LogViewerWidget()
                viewer.set_path(fname)
                tabs.addTab(viewer, get_icon("logs.svg"), osp.basename(fname))
        self.setCentralWidget(tabs)
        self.resize(900, 400)


def create_codraft_logviewer_window(parent=None):
    """View CodraFT logs"""
    fnames = [
        osp.normpath(fname)
        for fname in (
            get_old_log_fname(Conf.main.faulthandler_log_path.get()),
            get_old_log_fname(Conf.main.traceback_log_path.get()),
        )
        if osp.isfile(fname)
    ]
    if fnames:
        win = LogViewerWindow(fnames, parent=parent)
        win.show()
        return win
    QW.QMessageBox.information(parent, APP_NAME, _("Log files are currently empty."))
    return None


def test_log_viewer():
    """Test log viewer window"""
    with qt_app_context(exec_loop=True):
        _win = create_codraft_logviewer_window()


if __name__ == "__main__":
    test_log_viewer()
