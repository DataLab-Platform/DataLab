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
from codraft.env import execenv
from codraft.utils.qthelpers import exec_dialog


def get_title_contents(path):
    """Get title and contents for log filename"""
    with open(path, "r", encoding="utf-8") as fdesc:
        contents = fdesc.read()
    pathobj = Path(path)
    uri_path = pathobj.absolute().as_uri()
    text = f'{_("Contents of file")} <a href="{uri_path}">{path}</a>:'
    return text, contents


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

    def set_data(self, text, contents):
        """Set log data"""
        self.label.setText(text)
        self.label.setOpenExternalLinks(True)
        self.editor.setPlainText(contents)


class LogViewerWindow(QW.QDialog):
    """Log viewer window"""

    def __init__(self, fnames, parent=None):
        super().__init__(parent)
        self.setObjectName("logviewer")
        self.setWindowTitle(_("CodraFT log files"))
        self.setWindowIcon(get_icon("codraft.svg"))
        self.tabs = QW.QTabWidget(self)
        for fname in fnames:
            if osp.isfile(fname):
                viewer = LogViewerWidget()
                title, contents = get_title_contents(fname)
                if not contents.strip():
                    continue
                viewer.set_data(title, contents)
                self.tabs.addTab(viewer, get_icon("logs.svg"), osp.basename(fname))
        layout = QW.QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        self.resize(900, 400)

    @property
    def is_empty(self):
        """Return True if there is no log available"""
        return self.tabs.count() == 0


def exec_codraft_logviewer_dialog(parent=None):
    """View CodraFT logs"""
    fnames = [
        osp.normpath(fname)
        for fname in (
            Conf.main.traceback_log_path.get(),
            Conf.main.faulthandler_log_path.get(),
            get_old_log_fname(Conf.main.traceback_log_path.get()),
            get_old_log_fname(Conf.main.faulthandler_log_path.get()),
        )
        if osp.isfile(fname)
    ]
    dlg = LogViewerWindow(fnames, parent=parent)
    if dlg.is_empty:
        if not execenv.unattended:
            QW.QMessageBox.information(
                dlg, APP_NAME, _("Log files are currently empty.")
            )
        dlg.close()
    else:
        exec_dialog(dlg)
