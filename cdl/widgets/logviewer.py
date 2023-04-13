# -*- coding: utf-8 -*-
#
# Copyright © 2022 Codra
# Pierre Raybaut

"""
Module providing a log viewer widget, a log viewer window and CobraDataLab's log viewer
"""

import os.path as osp
from pathlib import Path
from typing import List, Optional

from guidata.configtools import get_icon
from guidata.widgets.codeeditor import CodeEditor
from qtpy import QtWidgets as QW

from cdl.config import APP_NAME, Conf, _, get_old_log_fname
from cdl.env import execenv
from cdl.utils.qthelpers import exec_dialog


def read_text_file(path: str) -> str:
    """Read text file using multiple encodings"""
    encodings = ["utf-8", "latin1", "cp1252", "utf-16", "utf-32", "ascii"]
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as fdesc:
                return fdesc.read()
        except UnicodeDecodeError:
            pass
    raise UnicodeDecodeError(
        "Unable to read file using the following encodings: {}".format(encodings)
    )


def get_title_contents(path):
    """Get title and contents for log filename"""
    contents = read_text_file(path)
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
        self.setWindowTitle(_("CobraDataLab log files"))
        self.setWindowIcon(get_icon("CobraDataLab.svg"))
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


def get_log_filenames() -> List[str]:
    """Return log filenames"""
    return [
        Conf.main.traceback_log_path.get(),
        Conf.main.faulthandler_log_path.get(),
        get_old_log_fname(Conf.main.traceback_log_path.get()),
        get_old_log_fname(Conf.main.faulthandler_log_path.get()),
    ]


def get_log_prompt_message() -> Optional[str]:
    """Return prompt message for log files, i.e. a message informing the user
    whether log files were generated during last session or current session."""
    avail = [osp.isfile(fname) for fname in get_log_filenames()]
    if avail[0] or avail[1]:
        return _("Log files were generated during current session.")
    elif avail[2] or avail[3]:
        return _("Log files were generated during last session.")
    return None


def exec_cdl_logviewer_dialog(parent=None):
    """View CobraDataLab logs"""
    fnames = [osp.normpath(fname) for fname in get_log_filenames() if osp.isfile(fname)]
    dlg = LogViewerWindow(fnames, parent=parent)
    if dlg.is_empty:
        if not execenv.unattended:
            QW.QMessageBox.information(
                dlg, APP_NAME, _("Log files are currently empty.")
            )
        dlg.close()
    else:
        exec_dialog(dlg)