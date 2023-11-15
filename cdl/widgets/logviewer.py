# -*- coding: utf-8 -*-
#
# Copyright © 2022 Codra
# Pierre Raybaut

"""
Module providing a log viewer widget, a log viewer window and DataLab's log viewer
"""

from __future__ import annotations

import os.path as osp

from guidata.configtools import get_icon
from guidata.qthelpers import exec_dialog
from qtpy import QtWidgets as QW

from cdl.config import APP_NAME, Conf, _, get_old_log_fname
from cdl.env import execenv
from cdl.widgets.fileviewer import FileViewerWidget, get_title_contents


class LogViewerWindow(QW.QDialog):
    """Log viewer window"""

    def __init__(self, fnames: list[str], parent: QW.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("logviewer")
        self.setWindowTitle(APP_NAME + " - " + _("Log files"))
        self.setWindowIcon(get_icon("DataLab.svg"))
        self.tabs = QW.QTabWidget(self)
        for fname in fnames:
            if osp.isfile(fname):
                title, contents = get_title_contents(fname)
                if not contents.strip():
                    continue
                viewer = FileViewerWidget(language="Python")
                viewer.set_data(title, contents)
                self.tabs.addTab(viewer, get_icon("logs.svg"), osp.basename(fname))
        layout = QW.QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        self.resize(900, 400)

    @property
    def is_empty(self) -> bool:
        """Return True if there is no log available"""
        return self.tabs.count() == 0


def get_log_filenames() -> list[str]:
    """Return log filenames"""
    return [
        Conf.main.traceback_log_path.get(),
        Conf.main.faulthandler_log_path.get(),
        get_old_log_fname(Conf.main.traceback_log_path.get()),
        get_old_log_fname(Conf.main.faulthandler_log_path.get()),
    ]


def get_log_prompt_message() -> str | None:
    """Return prompt message for log files, i.e. a message informing the user
    whether log files were generated during last session or current session."""
    avail = [osp.isfile(fname) for fname in get_log_filenames()]
    if avail[0] or avail[1]:
        return _("Log files were generated during current session.")
    if avail[2] or avail[3]:
        return _("Log files were generated during last session.")
    return None


def exec_cdl_logviewer_dialog(parent: QW.QWidget | None = None) -> None:
    """View DataLab logs"""
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
