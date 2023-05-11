# -*- coding: utf-8 -*-
#
# Copyright © 2022 Codra
# Pierre Raybaut

"""
Module providing DataLab Installation configuration widget
"""

from __future__ import annotations

import locale
import os
import platform
import sys
from subprocess import PIPE, Popen

from guidata.configtools import get_icon
from guidata.widgets.codeeditor import CodeEditor
from qtpy import QtWidgets as QW

from cdl import __version__
from cdl.config import DATAPATH, _
from cdl.utils import dephash
from cdl.utils.qthelpers import exec_dialog


def decode_fs_string(string: bytes) -> str:
    """Convert string from file system charset to unicode"""
    charset = sys.getfilesystemencoding()
    if charset is None:
        charset = locale.getpreferredencoding()
    return string.decode(charset)


def get_pip_list() -> str:
    """Get pip list result"""
    command = " ".join([sys.executable, "-m", "pip list"])
    with Popen(command, shell=True, stdout=PIPE, stderr=PIPE) as proc:
        output = proc.stdout.read()
    return decode_fs_string(output)


class InstallConfigViewerWidget(QW.QWidget):
    """Installation configuration widget"""

    def __init__(
        self, label: str, contents: str, parent: QW.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.editor = CodeEditor()
        self.editor.setReadOnly(True)
        self.editor.setPlainText(contents)
        layout = QW.QVBoxLayout()
        layout.addWidget(QW.QLabel(label))
        layout.addWidget(self.editor)
        self.setLayout(layout)


class InstallConfigViewerWindow(QW.QDialog):
    """Installation configuration window"""

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dependencies")
        self.setWindowTitle(_("About DataLab installation"))
        self.setWindowIcon(get_icon("DataLab.svg"))
        self.tabs = QW.QTabWidget()
        label = "Information on current DataLab installation:"
        more_infos = ""
        try:
            state = dephash.check_dependencies_hash(DATAPATH)
            bad_deps = [name for name in state if not state[name]]
            if bad_deps:
                more_infos += "Invalid dependencies: "
                more_infos += ", ".join(bad_deps)
            else:
                more_infos += "Dependencies hash file: checked."
        except IOError:
            more_infos += "Unable to open dependencies hash file."
        more_infos += os.linesep * 2
        if sys.executable.lower().endswith("cdl.exe"):
            #  Stand-alone version
            more_infos += "This is the Stand-alone version of DataLab"
        else:
            more_infos += get_pip_list()
        infos = os.linesep.join(
            [
                f"DataLab v{__version__}",
                "",
                f"Machine type: {platform.machine()}",
                f"Platform: {platform.platform()}",
                f"Python v{sys.version}",
                "",
                more_infos,
            ]
        )
        viewer = InstallConfigViewerWidget(label, infos, parent=self)
        layout = QW.QVBoxLayout()
        layout.addWidget(viewer)
        self.setLayout(layout)
        self.resize(800, 500)


def exec_cdl_installconfig_dialog(parent: QW.QWidget | None = None) -> None:
    """View DataLab installation configuration"""
    dlg = InstallConfigViewerWindow(parent=parent)
    exec_dialog(dlg)
