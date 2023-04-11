# -*- coding: utf-8 -*-
#
# Copyright © 2022 Codra
# Pierre Raybaut

"""
Module providing CodraFT Installation configuration widget
"""

import locale
import os
import platform
import sys
from subprocess import PIPE, Popen

from guidata.configtools import get_icon
from guidata.widgets.codeeditor import CodeEditor
from qtpy import QtWidgets as QW

from codraft import __version__
from codraft.config import DATAPATH, _
from codraft.utils import dephash
from codraft.utils.qthelpers import exec_dialog


def decode_fs_string(string):
    """Convert string from file system charset to unicode"""
    charset = sys.getfilesystemencoding()
    if charset is None:
        charset = locale.getpreferredencoding()
    return string.decode(charset)


def get_pip_list():
    """Get pip list result"""
    command = " ".join([sys.executable, "-m", "pip list"])
    with Popen(command, shell=True, stdout=PIPE, stderr=PIPE) as proc:
        output = proc.stdout.read()
    return decode_fs_string(output)


class InstallConfigViewerWidget(QW.QWidget):
    """Installation configuration widget"""

    def __init__(self, label, contents, parent=None):
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dependencies")
        self.setWindowTitle(_("About CodraFT installation"))
        self.setWindowIcon(get_icon("codraft.svg"))
        self.tabs = QW.QTabWidget()
        label = "Information on current CodraFT installation:"
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
        if sys.executable.lower().endswith("codraft.exe"):
            #  Stand-alone version
            more_infos += "This is the Stand-alone version of CodraFT"
        else:
            more_infos += get_pip_list()
        infos = os.linesep.join(
            [
                f"CodraFT v{__version__}",
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


def exec_codraft_installconfig_dialog(parent=None):
    """View CodraFT installation configuration"""
    dlg = InstallConfigViewerWindow(parent=parent)
    exec_dialog(dlg)
