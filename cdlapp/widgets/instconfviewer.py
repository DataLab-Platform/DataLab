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
from qtpy import QtWidgets as QW

from cdlapp import __version__
from cdlapp.config import APP_NAME, DATAPATH, IS_FROZEN, Conf, _
from cdlapp.utils import dephash
from cdlapp.utils.qthelpers import exec_dialog
from cdlapp.widgets.fileviewer import FileViewerWidget, get_title_contents


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


def get_install_infos() -> str:
    """Get DataLab installation informations

    Returns:
        str: installation informations
    """
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
    if IS_FROZEN:
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
    return infos


class InstallConfigViewerWindow(QW.QDialog):
    """Installation configuration window"""

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("instviewer")
        self.setWindowTitle(APP_NAME + " - " + _("Installation and configuration"))
        self.setWindowIcon(get_icon("DataLab.svg"))
        self.tabs = QW.QTabWidget(self)
        uc_title, uc_contents = get_title_contents(Conf.get_filename())
        for title, contents, tab_icon, tab_title in (
            (
                _("Installation configuration"),
                get_install_infos(),
                get_icon("libre-toolbox.svg"),
                _("Installation configuration"),
            ),
            (
                uc_title,
                uc_contents,
                get_icon("libre-gui-settings.svg"),
                _("User configuration"),
            ),
        ):
            viewer = FileViewerWidget(language="Python")
            viewer.set_data(title, contents)
            self.tabs.addTab(viewer, tab_icon, tab_title)
        layout = QW.QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        self.resize(800, 500)


def exec_cdl_installconfig_dialog(parent: QW.QWidget | None = None) -> None:
    """View DataLab installation configuration"""
    dlg = InstallConfigViewerWindow(parent=parent)
    exec_dialog(dlg)
