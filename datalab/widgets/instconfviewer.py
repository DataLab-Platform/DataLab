# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

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
from guidata.qthelpers import exec_dialog
from qtpy import QtWidgets as QW
from sigima.io.image import ImageIORegistry
from sigima.io.signal import SignalIORegistry

import datalab
from datalab.config import APP_NAME, DATAPATH, IS_FROZEN, Conf, _
from datalab.plugins import PluginRegistry
from datalab.utils import dephash
from datalab.widgets.fileviewer import FileViewerWidget, get_title_contents


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


def get_install_info() -> str:
    """Get DataLab installation informations

    Returns:
        str: installation informations
    """
    if IS_FROZEN:
        #  Stand-alone version
        more_info = "This is the Stand-alone version of DataLab"
    else:
        more_info = ""
        try:
            state = dephash.check_dependencies_hash(DATAPATH)
            bad_deps = [name for name in state if not state[name]]
            if bad_deps:
                more_info += "Invalid dependencies: "
                more_info += ", ".join(bad_deps)
            else:
                more_info += "Dependencies hash file: checked."
        except IOError:
            more_info += "Unable to open dependencies hash file."
        more_info += os.linesep * 2
        more_info += get_pip_list()
    info = os.linesep.join(
        [
            f"DataLab v{datalab.__version__}",
            "",
            f"Machine type: {platform.machine()}",
            f"Platform: {platform.platform()}",
            f"Python v{sys.version}",
            "",
            more_info,
        ]
    )
    return info


class InstallConfigViewerWindow(QW.QDialog):
    """Installation configuration window"""

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("instviewer")
        self.setWindowTitle(APP_NAME + " - " + _("Installation and configuration"))
        self.setWindowIcon(get_icon("DataLab.svg"))
        self.tabs = QW.QTabWidget(self)
        uc_title, uc_contents = get_title_contents(Conf.get_filename())
        plugins_io_contents = PluginRegistry.get_plugin_info(html=False)
        for registry in (SignalIORegistry, ImageIORegistry):
            plugins_io_contents += os.linesep + ("_" * 80) + os.linesep * 2
            plugins_io_contents += registry.get_format_info(mode="text")
        for title, contents, tab_icon, tab_title in (
            (
                _("Installation configuration"),
                get_install_info(),
                get_icon("libre-toolbox.svg"),
                _("Installation configuration"),
            ),
            (
                uc_title,
                uc_contents,
                get_icon("libre-gui-settings.svg"),
                _("User configuration"),
            ),
            (
                _("Plugins and I/O features"),
                plugins_io_contents,
                get_icon("libre-gui-plugin.svg"),
                _("Plugins and I/O features"),
            ),
        ):
            viewer = FileViewerWidget()
            viewer.set_data(title, contents)
            self.tabs.addTab(viewer, tab_icon, tab_title)
        layout = QW.QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        self.resize(800, 500)


def exec_datalab_installconfig_dialog(parent: QW.QWidget | None = None) -> None:
    """View DataLab installation configuration"""
    dlg = InstallConfigViewerWindow(parent=parent)
    exec_dialog(dlg)
