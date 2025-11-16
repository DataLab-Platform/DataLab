# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Module providing DataLab Installation configuration widget
"""

from __future__ import annotations

import json
import locale
import os
import platform
import sys
from importlib.metadata import distributions
from pathlib import Path

from guidata.configtools import get_icon
from guidata.qthelpers import exec_dialog
from qtpy import QtWidgets as QW
from sigima.io.image import ImageIORegistry
from sigima.io.signal import SignalIORegistry

import datalab
from datalab.config import APP_NAME, IS_FROZEN, Conf, _
from datalab.plugins import PluginRegistry
from datalab.widgets.fileviewer import FileViewerWidget, get_title_contents


def decode_fs_string(string: bytes) -> str:
    """Convert string from file system charset to unicode"""
    charset = sys.getfilesystemencoding()
    if charset is None:
        charset = locale.getpreferredencoding()
    return string.decode(charset)


def get_installed_package_info() -> str:
    """Get the list of installed packages with their versions"""
    packages = [(dist.metadata["Name"], dist.version) for dist in distributions()]

    # Sort alphabetically by package name
    packages.sort(key=lambda x: x[0].lower())

    # Determine column widths
    name_width = max(len(name) for name, _ in packages)
    version_width = max(len(version) for _, version in packages)

    header = f"{'Package':{name_width}}   {'Version':{version_width}}"
    separator = f"{'-' * name_width}   {'-' * version_width}"
    result_lines = [header, separator]
    for name, version in packages:
        result_lines.append(f"{name:{name_width}}   {version:{version_width}}")

    return os.linesep.join(result_lines)


def get_manifest_package_info(manifest_path: Path) -> str:
    """Get the list of packages from the build manifest file

    Args:
        manifest_path: Path to the manifest.json file

    Returns:
        Formatted string with package list and build information
    """
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        packages = list(manifest["packages"].items())
        packages.sort(key=lambda x: x[0].lower())

        # Determine column widths
        name_width = max(len(name) for name, _ in packages)
        version_width = max(len(version) for _, version in packages)

        header = f"{'Package':{name_width}}   {'Version':{version_width}}"
        separator = f"{'-' * name_width}   {'-' * version_width}"
        result_lines = [
            f"Build time: {manifest['build_time']}",
            f"Python version: {manifest['python_version']}",
            f"Build system: {manifest['system']} {manifest['release']}",
            f"Architecture: {manifest['architecture']}",
            "",
            header,
            separator,
        ]
        for name, version in packages:
            result_lines.append(f"{name:{name_width}}   {version:{version_width}}")

        return os.linesep.join(result_lines)
    except Exception as e:
        return f"Error reading manifest file: {e}"


def get_install_info() -> str:
    """Get DataLab installation informations

    Returns:
        str: installation informations
    """
    if IS_FROZEN:
        #  Stand-alone version
        more_info = "This is the Stand-alone version of DataLab."
        more_info += os.linesep * 2

        # Try to read manifest file from executable root directory
        manifest_path = Path(sys.executable).parent / "manifest.json"
        if manifest_path.exists():
            more_info += get_manifest_package_info(manifest_path)
        else:
            more_info += "Manifest file not found."
    else:
        more_info = get_installed_package_info()
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
