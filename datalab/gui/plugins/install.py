# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Plugin installation
-------------------

Install plugins from a wheel or module file, with the user's consent, and list
the plugins installed this way.
"""

from __future__ import annotations

import os.path as osp
from collections.abc import Callable
from datetime import datetime
from html import escape

from guidata.configtools import get_icon
from guidata.qthelpers import win32_fix_title_bar_background
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from qtpy.compat import getopenfilename

from datalab.config import Conf, _
from datalab.plugins import PluginRegistry
from datalab.plugins.base import get_installed_plugin_store
from datalab.plugins.store import (
    WHEEL_KIND,
    InstalledPlugin,
    InstalledPluginStore,
    InstallResult,
    PluginInstallError,
)
from datalab.utils.qthelpers import show_in_folder as _show_in_folder
from datalab.widgets.expandabletext import apply_subdued_color


def _format_size(size_bytes: int) -> str:
    return QC.QLocale.system().formattedDataSize(size_bytes)


def _format_date(timestamp: str) -> str:
    try:
        moment = datetime.fromisoformat(timestamp).astimezone()
    except ValueError:
        return timestamp
    return QC.QLocale.system().toString(
        QC.QDate(moment.year, moment.month, moment.day), QC.QLocale.ShortFormat
    )


def _plugin_title(plugin: InstalledPlugin) -> str:
    if plugin.kind == WHEEL_KIND:
        return f"{plugin.name} {plugin.version}"
    return plugin.filename


class PluginConsentDialog(QW.QDialog):
    """Dialog asking the user to confirm the installation of a plugin file.

    Args:
        manifest: Wheel or module manifest returned by the installed plugin store
        parent: Parent widget
        origin: Where the file comes from, when not chosen by the user
    """

    def __init__(
        self,
        manifest: dict,
        parent: QW.QWidget | None = None,
        origin: str | None = None,
    ) -> None:
        super().__init__(parent)
        win32_fix_title_bar_background(self)
        self.setWindowTitle(_("Install plugin"))
        layout = QW.QVBoxLayout()
        self.setLayout(layout)

        form = QW.QFormLayout()
        if origin:
            form.addRow(_("Source:"), QW.QLabel(origin))
        if "distribution" in manifest:
            form.addRow(_("Plugin:"), QW.QLabel(manifest["distribution"]))
            form.addRow(_("Version:"), QW.QLabel(manifest["version"]))
            if manifest["summary"]:
                summary = QW.QLabel(manifest["summary"])
                summary.setWordWrap(True)
                form.addRow(_("Description:"), summary)
            entry_points = "<br>".join(
                escape(f"{entry['module']}:{entry['attribute']}")
                for entry in manifest["entry_points"]
            )
            form.addRow(_("Plugin classes:"), QW.QLabel(entry_points))
            dependencies = "<br>".join(
                escape(
                    f"{dependency['requirement']} "
                    f"({dependency['installed_version'] or _('not applicable')})"
                )
                for dependency in manifest["dependencies"]
            )
            form.addRow(_("Dependencies:"), QW.QLabel(dependencies or _("None")))
        else:
            form.addRow(_("Plugin module:"), QW.QLabel(manifest["name"]))
        form.addRow(_("File:"), QW.QLabel(manifest["filename"]))
        form.addRow(_("Size:"), QW.QLabel(_format_size(manifest["size_bytes"])))
        digest = QW.QLabel(manifest["sha256"])
        digest.setTextInteractionFlags(QC.Qt.TextSelectableByMouse)
        form.addRow(_("SHA-256:"), digest)
        layout.addLayout(form)

        warning = QW.QLabel(
            _(
                "A plugin runs with the same rights as DataLab: it may read and "
                "modify your files. Install only plugins from authors you trust."
            )
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)

        button_box = QW.QDialogButtonBox(QW.QDialogButtonBox.Cancel)
        self.install_button = button_box.addButton(
            _("Install"), QW.QDialogButtonBox.AcceptRole
        )
        self.install_button.setAutoDefault(False)
        button_box.button(QW.QDialogButtonBox.Cancel).setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)


class InstalledPluginItemWidget(QW.QWidget):
    """Row describing a plugin installed from a file.

    Args:
        plugin: Installed plugin
        path: Installed plugin file path
        parent: Parent widget
    """

    def __init__(
        self, plugin: InstalledPlugin, path: str, parent: QW.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.plugin = plugin
        layout = QW.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        label = QW.QLabel(
            f"<b>{escape(_plugin_title(plugin))}</b><br>"
            + escape(
                _("Installed on %s from %s")
                % (_format_date(plugin.installed_at), plugin.filename)
            )
        )
        label.setWordWrap(True)
        layout.addWidget(label, 1)

        self.show_in_folder_button = QW.QToolButton()
        self.show_in_folder_button.setIcon(get_icon("show_in_folder.svg"))
        self.show_in_folder_button.setToolTip(_("Show in folder"))
        self.show_in_folder_button.setAutoRaise(True)
        self.show_in_folder_button.clicked.connect(lambda: _show_in_folder(path))
        layout.addWidget(self.show_in_folder_button)

        self.uninstall_button = QW.QPushButton(_("Uninstall"))
        layout.addWidget(self.uninstall_button)


class InstalledPluginsWidget(QW.QWidget):
    """Tab installing plugins from a file and listing the installed ones.

    Args:
        reload_plugins: Callback applying the configuration and reloading plugins
        parent: Parent widget
    """

    def __init__(
        self, reload_plugins: Callable[[], None], parent: QW.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._reload_plugins = reload_plugins
        self.item_widgets: list[InstalledPluginItemWidget] = []
        layout = QW.QVBoxLayout()
        self.setLayout(layout)

        intro = QW.QLabel(
            _(
                "Install a plugin shared as a wheel file (<code>.whl</code>) or as "
                "a Python module (<code>datalab_*.py</code>). Wheels must be pure "
                "Python and rely only on packages already provided by DataLab."
            )
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        install_layout = QW.QHBoxLayout()
        self.install_button = QW.QPushButton(
            get_icon("metadata_add.svg"), _("Install from file...")
        )
        self.install_button.clicked.connect(self.install_from_file)
        install_layout.addWidget(self.install_button)
        install_layout.addStretch()
        layout.addLayout(install_layout)
        layout.addSpacing(12)

        title = QW.QLabel(_("Plugins installed from a file"))
        title_font = title.font()
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        self.items_layout = QW.QVBoxLayout()
        self.items_layout.setContentsMargins(18, 0, 0, 0)
        layout.addLayout(self.items_layout)
        self.placeholder = QW.QLabel(_("No plugin has been installed from a file."))
        self.placeholder.setWordWrap(True)
        apply_subdued_color(self.placeholder)
        self.items_layout.addWidget(self.placeholder)
        layout.addStretch()
        self.refresh()

    @property
    def store(self) -> InstalledPluginStore:
        """Return the installed plugin store."""
        return get_installed_plugin_store()

    def refresh(self) -> None:
        """Rebuild the list of installed plugins."""
        for widget in self.item_widgets:
            self.items_layout.removeWidget(widget)
            widget.deleteLater()
        self.item_widgets.clear()
        try:
            plugins = self.store.get_plugins()
        except PluginInstallError as exc:
            self.placeholder.setText(str(exc))
            self.placeholder.show()
            return
        self.placeholder.setText(_("No plugin has been installed from a file."))
        self.placeholder.setVisible(not plugins)
        for plugin in plugins:
            widget = InstalledPluginItemWidget(plugin, self.store.get_path(plugin))
            widget.uninstall_button.clicked.connect(
                lambda _checked=False, item=plugin: self.uninstall(item)
            )
            self.item_widgets.append(widget)
            self.items_layout.addWidget(widget)

    def select_file(self) -> str | None:
        """Ask the user for the plugin file to install."""
        filename, _filter = getopenfilename(
            self,
            _("Install plugin from file"),
            Conf.base_dir.get(osp.expanduser("~")),
            _("DataLab plugins") + " (*.whl *.py)",
        )
        return filename or None

    def confirm_installation(self, manifest: dict, origin: str | None = None) -> bool:
        """Return True if the user agrees to install the inspected file."""
        return bool(PluginConsentDialog(manifest, self, origin).exec())

    def install_from_file(self) -> InstallResult | None:
        """Install a plugin file chosen by the user.

        Returns:
            Installation result, or None if cancelled or refused
        """
        path = self.select_file()
        if not path:
            return None
        return self.install_path(path)

    def install_path(
        self, path: str, origin: str | None = None
    ) -> InstallResult | None:
        """Install a plugin file after the user's consent.

        Args:
            path: Wheel or module file path
            origin: Where the file comes from, shown in the consent dialog

        Returns:
            Installation result, or None if refused
        """
        is_wheel = path.lower().endswith(".whl")
        try:
            if is_wheel:
                manifest = self.store.inspect_wheel_file(path)
            else:
                manifest = self.store.inspect_module_file(path)
            if not self.confirm_installation(manifest, origin=origin):
                return None
            install = (
                self.store.install_wheel if is_wheel else self.store.install_module
            )
            result = install(path, expected_sha256=manifest["sha256"])
        except (OSError, ValueError) as exc:
            QW.QMessageBox.warning(
                self,
                _("Install plugin"),
                _("This file cannot be installed as a plugin:") + f"\n\n{exc}",
            )
            return None
        self.refresh()
        title = _plugin_title(result.plugin)
        if result.restart_required:
            QW.QMessageBox.information(
                self,
                _("Install plugin"),
                _("%s will be loaded at the next start of DataLab.") % title,
            )
        elif self._ask_reload(_("%s has been installed.") % title):
            self._reload_plugins()
            if self._enable_plugins_from(result.plugin):
                self._reload_plugins()
        return result

    def uninstall(self, plugin: InstalledPlugin) -> None:
        """Uninstall a plugin after confirmation."""
        title = _plugin_title(plugin)
        reply = QW.QMessageBox.question(
            self,
            _("Uninstall plugin"),
            _("Do you want to uninstall %s?") % title,
            QW.QMessageBox.Yes | QW.QMessageBox.No,
            QW.QMessageBox.No,
        )
        if reply != QW.QMessageBox.Yes:
            return
        try:
            self.store.uninstall(plugin)
        except (OSError, ValueError) as exc:
            QW.QMessageBox.warning(self, _("Uninstall plugin"), str(exc))
            return
        self.refresh()
        if self._ask_reload(_("%s has been uninstalled.") % title):
            self._reload_plugins()

    def _ask_reload(self, message: str) -> bool:
        reply = QW.QMessageBox.question(
            self,
            _("Reload Plugins"),
            message + "\n\n" + _("Do you want to reload plugins now?"),
            QW.QMessageBox.Yes | QW.QMessageBox.No,
            QW.QMessageBox.Yes,
        )
        return reply == QW.QMessageBox.Yes

    def _enable_plugins_from(self, plugin: InstalledPlugin) -> bool:
        """Enable the classes of a newly installed plugin in an explicit list.

        Returns:
            True if the list of enabled plugins changed
        """
        enabled_ids = Conf.plugins_enabled_list.get(None)
        if enabled_ids is None:
            return False
        new_ids = [
            plugin_class.get_plugin_id()
            for plugin_class in PluginRegistry.get_plugin_classes()
            if getattr(plugin_class, "__plugin_filepath__", None)
            and self.store.find_plugin(plugin_class.__plugin_filepath__) == plugin
            and plugin_class.get_plugin_id() not in enabled_ids
        ]
        if not new_ids:
            return False
        Conf.plugins_enabled_list.set(list(enabled_ids) + new_ids)
        return True
