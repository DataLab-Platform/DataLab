# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Plugin installation
-------------------

Install plugins from a wheel or module file, with the user's consent, and list
the plugins installed this way.
"""

from __future__ import annotations

import logging
import os.path as osp
from collections.abc import Callable
from datetime import datetime
from html import escape
from typing import TypeVar

from guidata.configtools import get_icon
from guidata.qthelpers import win32_fix_title_bar_background
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from qtpy.compat import getopenfilename

from datalab.config import Conf, _
from datalab.plugins.base import (
    enable_installed_plugin_classes,
    get_installed_plugin_store,
)
from datalab.plugins.catalog import CatalogPlugin
from datalab.plugins.dependencies import DependencyPlan
from datalab.plugins.modules import apply_catalog_metadata
from datalab.plugins.store import (
    WHEEL_KIND,
    InstalledPlugin,
    InstalledPluginStore,
    InstallResult,
    PluginInstallError,
)
from datalab.utils.qthelpers import show_in_folder as _show_in_folder
from datalab.widgets.expandabletext import apply_subdued_color

_Result = TypeVar("_Result")
_RUNNING_TASKS: set[_Task] = set()


class _Task(QC.QThread):
    """Run file and dependency operations outside the GUI thread."""

    succeeded = QC.Signal(object)
    failed = QC.Signal(str)

    def __init__(self, function: Callable[[], object]) -> None:
        super().__init__()
        self._function = function

    def run(self) -> None:
        """Call the operation and report its errors, expected or not."""
        try:
            result = self._function()
        except (OSError, ValueError) as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Escaping this thread, the exception would emit no signal at all: the
            # caller would wait forever for the end of the operation
            logging.getLogger(__name__).exception("Plugin operation failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.succeeded.emit(result)


def run_plugin_task(
    function: Callable[[], _Result],
    on_success: Callable[[_Result], None],
    on_failure: Callable[[str], None],
) -> None:
    """Keep an asynchronous plugin operation alive until its thread finishes."""
    task = _Task(function)
    _RUNNING_TASKS.add(task)
    task.succeeded.connect(on_success)
    task.failed.connect(on_failure)
    task.finished.connect(lambda: _RUNNING_TASKS.discard(task))
    task.start()


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
    if plugin.version:
        return f"{plugin.filename} {plugin.version}"
    return f"{plugin.filename} ({_('revision %s') % plugin.sha256[:12]})"


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
            dependencies = []
            for dependency in manifest["dependencies"]:
                version = dependency["installed_version"]
                if version is None:
                    version = (
                        _("not installed")
                        if dependency["applies"]
                        else _("not applicable")
                    )
                dependencies.append(escape(f"{dependency['requirement']} ({version})"))
            form.addRow(
                _("Dependencies:"), QW.QLabel("<br>".join(dependencies) or _("None"))
            )
            additional = manifest.get("additional_dependencies", [])
            if additional:
                details = "<br>".join(
                    escape(
                        f"{wheel['distribution']} {wheel['version']} "
                        f"({_format_size(wheel['size_bytes'])})"
                    )
                    + f"<br><small>SHA-256: {escape(wheel['sha256'])}</small>"
                    for wheel in additional
                )
                label = QW.QLabel(details)
                label.setWordWrap(True)
                label.setTextInteractionFlags(QC.Qt.TextSelectableByMouse)
                form.addRow(_("Additional packages from the local wheelhouse:"), label)
                source = QW.QLabel(escape(manifest["dependency_source"]))
                source.setWordWrap(True)
                form.addRow(_("Dependency source:"), source)
        else:
            form.addRow(_("Plugin module:"), QW.QLabel(manifest["name"]))
            if manifest["display_name"]:
                form.addRow(_("Plugin:"), QW.QLabel(escape(manifest["display_name"])))
            if manifest["version"]:
                form.addRow(_("Version:"), QW.QLabel(escape(manifest["version"])))
            else:
                form.addRow(_("Revision:"), QW.QLabel(manifest["sha256"][:12]))
            if manifest["summary"]:
                description = QW.QLabel(escape(manifest["summary"]))
                description.setWordWrap(True)
                form.addRow(_("Description:"), description)
            if manifest["warnings"]:
                warning = QW.QLabel(
                    _(
                        "Some metadata cannot be read without executing the file. "
                        "Catalog metadata is used when provided; dependencies are "
                        "not installed for single-file plugins."
                    )
                )
                warning.setWordWrap(True)
                form.addRow(warning)
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

    busy_changed = QC.Signal(bool)

    def __init__(
        self, reload_plugins: Callable[[], None], parent: QW.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._reload_plugins = reload_plugins
        self._pending_install: tuple[str, str | None, str] | None = None
        self._pending_finish: Callable[[], None] | None = None
        self._dependency_plan: DependencyPlan | None = None
        self._manifest: dict | None = None
        self.item_widgets: list[InstalledPluginItemWidget] = []
        layout = QW.QVBoxLayout()
        self.setLayout(layout)

        intro = QW.QLabel(
            _(
                "Install a plugin shared as a wheel file (<code>.whl</code>) or as "
                "a Python module (<code>datalab_*.py</code>). Wheels must be pure "
                "Python. Missing dependencies may be supplied as wheels in an "
                "adjacent <code>dependencies</code> folder, without Internet "
                "access or changing packages already installed."
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
        self.status_label = QW.QLabel()
        self.status_label.setWordWrap(True)
        apply_subdued_color(self.status_label)
        layout.addWidget(self.status_label)

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
        self,
        path: str,
        origin: str | None = None,
        *,
        wheelhouse: str | None = None,
        on_finished: Callable[[], None] | None = None,
        catalog_plugin: CatalogPlugin | None = None,
    ) -> InstallResult | None:
        """Install a plugin file after the user's consent.

        Args:
            path: Wheel or module file path
            origin: Where the file comes from, shown in the consent dialog

        Returns:
            Installation result, or None if refused
        """
        is_wheel = path.lower().endswith(".whl")
        asynchronous = False
        try:
            if self._pending_install is not None:
                raise PluginInstallError("Another plugin installation is in progress")
            if is_wheel:
                manifest = self.store.inspect_wheel_file(
                    path, allow_missing_dependencies=True
                )
                if any(
                    item["applies"] and not item["compatible"]
                    for item in manifest["dependencies"]
                ):
                    self._pending_install = (
                        path,
                        origin,
                        catalog_plugin.id if catalog_plugin else "",
                    )
                    self._pending_finish = on_finished
                    self._manifest = manifest
                    self._set_busy(True, _("Checking the local dependency wheels..."))
                    run_plugin_task(
                        lambda: self.store.prepare_wheel_file(path, wheelhouse),
                        self._on_prepared,
                        self._on_install_failed,
                    )
                    asynchronous = True
                    return None
            else:
                manifest = self.store.inspect_module_file(path)
                if catalog_plugin is not None:
                    release = catalog_plugin.get_installable_release()
                    if release is None or catalog_plugin.module != manifest["name"]:
                        raise PluginInstallError(
                            "The module does not match the catalog"
                        )
                    manifest = apply_catalog_metadata(
                        manifest,
                        plugin_id=catalog_plugin.id,
                        name=catalog_plugin.name,
                        version=release.version,
                        summary=catalog_plugin.summary,
                    )
            if not self.confirm_installation(manifest, origin=origin):
                return None
            if is_wheel:
                result = self.store.install_wheel(
                    path,
                    expected_sha256=manifest["sha256"],
                    catalog_id=catalog_plugin.id if catalog_plugin else "",
                )
            else:
                result = self.store.install_module(
                    path,
                    expected_sha256=manifest["sha256"],
                    catalog_id=catalog_plugin.id if catalog_plugin else "",
                    catalog_name=catalog_plugin.name if catalog_plugin else "",
                    catalog_version=manifest["version"],
                )
        except (OSError, ValueError) as exc:
            QW.QMessageBox.warning(
                self,
                _("Install plugin"),
                _("This file cannot be installed as a plugin:") + f"\n\n{exc}",
            )
            return None
        finally:
            if not asynchronous and on_finished is not None:
                on_finished()
        self._present_result(result)
        return result

    def _present_result(self, result: InstallResult) -> None:
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

    def _set_busy(self, busy: bool, text: str = "") -> None:
        self.install_button.setEnabled(not busy)
        for item in self.item_widgets:
            item.uninstall_button.setEnabled(not busy)
        self.status_label.setText(text)
        self.busy_changed.emit(busy)

    def _on_prepared(self, prepared: tuple[dict, DependencyPlan | None]) -> None:
        manifest, self._dependency_plan = prepared
        assert self._pending_install is not None
        path, origin, catalog_id = self._pending_install
        if self._manifest is None or manifest["sha256"] != self._manifest["sha256"]:
            self._on_install_failed(
                _("The plugin changed while checking dependencies.")
            )
            return
        if not self.confirm_installation(manifest, origin=origin):
            self._finish_installation()
            return
        self._set_busy(True, _("Installing the plugin and its dependencies..."))
        run_plugin_task(
            lambda: self.store.install_wheel(
                path,
                expected_sha256=manifest["sha256"],
                dependency_plan=self._dependency_plan,
                catalog_id=catalog_id,
            ),
            self._on_installed,
            self._on_install_failed,
        )

    def _on_installed(self, result: InstallResult) -> None:
        try:
            self._present_result(result)
        finally:
            self._finish_installation()

    def _on_install_failed(self, message: str) -> None:
        self._finish_installation()
        QW.QMessageBox.warning(self, _("Install plugin"), message)

    def _finish_installation(self) -> None:
        if self._dependency_plan is not None:
            self._dependency_plan.close()
        self._dependency_plan = None
        self._pending_install = None
        self._manifest = None
        self._set_busy(False)
        callback = self._pending_finish
        self._pending_finish = None
        if callback is not None:
            callback()

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
        changed, _recognized = enable_installed_plugin_classes(
            self.store, [plugin.name]
        )
        return changed
