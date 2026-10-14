# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Available plugins
-----------------

Browse the DataLab plugin catalog and install or update its plugins.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from collections.abc import Callable
from html import escape
from importlib import metadata as importlib_metadata

from guidata.configtools import get_icon
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import PLUGIN_ERROR_COLOR, Conf, _
from datalab.gui.plugins.install import InstalledPluginsWidget
from datalab.plugins.catalog import (
    CatalogPlugin,
    CatalogRelease,
    download_release,
    fetch_catalog,
)
from datalab.plugins.store import WHEEL_KIND
from datalab.widgets.expandabletext import apply_palette_color, apply_subdued_color

CATALOG_PAGE_URL = "https://datalab-platform.com/plugins/"

#: Running tasks, kept alive until their thread finishes
_RUNNING_TASKS: set[_Task] = set()


class _Task(QC.QThread):
    """Run a function in a background thread."""

    succeeded = QC.Signal(object)
    failed = QC.Signal(str)

    def __init__(self, function: Callable[[], object]) -> None:
        super().__init__()
        self._function = function

    def run(self) -> None:  # pragma: no cover - thread entry point
        """Call the function and emit its result or error message."""
        try:
            result = self._function()
        except (OSError, ValueError) as exc:
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(result)


def _is_newer(version: str, installed: str) -> bool:
    try:
        return Version(version) > Version(installed)
    except InvalidVersion:
        return version != installed


def _tier_label(plugin: CatalogPlugin) -> str:
    return _("Official") if plugin.tier == "official" else _("Community")


class AvailablePluginItemWidget(QW.QWidget):
    """Row describing a catalog plugin and the action it allows.

    Args:
        plugin: Catalog plugin
        installed: How the plugin is installed (``""``, ``"file"`` or
         ``"environment"``) and its installed version
        parent: Parent widget
    """

    def __init__(
        self,
        plugin: CatalogPlugin,
        installed: tuple[str, str],
        parent: QW.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.plugin = plugin
        self.action_button: QW.QPushButton | None = None
        release = plugin.get_installable_release()
        layout = QW.QHBoxLayout()
        layout.setContentsMargins(0, 4, 0, 4)
        self.setLayout(layout)

        version = f" {escape(release.version)}" if release else ""
        details = " &middot; ".join(
            part
            for part in (
                escape(plugin.license),
                f'<a href="{escape(plugin.repository, quote=True)}">'
                f"{escape(_('Source code'))}</a>",
            )
            if part
        )
        label = QW.QLabel(
            f"<b>{escape(plugin.name)}</b>{version} "
            f"<small>({escape(_tier_label(plugin))})</small><br>"
            f"{escape(plugin.summary)}<br><small>{details}</small>"
        )
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        label.setTextInteractionFlags(QC.Qt.TextBrowserInteraction)
        layout.addWidget(label, 1)

        status, action = self._describe(release, installed)
        self.status_label = QW.QLabel(status)
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(QC.Qt.AlignRight | QC.Qt.AlignVCenter)
        if plugin.status == "revoked":
            apply_palette_color(self.status_label, QG.QColor(PLUGIN_ERROR_COLOR))
        else:
            apply_subdued_color(self.status_label)
        layout.addWidget(self.status_label)
        if action:
            self.action_button = QW.QPushButton(action)
            layout.addWidget(self.action_button)

    def _describe(
        self, release: CatalogRelease | None, installed: tuple[str, str]
    ) -> tuple[str, str | None]:
        """Return the status text and the action button text, if any."""
        plugin = self.plugin
        if plugin.status == "revoked":
            return _("Withdrawn: %s") % plugin.status_reason, None
        if release is None:
            if plugin.get_installable_release(python_version="") is not None:
                python = f"{sys.version_info.major}.{sys.version_info.minor}"
                return _("Not available for Python %s") % python, None
            return _("Not available for DataLab desktop"), None
        how, version = installed
        if how == "environment":
            return _("Installed in the Python environment (%s)") % version, None
        if how == "file" and not _is_newer(release.version, version):
            return _("Installed"), None
        status = ""
        if plugin.status == "deprecated":
            status = _("Deprecated: %s") % plugin.status_reason
        if how == "file":
            status = status or _("Installed: %s") % version
            return status, _("Update to %s") % release.version
        return status, _("Install")


class AvailablePluginsWidget(QW.QWidget):
    """Tab listing the catalog plugins, with install and update actions.

    Args:
        installer: Tab installing plugin files, used once a wheel is downloaded
        parent: Parent widget
    """

    def __init__(
        self, installer: InstalledPluginsWidget, parent: QW.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._installer = installer
        self.plugins: list[CatalogPlugin] | None = None
        self.item_widgets: list[AvailablePluginItemWidget] = []
        self._download: tuple[CatalogPlugin, str] | None = None
        self._busy = False
        layout = QW.QVBoxLayout()
        self.setLayout(layout)

        intro = QW.QLabel(
            _(
                'Plugins listed in the <a href="%s">DataLab plugin catalog</a>. '
                "Official plugins are maintained by the DataLab team, community "
                "plugins by their authors: install only plugins from authors you "
                "trust."
            )
            % CATALOG_PAGE_URL
        )
        intro.setWordWrap(True)
        intro.setOpenExternalLinks(True)
        layout.addWidget(intro)

        controls = QW.QHBoxLayout()
        self.search_edit = QW.QLineEdit()
        self.search_edit.setPlaceholderText(_("Search plugins"))
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._populate)
        controls.addWidget(self.search_edit, 1)
        self.refresh_button = QW.QPushButton(get_icon("refresh-auto.svg"), _("Refresh"))
        self.refresh_button.clicked.connect(self.refresh)
        controls.addWidget(self.refresh_button)
        layout.addLayout(controls)

        self.status_label = QW.QLabel()
        self.status_label.setWordWrap(True)
        apply_subdued_color(self.status_label)
        layout.addWidget(self.status_label)

        scroll = QW.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
        container = QW.QWidget()
        self.items_layout = QW.QVBoxLayout()
        self.items_layout.setContentsMargins(6, 0, 6, 0)
        self.items_layout.addStretch()
        container.setLayout(self.items_layout)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

    def showEvent(self, event: QG.QShowEvent) -> None:  # pylint: disable=C0103
        """Load the catalog the first time the tab is shown."""
        super().showEvent(event)
        if self.plugins is None and not self._busy:
            self.refresh()

    def run_task(
        self,
        function: Callable[[], object],
        on_success: Callable[[object], None],
        on_failure: Callable[[str], None],
    ) -> None:
        """Run a function in a background thread and report its outcome."""
        task = _Task(function)
        _RUNNING_TASKS.add(task)
        task.succeeded.connect(on_success)
        task.failed.connect(on_failure)
        task.finished.connect(lambda: _RUNNING_TASKS.discard(task))
        task.start()

    def refresh(self) -> None:
        """Download the catalog again."""
        url = Conf.plugins_catalog_url.get()
        if not url:
            self.status_label.setText(_("No plugin catalog is configured."))
            return
        self._set_busy(_("Loading the plugin catalog..."))
        self.run_task(lambda: fetch_catalog(url), self._on_catalog, self._on_error)

    def install(self, plugin: CatalogPlugin) -> None:
        """Download the newest release of a plugin, then install it."""
        release = plugin.get_installable_release()
        if release is None or self._busy:
            return
        directory = tempfile.mkdtemp(prefix="datalab-plugin-")
        self._download = (plugin, directory)
        self._set_busy(_("Downloading %s...") % release.filename)
        self.run_task(
            lambda: download_release(release, directory),
            self._on_downloaded,
            self._on_download_failed,
        )

    def _set_busy(self, message: str | None) -> None:
        self._busy = message is not None
        self.refresh_button.setEnabled(not self._busy)
        for widget in self.item_widgets:
            if widget.action_button is not None:
                widget.action_button.setEnabled(not self._busy)
        if message is not None:
            apply_subdued_color(self.status_label)
            self.status_label.setText(message)

    def _on_catalog(self, plugins: list[CatalogPlugin]) -> None:
        self.plugins = plugins
        self._set_busy(None)
        self.status_label.setText(_("Plugins in the catalog: %d") % len(plugins))
        self._populate()

    def _on_error(self, message: str) -> None:
        self._set_busy(None)
        apply_palette_color(self.status_label, QG.QColor(PLUGIN_ERROR_COLOR))
        self.status_label.setText(
            _("The plugin catalog cannot be loaded:") + f" {message}"
        )

    def _on_downloaded(self, path: str) -> None:
        plugin, directory = self._download
        self._download = None
        self._set_busy(None)
        self.status_label.setText("")
        try:
            self._installer.install_path(
                path, origin=_("DataLab plugin catalog (%s)") % _tier_label(plugin)
            )
        finally:
            shutil.rmtree(directory, ignore_errors=True)
        self._populate()

    def _on_download_failed(self, message: str) -> None:
        _plugin, directory = self._download
        self._download = None
        shutil.rmtree(directory, ignore_errors=True)
        self._set_busy(None)
        self.status_label.setText("")
        QW.QMessageBox.warning(self, _("Install plugin"), message)

    def _get_installed(self, plugin: CatalogPlugin) -> tuple[str, str]:
        """Return how a catalog plugin is installed, and its version."""
        if not plugin.distribution:
            return "", ""
        key = canonicalize_name(plugin.distribution)
        try:
            installed_plugins = self._installer.store.get_plugins()
        except ValueError:
            installed_plugins = []
        for installed in installed_plugins:
            if (
                installed.kind == WHEEL_KIND
                and canonicalize_name(installed.name) == key
            ):
                return "file", installed.version
        try:
            return "environment", importlib_metadata.version(plugin.distribution)
        except importlib_metadata.PackageNotFoundError:
            return "", ""

    def _populate(self) -> None:
        """Rebuild the plugin rows matching the search text."""
        for widget in self.item_widgets:
            self.items_layout.removeWidget(widget)
            widget.deleteLater()
        self.item_widgets.clear()
        if self.plugins is None:
            return
        text = self.search_edit.text()
        for plugin in self.plugins:
            if not plugin.matches(text):
                continue
            widget = AvailablePluginItemWidget(plugin, self._get_installed(plugin))
            if widget.action_button is not None:
                widget.action_button.clicked.connect(
                    lambda _checked=False, item=plugin: self.install(item)
                )
            self.item_widgets.append(widget)
            self.items_layout.insertWidget(self.items_layout.count() - 1, widget)
