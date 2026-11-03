# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Available plugins
-----------------

Browse the DataLab plugin catalog and the local plugin catalogs, and install or
update their plugins.
"""

from __future__ import annotations

import os.path as osp
import sys
import tempfile
from collections.abc import Callable, Sequence
from html import escape
from importlib import metadata as importlib_metadata
from importlib.machinery import PathFinder

from guidata.configtools import get_icon
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import (
    PLUGIN_ERROR_COLOR,
    Conf,
    _,
    get_env_plugin_catalog_paths,
    get_plugin_catalog_paths,
    normalize_plugin_paths,
)
from datalab.gui.plugins.install import InstalledPluginsWidget, run_plugin_task
from datalab.plugins.catalog import (
    CatalogPlugin,
    CatalogRelease,
    CatalogsContent,
    download_dependency_wheels,
    download_release,
    load_catalogs,
)
from datalab.plugins.dependencies import DEPENDENCIES_DIRECTORY
from datalab.plugins.store import MODULE_KIND, WHEEL_KIND
from datalab.widgets.expandabletext import apply_palette_color, apply_subdued_color

CATALOG_PAGE_URL = "https://datalab-platform.com/plugins/"


def _is_newer(version: str, installed: str) -> bool:
    try:
        return Version(version) > Version(installed)
    except InvalidVersion:
        return version != installed


def _tier_label(plugin: CatalogPlugin) -> str:
    return _("Official") if plugin.tier == "official" else _("Community")


def _source_label(plugin: CatalogPlugin) -> str:
    return _("Local catalog") if plugin.source else _tier_label(plugin)


def merge_catalog_directories(configured: Sequence[str]) -> list[str]:
    """Return the local catalog folders, by priority.

    Folders of the ``DATALAB_PLUGINS_CATALOG`` environment variable come first,
    then the folders configured in DataLab.

    Args:
        configured: Folders configured in DataLab
    """
    return normalize_plugin_paths([*get_env_plugin_catalog_paths(), *configured])


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
        installed: tuple[str, str, str],
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
        source_link = (
            f'<a href="{escape(plugin.repository, quote=True)}">'
            f"{escape(_('Source code'))}</a>"
            if plugin.repository
            else ""
        )
        details = " &middot; ".join(
            part for part in (escape(plugin.license), source_link) if part
        )
        label = QW.QLabel(
            f"<b>{escape(plugin.name)}</b>{version} "
            f"<small>({escape(_source_label(plugin))})</small><br>"
            f"{escape(plugin.summary)}<br><small>{details}</small>"
        )
        if plugin.source:
            label.setToolTip(_("Local catalog: %s") % plugin.source)
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
        self, release: CatalogRelease | None, installed: tuple[str, str, str]
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
        how, version, digest = installed
        if how == "unmanaged":
            return _("Provided outside the plugin store"), None
        if how == "conflict":
            return _("Conflicting installed plugin: %s") % version, None
        if how == "error":
            return version, None
        if how == "environment":
            return _("Installed in the Python environment (%s)") % version, None
        if how == "file" and plugin.kind == MODULE_KIND:
            notice = (
                _("Deprecated: %s") % plugin.status_reason
                if plugin.status == "deprecated"
                else ""
            )
            if digest == release.sha256:
                return notice or _("Installed"), None
            if version and release.version:
                try:
                    if Version(release.version) < Version(version):
                        return _("Installed: %s") % version, None
                    if Version(release.version) > Version(version):
                        return notice or _("Installed: %s") % version, _(
                            "Update to %s"
                        ) % release.version
                except InvalidVersion:
                    pass
            return notice or _("Different installed revision"), _("Replace...")
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

    The local catalogs are listed first: a plugin they provide hides the same
    plugin in the online catalog.

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
        self._download: tuple[CatalogPlugin, tempfile.TemporaryDirectory] | None = None
        self._busy = False
        self._installer.busy_changed.connect(
            lambda busy: self._set_busy(_("Installing plugin...") if busy else None)
        )
        self._directories = merge_catalog_directories(get_plugin_catalog_paths())
        #: Catalog URL and local folders of the last loading
        self._loaded_sources: tuple[str, tuple[str, ...]] | None = None
        layout = QW.QVBoxLayout()
        self.setLayout(layout)

        intro = QW.QLabel(
            _(
                'Plugins listed in the <a href="%s">DataLab plugin catalog</a> '
                "and in the local catalogs of the <b>Advanced</b> tab, "
                "which take precedence. Official plugins are maintained by the "
                "DataLab team, community plugins by their authors, and local "
                "catalogs by your organization: install only plugins from "
                "authors you trust."
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

        self.errors_label = QW.QLabel()
        self.errors_label.setWordWrap(True)
        self.errors_label.setTextFormat(QC.Qt.PlainText)
        apply_palette_color(self.errors_label, QG.QColor(PLUGIN_ERROR_COLOR))
        self.errors_label.hide()
        layout.addWidget(self.errors_label)

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
        """Load the catalogs when the tab is shown, unless already loaded."""
        super().showEvent(event)
        if not self._busy and self._loaded_sources != self._get_sources():
            self.refresh()

    def get_catalog_directories(self) -> list[str]:
        """Return the local catalog folders, by priority."""
        return list(self._directories)

    def set_catalog_directories(self, directories: Sequence[str]) -> None:
        """Set the local catalog folders, by priority.

        The catalogs are loaded again when the tab is visible, otherwise when it
        is shown.
        """
        self._directories = list(directories)
        if self.isVisible() and not self._busy:
            self.refresh()

    def _get_sources(self) -> tuple[str, tuple[str, ...]]:
        return Conf.plugins_catalog_url.get(), tuple(self._directories)

    def run_task(
        self,
        function: Callable[[], object],
        on_success: Callable[[object], None],
        on_failure: Callable[[str], None],
    ) -> None:
        """Run a function in a background thread and report its outcome."""
        run_plugin_task(function, on_success, on_failure)

    def refresh(self) -> None:
        """Load the catalogs again."""
        sources = self._get_sources()
        self._loaded_sources = sources
        url, directories = sources
        self.errors_label.hide()
        if not url and not directories:
            self.plugins = None
            self._populate()
            apply_subdued_color(self.status_label)
            self.status_label.setText(_("No plugin catalog is configured."))
            return
        self._set_busy(_("Loading the plugin catalogs..."))
        self.run_task(
            lambda: load_catalogs(url, directories), self._on_catalog, self._on_error
        )

    def install(self, plugin: CatalogPlugin) -> None:
        """Download the newest release of a plugin, then install it."""
        release = plugin.get_installable_release()
        if release is None or self._busy:
            return
        directory = tempfile.TemporaryDirectory(prefix="datalab-plugin-")
        self._download = (plugin, directory)
        self._set_busy(_("Downloading %s...") % release.filename)
        self.run_task(
            lambda: self._download_release(release, directory.name),
            self._on_downloaded,
            self._on_download_failed,
        )

    @staticmethod
    def _download_release(release: CatalogRelease, directory: str) -> str:
        path = download_release(release, directory)
        if release.dependency_wheels:
            download_dependency_wheels(
                release, str(osp.join(directory, DEPENDENCIES_DIRECTORY))
            )
        return path

    def _set_busy(self, message: str | None) -> None:
        self._busy = message is not None
        self.refresh_button.setEnabled(not self._busy)
        for widget in self.item_widgets:
            if widget.action_button is not None:
                widget.action_button.setEnabled(not self._busy)
        if message is not None:
            apply_subdued_color(self.status_label)
            self.status_label.setText(message)

    def _on_catalog(self, content: CatalogsContent) -> None:
        self.plugins = list(content.plugins)
        self._set_busy(None)
        url, directories = self._loaded_sources
        if len(directories) + bool(url) > 1:
            text = _("Plugins in the catalogs: %d") % len(self.plugins)
        else:
            text = _("Plugins in the catalog: %d") % len(self.plugins)
        self.status_label.setText(text)
        self.errors_label.setText(
            "\n".join(f"\u26a0 {message}" for message in content.errors)
        )
        self.errors_label.setVisible(bool(content.errors))
        self._populate()
        if self.isVisible() and self._loaded_sources != self._get_sources():
            # The sources changed during the loading
            self.refresh()

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
        if plugin.source:
            origin = _("Local plugin catalog (%s)") % plugin.source
        else:
            origin = _("DataLab plugin catalog (%s)") % _tier_label(plugin)
        wheelhouse = osp.join(directory.name, DEPENDENCIES_DIRECTORY)
        if plugin.source and not osp.isdir(wheelhouse):
            wheelhouse = osp.join(plugin.source, DEPENDENCIES_DIRECTORY)

        def finished() -> None:
            directory.cleanup()
            self._populate()

        self._installer.install_path(
            path,
            origin=origin,
            wheelhouse=wheelhouse,
            on_finished=finished,
            catalog_plugin=plugin,
        )

    def _on_download_failed(self, message: str) -> None:
        _plugin, directory = self._download
        self._download = None
        directory.cleanup()
        self._set_busy(None)
        self.status_label.setText("")
        QW.QMessageBox.warning(self, _("Install plugin"), message)

    def _get_installed(self, plugin: CatalogPlugin) -> tuple[str, str, str]:
        """Return how a catalog plugin is installed, and its version."""
        try:
            installed_plugins = self._installer.store.get_plugins()
        except (OSError, ValueError) as exc:
            return "error", str(exc), ""
        if plugin.kind == MODULE_KIND:
            for installed in installed_plugins:
                same_module = installed.kind == MODULE_KIND and osp.normcase(
                    installed.name
                ) == osp.normcase(plugin.module)
                if same_module:
                    if installed.name != plugin.module:
                        return "conflict", installed.name, ""
                    if installed.catalog_id and installed.catalog_id != plugin.id:
                        return "conflict", installed.catalog_id, ""
                    return "file", installed.version, installed.sha256
                if installed.catalog_id == plugin.id:
                    return "conflict", installed.name, ""
            managed = osp.normcase(osp.realpath(self._installer.store.modules_dir))
            paths = [
                entry
                for entry in sys.path
                if osp.normcase(osp.realpath(entry)) != managed
            ]
            if PathFinder.find_spec(plugin.module, paths) is not None:
                return "unmanaged", "", ""
            return "", "", ""
        if not plugin.distribution:
            return "", "", ""
        key = canonicalize_name(plugin.distribution)
        for installed in installed_plugins:
            if (
                installed.kind == WHEEL_KIND
                and canonicalize_name(installed.name) == key
            ):
                return "file", installed.version, installed.sha256
            if installed.catalog_id == plugin.id and installed.kind != WHEEL_KIND:
                return "conflict", installed.name, ""
        try:
            return "environment", importlib_metadata.version(plugin.distribution), ""
        except importlib_metadata.PackageNotFoundError:
            return "", "", ""

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
