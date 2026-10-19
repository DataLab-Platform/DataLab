# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Installing plugins from the catalog in the configuration dialog."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from qtpy import QtWidgets as QW

from datalab import __version__
from datalab.config import (
    DATALAB_PLUGINS_CATALOG_ENV_VAR,
    Conf,
    get_plugin_catalog_paths,
    set_plugin_catalog_paths,
)
from datalab.config.persistence import has_persisted_option, remove_persisted_option
from datalab.gui.plugins.config import PluginConfigDialog
from datalab.plugins import PluginRegistry
from datalab.plugins import base as plugin_base
from datalab.plugins.store import InstalledPluginStore
from datalab.tests import datalab_test_app_context
from datalab.tests.backbone.plugins.wheel_factory import (
    make_plugin_wheel,
    wheel_filename,
)

PACKAGE = "datalab_catalog_ui_probe"
DISTRIBUTION = "datalab-catalog-ui-probe"
PLUGIN_ID = "org.example.catalog-ui-probe"
OPTIONS = ("plugins_catalog_url", "plugins_catalog_path_list")


@pytest.fixture(name="store")
def fixture_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use an empty installed plugin store and restore the import state."""
    store = InstalledPluginStore(
        str(tmp_path / "installed_plugins"),
        host_distributions={"datalab-platform": __version__},
    )
    monkeypatch.setattr(plugin_base, "_INSTALLED_PLUGIN_STORE", store)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv(DATALAB_PLUGINS_CATALOG_ENV_VAR, raising=False)
    saved = {
        name: (has_persisted_option(Conf, name), getattr(Conf, name).get())
        for name in OPTIONS
    }
    set_plugin_catalog_paths([])
    yield store
    for name, (persisted, value) in saved.items():
        if persisted:
            getattr(Conf, name).set(value)
        else:
            remove_persisted_option(Conf, name)
    for name in [name for name in sys.modules if name.startswith(PACKAGE)]:
        del sys.modules[name]


def publish_catalog(
    site: Path, versions: tuple[str, ...], requires_python: dict[str, str] | None = None
) -> str:
    """Publish probe plugin releases in a catalog folder and return its URL."""
    releases = []
    for version in sorted(versions, reverse=True):
        python = (requires_python or {}).get(version, ">=3.9")
        data = make_plugin_wheel(
            distribution=DISTRIBUTION,
            version=version,
            package=PACKAGE,
            plugin_id=PLUGIN_ID,
            requires_dist=("datalab-platform>=1.0",),
            requires_python=python,
        )
        digest = hashlib.sha256(data).hexdigest()
        filename = wheel_filename(DISTRIBUTION, version)
        path = site / "wheels" / digest / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        releases.append(
            {
                "version": version,
                "filename": filename,
                "url": f"wheels/{digest}/{filename}",
                "sha256": digest,
                "size": len(data),
                "requires_python": python,
                "targets": ["desktop"],
            }
        )
    catalog = {
        "schema_version": 1,
        "plugins": [
            {
                "id": PLUGIN_ID,
                "name": "Catalog probe",
                "tier": "community",
                "status": "active",
                "repository": "https://github.com/someone/catalog-probe",
                "distribution": DISTRIBUTION,
                "summary": "Probe plugin listed in a test catalog",
                "license": "MIT",
                "releases": releases,
            }
        ],
    }
    (site / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    return (site / "catalog.json").as_uri()


def run_now(function, on_success, on_failure) -> None:
    """Run a catalog task synchronously."""
    try:
        result = function()
    except (OSError, ValueError) as exc:
        on_failure(str(exc))
        return
    on_success(result)


def test_catalog_plugin_is_installed_then_offered_as_update(
    store: InstalledPluginStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A catalog plugin is downloaded, verified, installed and later updated."""
    site = tmp_path / "site"
    Conf.plugins_catalog_url.set(publish_catalog(site, ("1.0.0",)))
    origins: list[str] = []
    monkeypatch.setattr(
        QW.QMessageBox, "question", lambda *_args, **_kwargs: QW.QMessageBox.Yes
    )
    with datalab_test_app_context(console=False) as win:
        dialog = PluginConfigDialog(win)
        available = dialog.available_plugins_widget
        assert dialog.tabs.indexOf(available) == 2
        monkeypatch.setattr(available, "run_task", run_now)
        monkeypatch.setattr(
            dialog.installed_plugins_widget,
            "confirm_installation",
            lambda _manifest, origin=None: origins.append(origin) or True,
        )

        available.refresh()
        (row,) = available.item_widgets
        assert row.action_button.text() == "Install"
        row.action_button.click()
        QW.QApplication.processEvents()

        assert origins == ["DataLab plugin catalog (Community)"]
        assert PluginRegistry.get_plugin(PLUGIN_ID) is not None
        assert [plugin.version for plugin in store.get_plugins()] == ["1.0.0"]
        (row,) = available.item_widgets
        assert row.action_button is None
        assert row.status_label.text() == "Installed"

        available.search_edit.setText("camera")
        assert not available.item_widgets
        available.search_edit.clear()

        Conf.plugins_catalog_url.set(publish_catalog(site, ("1.0.0", "1.1.0")))
        available.refresh()
        (row,) = available.item_widgets
        assert row.action_button.text() == "Update to 1.1.0"

        future = {"2.0.0": ">=4"}
        Conf.plugins_catalog_url.set(publish_catalog(site, ("1.1.0", "2.0.0"), future))
        available.refresh()
        (row,) = available.item_widgets
        assert row.action_button.text() == "Update to 1.1.0"
        Conf.plugins_catalog_url.set(publish_catalog(site, ("2.0.0",), future))
        available.refresh()
        (row,) = available.item_widgets
        assert row.action_button is None
        assert row.status_label.text().startswith("Not available for Python 3.")

        Conf.plugins_catalog_url.set((tmp_path / "missing.json").as_uri())
        available.refresh()
        assert "Cannot download" in available.errors_label.text()
        assert not available.item_widgets
        dialog.close()
        dialog.deleteLater()


def publish_wheel(folder: Path, version: str) -> None:
    """Drop a probe plugin wheel in a local catalog folder."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / wheel_filename(DISTRIBUTION, version)).write_bytes(
        make_plugin_wheel(
            distribution=DISTRIBUTION,
            version=version,
            package=PACKAGE,
            plugin_id=PLUGIN_ID,
            requires_dist=("datalab-platform>=1.0",),
        )
    )


def test_local_catalog_plugin_takes_precedence_and_is_installed(
    store: InstalledPluginStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plugin of a local catalog hides the online one and is installed from it."""
    local = tmp_path / "shared" / "plugins"
    publish_wheel(local, "1.1.0")
    Conf.plugins_catalog_url.set(publish_catalog(tmp_path / "site", ("1.0.0",)))
    set_plugin_catalog_paths([str(local)])
    origins: list[str] = []
    monkeypatch.setattr(
        QW.QMessageBox, "question", lambda *_args, **_kwargs: QW.QMessageBox.Yes
    )
    with datalab_test_app_context(console=False) as win:
        dialog = PluginConfigDialog(win)
        available = dialog.available_plugins_widget
        assert available.get_catalog_directories() == [str(local)]
        monkeypatch.setattr(available, "run_task", run_now)
        monkeypatch.setattr(
            dialog.installed_plugins_widget,
            "confirm_installation",
            lambda _manifest, origin=None: origins.append(origin) or True,
        )

        available.refresh()
        (row,) = available.item_widgets
        assert row.plugin.source == str(local)
        assert row.action_button.text() == "Install"
        assert available.status_label.text() == "Plugins in the catalogs: 1"
        assert available.errors_label.isHidden()
        row.action_button.click()
        QW.QApplication.processEvents()

        assert origins == [f"Local plugin catalog ({local})"]
        assert [plugin.version for plugin in store.get_plugins()] == ["1.1.0"]

        Conf.plugins_catalog_url.set((tmp_path / "missing.json").as_uri())
        available.refresh()
        (row,) = available.item_widgets
        assert row.status_label.text() == "Installed"
        assert "Cannot download" in available.errors_label.text()
        dialog.close()
        dialog.deleteLater()


def test_local_catalog_folders_are_set_in_plugin_settings(
    store: InstalledPluginStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local catalog folders are edited without reloading plugins."""
    del store
    env_folder, added, edited = (tmp_path / name for name in ("env", "a", "b"))
    monkeypatch.setenv(DATALAB_PLUGINS_CATALOG_ENV_VAR, str(env_folder))
    Conf.plugins_catalog_url.set("")

    def no_reload_question(*_args, **_kwargs):
        raise AssertionError("Changing catalogs must not reload plugins")

    monkeypatch.setattr(QW.QMessageBox, "question", no_reload_question)
    with datalab_test_app_context(console=False) as win:
        dialog = PluginConfigDialog(win)
        available = dialog.available_plugins_widget
        (env_widget,) = dialog.env_catalog_path_widgets
        assert env_widget.path == str(env_folder) and env_widget.edit_button is None
        assert DATALAB_PLUGINS_CATALOG_ENV_VAR in env_widget.path_label.text()
        assert dialog.catalog_paths_placeholder.isHidden()
        monkeypatch.setattr(available, "run_task", run_now)
        selected = iter([str(added), str(env_folder), str(edited)])
        monkeypatch.setattr(
            dialog, "_browse_catalog_directory", lambda _path=None: next(selected)
        )

        dialog.add_catalog_button.click()
        dialog.add_catalog_button.click()
        assert dialog.catalog_paths == [str(added)]
        assert available.get_catalog_directories() == [str(env_folder), str(added)]
        dialog.catalog_path_widgets[0].edit_button.click()
        assert available.get_catalog_directories() == [str(env_folder), str(edited)]

        available.refresh()
        assert available.errors_label.text().count("not found") == 2

        dialog.accept()
    assert get_plugin_catalog_paths() == [str(edited)]

    with datalab_test_app_context(console=False) as win:
        dialog = PluginConfigDialog(win)
        dialog.catalog_path_widgets[0].delete_button.click()
        assert not dialog.catalog_paths
        dialog.reject()
    assert get_plugin_catalog_paths() == [str(edited)]


if __name__ == "__main__":
    pytest.main([__file__])
