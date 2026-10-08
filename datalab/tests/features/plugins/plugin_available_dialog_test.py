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
from datalab.config import Conf
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


@pytest.fixture(name="store")
def fixture_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use an empty installed plugin store and restore the import state."""
    store = InstalledPluginStore(
        str(tmp_path / "installed_plugins"),
        host_distributions={"datalab-platform": __version__},
    )
    monkeypatch.setattr(plugin_base, "_INSTALLED_PLUGIN_STORE", store)
    monkeypatch.setattr(sys, "path", list(sys.path))
    had_url = has_persisted_option(Conf, "plugins_catalog_url")
    original_url = Conf.plugins_catalog_url.get()
    yield store
    if had_url:
        Conf.plugins_catalog_url.set(original_url)
    else:
        remove_persisted_option(Conf, "plugins_catalog_url")
    for name in [name for name in sys.modules if name.startswith(PACKAGE)]:
        del sys.modules[name]


def publish_catalog(site: Path, versions: tuple[str, ...]) -> str:
    """Publish probe plugin releases in a catalog folder and return its URL."""
    releases = []
    for version in sorted(versions, reverse=True):
        data = make_plugin_wheel(
            distribution=DISTRIBUTION,
            version=version,
            package=PACKAGE,
            plugin_id=PLUGIN_ID,
            requires_dist=("datalab-platform>=1.0",),
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

        Conf.plugins_catalog_url.set((tmp_path / "missing.json").as_uri())
        available.refresh()
        assert "cannot be loaded" in available.status_label.text()
        dialog.close()
        dialog.deleteLater()


if __name__ == "__main__":
    pytest.main([__file__])
