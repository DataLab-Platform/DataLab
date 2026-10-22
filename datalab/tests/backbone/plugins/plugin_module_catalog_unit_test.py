# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Single-file plugins in mixed catalogs, with old installed indexes too."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from qtpy import QtWidgets as QW

from datalab.gui.plugins.available import (
    AvailablePluginItemWidget,
    AvailablePluginsWidget,
)
from datalab.gui.plugins.install import InstalledPluginsWidget
from datalab.plugins import base as plugin_base
from datalab.plugins import catalog as catalog_client
from datalab.plugins.catalog import (
    CatalogError,
    download_release,
    read_folder_catalog,
)
from datalab.plugins.catalogtool import (
    build_local_catalog,
    check_local_catalog,
    prepare_local_catalog,
)
from datalab.plugins.generator import PluginProject, render_single_file_plugin
from datalab.plugins.store import InstalledPluginStore, PluginInstallError
from datalab.tests.backbone.plugins.plugin_local_catalog_unit_test import add_wheel
from datalab.tests.backbone.plugins.plugin_modules_unit_test import FILENAME, SOURCE


def write_module(folder: Path, source: bytes = SOURCE) -> Path:
    path = folder / FILENAME
    path.write_bytes(source)
    return path


@pytest.fixture(name="store")
def fixture_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> InstalledPluginStore:
    monkeypatch.setattr(sys, "path", list(sys.path))
    store = InstalledPluginStore(str(tmp_path / "installed"))
    monkeypatch.setattr(plugin_base, "_INSTALLED_PLUGIN_STORE", store)
    return store


def test_mixed_folder_and_declared_catalog_are_read_without_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "side-effect"
    path = write_module(
        tmp_path, SOURCE + f"\nopen({str(marker)!r}, 'w').write('executed')\n".encode()
    )
    add_wheel(tmp_path, "1.0")
    plugins, warnings = read_folder_catalog(str(tmp_path))
    assert not warnings
    module = next(plugin for plugin in plugins if plugin.kind == "module")
    assert module.id == "org.example.module"
    assert module.module == path.stem
    release = module.get_installable_release()
    assert release is not None and release.version == "1.2"
    assert release.kind == "module"
    assert not marker.exists()
    report = build_local_catalog(str(tmp_path))
    assert report.written and not report.errors
    assert not check_local_catalog(str(tmp_path)).errors
    with (tmp_path / "catalog.json").open(encoding="utf-8") as stream:
        payload = json.load(stream)
    record = next(
        plugin for plugin in payload["plugins"] if plugin.get("kind") == "module"
    )
    record["name"] = "An administered name"
    record["license"] = "BSD-3-Clause"
    record["releases"][0]["yanked"] = "Needs review"
    (tmp_path / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")
    rebuilt = build_local_catalog(str(tmp_path))
    assert not rebuilt.errors
    record = next(
        plugin
        for plugin in rebuilt.catalog["plugins"]
        if plugin.get("kind") == "module"
    )
    assert record["name"] == "An administered name"
    assert record["license"] == "BSD-3-Clause"
    assert record["releases"][0]["yanked"] == "Needs review"
    assert not marker.exists()

    # A mirror may be consumed with no URL opener available at all.
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("Offline catalog contacted the network"),
    )
    offline, _ = read_folder_catalog(str(tmp_path))
    release = next(plugin for plugin in offline if plugin.kind == "module").releases[0]
    target = tmp_path / "copy"
    target.mkdir()
    assert (
        Path(download_release(release, str(target))).read_bytes() == path.read_bytes()
    )


def test_generated_and_unversioned_modules_need_no_packaging(
    tmp_path: Path, store: InstalledPluginStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = PluginProject(
        destination=tmp_path,
        name="Example",
        package=Path(FILENAME).stem,
        plugin_id="org.example.generated",
        description="Generated example",
    )
    path = write_module(tmp_path, render_single_file_plugin(project).encode("utf-8"))
    report = prepare_local_catalog(str(tmp_path))
    assert not report.errors and report.written
    (plugin,) = read_folder_catalog(str(tmp_path))[0]
    assert plugin.releases[0].version == "0.1.0"
    (tmp_path / "catalog.json").unlink()
    path.write_text("VALUE = 1\n", encoding="utf-8")
    report = build_local_catalog(str(tmp_path))
    assert not report.errors and report.written
    (plugin,) = read_folder_catalog(str(tmp_path))[0]
    assert plugin.releases[0].version == ""
    first = store.install_module(str(path)).plugin
    assert first.version == "" and first.catalog_id == ""
    index_path = Path(store.index_path)
    index = json.loads(index_path.read_text("utf-8"))
    index["plugins"][0].pop("catalog_id")
    index_path.write_text(json.dumps(index), encoding="utf-8")
    assert store.get_plugins()[0] == first
    _app = QW.QApplication.instance() or QW.QApplication([])
    installer = InstalledPluginsWidget(lambda: None)
    browser = AvailablePluginsWidget(installer)
    installed = browser._get_installed(plugin)
    item = AvailablePluginItemWidget(plugin, installed)
    assert item.action_button is None
    path.write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "catalog.json").unlink()
    (changed,) = read_folder_catalog(str(tmp_path))[0]
    item = AvailablePluginItemWidget(changed, browser._get_installed(changed))
    assert item.action_button is not None
    assert item.action_button.text() == "Replace..."
    consent: list[dict] = []
    monkeypatch.setattr(
        installer,
        "confirm_installation",
        lambda manifest, origin=None: consent.append(manifest) or False,
    )
    assert installer.install_path(str(path), catalog_plugin=changed) is None
    assert store.get_plugins() == [first]
    assert consent[0]["version"] == ""
    monkeypatch.setattr(installer, "confirm_installation", lambda *a, **k: True)
    monkeypatch.setattr(installer, "_ask_reload", lambda *_args: False)
    result = installer.install_path(str(path), catalog_plugin=changed)
    assert result is not None and not result.restart_required
    assert result.plugin.catalog_id == changed.id
    assert result.plugin.sha256 != first.sha256
    assert not result.plugin.version
    installer.deleteLater()
    browser.deleteLater()
    item.deleteLater()


def test_versioned_module_updates_and_conflicting_id(
    tmp_path: Path, store: InstalledPluginStore
) -> None:
    path = write_module(tmp_path)
    first = store.install_module(str(path))
    path.write_bytes(SOURCE.replace(b'version="1.2"', b'version="1.3"'))
    (plugin,) = read_folder_catalog(str(tmp_path))[0]
    _app = QW.QApplication.instance() or QW.QApplication([])
    item = AvailablePluginItemWidget(
        plugin, ("file", first.plugin.version, first.plugin.sha256)
    )
    assert item.action_button is not None
    assert item.action_button.text() == "Update to 1.3"
    result = store.install_module(
        str(path), catalog_id=plugin.id, catalog_name=plugin.name, catalog_version="1.3"
    )
    assert result.replaced == first.plugin
    assert result.plugin.version == "1.3"
    with pytest.raises(ValueError, match="plugin_id"):
        store.install_module(str(path), catalog_id="org.example.other")
    store.uninstall(result.plugin)
    assert not store.get_plugins()
    item.deleteLater()


def test_failed_index_write_restores_the_previous_module(
    tmp_path: Path, store: InstalledPluginStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_module(tmp_path)
    first = store.install_module(str(path)).plugin
    original = Path(store.get_path(first)).read_bytes()
    path.write_bytes(SOURCE + b"\nVALUE = 2\n")

    def fail(*_args, **_kwargs):
        raise OSError("Index write failed")

    monkeypatch.setattr(store, "_write_index", fail)
    with pytest.raises(OSError, match="Index write"):
        store.install_module(str(path))
    assert Path(store.get_path(first)).read_bytes() == original
    assert store.get_plugins() == [first]


def test_module_changed_after_consent_does_not_replace_the_installed_file(
    tmp_path: Path, store: InstalledPluginStore
) -> None:
    path = write_module(tmp_path)
    first = store.install_module(str(path)).plugin
    manifest = store.inspect_module_file(str(path))
    path.write_bytes(SOURCE + b"\nVALUE = 2\n")
    with pytest.raises(PluginInstallError, match="changed since it was inspected"):
        store.install_module(str(path), expected_sha256=manifest["sha256"])
    assert store.get_plugins() == [first]
    assert Path(store.get_path(first)).read_bytes() == SOURCE


def test_no_script_execution_and_explicit_metadata_for_dynamic_scripts(
    tmp_path: Path,
) -> None:
    path = write_module(tmp_path, b"raise RuntimeError('not executed')\n")
    report = build_local_catalog(str(tmp_path))
    assert report.written and report.warnings
    record = report.catalog["plugins"][0]
    record.update(id="org.example.dynamic", name="Dynamic plugin", license="MIT")
    record["releases"][0]["version"] = "2.0"
    (tmp_path / "catalog.json").write_text(json.dumps(report.catalog), encoding="utf-8")
    rebuilt = build_local_catalog(str(tmp_path))
    assert rebuilt.written and not rebuilt.errors
    assert rebuilt.catalog["plugins"][0]["id"] == "org.example.dynamic"
    assert rebuilt.catalog["plugins"][0]["releases"][0]["version"] == "2.0"
    assert not check_local_catalog(str(tmp_path)).errors
    path.write_text("invalid = (", encoding="utf-8")
    failed = build_local_catalog(str(tmp_path))
    assert failed.errors and not failed.written


def test_modules_keep_withdrawal_and_source_priority_rules(tmp_path: Path) -> None:
    write_module(tmp_path)
    report = build_local_catalog(str(tmp_path))
    record = report.catalog["plugins"][0]
    record.update(status="revoked", status_reason="Withdrawn")
    (tmp_path / "catalog.json").write_text(json.dumps(report.catalog), encoding="utf-8")
    report = build_local_catalog(str(tmp_path))
    (plugin,) = read_folder_catalog(str(tmp_path))[0]
    assert plugin.status == "revoked"
    assert plugin.get_installable_release() is None
    assert set(report.catalog["plugins"][0]["releases"][0]) == {"version", "sha256"}


def test_module_release_checks_compatibility_size_and_identity(tmp_path: Path) -> None:
    path = write_module(tmp_path)
    report = build_local_catalog(str(tmp_path))
    record = report.catalog["plugins"][0]
    release = record["releases"][0]
    release["requires_python"] = ">=99"
    (tmp_path / "catalog.json").write_text(json.dumps(report.catalog), encoding="utf-8")
    (plugin,) = read_folder_catalog(str(tmp_path))[0]
    assert plugin.get_installable_release() is None
    assert plugin.get_installable_release(python_version="") is not None
    release["dependency_wheels"] = [
        {
            "filename": "extra-1.0-py3-none-any.whl",
            "url": "extra.whl",
            "size": 1,
            "sha256": "a" * 64,
        }
    ]
    (tmp_path / "catalog.json").write_text(json.dumps(report.catalog), encoding="utf-8")
    with pytest.raises(CatalogError, match="Invalid plugin catalog"):
        read_folder_catalog(str(tmp_path))
    release.pop("dependency_wheels")
    release["requires_python"] = ">=3.9"
    record["id"] = "org.example.wrong"
    (tmp_path / "catalog.json").write_text(json.dumps(report.catalog), encoding="utf-8")
    assert check_local_catalog(str(tmp_path)).errors
    assert path.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file names ignore case")
def test_case_alias_cannot_overwrite_another_module(
    tmp_path: Path, store: InstalledPluginStore
) -> None:
    path = tmp_path / "datalab_Case.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    first = store.install_module(str(path)).plugin
    replacement = tmp_path / "datalab_case.py"
    replacement.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(PluginInstallError, match="file name conflicts"):
        store.install_module(str(replacement))
    assert Path(store.get_path(first)).read_text("utf-8") == "VALUE = 1\n"
    assert store.get_plugins() == [first]


def test_module_prepare_never_invokes_pip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_module(tmp_path)
    monkeypatch.setattr(
        "datalab.plugins.dependencies._pip",
        lambda *_args, **_kwargs: pytest.fail("Modules must not invoke pip"),
    )
    report = prepare_local_catalog(str(tmp_path))
    assert report.written and not report.errors


def test_module_download_refuses_changed_bytes_and_excessive_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_module(tmp_path)
    (plugin,) = read_folder_catalog(str(tmp_path))[0]
    release = plugin.get_installable_release()
    assert release is not None
    target = tmp_path / "download"
    target.mkdir()
    path.write_bytes(SOURCE.replace(b"An example", b"An altered"))
    with pytest.raises(CatalogError):
        download_release(release, str(target))
    assert not list(target.iterdir())
    path.write_bytes(SOURCE)
    monkeypatch.setattr(catalog_client, "MAX_MODULE_BYTES", len(SOURCE) - 1)
    with pytest.raises(CatalogError, match="size"):
        download_release(release, str(target))
    assert not list(target.iterdir())


def test_module_sources_keep_first_source_and_report_masked_revocation(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    write_module(first)
    write_module(second, SOURCE.replace(b'version="1.2"', b'version="2.0"'))
    content = catalog_client.load_catalogs("", [str(first), str(second)])
    assert not content.errors
    assert len(content.plugins) == 1
    assert content.plugins[0].releases[0].version == "1.2"
    report = build_local_catalog(str(second))
    report.catalog["plugins"][0].update(status="revoked", status_reason="Withdrawn")
    (second / "catalog.json").write_text(json.dumps(report.catalog), encoding="utf-8")
    content = catalog_client.load_catalogs("", [str(first), str(second)])
    assert len(content.plugins) == 1
    assert len(content.errors) == 1 and "Withdrawn" in content.errors[0]
