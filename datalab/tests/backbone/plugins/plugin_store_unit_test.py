# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the store of plugins installed from a file."""

from __future__ import annotations

import importlib
import json
import os.path as osp
import sys
from collections.abc import Iterator
from importlib import metadata as importlib_metadata
from pathlib import Path

import pytest

from datalab.config import Conf
from datalab.plugins import PluginRegistry, discover_plugins
from datalab.plugins import base as plugin_base
from datalab.plugins.store import (
    InstalledPluginStore,
    PluginInstallError,
)
from datalab.plugins.wheels import WheelInspectionError
from datalab.tests.backbone.plugins.wheel_factory import (
    make_plugin_wheel,
    wheel_filename,
)

PACKAGE_PREFIX = "datalab_store_probe"


@pytest.fixture(autouse=True)
def isolate_imports(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore ``sys.path``, probe modules and the plugin registry."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    plugin_classes = list(PluginRegistry.get_plugin_classes())
    discovery_errors = PluginRegistry.get_discovery_errors()
    failed_plugins = PluginRegistry.get_failed_plugins()
    traceback_log_available = Conf.traceback_log_available.get()
    try:
        yield
    finally:
        for name in [name for name in sys.modules if name.startswith(PACKAGE_PREFIX)]:
            del sys.modules[name]
        importlib.invalidate_caches()
        PluginRegistry.clear_plugin_classes()
        PluginRegistry.get_plugin_classes().extend(plugin_classes)
        PluginRegistry.clear_discovery_errors()
        for tb_text in discovery_errors:
            PluginRegistry.add_discovery_error(tb_text)
        PluginRegistry.clear_failed_plugins()
        for failed in failed_plugins:
            PluginRegistry.add_failed_plugin(
                failed.name, failed.filepath, failed.traceback, failed.source
            )
        Conf.traceback_log_available.set(traceback_log_available)


@pytest.fixture(name="store")
def fixture_store(tmp_path: Path) -> InstalledPluginStore:
    """Return an empty store, as DataLab creates it."""
    return InstalledPluginStore(
        str(tmp_path / "installed_plugins"),
        host_distributions={"datalab-platform": "1.3.0"},
    )


def write_wheel(directory: Path, suffix: str, version: str = "1.0.0", **kwargs) -> str:
    """Write a probe plugin wheel and return its path."""
    distribution = f"datalab-store-probe-{suffix}"
    options = {"requires_dist": ("datalab-platform>=1.3",)}
    options.update(kwargs)
    data = make_plugin_wheel(
        distribution=distribution,
        version=version,
        package=f"{PACKAGE_PREFIX}_{suffix}",
        plugin_id=f"org.example.store-probe-{suffix}",
        **options,
    )
    path = directory / wheel_filename(distribution, version)
    path.write_bytes(data)
    return str(path)


def select_entry_points(group: str) -> list[importlib_metadata.EntryPoint]:
    """Return entry points of a group, with the Python 3.9 API too."""
    entry_points = importlib_metadata.entry_points()
    if hasattr(entry_points, "select"):
        return list(entry_points.select(group=group))
    return list(entry_points.get(group, ()))  # Python 3.9 compatibility


def test_installed_wheel_is_discovered_from_its_archive(
    store: InstalledPluginStore, tmp_path: Path
) -> None:
    """An installed wheel stays a ZIP archive, discovered by its entry point."""
    path = write_wheel(tmp_path, "discovered")
    manifest = store.inspect_wheel_file(path)

    result = store.install_wheel(path, expected_sha256=manifest["sha256"])

    assert result.replaced is None and not result.restart_required
    assert store.get_plugins() == [result.plugin]
    assert result.plugin.name == "datalab-store-probe-discovered"
    assert result.plugin.packages == (f"{PACKAGE_PREFIX}_discovered",)
    wheel_path = store.get_path(result.plugin)
    assert Path(wheel_path).read_bytes() == Path(path).read_bytes()

    assert store.activate() == [wheel_path]
    assert store.activate() == []
    assert any(
        entry_point.value.startswith(f"{PACKAGE_PREFIX}_discovered.")
        for entry_point in select_entry_points("datalab.plugins")
    )


def test_discovery_registers_installed_wheel_plugin(
    store: InstalledPluginStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plugin discovery activates the store before reading entry points."""
    store.install_wheel(write_wheel(tmp_path, "registered"))
    get_entry_points = plugin_base._get_plugin_entry_points
    monkeypatch.setattr(plugin_base, "_INSTALLED_PLUGIN_STORE", store)
    monkeypatch.setattr(
        plugin_base,
        "_get_plugin_entry_points",
        lambda: [
            entry_point
            for entry_point in get_entry_points()
            if entry_point.value.startswith(PACKAGE_PREFIX)
        ],
    )
    monkeypatch.setattr(plugin_base.pkgutil, "iter_modules", lambda: [])
    monkeypatch.setattr(Conf.plugins_enabled, "get", lambda: True)
    PluginRegistry.clear_plugin_classes()

    discover_plugins()

    (plugin_class,) = PluginRegistry.get_plugin_classes()
    assert plugin_class.get_plugin_id() == "org.example.store-probe-registered"
    filepath = plugin_class.__plugin_filepath__
    assert store.find_plugin(filepath) == store.get_plugins()[0]
    assert not PluginRegistry.get_failed_plugins()


def test_corrupt_index_is_reported_without_aborting_discovery(
    store: InstalledPluginStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A damaged store becomes a plugin failure, not a startup error."""
    Path(store.root).mkdir(parents=True)
    Path(store.index_path).write_text("{", encoding="utf-8")
    monkeypatch.setattr(plugin_base, "_INSTALLED_PLUGIN_STORE", store)
    monkeypatch.setattr(plugin_base, "_get_plugin_entry_points", lambda: [])
    monkeypatch.setattr(plugin_base.pkgutil, "iter_modules", lambda: [])
    monkeypatch.setattr(Conf.plugins_enabled, "get", lambda: True)

    with pytest.raises(PluginInstallError, match="Unreadable"):
        store.get_plugins()
    discover_plugins()

    (failed,) = PluginRegistry.get_failed_plugins()
    assert failed.name == "Plugins installed from a file"
    assert "PluginInstallError" in failed.traceback


def test_install_rejects_changed_unsatisfied_or_conflicting_wheels(
    store: InstalledPluginStore, tmp_path: Path
) -> None:
    """Consent applies to inspected bytes; conflicts are refused."""
    path = write_wheel(tmp_path, "refused")
    with pytest.raises(PluginInstallError, match="changed since"):
        store.install_wheel(path, expected_sha256="0" * 64)

    unsatisfied = write_wheel(
        tmp_path, "unsatisfied", requires_dist=("datalab-platform>=99",)
    )
    with pytest.raises(WheelInspectionError, match="incompatible installed version"):
        store.install_wheel(unsatisfied)

    # pytest is installed in the test environment, outside the store
    environment = tmp_path / wheel_filename("pytest", "99.0")
    environment.write_bytes(
        make_plugin_wheel(
            distribution="pytest",
            version="99.0",
            package=f"{PACKAGE_PREFIX}_environment",
            requires_dist=(),
        )
    )
    with pytest.raises(PluginInstallError, match="Python environment"):
        store.install_wheel(str(environment))

    store.install_wheel(path)
    with pytest.raises(PluginInstallError, match="already installed"):
        store.install_wheel(path)
    assert len(store.get_plugins()) == 1


def test_replacing_an_imported_wheel_waits_for_restart(
    store: InstalledPluginStore, tmp_path: Path
) -> None:
    """A new version of imported packages is only activated at next start."""
    old = store.install_wheel(write_wheel(tmp_path, "updated")).plugin
    store.activate()
    importlib.import_module(f"{PACKAGE_PREFIX}_updated.desktop")

    result = store.install_wheel(write_wheel(tmp_path, "updated", version="2.0.0"))

    assert result.replaced == old and result.restart_required
    assert store.get_plugins() == [result.plugin]
    assert store.activate() == []
    assert osp.isfile(store.get_path(old))

    # Next start: the previous version is no longer imported nor importable
    sys.path.remove(store.get_path(old))
    for name in [name for name in sys.modules if name.startswith(PACKAGE_PREFIX)]:
        del sys.modules[name]
    restarted = InstalledPluginStore(store.root)
    assert restarted.activate() == [store.get_path(result.plugin)]
    assert not osp.exists(store.get_path(old))


def test_uninstalled_wheel_leaves_the_import_path(
    store: InstalledPluginStore, tmp_path: Path
) -> None:
    """Uninstalling removes the archive from ``sys.path`` and from disk."""
    path = write_wheel(tmp_path, "removed")
    plugin = store.install_wheel(path).plugin
    store.activate()

    store.uninstall(plugin)

    assert not store.get_plugins()
    assert store.get_path(plugin) not in sys.path
    assert not osp.exists(store.get_path(plugin))
    assert not any(
        entry_point.value.startswith(f"{PACKAGE_PREFIX}_removed.")
        for entry_point in select_entry_points("datalab.plugins")
    )
    with pytest.raises(PluginInstallError, match="not installed"):
        store.uninstall(plugin)
    assert store.install_wheel(path).plugin.sha256 == plugin.sha256


def test_module_plugin_install_replace_and_uninstall(
    store: InstalledPluginStore, tmp_path: Path
) -> None:
    """A single ``datalab_*.py`` file is installed in the modules folder."""
    name = f"{PACKAGE_PREFIX}_module"
    source = tmp_path / f"{name}.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    plugin = store.install_module(str(source)).plugin
    assert store.activate() == [store.modules_dir]
    assert importlib.import_module(name).VALUE == 1

    source.write_text("VALUE = 2\n", encoding="utf-8")
    result = store.install_module(str(source))
    assert result.replaced == plugin and not result.restart_required
    assert importlib.reload(sys.modules[name]).VALUE == 2

    store.uninstall(result.plugin)
    assert not osp.exists(store.get_path(result.plugin))
    assert not store.get_plugins()


@pytest.mark.parametrize(
    "filename",
    ["probe.py", "datalab_.py", "datalab_bad-name.py", "datalab_probe.txt"],
)
def test_module_plugin_requires_a_datalab_module_name(
    store: InstalledPluginStore, tmp_path: Path, filename: str
) -> None:
    """Only files discovered by the ``datalab_`` convention are accepted."""
    path = tmp_path / filename
    path.write_text("", encoding="utf-8")

    with pytest.raises(PluginInstallError, match="datalab_<name>.py"):
        store.install_module(str(path))


def test_module_plugin_must_not_shadow_an_importable_module(
    store: InstalledPluginStore, tmp_path: Path
) -> None:
    """An installed module would be hidden by an existing one."""
    name = f"{PACKAGE_PREFIX}_clash"
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / f"{name}.py").write_text("", encoding="utf-8")
    sys.path.append(str(existing))
    source = tmp_path / f"{name}.py"
    source.write_text("", encoding="utf-8")

    with pytest.raises(PluginInstallError, match="already exists"):
        store.install_module(str(source))


def test_activation_deletes_leftover_files(
    store: InstalledPluginStore, tmp_path: Path
) -> None:
    """Files left by an interrupted operation are deleted at activation."""
    plugin = store.install_wheel(write_wheel(tmp_path, "kept")).plugin
    leftover = Path(store.wheels_dir) / f"{'a' * 64}.whl.tmp"
    leftover.write_bytes(b"partial")

    store.activate()

    assert not leftover.exists()
    assert osp.isfile(store.get_path(plugin))
    index = json.loads(Path(store.index_path).read_text(encoding="utf-8"))
    assert index["schema_version"] == 1
    assert index["plugins"][0]["sha256"] == plugin.sha256


if __name__ == "__main__":
    pytest.main([__file__])
