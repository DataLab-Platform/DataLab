# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Offline dependency planning and transactional installation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from datalab.plugins import dependencies, wheels
from datalab.plugins.dependencies import DependencyError
from datalab.plugins.store import InstalledPluginStore
from datalab.plugins.wheels import (
    DESKTOP_ENTRY_POINT_GROUP,
    WheelInspectionError,
    get_reference_distributions,
    inspect_dependency_wheel,
    inspect_wheel,
)
from datalab.tests.backbone.plugins.wheel_factory import (
    make_plugin_wheel,
    wheel_filename,
)

PACKAGE = "datalab_offline_dependency_probe"
DISTRIBUTION = "datalab-offline-dependency-probe"


@pytest.fixture(name="store")
def fixture_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> InstalledPluginStore:
    """Keep imported test dependencies private to each test."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    store = InstalledPluginStore(str(tmp_path / "installed"))
    yield store
    for name in [
        name
        for name in sys.modules
        if name.startswith(
            (PACKAGE, "datalab_offline_plugin_probe", "datalab_source_sdk_probe")
        )
    ]:
        del sys.modules[name]
    importlib.invalidate_caches()


def write_files(
    tmp_path: Path, *, requires: tuple[str, ...] = (), native: bool = False
) -> Path:
    """Write a plugin and an ordinary dependency into a local catalog."""
    wheelhouse = tmp_path / "dependencies"
    wheelhouse.mkdir()
    (wheelhouse / wheel_filename(DISTRIBUTION, "1.0.0")).write_bytes(
        make_plugin_wheel(
            distribution=DISTRIBUTION,
            version="1.0.0",
            package=PACKAGE,
            entry_points="",
            files={f"{PACKAGE}/__init__.py": "VALUE = 42\n"},
            extra_files={f"{PACKAGE}/library.dll": b"fixture"} if native else None,
            requires_dist=requires,
            record=True,
        )
    )
    plugin = tmp_path / wheel_filename("datalab-offline-plugin-probe", "1.0.0")
    plugin.write_bytes(
        make_plugin_wheel(
            distribution="datalab-offline-plugin-probe",
            version="1.0.0",
            package="datalab_offline_plugin_probe",
            requires_dist=(f"{DISTRIBUTION}>=1",),
            record=True,
        )
    )
    return plugin


def test_dependency_wheels_are_distinct_from_plugin_wheels() -> None:
    """Native dependencies need no plugin entry point; plugins stay pure Python."""
    data = make_plugin_wheel(
        entry_points="",
        extra_files={"example_datalab_plugin/library.dll": b"native"},
    )
    filename = wheel_filename("example-datalab-plugin", "1.2.0")
    assert inspect_dependency_wheel(data, filename=filename)["compatible"]
    with pytest.raises(WheelInspectionError, match="Native payloads"):
        inspect_wheel(
            data,
            filename=filename,
            entry_point_group=DESKTOP_ENTRY_POINT_GROUP,
            available_distributions={"sigima": "1.3.0"},
        )


def test_reference_snapshot_excludes_ambient_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An author's installed plugin must not become part of the host baseline."""
    distributions = [
        SimpleNamespace(
            metadata={"Name": "datalab-platform"},
            version="1.4.0",
            requires=["core[fast]>=1", "test-only; extra == 'test'"],
        ),
        SimpleNamespace(
            metadata={"Name": "core"},
            version="1.0",
            requires=["accelerator; extra == 'fast'"],
        ),
        SimpleNamespace(metadata={"Name": "accelerator"}, version="2.0", requires=[]),
        SimpleNamespace(metadata={"Name": "pytango"}, version="10.3", requires=[]),
    ]
    monkeypatch.setattr(
        wheels.importlib_metadata, "distributions", lambda: distributions
    )
    assert get_reference_distributions() == {
        "datalab-platform": "1.4.0",
        "core": "1.0",
        "accelerator": "2.0",
    }


def test_catalog_candidates_do_not_override_installed_versions() -> None:
    """Catalog metadata accepts any matching artifact, not only its newest one."""
    data = make_plugin_wheel(requires_dist=("optional-helper<2",))
    filename = wheel_filename("example-datalab-plugin", "1.2.0")
    candidates = {"optional-helper": ["1.0", "2.0"]}
    manifest = inspect_wheel(
        data,
        filename=filename,
        entry_point_group=DESKTOP_ENTRY_POINT_GROUP,
        available_distributions={},
        additional_distributions=candidates,
    )
    assert manifest["dependencies"][0]["compatible"]
    with pytest.raises(WheelInspectionError, match="incompatible installed version"):
        inspect_wheel(
            data,
            filename=filename,
            entry_point_group=DESKTOP_ENTRY_POINT_GROUP,
            available_distributions={"optional-helper": "2.0"},
            additional_distributions=candidates,
        )


def test_offline_installation_is_shared_deferred_and_kept_on_uninstall(
    store: InstalledPluginStore, tmp_path: Path
) -> None:
    """Resolve a real pip graph without an index and activate only after restart."""
    plugin = write_files(tmp_path)
    protected = store.get_available_distributions()
    manifest, plan = store.prepare_wheel_file(str(plugin))
    assert plan is not None
    try:
        assert [wheel.distribution for wheel in plan.wheels] == [DISTRIBUTION]
        result = store.install_wheel(
            str(plugin), expected_sha256=manifest["sha256"], dependency_plan=plan
        )
    finally:
        plan.close()
    assert result.restart_required
    assert not store.activate()
    assert importlib.util.find_spec(PACKAGE) is None
    assert all(
        store.get_available_distributions()[name] == version
        for name, version in protected.items()
    )
    restarted = InstalledPluginStore(store.root)
    assert restarted.activate()
    assert importlib.import_module(PACKAGE).VALUE == 42
    paths = restarted.get_dependency_paths()
    restarted.uninstall(result.plugin)
    assert restarted.get_dependency_paths() == paths
    assert all(Path(path).is_dir() for path in paths)
    assert not restarted.get_plugins()


def test_missing_transitive_wheel_leaves_the_store_unchanged(
    store: InstalledPluginStore, tmp_path: Path
) -> None:
    plugin = write_files(
        tmp_path, requires=("datalab-deliberately-unavailable-dependency>=1",)
    )
    with pytest.raises(DependencyError, match="Dependency installer failed"):
        store.prepare_wheel_file(str(plugin))
    assert not store.get_plugins()
    assert not store.get_dependency_paths()


def test_resolution_preserves_the_hosts_development_import_paths(
    store: InstalledPluginStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real SDK exposed through sys.path is already provided, not downloadable."""
    sdk = tmp_path / "sdk"
    package = sdk / "datalab_source_sdk_probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")
    info = sdk / "datalab_source_sdk_probe-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: datalab-source-sdk-probe\nVersion: 1.0\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(sdk))
    assert importlib.import_module("datalab_source_sdk_probe").VALUE == 7
    plugin = write_files(tmp_path, requires=("datalab-source-sdk-probe>=1",))
    manifest, plan = store.prepare_wheel_file(str(plugin))
    assert plan is not None
    try:
        assert [wheel.distribution for wheel in plan.wheels] == [DISTRIBUTION]
        store.install_wheel(
            str(plugin), expected_sha256=manifest["sha256"], dependency_plan=plan
        )
    finally:
        plan.close()


def test_restarted_plugin_is_enabled_in_an_explicit_list(
    store: InstalledPluginStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pending enablement survives the restart required by dependencies."""
    from datalab.plugins import base as plugin_base

    plugin = write_files(tmp_path)
    manifest, plan = store.prepare_wheel_file(str(plugin))
    assert plan is not None
    try:
        store.install_wheel(
            str(plugin), expected_sha256=manifest["sha256"], dependency_plan=plan
        )
    finally:
        plan.close()
    restarted = InstalledPluginStore(store.root)
    registry = plugin_base.PluginRegistry
    classes = list(registry.get_plugin_classes())
    errors = list(registry.get_discovery_errors())
    failed = list(registry.get_failed_plugins())
    traceback_available = plugin_base.Conf.traceback_log_available.get()
    enabled: list[str] = []
    monkeypatch.setattr(plugin_base, "_INSTALLED_PLUGIN_STORE", restarted)
    monkeypatch.setattr(plugin_base, "_extend_sys_path_with_plugin_paths", lambda: None)
    monkeypatch.setattr(plugin_base.pkgutil, "iter_modules", lambda: iter(()))
    entry_points = plugin_base._get_plugin_entry_points
    monkeypatch.setattr(
        plugin_base,
        "_get_plugin_entry_points",
        lambda: [
            entry
            for entry in entry_points()
            if entry.module.startswith("datalab_offline_plugin_probe")
        ],
    )
    monkeypatch.setattr(plugin_base.Conf.plugins_enabled, "get", lambda *_args: True)
    monkeypatch.setattr(
        plugin_base.Conf.plugins_enabled_list, "get", lambda *_args: list(enabled)
    )
    monkeypatch.setattr(
        plugin_base.Conf.plugins_enabled_list,
        "set",
        lambda values: enabled.__setitem__(slice(None), values),
    )
    registry.clear_plugin_classes()
    try:
        plugin_base.discover_plugins()
        assert enabled == ["org.example.wheel-plugin"]
        assert not restarted.get_pending_enabled_names()
        assert not registry.get_failed_plugins()
    finally:
        registry.clear_plugin_classes()
        registry.get_plugin_classes().extend(classes)
        registry.clear_discovery_errors()
        for error in errors:
            registry.add_discovery_error(error)
        registry.clear_failed_plugins()
        for item in failed:
            registry.add_failed_plugin(
                item.name, item.filepath, item.traceback, item.source
            )
        plugin_base.Conf.traceback_log_available.set(traceback_available)


def test_offline_resolution_rejects_dependency_urls(
    store: InstalledPluginStore, tmp_path: Path
) -> None:
    """A transitive direct URL must not bypass the offline-only policy."""
    plugin = write_files(
        tmp_path, requires=("remote-helper @ https://must-not-be-requested.test/x.whl",)
    )
    with pytest.raises(DependencyError, match="Direct dependency URLs"):
        store.prepare_wheel_file(str(plugin))
    assert not store.get_plugins()


def test_changed_dependency_is_rejected_after_consent(
    store: InstalledPluginStore, tmp_path: Path
) -> None:
    plugin = write_files(tmp_path)
    manifest, plan = store.prepare_wheel_file(str(plugin))
    assert plan is not None
    try:
        Path(plan.wheels[0].path).write_bytes(b"changed")
        with pytest.raises(DependencyError, match="changed after consent"):
            store.install_wheel(
                str(plugin), expected_sha256=manifest["sha256"], dependency_plan=plan
            )
    finally:
        plan.close()
    assert not store.get_plugins()
    assert not store.get_dependency_paths()


def test_windows_native_path_limits_do_not_publish_a_broken_installation(
    store: InstalledPluginStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DLL loader path failure is rejected before publishing the store index."""
    plugin = write_files(tmp_path, native=True)
    manifest, plan = store.prepare_wheel_file(str(plugin))
    assert plan is not None
    monkeypatch.setattr(dependencies, "MAX_NATIVE_PATH", 1)
    try:
        with pytest.raises(DependencyError, match="Windows DLL path limit"):
            store.install_wheel(
                str(plugin), expected_sha256=manifest["sha256"], dependency_plan=plan
            )
    finally:
        plan.close()
    assert not store.get_plugins()
    assert not store.get_dependency_paths()


def test_missing_dependency_permission_does_not_allow_host_conflicts(
    store: InstalledPluginStore, tmp_path: Path
) -> None:
    plugin = write_files(tmp_path)
    plugin.write_bytes(
        make_plugin_wheel(
            distribution="datalab-offline-plugin-probe",
            version="1.0.0",
            requires_dist=("numpy<1",),
        )
    )
    with pytest.raises(WheelInspectionError, match="incompatible installed version"):
        store.prepare_wheel_file(str(plugin))
