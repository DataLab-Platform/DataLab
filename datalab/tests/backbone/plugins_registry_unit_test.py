# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for stable plugin identity and registry lookups."""

from __future__ import annotations

import importlib
import sys
import textwrap
from collections.abc import Iterator
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import ModuleType

import pytest

from datalab.config import Conf
from datalab.plugins import (
    PluginBase,
    PluginCapability,
    PluginInfo,
    PluginRegistry,
    discover_plugins,
    migrate_enabled_plugin_ids,
)


@pytest.fixture(autouse=True)
def preserve_plugin_registry() -> Iterator[None]:
    """Restore global plugin registry state after each unit test."""
    plugin_classes = list(PluginRegistry.get_plugin_classes())
    plugin_instances = list(PluginRegistry.get_plugins())
    discovery_errors = PluginRegistry.get_discovery_errors()
    failed_plugins = PluginRegistry.get_failed_plugins()
    traceback_log_available = Conf.main.traceback_log_available.get()
    try:
        yield
    finally:
        PluginRegistry.clear_plugin_classes()
        PluginRegistry.get_plugin_classes().extend(plugin_classes)
        PluginRegistry.get_plugins().clear()
        PluginRegistry.get_plugins().extend(plugin_instances)
        PluginRegistry.clear_discovery_errors()
        for tb_text in discovery_errors:
            PluginRegistry.add_discovery_error(tb_text)
        PluginRegistry.clear_failed_plugins()
        for failed_plugin in failed_plugins:
            PluginRegistry.add_failed_plugin(
                failed_plugin.name,
                failed_plugin.filepath,
                failed_plugin.traceback,
                failed_plugin.source,
            )
            Conf.main.traceback_log_available.set(traceback_log_available)


def create_plugin_class(
    class_name: str,
    module_name: str,
    *,
    plugin_id: str | None,
    display_name: str,
) -> type[PluginBase]:
    """Create an isolated concrete plugin class for registry tests."""

    def create_actions(self: PluginBase) -> None:  # pylint: disable=unused-argument
        """Dummy action creation method for test plugin."""

    return type(
        class_name,
        (PluginBase,),
        {
            "__module__": module_name,
            "PLUGIN_INFO": PluginInfo(id=plugin_id, name=display_name),
            "create_actions": create_actions,
        },
    )


def test_legacy_plugin_identity_and_name_lookup() -> None:
    """Legacy plugins get a deterministic ID and retain name lookup."""
    plugin_class = create_plugin_class(
        "LegacyPlugin",
        "datalab_legacy_plugin",
        plugin_id=None,
        display_name="Legacy display name",
    )
    plugin = plugin_class()

    assert plugin.plugin_id == "datalab_legacy_plugin.LegacyPlugin"
    assert plugin.info.id is None

    PluginRegistry.register_plugin(plugin)

    assert PluginRegistry.get_plugin(plugin.plugin_id) is plugin
    assert PluginRegistry.get_plugin("Legacy display name") is plugin
    assert PluginRegistry.get_plugin(plugin_class) is plugin


def test_plugin_capabilities_are_typed_and_immutable() -> None:
    """Capabilities use stable enum values and immutable plugin metadata."""
    info = PluginInfo(
        id="org.example.application",
        name="Application plugin",
        capabilities=(
            PluginCapability.APPLICATION,
            PluginCapability.PROCESSING,
        ),
    )

    assert isinstance(info.capabilities, frozenset)
    assert info.capabilities == frozenset(
        {PluginCapability.APPLICATION, PluginCapability.PROCESSING}
    )
    assert PluginInfo(name="Legacy plugin").capabilities == frozenset()

    with pytest.raises(TypeError, match="PluginCapability"):
        PluginInfo(name="Invalid plugin", capabilities={"processing"})


def test_entry_point_discovers_plugin_class_once_with_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed entry point registers its plugin class exactly once."""
    plugin_class = create_plugin_class(
        "InstalledPlugin",
        "installed_plugin",
        plugin_id="org.example.installed",
        display_name="Installed plugin",
    )
    PluginRegistry.clear_plugin_classes()

    class EntryPoint:
        """Minimal importlib.metadata entry-point test double."""

        name = "installed-plugin"
        value = "installed_plugin:InstalledPlugin"

        @staticmethod
        def load() -> type[PluginBase]:
            """Return the installed plugin class after simulating import."""
            PluginRegistry.get_plugin_classes().append(plugin_class)
            return plugin_class

    class EntryPoints(list):
        """Entry-point collection supporting the modern selection API."""

        def select(self, *, group: str):
            """Select entry points belonging to the DataLab plugin group."""
            assert group == "datalab.plugins"
            return self

    monkeypatch.setattr(
        "datalab.plugins.importlib_metadata.entry_points",
        lambda: EntryPoints([EntryPoint()]),
    )
    monkeypatch.setattr("datalab.plugins.pkgutil.iter_modules", lambda: [])
    monkeypatch.setattr("datalab.plugins.Conf.main.plugins_enabled.get", lambda: True)

    discovered = discover_plugins()

    assert not discovered
    assert PluginRegistry.get_plugin_classes() == [plugin_class]
    assert getattr(plugin_class, "__plugin_discovery_sources__") == (
        "entry point 'installed-plugin' (installed_plugin:InstalledPlugin)",
    )


def test_real_distribution_entry_point_is_discovered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Importlib metadata discovers a plugin from installed-package metadata."""
    module_name = "installed_entry_point_probe"
    (tmp_path / f"{module_name}.py").write_text(
        textwrap.dedent(
            """
            from datalab.plugins import PluginBase, PluginInfo


            class InstalledProbe(PluginBase):
                PLUGIN_INFO = PluginInfo(
                    id="org.example.installed-probe",
                    name="Installed probe",
                )

                def create_actions(self):
                    pass
            """
        ),
        encoding="utf-8",
    )
    dist_info = tmp_path / "installed_entry_point_probe-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: installed-entry-point-probe\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        f"[datalab.plugins]\ninstalled-probe = {module_name}:InstalledProbe\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr("datalab.plugins.pkgutil.iter_modules", lambda: [])
    monkeypatch.setattr("datalab.plugins.Conf.main.plugins_enabled.get", lambda: True)
    distribution = next(importlib_metadata.distributions(path=[str(tmp_path)]))
    entry_points = list(distribution.entry_points)
    monkeypatch.setattr(
        "datalab.plugins._get_plugin_entry_points", lambda: entry_points
    )
    PluginRegistry.clear_plugin_classes()
    sys.modules.pop(module_name, None)
    importlib.invalidate_caches()
    try:
        discovered = discover_plugins()

        assert all(isinstance(module, ModuleType) for module in discovered)
        assert len(PluginRegistry.get_plugin_classes()) == 1
        plugin_class = PluginRegistry.get_plugin_classes()[0]
        assert plugin_class.get_plugin_id() == "org.example.installed-probe"
        assert plugin_class.__plugin_discovery_sources__ == (
            "entry point 'installed-probe' "
            "(installed_entry_point_probe:InstalledProbe)",
        )
    finally:
        sys.modules.pop(module_name, None)


def test_python39_entry_point_mapping_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Python 3.9 mapping-shaped entry-point API remains supported."""
    plugin_class = create_plugin_class(
        "Python39Plugin",
        "python39_plugin",
        plugin_id="org.example.python39",
        display_name="Python 3.9 plugin",
    )
    PluginRegistry.clear_plugin_classes()

    class EntryPoint:
        """Minimal mapping entry-point value."""

        name = "python39-plugin"
        value = "python39_plugin:Python39Plugin"

        @staticmethod
        def load() -> type[PluginBase]:
            """Return the Python 3.9 plugin class."""
            return plugin_class

    monkeypatch.setattr(
        "datalab.plugins.importlib_metadata.entry_points",
        lambda: {"datalab.plugins": [EntryPoint()]},
    )
    monkeypatch.setattr("datalab.plugins.pkgutil.iter_modules", lambda: [])
    monkeypatch.setattr("datalab.plugins.Conf.main.plugins_enabled.get", lambda: True)

    discover_plugins()

    assert PluginRegistry.get_plugin_classes() == [plugin_class]


def test_shared_entry_point_module_is_reloaded_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple entry points from one module trigger a single hot reload."""
    module_name = "shared_entry_point_module"
    module = ModuleType(module_name)
    first_plugin_class = create_plugin_class(
        "FirstPlugin",
        module_name,
        plugin_id="org.example.shared.first",
        display_name="First shared plugin",
    )
    second_plugin_class = create_plugin_class(
        "SecondPlugin",
        module_name,
        plugin_id="org.example.shared.second",
        display_name="Second shared plugin",
    )
    module.FirstPlugin = first_plugin_class
    module.SecondPlugin = second_plugin_class
    PluginRegistry.clear_plugin_classes()

    class EntryPoint:
        """Entry point resolving one class from the shared module."""

        def __init__(self, name: str, class_name: str):
            """Initialize an entry point targeting one class."""
            self.name = name
            self.value = f"{module_name}:{class_name}"
            self.module = module_name
            self.class_name = class_name

        def load(self) -> type[PluginBase]:
            """Load the targeted class from the shared module."""
            return getattr(module, self.class_name)

    reload_count = 0

    def reload_module(loaded_module: ModuleType) -> ModuleType:
        """Count and return the shared module reload."""
        nonlocal reload_count
        assert loaded_module is module
        reload_count += 1
        return module

    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    monkeypatch.setattr(
        "datalab.plugins._get_plugin_entry_points",
        lambda: [
            EntryPoint("first-plugin", "FirstPlugin"),
            EntryPoint("second-plugin", "SecondPlugin"),
        ],
    )
    monkeypatch.setattr("datalab.plugins.importlib.reload", reload_module)
    monkeypatch.setattr("datalab.plugins.pkgutil.iter_modules", lambda: [])
    monkeypatch.setattr("datalab.plugins.Conf.main.plugins_enabled.get", lambda: True)
    try:
        discover_plugins()
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module

    assert reload_count == 1
    assert PluginRegistry.get_plugin_classes() == [
        first_plugin_class,
        second_plugin_class,
    ]


def test_entry_point_metadata_failure_preserves_convention_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unreadable package metadata does not suppress local plugin discovery."""
    module_name = "datalab_metadata_fallback"
    module = ModuleType(module_name)
    module.__file__ = __file__

    def entry_points():
        """Simulate unreadable installed distribution metadata."""
        raise RuntimeError("unreadable distribution metadata")

    def import_module(name: str) -> ModuleType:
        """Import the convention plugin after metadata discovery fails."""
        assert name == module_name
        create_plugin_class(
            "ConventionFallbackPlugin",
            module_name,
            plugin_id="org.example.convention-fallback",
            display_name="Convention fallback plugin",
        )
        return module

    PluginRegistry.clear_plugin_classes()
    monkeypatch.setattr("datalab.plugins.importlib_metadata.entry_points", entry_points)
    monkeypatch.setattr(
        "datalab.plugins.pkgutil.iter_modules",
        lambda: [(object(), module_name, False)],
    )
    monkeypatch.setattr("datalab.plugins.importlib.import_module", import_module)
    monkeypatch.setattr("datalab.plugins.Conf.main.plugins_enabled.get", lambda: True)

    assert discover_plugins() == [module]
    assert len(PluginRegistry.get_plugin_classes()) == 1
    failed_plugin = PluginRegistry.get_failed_plugins()[0]
    assert failed_plugin.name == "datalab.plugins"
    assert failed_plugin.source == "entry point group 'datalab.plugins'"
    assert "unreadable distribution metadata" in failed_plugin.traceback


def test_invalid_plugin_id_is_rejected_during_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed stable IDs are diagnosed before plugin configuration opens."""
    plugin_class = create_plugin_class(
        "InvalidIdPlugin",
        "invalid_id_plugin",
        plugin_id=" ",
        display_name="Invalid ID plugin",
    )
    PluginRegistry.clear_plugin_classes()

    class EntryPoint:
        """Entry point exposing the malformed plugin class."""

        name = "invalid-id-plugin"
        value = "invalid_id_plugin:InvalidIdPlugin"

        @staticmethod
        def load() -> type[PluginBase]:
            """Return the plugin class with malformed metadata."""
            return plugin_class

    monkeypatch.setattr(
        "datalab.plugins._get_plugin_entry_points", lambda: [EntryPoint()]
    )
    monkeypatch.setattr("datalab.plugins.pkgutil.iter_modules", lambda: [])
    monkeypatch.setattr("datalab.plugins.Conf.main.plugins_enabled.get", lambda: True)

    discover_plugins()

    assert not PluginRegistry.get_plugin_classes()
    failed_plugin = PluginRegistry.get_failed_plugins()[0]
    assert failed_plugin.name == "InvalidIdPlugin"
    assert "Plugin ID not set" in failed_plugin.traceback
    assert failed_plugin.source == (
        "entry point 'invalid-id-plugin' (invalid_id_plugin:InvalidIdPlugin)"
    )


def test_broken_entry_point_does_not_abort_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken installed plugin is recorded without blocking other plugins."""

    class BrokenEntryPoint:
        """Entry point that fails while importing third-party code."""

        name = "broken-plugin"
        value = "broken_plugin:BrokenPlugin"

        @staticmethod
        def load() -> type[PluginBase]:
            """Raise the third-party import failure."""
            raise RuntimeError("broken entry point")

    monkeypatch.setattr(
        "datalab.plugins._get_plugin_entry_points", lambda: [BrokenEntryPoint()]
    )
    monkeypatch.setattr("datalab.plugins.pkgutil.iter_modules", lambda: [])
    monkeypatch.setattr("datalab.plugins.Conf.main.plugins_enabled.get", lambda: True)

    assert not discover_plugins()
    failed_plugin = PluginRegistry.get_failed_plugins()[0]
    assert failed_plugin.name == "broken-plugin"
    assert failed_plugin.source == (
        "entry point 'broken-plugin' (broken_plugin:BrokenPlugin)"
    )
    assert "broken entry point" in failed_plugin.traceback


def test_entry_point_id_collision_rejects_all_contributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct entry-point targets with one stable ID are all rejected."""
    first_class = create_plugin_class(
        "FirstInstalledPlugin",
        "first_installed_plugin",
        plugin_id="org.example.collision",
        display_name="First installed plugin",
    )
    second_class = create_plugin_class(
        "SecondInstalledPlugin",
        "second_installed_plugin",
        plugin_id="org.example.collision",
        display_name="Second installed plugin",
    )
    PluginRegistry.clear_plugin_classes()

    class EntryPoint:
        """Minimal entry point returning one configured plugin class."""

        def __init__(self, name: str, value: str, plugin_class: type[PluginBase]):
            """Initialize an entry point with its target class."""
            self.name = name
            self.value = value
            self.plugin_class = plugin_class

        def load(self) -> type[PluginBase]:
            """Return the configured plugin class."""
            return self.plugin_class

    monkeypatch.setattr(
        "datalab.plugins._get_plugin_entry_points",
        lambda: [
            EntryPoint(
                "first-plugin",
                "first_installed_plugin:FirstInstalledPlugin",
                first_class,
            ),
            EntryPoint(
                "second-plugin",
                "second_installed_plugin:SecondInstalledPlugin",
                second_class,
            ),
        ],
    )
    monkeypatch.setattr("datalab.plugins.pkgutil.iter_modules", lambda: [])
    monkeypatch.setattr("datalab.plugins.Conf.main.plugins_enabled.get", lambda: True)

    discover_plugins()

    assert not PluginRegistry.get_plugin_classes()
    failed_plugins = PluginRegistry.get_failed_plugins()
    assert {failed.name for failed in failed_plugins} == {
        "FirstInstalledPlugin",
        "SecondInstalledPlugin",
    }
    assert all("org.example.collision" in failed.traceback for failed in failed_plugins)
    assert {failed.source for failed in failed_plugins} == {
        "entry point 'first-plugin' (first_installed_plugin:FirstInstalledPlugin)",
        "entry point 'second-plugin' (second_installed_plugin:SecondInstalledPlugin)",
    }


def test_same_target_from_entry_point_and_convention_is_merged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same target discovered twice retains both sources without collision."""
    entry_point_class = create_plugin_class(
        "SharedPlugin",
        "datalab_shared_plugin",
        plugin_id="org.example.shared",
        display_name="Shared plugin",
    )
    PluginRegistry.clear_plugin_classes()

    class EntryPoint:
        """Entry point targeting the convention-discovered module."""

        name = "shared-plugin"
        value = "datalab_shared_plugin:SharedPlugin"

        @staticmethod
        def load() -> type[PluginBase]:
            """Return the class also discovered by module convention."""
            return entry_point_class

    module_name = "datalab_shared_plugin"
    module = ModuleType(module_name)
    module.__file__ = __file__

    def import_module(name: str) -> ModuleType:
        """Simulate re-importing the plugin through module convention."""
        assert name == module_name
        create_plugin_class(
            "SharedPlugin",
            module_name,
            plugin_id="org.example.shared",
            display_name="Shared plugin",
        )
        return module

    monkeypatch.setattr(
        "datalab.plugins._get_plugin_entry_points", lambda: [EntryPoint()]
    )
    monkeypatch.setattr(
        "datalab.plugins.pkgutil.iter_modules",
        lambda: [(object(), module_name, False)],
    )
    monkeypatch.setattr("datalab.plugins.importlib.import_module", import_module)
    monkeypatch.setattr("datalab.plugins.Conf.main.plugins_enabled.get", lambda: True)

    discovered = discover_plugins()

    assert discovered == [module]
    assert len(PluginRegistry.get_plugin_classes()) == 1
    plugin_class = PluginRegistry.get_plugin_classes()[0]
    assert getattr(plugin_class, "__plugin_discovery_sources__") == (
        "entry point 'shared-plugin' (datalab_shared_plugin:SharedPlugin)",
        "module convention 'datalab_shared_plugin'",
    )


def test_registry_rejects_id_collisions_not_display_name_collisions() -> None:
    """Stable IDs are unique while duplicate display names are non-structural."""
    first_class = create_plugin_class(
        "FirstPlugin",
        "datalab_first_plugin",
        plugin_id="org.example.first",
        display_name="Shared display name",
    )
    second_class = create_plugin_class(
        "SecondPlugin",
        "datalab_second_plugin",
        plugin_id="org.example.second",
        display_name="Shared display name",
    )
    duplicate_id_class = create_plugin_class(
        "DuplicateIdPlugin",
        "datalab_duplicate_id_plugin",
        plugin_id="org.example.first",
        display_name="Another display name",
    )
    first = first_class()
    second = second_class()

    PluginRegistry.register_plugin(first)
    PluginRegistry.register_plugin(second)

    assert PluginRegistry.get_plugin("org.example.first") is first
    assert PluginRegistry.get_plugin("org.example.second") is second
    assert PluginRegistry.get_plugin("Shared display name") is None

    with pytest.raises(ValueError, match="org.example.first"):
        PluginRegistry.register_plugin(duplicate_id_class())


def test_enabled_plugin_names_migrate_to_ids_without_losing_unknowns() -> None:
    """Legacy names migrate idempotently while unavailable plugins are retained."""
    first_class = create_plugin_class(
        "FirstPlugin",
        "datalab_first_plugin",
        plugin_id="org.example.first",
        display_name="Shared display name",
    )
    second_class = create_plugin_class(
        "SecondPlugin",
        "datalab_second_plugin",
        plugin_id="org.example.second",
        display_name="Shared display name",
    )
    enabled_plugins = [
        "Shared display name",
        "org.example.first",
        "temporarily.unavailable",
    ]

    migrated_plugins = migrate_enabled_plugin_ids(
        enabled_plugins, [first_class, second_class]
    )

    assert migrated_plugins == [
        "org.example.first",
        "org.example.second",
        "temporarily.unavailable",
    ]
    assert (
        migrate_enabled_plugin_ids(migrated_plugins, [first_class, second_class])
        == migrated_plugins
    )
    assert migrate_enabled_plugin_ids(None, [first_class, second_class]) is None


def test_unregister_all_plugins_continues_after_hook_error() -> None:
    """Bulk unregistration leaves the registry coherent after a hook error."""
    unregistered_plugin_ids: list[str] = []

    def unregister_hooks(self: PluginBase) -> None:
        unregistered_plugin_ids.append(self.plugin_id)
        if self.plugin_id == "org.example.failing":
            raise RuntimeError("unregister hook failed")

    failing_class = create_plugin_class(
        "FailingPlugin",
        "datalab_failing_plugin",
        plugin_id="org.example.failing",
        display_name="Failing plugin",
    )
    succeeding_class = create_plugin_class(
        "SucceedingPlugin",
        "datalab_succeeding_plugin",
        plugin_id="org.example.succeeding",
        display_name="Succeeding plugin",
    )
    failing_class.unregister_hooks = unregister_hooks
    succeeding_class.unregister_hooks = unregister_hooks
    failing_plugin = failing_class()
    succeeding_plugin = succeeding_class()
    main = object()
    failing_plugin.register(main)
    succeeding_plugin.register(main)

    with pytest.raises(RuntimeError, match="unregister hook failed"):
        PluginRegistry.unregister_all_plugins()

    assert unregistered_plugin_ids == [
        "org.example.failing",
        "org.example.succeeding",
    ]
    assert not PluginRegistry.get_plugins()
    assert not failing_plugin.is_registered()
    assert not succeeding_plugin.is_registered()


def test_plugin_registration_rollback_on_hook_abort() -> None:
    """An interrupted registration leaves neither instance nor stale state."""

    class RegistrationAbort(BaseException):
        """Custom exception to simulate a registration abort."""

    plugin_class = create_plugin_class(
        "InterruptedPlugin",
        "datalab_interrupted_plugin",
        plugin_id="org.example.interrupted",
        display_name="Interrupted plugin",
    )

    def register_hooks(self: PluginBase) -> None:
        raise RegistrationAbort

    plugin_class.register_hooks = register_hooks
    plugin = plugin_class()

    with pytest.raises(RegistrationAbort):
        plugin.register(object())

    assert PluginRegistry.get_plugin(plugin.plugin_id) is None
    assert not plugin.is_registered()
    assert plugin.main is None
    assert plugin.proxy is None
