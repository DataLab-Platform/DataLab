# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for stable plugin identity and registry lookups."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from datalab.plugins import (
    PluginBase,
    PluginCapability,
    PluginInfo,
    PluginRegistry,
    migrate_enabled_plugin_ids,
)


@pytest.fixture(autouse=True)
def preserve_plugin_registry() -> Iterator[None]:
    """Restore global plugin registry state after each unit test."""
    plugin_classes = list(PluginRegistry.get_plugin_classes())
    plugin_instances = list(PluginRegistry.get_plugins())
    try:
        yield
    finally:
        PluginRegistry.clear_plugin_classes()
        PluginRegistry.get_plugin_classes().extend(plugin_classes)
        PluginRegistry.get_plugins().clear()
        PluginRegistry.get_plugins().extend(plugin_instances)


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
