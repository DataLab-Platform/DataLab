# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for computing-feature identity and ownership."""

from __future__ import annotations

import pytest
import sigima.proc.signal as sips

from datalab.gui.processor.base import BaseProcessor, ComputingFeature


class ProcessorRegistryHarness:
    """Minimal harness for testing registry methods without constructing Qt UI."""

    add_feature = BaseProcessor.add_feature
    get_feature = BaseProcessor.get_feature
    remove_feature = BaseProcessor.remove_feature
    remove_features_by_owner = BaseProcessor.remove_features_by_owner

    def __init__(self) -> None:
        self.computing_registry: dict[str, ComputingFeature] = {}


def test_feature_registry_uses_stable_ids_and_legacy_aliases() -> None:
    """Features resolve by ID, function, and unambiguous legacy name."""
    processor = ProcessorRegistryHarness()
    feature = ComputingFeature(
        pattern="1_to_1",
        function=sips.derivative,
        title="Derivative",
        feature_id="org.example.derivative",
    )

    processor.add_feature(feature, owner="org.example.plugin")

    assert feature.owner_plugin_id == "org.example.plugin"
    assert processor.get_feature("org.example.derivative") is feature
    assert processor.get_feature("derivative") is feature
    assert processor.get_feature(sips.derivative) is feature


def test_feature_collision_is_atomic_and_owner_removal_is_scoped() -> None:
    """ID collisions preserve the registry and owner cleanup is selective."""
    processor = ProcessorRegistryHarness()
    internal_feature = ComputingFeature(
        pattern="1_to_1",
        function=sips.inverse,
        title="Inverse",
    )
    plugin_feature = ComputingFeature(
        pattern="1_to_1",
        function=sips.derivative,
        title="Derivative",
        feature_id="org.example.derivative",
        owner_plugin_id="org.example.plugin",
    )
    conflicting_feature = ComputingFeature(
        pattern="1_to_1",
        function=sips.absolute,
        title="Absolute value",
        feature_id="org.example.derivative",
        owner_plugin_id="org.example.other-plugin",
    )
    processor.add_feature(internal_feature)
    processor.add_feature(plugin_feature)

    with pytest.raises(ValueError, match="org.example.derivative"):
        processor.add_feature(conflicting_feature)

    assert len(processor.computing_registry) == 2
    assert processor.get_feature("org.example.derivative") is plugin_feature

    removed = processor.remove_features_by_owner("org.example.plugin")

    assert removed == [plugin_feature]
    assert processor.get_feature(sips.inverse) is internal_feature
    with pytest.raises(ValueError, match="org.example.derivative"):
        processor.get_feature("org.example.derivative")


def test_duplicate_function_alias_requires_stable_id() -> None:
    """A function reused by multiple contributions is addressable only by ID."""
    processor = ProcessorRegistryHarness()
    built_in = ComputingFeature(
        pattern="1_to_1",
        function=sips.derivative,
        title="Built-in derivative",
    )
    plugin_feature = ComputingFeature(
        pattern="1_to_1",
        function=sips.derivative,
        title="Plugin derivative",
        feature_id="org.example.derivative",
        owner_plugin_id="org.example.plugin",
    )
    processor.add_feature(built_in)
    processor.add_feature(plugin_feature)

    assert processor.get_feature(built_in.feature_id) is built_in
    assert processor.get_feature(plugin_feature.feature_id) is plugin_feature
    with pytest.raises(ValueError, match="Ambiguous"):
        processor.get_feature(sips.derivative)
    with pytest.raises(ValueError, match="Ambiguous"):
        processor.get_feature("derivative")
