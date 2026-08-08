# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for package-backed plugin example declarations."""

from __future__ import annotations

import importlib
import sys
import zipfile

import pytest

from datalab.plugin_examples import PluginExample
from datalab.plugins import PluginBase, PluginInfo, PluginRegistry
from datalab.recipes import RecipeDescriptor, RecipeOutcome


def test_plugin_example_resolves_resource_from_zip_package(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Example resources work without a persistent filesystem path."""
    archive_path = tmp_path / "example-plugin.zip"
    package_name = "zipped_plugin_example"
    payload = b"packaged workspace"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{package_name}/__init__.py", "")
        archive.writestr(f"{package_name}/examples/quickstart.h5", payload)

    monkeypatch.syspath_prepend(str(archive_path))
    importlib.invalidate_caches()
    example = PluginExample(
        id="quickstart",
        title="Quick start",
        description="Packaged test workspace",
        resource=f"{package_name}:examples/quickstart.h5",
        recipe_id="org.example.application:quick-check",
        expected_checks=["signal-count", "summary-table"],
    )

    try:
        assert example.package == package_name
        assert example.resource_path == "examples/quickstart.h5"
        assert example.expected_checks == ("signal-count", "summary-table")
        assert example.resolve().read_bytes() == payload
        with example.as_file() as filename:
            materialized_path = filename
            assert filename.is_file()
            assert filename.read_bytes() == payload
        assert not materialized_path.exists()
    finally:
        sys.modules.pop(package_name, None)


def test_plugin_validates_owned_examples_and_recipe_references() -> None:
    """Example IDs are unique and optional recipe links stay plugin-owned."""

    def run_recipe(*_args) -> RecipeOutcome:
        return RecipeOutcome()

    recipe = RecipeDescriptor(
        recipe_id="org.example.application:quick-check",
        plugin_version="1.2.0",
        title="Quick check",
        version="1.0.0",
        run=run_recipe,
    )
    example = PluginExample(
        id="quickstart",
        title="Quick start",
        resource="datalab:data/tests/reordering_test.h5",
        recipe_id=recipe.recipe_id,
    )

    class ExamplePlugin(PluginBase):
        """Plugin exposing one recipe-backed example."""

        PLUGIN_INFO = PluginInfo(
            id="org.example.application",
            name="Example application",
            version="1.2.0",
        )
        RECIPES = (recipe,)
        EXAMPLES = (example,)

        def create_actions(self) -> None:
            """Create no actions for this contract test."""

    try:
        assert ExamplePlugin.get_examples() == (example,)
        assert ExamplePlugin.get_example("quickstart") is example
        with pytest.raises(KeyError, match="missing"):
            ExamplePlugin.get_example("missing")

        ExamplePlugin.EXAMPLES = (example, example)
        with pytest.raises(ValueError, match="Duplicate plugin example ID"):
            ExamplePlugin.get_examples()

        ExamplePlugin.EXAMPLES = (
            PluginExample(
                id="orphan",
                title="Orphan",
                resource="datalab:data/tests/reordering_test.h5",
                recipe_id="org.example.application:missing",
            ),
        )
        with pytest.raises(ValueError, match="unknown recipe"):
            ExamplePlugin.get_examples()
    finally:
        PluginRegistry.get_plugin_classes().remove(ExamplePlugin)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"id": "Invalid ID"}, "example ID"),
        ({"resource": "missing-separator"}, "package:path"),
        ({"resource": "datalab:/absolute.h5"}, "relative"),
        ({"resource": "datalab:data/../secret.h5"}, "relative"),
        ({"expected_checks": "not-a-sequence"}, "expected checks"),
    ],
)
def test_plugin_example_rejects_invalid_declarations(kwargs, message) -> None:
    """Invalid resource declarations fail before package resolution."""
    values = {
        "id": "quickstart",
        "title": "Quick start",
        "resource": "datalab:data/tests/reordering_test.h5",
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        PluginExample(**values)
