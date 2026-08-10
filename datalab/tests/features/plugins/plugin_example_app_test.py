# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Application test for opening package-backed plugin examples."""

from __future__ import annotations

import pytest

from datalab.plugin_examples import PluginExample
from datalab.plugins import PluginBase, PluginInfo, PluginRegistry
from datalab.tests import datalab_test_app_context


def test_plugin_opens_packaged_h5_workspace() -> None:
    """A plugin opens its native HDF5 example independently of the CWD."""
    example = PluginExample(
        id="quickstart",
        title="Quick start",
        resource="datalab:data/tests/reordering_test.h5",
    )

    class ExamplePlugin(PluginBase):
        """Plugin exposing a packaged DataLab workspace."""

        PLUGIN_INFO = PluginInfo(
            id="org.example.application",
            name="Example application",
            version="1.0.0",
        )
        EXAMPLES = (example,)

        def create_actions(self) -> None:
            """Create no actions for this integration test."""

    try:
        with datalab_test_app_context(console=False) as win:
            plugin = ExamplePlugin()
            plugin.register(win)
            try:
                opened = plugin.open_example("quickstart", reset_all=True)
                assert opened is example
                assert len(win.signalpanel) > 0
                assert win.signalpanel.objmodel.get_groups()
            finally:
                plugin.unregister()
    finally:
        if ExamplePlugin in PluginRegistry.get_plugin_classes():
            PluginRegistry.get_plugin_classes().remove(ExamplePlugin)


def test_plugin_launch_example_confirms_workspace_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default Applications launcher preserves workspace safeguards."""
    example = PluginExample(
        id="quickstart",
        title="Quick start",
        resource="datalab:data/tests/reordering_test.h5",
    )

    class ExamplePlugin(PluginBase):
        """Plugin exposing the default Applications example workflow."""

        PLUGIN_INFO = PluginInfo(
            id="org.example.launch-application",
            name="Example launch application",
            version="1.0.0",
        )
        EXAMPLES = (example,)

        def create_actions(self) -> None:
            """Create no actions for this integration test."""

    try:
        with datalab_test_app_context(console=False) as win:
            plugin = ExamplePlugin()
            plugin.register(win)
            try:
                plugin.open_example(example.id, reset_all=True)
                confirmations: list[str] = []
                monkeypatch.setattr(
                    plugin,
                    "ask_yesno",
                    lambda message, **_kwargs: confirmations.append(message) or False,
                )

                monkeypatch.setattr(win, "confirm_memory_state", lambda: False)
                assert plugin.launch_example(example.id) is None
                assert not confirmations

                monkeypatch.setattr(win, "confirm_memory_state", lambda: True)
                assert plugin.launch_example(example.id) is None
                assert confirmations

                monkeypatch.setattr(plugin, "ask_yesno", lambda *_args, **_kwargs: True)
                assert plugin.launch_example(example.id) is example
                assert len(win.signalpanel) > 0
            finally:
                plugin.unregister()
    finally:
        if ExamplePlugin in PluginRegistry.get_plugin_classes():
            PluginRegistry.get_plugin_classes().remove(ExamplePlugin)
