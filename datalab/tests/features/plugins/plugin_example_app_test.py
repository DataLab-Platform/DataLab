# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Application test for opening package-backed plugin examples."""

from __future__ import annotations

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
