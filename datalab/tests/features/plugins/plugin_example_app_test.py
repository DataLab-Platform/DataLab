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


def test_plugin_opens_generated_example_in_panels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generated example loads its in-memory objects without any resource."""
    import numpy as np
    from sigima.objects import create_signal

    from datalab.plugin_examples import PluginExampleData

    example = PluginExample(id="generated", title="Generated campaign")

    class GeneratedExamplePlugin(PluginBase):
        """Plugin materializing a two-signal campaign in memory."""

        PLUGIN_INFO = PluginInfo(
            id="org.example.generated-application",
            name="Generated example application",
            version="1.0.0",
        )
        EXAMPLES = (example,)

        @classmethod
        def materialize_example(cls, example_id: str) -> PluginExampleData | None:
            cls.get_example(example_id)
            x = np.linspace(0.0, 1.0, 11)
            return PluginExampleData(
                tuple(
                    create_signal(f"Generated {index}", x, x * index)
                    for index in (1, 2)
                ),
                {"gain": 2.0},
            )

        def create_actions(self) -> None:
            """Create no actions for this integration test."""

    try:
        with datalab_test_app_context(console=False) as win:
            plugin = GeneratedExamplePlugin()
            plugin.register(win)
            try:
                batches: list[tuple] = []
                original_add_objects = win.signalpanel._add_objects

                def track_add_objects(objects, *args, **kwargs) -> None:
                    batch = tuple(objects)
                    batches.append(batch)
                    original_add_objects(batch, *args, **kwargs)

                monkeypatch.setattr(
                    win.signalpanel,
                    "_add_objects",
                    track_add_objects,
                )
                opened = plugin.open_example("generated", reset_all=True)
                assert opened is example
                assert len(win.signalpanel) == 2
                assert len(batches) == 1
                assert len(batches[0]) == 2
                assert win.get_current_panel() == "signal"
                assert plugin.last_example_data is not None
                assert plugin.last_example_data.parameter_values == {"gain": 2.0}
            finally:
                plugin.unregister()
    finally:
        if GeneratedExamplePlugin in PluginRegistry.get_plugin_classes():
            PluginRegistry.get_plugin_classes().remove(GeneratedExamplePlugin)


def test_generated_example_rolls_back_cross_panel_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed second panel leaves no generated objects or implicit groups."""
    import numpy as np
    from sigima.objects import create_image, create_signal

    from datalab.plugin_examples import PluginExampleData

    example = PluginExample(id="mixed", title="Mixed campaign")

    class MixedExamplePlugin(PluginBase):
        """Plugin materializing objects in both data panels."""

        PLUGIN_INFO = PluginInfo(
            id="org.example.mixed-application",
            name="Mixed example application",
            version="1.0.0",
        )
        EXAMPLES = (example,)

        @classmethod
        def materialize_example(cls, example_id: str) -> PluginExampleData | None:
            cls.get_example(example_id)
            return PluginExampleData(
                (
                    create_signal("Signal", [0.0], [1.0]),
                    create_image("Image", np.ones((2, 2))),
                )
            )

        def create_actions(self) -> None:
            """Create no actions for this integration test."""

    try:
        with datalab_test_app_context(console=False) as win:
            plugin = MixedExamplePlugin()
            plugin.register(win)
            try:

                def fail_image_batch(*_args, **_kwargs) -> None:
                    raise RuntimeError("injected image example failure")

                monkeypatch.setattr(win.imagepanel, "_add_objects", fail_image_batch)

                with pytest.raises(
                    RuntimeError, match="injected image example failure"
                ):
                    plugin.open_example("mixed", reset_all=True)

                assert len(win.signalpanel) == 0
                assert len(win.imagepanel) == 0
                assert win.signalpanel.objmodel.get_groups() == []
                assert win.imagepanel.objmodel.get_groups() == []
                assert win.signalpanel.objview.topLevelItemCount() == 0
                assert win.imagepanel.objview.topLevelItemCount() == 0
                assert not win.is_modified()
            finally:
                plugin.unregister()
    finally:
        if MixedExamplePlugin in PluginRegistry.get_plugin_classes():
            PluginRegistry.get_plugin_classes().remove(MixedExamplePlugin)
