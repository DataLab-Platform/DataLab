# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Application test for plugin tools: Plugins menu entries and instruments."""

# guitest: show

from __future__ import annotations

import textwrap

import numpy as np
import pytest
from qtpy import QtWidgets as QW
from sigima.objects import create_image, create_signal

from datalab.config import Conf
from datalab.gui.main import DLMainWindow
from datalab.gui.plugins.instrument import InstrumentWindow
from datalab.plugins import PluginRegistry
from datalab.plugins.instruments import InstrumentFrame
from datalab.tests import datalab_test_app_context
from datalab.tests.features.plugins.plugin_test_dataset import temporary_plugin_dir

PLUGIN_SOURCE = textwrap.dedent(
    '''
    """Application plugin declaring a launcher tool and an instrument tool."""

    import guidata.dataset as gds
    import numpy as np
    from sigima.objects import create_signal

    from datalab.plugins import PluginBase, PluginCapability, PluginInfo
    from datalab.plugins.instruments import (
        InstrumentAcquisition,
        InstrumentFrame,
        PluginInstrument,
    )
    from datalab.plugins.tools import PluginTool


    class GeneratorSettings(gds.DataSet):
        """Signal generator settings."""

        amplitude = gds.FloatItem("Amplitude", default=1.0)
        count = gds.IntItem("Shots", default=3, min=1)


    class SignalGenerator(PluginInstrument):
        """Instrument generating ramps."""

        live_interval_ms = 100

        def __init__(self):
            super().__init__(GeneratorSettings())
            self.frames = 0

        def preview(self):
            self.frames += 1
            x = np.linspace(0.0, 1.0, 11)
            signal = create_signal("Live", x, self.settings.amplitude * x)
            return InstrumentFrame(
                (signal,), summary=f"Frame {self.frames}", value_range=(-2.0, 2.0)
            )

        def acquire(self):
            x = np.linspace(0.0, 1.0, 11)
            return InstrumentAcquisition(
                "Generator acquisition",
                [
                    create_signal(f"Shot {index}", x, x * index)
                    for index in range(self.settings.count)
                ],
            )


    class ToolsTestPlugin(PluginBase):
        """Plugin exposing tools only."""

        PLUGIN_INFO = PluginInfo(
            id="org.example.tools-app",
            name="Tool test application",
            version="1.0.0",
            capabilities=(PluginCapability.APPLICATION,),
        )
        TOOLS = (
            PluginTool(
                id="generator",
                title="Signal generator...",
                instrument="create_generator",
                object_type="signal",
            ),
            PluginTool(
                id="inspect",
                title="Inspect images...",
                launcher="inspect_images",
                object_type="image",
                selection="at_least_one",
            ),
        )

        def create_generator(self):
            return SignalGenerator()

        def inspect_images(self):
            return len(self.get_selected_objects())

        def create_actions(self):
            pass
    '''
)


def _plugin_submenu_actions(win: DLMainWindow) -> dict[str, QW.QAction]:
    """Return the actions of the test plugin's submenu in the Plugins menu."""
    win.plugins_menu.aboutToShow.emit()
    for action in win.plugins_menu.actions():
        menu = action.menu()
        if menu is not None and menu.title() == "Tool test application":
            return {
                sub_action.text(): sub_action
                for sub_action in menu.actions()
                if not sub_action.isSeparator()
            }
    return {}


def test_plugin_tools_in_menus_and_instrument_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tools are listed in their panel menu and open an instrument window."""
    with (
        Conf.plugins_enabled.context(True),
        Conf.plugins_enabled_list.context(None),
        temporary_plugin_dir() as plugin_dir,
    ):
        with open(
            f"{plugin_dir}/datalab_test_plugin_tools.py", "w", encoding="utf-8"
        ) as handle:
            handle.write(PLUGIN_SOURCE)
        with datalab_test_app_context(console=False) as win:
            plugin = PluginRegistry.get_plugin("org.example.tools-app")
            assert plugin is not None

            win.set_current_panel("signal")
            QW.QApplication.processEvents()
            assert list(_plugin_submenu_actions(win)) == ["Signal generator..."]

            win.set_current_panel("image")
            QW.QApplication.processEvents()
            inspect_action = _plugin_submenu_actions(win)["Inspect images..."]
            assert not inspect_action.isEnabled()
            assert plugin.assess_tool("inspect") == "Select at least one image"
            win.imagepanel.add_object(create_image("Frame", np.zeros((4, 4))))
            QW.QApplication.processEvents()
            assert inspect_action.isEnabled()
            assert plugin.assess_tool("inspect") is None
            assert plugin.launch_tool("inspect") == 1

            win.set_current_panel("signal")
            with pytest.raises(ValueError, match="Select at least one image"):
                plugin.launch_tool("inspect")

            window = plugin.launch_tool("generator")
            assert isinstance(window, InstrumentWindow)
            assert plugin.launch_tool("generator") is window
            window.refresh()
            assert len(window.items) == 1
            assert window.summary_label.text().startswith("Frame")
            assert window.plotwidget.plot.get_axis_limits("left") == (-2.0, 2.0)

            # A new value range with the same X axis updates Y and keeps the X zoom
            plot = window.plotwidget.plot
            x = np.linspace(0.0, 1.0, 11)
            plot.set_axis_limits("bottom", 0.2, 0.6)
            window.render(
                InstrumentFrame((create_signal("Live", x, x),), value_range=(-5, 5))
            )
            assert plot.get_axis_limits("left") == (-5.0, 5.0)
            assert plot.get_axis_limits("bottom") == (0.2, 0.6)
            window.render(InstrumentFrame((create_signal("Live", x, x),)))
            assert plot.get_axis_limits("left")[1] < 5.0
            assert plot.get_axis_limits("bottom") == (0.2, 0.6)
            wider = np.linspace(0.0, 2.0, 11)
            window.render(InstrumentFrame((create_signal("Live", wider, wider),)))
            assert plot.get_axis_limits("bottom")[1] >= 2.0

            window.live_button.setChecked(True)
            assert window._live_timer.isActive()  # pylint: disable=protected-access
            window.live_button.setChecked(False)

            instrument = plugin.get_instrument("generator")

            def failing_preview():
                raise ValueError("Amplitude out of range")

            monkeypatch.setattr(instrument, "preview", failing_preview)
            window.refresh()
            assert window.status_label.text() == "Amplitude out of range"
            monkeypatch.undo()

            window.acquire()
            groups = win.signalpanel.objmodel.get_groups()
            assert groups[-1].title == "Generator acquisition"
            assert len(groups[-1]) == 3
            assert win.get_current_panel() == "signal"
            assert "3 objects" in window.status_label.text()

            closed: list[int] = []
            window.finished.connect(closed.append)
            win.reload_plugins()
            QW.QApplication.processEvents()
            assert closed
            reloaded = PluginRegistry.get_plugin("org.example.tools-app")
            assert reloaded.launch_tool("generator") is not window
            win.close_plugin_instruments()
