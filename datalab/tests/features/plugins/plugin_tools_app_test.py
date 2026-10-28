# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Application test for plugin tools: Plugins menu entries and instruments."""

# guitest: show

from __future__ import annotations

import textwrap
import threading
import time

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
            self.events = []

        def open(self):
            self.events.append("open")

        def close(self):
            self.events.append("close")

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
            assert window.wait_until_idle()
            instrument = plugin.get_instrument("generator")
            assert instrument.events == ["open"]
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

            def failing_preview():
                raise ValueError("Amplitude out of range")

            monkeypatch.setattr(instrument, "preview", failing_preview)
            window.refresh()
            assert window.wait_until_idle()
            assert window.status_label.text() == "Amplitude out of range"
            monkeypatch.undo()

            window.acquire()
            assert window.wait_until_idle()
            groups = win.signalpanel.objmodel.get_groups()
            assert groups[-1].title == "Generator acquisition"
            assert len(groups[-1]) == 3
            assert win.get_current_panel() == "signal"
            assert "3 objects" in window.status_label.text()

            # A window opened again calls the instrument after the closed one
            window.close()
            QW.QApplication.processEvents()
            reopened = plugin.launch_tool("generator")
            assert reopened is not window
            assert reopened.wait_until_idle()
            assert instrument.events == ["open", "close", "open"]
            window = reopened

            # A window closed while hidden (Qt does not emit `finished`) is
            # replaced by a new one when the tool is launched again
            window.hide()
            window.close()
            assert window.wait_closed()
            replaced = plugin.launch_tool("generator")
            assert replaced is not window
            assert replaced.wait_until_idle()
            assert instrument.events == ["open", "close", "open", "close", "open"]
            window = replaced

            closed: list[int] = []
            window.finished.connect(closed.append)
            win.reload_plugins()
            QW.QApplication.processEvents()
            assert closed
            assert instrument.events == [
                "open",
                "close",
                "open",
                "close",
                "open",
                "close",
            ]
            reloaded = PluginRegistry.get_plugin("org.example.tools-app")
            assert reloaded.launch_tool("generator") is not window
            win.close_plugin_instruments()


def test_instrument_new_group_option() -> None:
    """Acquisitions go to a new group, or to the current group of the panel when
    the **New group** option is off; the option is remembered per instrument."""
    with (
        Conf.plugins_enabled.context(True),
        Conf.plugins_enabled_list.context(None),
        Conf.instrument_new_group.context({}),
        temporary_plugin_dir() as plugin_dir,
    ):
        with open(
            f"{plugin_dir}/datalab_test_plugin_tools.py", "w", encoding="utf-8"
        ) as handle:
            handle.write(PLUGIN_SOURCE)
        with datalab_test_app_context(console=False) as win:
            plugin = PluginRegistry.get_plugin("org.example.tools-app")
            instrument = plugin.get_instrument("generator")
            key = "org.example.tools-app/generator"
            model = win.signalpanel.objmodel

            def acquire(window: InstrumentWindow) -> None:
                window.acquire()
                assert window.wait_until_idle()

            # Default state set by the instrument, nothing remembered yet
            window = plugin.launch_tool("generator")
            assert window.wait_until_idle()
            assert window.new_group_box.isChecked()
            assert key not in Conf.instrument_new_group.get()

            # Option off, panel without current group: a group is created anyway
            window.new_group_box.setChecked(False)
            assert Conf.instrument_new_group.get()[key] is False
            assert not model.get_groups()
            acquire(window)
            groups = model.get_groups()
            assert [group.title for group in groups] == ["Generator acquisition"]
            assert len(groups[0]) == 3

            # Option off: the next acquisitions go to the current group
            acquire(window)
            assert len(model.get_groups()) == 1
            assert len(groups[0]) == 6
            assert "group 'Generator acquisition'" in window.status_label.text()
            target = win.signalpanel.add_group("Target", select=True)
            acquire(window)
            assert len(model.get_groups()) == 2
            assert len(groups[0]) == 6 and len(target) == 3
            assert "group 'Target'" in window.status_label.text()
            acquire(window)
            assert len(target) == 6

            # Option on: each acquisition goes to a new group
            window.new_group_box.setChecked(True)
            assert Conf.instrument_new_group.get()[key] is True
            acquire(window)
            groups = model.get_groups()
            assert len(groups) == 3 and len(groups[-1]) == 3
            assert len(target) == 6

            # The choice of the user is remembered, and prevails over the default
            window.new_group_box.setChecked(False)
            window.close()
            QW.QApplication.processEvents()
            window = plugin.launch_tool("generator")
            assert not window.new_group_box.isChecked()
            window.close()
            QW.QApplication.processEvents()

            # Without a remembered choice, the instrument sets the default state
            Conf.instrument_new_group.set({})
            instrument.new_group = False
            window = plugin.launch_tool("generator")
            assert not window.new_group_box.isChecked()
            assert key not in Conf.instrument_new_group.get()
            win.close_plugin_instruments()


def test_instrument_acquisition_target() -> None:
    """The destination of an acquisition is taken when **Acquire** is clicked:
    selecting another group or toggling the **New group** option while the
    acquisition is running does not change it, and a new group replaces a group
    removed meanwhile."""
    with (
        Conf.plugins_enabled.context(True),
        Conf.plugins_enabled_list.context(None),
        Conf.instrument_new_group.context({}),
        temporary_plugin_dir() as plugin_dir,
    ):
        with open(
            f"{plugin_dir}/datalab_test_plugin_tools.py", "w", encoding="utf-8"
        ) as handle:
            handle.write(PLUGIN_SOURCE)
        with datalab_test_app_context(console=False) as win:
            plugin = PluginRegistry.get_plugin("org.example.tools-app")
            instrument = plugin.get_instrument("generator")
            running, release = threading.Event(), threading.Event()
            acquire = instrument.acquire

            def slow_acquire():
                """Acquire once the test releases the acquisition."""
                running.set()
                assert release.wait(10.0)
                return acquire()

            instrument.acquire = slow_acquire
            panel = win.signalpanel
            window = plugin.launch_tool("generator")
            assert window.wait_until_idle()

            def start_acquisition() -> None:
                """Click Acquire, and wait until the acquisition is running."""
                running.clear()
                release.clear()
                window.acquire()
                deadline = time.monotonic() + 10.0
                while not running.is_set():
                    assert time.monotonic() < deadline
                    QW.QApplication.processEvents()
                    time.sleep(0.005)

            def finish_acquisition() -> None:
                """Release the acquisition, and wait until it is added."""
                release.set()
                assert window.wait_until_idle()

            # Option off and group A current when Acquire is clicked
            window.new_group_box.setChecked(False)
            group_a = panel.add_group("A", select=True)
            start_acquisition()
            group_b = panel.add_group("B", select=True)
            window.new_group_box.setChecked(True)
            finish_acquisition()
            assert len(panel.objmodel.get_groups()) == 2
            assert len(group_a) == 3 and len(group_b) == 0
            assert "group 'A'" in window.status_label.text()

            # Option on when Acquire is clicked
            panel.objview.select_groups([group_b])
            start_acquisition()
            window.new_group_box.setChecked(False)
            finish_acquisition()
            groups = panel.objmodel.get_groups()
            assert len(groups) == 3 and groups[-1].title == "Generator acquisition"
            assert len(group_a) == 3 and len(group_b) == 0 and len(groups[-1]) == 3

            # Current group removed while the acquisition is running
            panel.objview.select_groups([group_b])
            start_acquisition()
            panel.remove_object(force=True)
            assert group_b not in panel.objmodel.get_groups()
            finish_acquisition()
            groups = panel.objmodel.get_groups()
            assert len(groups) == 3 and groups[-1].title == "Generator acquisition"
            assert len(groups[-1]) == 3 and len(group_a) == 3
            win.close_plugin_instruments()
