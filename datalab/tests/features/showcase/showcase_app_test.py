# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase mode application tests

The showcase scenarios are played in check mode (speed 0): no delay and no visual
effect, with the demo mode accepting the dialogs.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.config import Conf
from datalab.env import execenv
from datalab.gui.showcase import (
    ScenarioSkipped,
    ShowcaseDialog,
    ShowcaseOptions,
    ShowcasePlayer,
    ShowcaseScenario,
    get_showcase_scenarios,
    play_showcase,
    player,
    show_showcase_dialog,
)
from datalab.gui.showcase import dialog as dialog_module
from datalab.gui.showcase.scenarios import get_native_scenarios
from datalab.plugins import PluginRegistry
from datalab.tests import datalab_test_app_context, record_shown_windows
from datalab.tests.features.plugins.applications_panel_test import _plugin


def accept_dialog(dialog: QW.QDialog) -> int:
    """Accept a dialog, as the unattended mode does when it executes a dialog"""
    dialog.accept()
    return dialog.result()


@pytest.fixture(name="registry")
def fixture_registry() -> Generator[list, None, None]:
    """Give the list of registered plugins, restored after the test"""
    plugins = PluginRegistry.get_plugins()
    previous_plugins = list(plugins)
    try:
        yield plugins
    finally:
        plugins[:] = previous_plugins


@pytest.mark.parametrize("scenario_id", [s.id for s in get_native_scenarios()])
def test_native_scenario(scenario_id: str) -> None:
    """Each native scenario plays without error in check mode"""
    with datalab_test_app_context(console=False, history=True) as win:
        options = ShowcaseOptions(scenario_ids=[scenario_id], speed=0.0)
        results = play_showcase(win, options)
        # The demo mode is disabled once the showcase is over
        assert execenv.unattended and not execenv.demo_mode
    assert [result.scenario_id for result in results] == [scenario_id]
    if results[0].status == "skipped":
        pytest.skip(results[0].message)
    assert results[0].status == "done", results[0].message


def test_showcase_goes_on_after_failures() -> None:
    """Failing and skipped scenarios do not stop the showcase, the operator does"""
    played: list[str] = []

    def fail(ctx) -> None:
        ctx.win.add_object(ctx.win.signalpanel.create_object())
        ctx.checkpoint(False, "Expected failure")

    def skip(_ctx) -> None:
        raise ScenarioSkipped("Not available")

    def check_clean_workspace(ctx) -> None:
        played.append("clean" if not ctx.win.has_objects() else "dirty")

    def stop(ctx) -> None:
        played.append("stop")
        ctx.request_stop()

    scenarios = [
        ShowcaseScenario("fail", "Failing", fail),
        ShowcaseScenario("skip", "Skipped", skip),
        ShowcaseScenario("missing", "Missing", stop, requires=("no_such_module",)),
        ShowcaseScenario("clean", "Clean", check_clean_workspace),
        ShowcaseScenario("stop", "Stop", stop),
        ShowcaseScenario("never", "Never played", stop),
    ]
    with datalab_test_app_context(console=False) as win:
        options = ShowcaseOptions(loop=True, speed=0.0)
        results = ShowcasePlayer(win, scenarios, options).play()
    statuses = [(result.scenario_id, result.status) for result in results]
    assert statuses == [
        ("fail", "failed"),
        ("skip", "skipped"),
        ("missing", "skipped"),
        ("clean", "done"),
        ("stop", "done"),
    ]
    assert "Expected failure" in results[0].message
    assert played == ["clean", "stop"]


def test_application_scenarios(registry: list) -> None:
    """A scenario is generated for each application plugin, between native ones"""
    with datalab_test_app_context(console=False) as win:
        # Registered once the main window is created, as the plugin double does
        # not create any action
        registry.append(_plugin("org.example.showcase", "Showcase application"))
        scenarios = get_showcase_scenarios(win)
        ids = [scenario.id for scenario in scenarios]
        # Native scenarios alternate with the scenarios of the applications
        assert [i for i in ids if not i.startswith("app:")] == ["d1", "d2", "d3", "d4"]
        assert ids[1].startswith("app:") and "app:org.example.showcase" in ids
        index = ids.index("app:org.example.showcase")
        assert scenarios[index].title == "Showcase application"
        options = ShowcaseOptions(["app:org.example.showcase"], speed=0.0)
        results = play_showcase(win, options, scenarios)
    assert results[0].status == "done", results[0].message


def test_showcase_dialog(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dialog shows no transient window, remembers the choices and plays"""
    with datalab_test_app_context(console=False) as win:
        action_texts = [action.text() for action in win.help_menu.actions()]
        assert any(text.startswith("Showcase mode") for text in action_texts)
        scenarios = get_showcase_scenarios(win)
        ids = [scenario.id for scenario in scenarios]
        with record_shown_windows() as shown:
            dialog = ShowcaseDialog(win, scenarios)
        assert not shown
        assert dialog.get_selected_ids() == ids
        dialog.scenario_list.item(1).setCheckState(QC.Qt.Unchecked)
        dialog.accept()
        assert dialog.options.scenario_ids == [ids[0]] + ids[2:]
        assert Conf.showcase_excluded_scenarios.get() == [ids[1]]
        dialog = ShowcaseDialog(win, scenarios)
        assert dialog.get_selected_ids() == [ids[0]] + ids[2:]
        # The dialog is accepted by the unattended mode: only D1 is played
        Conf.showcase_excluded_scenarios.set(ids[1:])
        monkeypatch.setitem(player.PACE_SPEEDS, "normal", 0.0)
        # A dialog executed after another test is rejected at once: accept it here
        monkeypatch.setattr(dialog_module, "exec_dialog", accept_dialog)
        results = show_showcase_dialog(win)
    assert [(result.scenario_id, result.status) for result in results] == [
        ("d1", "done")
    ]
