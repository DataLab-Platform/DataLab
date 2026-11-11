# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase mode unit tests

Command-line options and scenario selection of the showcase mode.
"""

from __future__ import annotations

import sys

import pytest

from datalab.env import SHOWCASE_WEB_URL, DLExecEnv
from datalab.gui.showcase import player
from datalab.gui.showcase.engine import ScenarioSkipped, ShowcaseScenario
from datalab.gui.showcase.player import (
    ShowcaseOptions,
    interleave,
    parse_scenario_ids,
    select_scenarios,
)
from datalab.gui.showcase.scenarios import get_native_scenarios


def _parse(monkeypatch: pytest.MonkeyPatch, *args: str) -> DLExecEnv:
    """Return an execution environment parsing command-line arguments"""
    monkeypatch.setattr(sys, "argv", ["datalab", *args])
    return DLExecEnv()


def test_showcase_command_line_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """All the showcase options are parsed, and give the showcase options"""
    env = _parse(
        monkeypatch,
        "--showcase",
        "d1,d3",
        "--showcase-loop",
        "--showcase-speed",
        "0.7",
        "--showcase-screen",
        "1",
        "--showcase-fullscreen",
        "--showcase-web",
        "http://localhost:5173/",
    )
    assert env.showcase == "d1,d3"
    assert env.showcase_loop and env.showcase_fullscreen
    assert env.showcase_speed == pytest.approx(0.7)
    assert env.showcase_screen == 1
    assert env.showcase_web == "http://localhost:5173/"
    monkeypatch.setattr(player, "execenv", env)
    assert ShowcaseOptions.from_execenv() == ShowcaseOptions(
        scenario_ids=["d1", "d3"],
        loop=True,
        speed=0.7,
        fullscreen=True,
        screen=1,
        web_url="http://localhost:5173/",
    )
    # Without URL, the public deployment of DataLab Web is played
    assert _parse(monkeypatch, "--showcase-web").showcase_web == SHOWCASE_WEB_URL


def test_showcase_command_line_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without IDS, --showcase plays all the scenarios, once, at normal pace"""
    env = _parse(monkeypatch, "--showcase")
    assert env.showcase == "all"
    assert not env.showcase_loop and not env.showcase_fullscreen
    assert env.showcase_speed is None and env.showcase_screen is None
    assert env.showcase_web is None
    monkeypatch.setattr(player, "execenv", env)
    assert ShowcaseOptions.from_execenv() == ShowcaseOptions()
    assert _parse(monkeypatch).showcase is None


def test_scenario_selection() -> None:
    """Scenarios are selected from their IDs, in the requested order"""
    assert parse_scenario_ids(None) is None
    assert parse_scenario_ids("all") is None
    assert parse_scenario_ids(" d1, ,d3 ") == ["d1", "d3"]
    assert interleave([1, 2, 3], ["a"]) == [1, "a", 2, 3]
    assert interleave([1], ["a", "b", "c"]) == [1, "a", "b", "c"]
    scenarios = get_native_scenarios()
    assert [scenario.id for scenario in scenarios] == ["d1", "d2", "d3", "d4"]
    assert all(scenario.title and scenario.subtitle for scenario in scenarios)
    selected = select_scenarios(scenarios, ["d3", "unknown", "d1", "d3"])
    assert [scenario.id for scenario in selected] == ["d3", "d1"]
    assert select_scenarios(scenarios, None) == scenarios


def test_scenario_requirements() -> None:
    """A scenario requiring a missing module is skipped"""
    scenario = ShowcaseScenario(
        "missing", "Missing", lambda ctx: None, requires=("json", "no_such_module")
    )
    assert scenario.missing_requirements() == ["no_such_module"]
    with pytest.raises(ScenarioSkipped):
        scenario.check_requirements()
