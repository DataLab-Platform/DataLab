# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase player
===============

The :mod:`datalab.gui.showcase.player` module gathers the showcase scenarios (the
native ones and those generated for the active application plugins) and plays
them in the DataLab main window, once or in a loop.
"""

from __future__ import annotations

import gc
import logging
import traceback
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.env import execenv
from datalab.gui.showcase import applications
from datalab.gui.showcase.engine import (
    ScenarioSkipped,
    ShowcaseContext,
    ShowcaseScenario,
    is_headless,
)
from datalab.gui.showcase.foreground import bring_to_front
from datalab.gui.showcase.scenarios import get_native_scenarios, web_interlude

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow

logger = logging.getLogger("datalab.showcase")

#: Speed of each pace proposed in the Showcase mode dialog
PACE_SPEEDS = {"slow": 1.5, "normal": 1.0, "fast": 0.7}

#: Maximum number of scenario results kept by a player (loops)
MAX_RESULTS = 200

Item = TypeVar("Item")


def parse_scenario_ids(text: str | None) -> list[str] | None:
    """Parse a comma-separated list of scenario IDs

    Args:
        text: Comma-separated IDs, ``'all'`` or None

    Returns:
        List of IDs, or None for all the scenarios
    """
    if text is None or text.strip().lower() == "all":
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


@dataclass
class ShowcaseOptions:
    """Showcase options

    Args:
        scenario_ids: IDs of the scenarios to play, in this order (None: all)
        loop: True to play the scenarios in a loop, until the operator stops it
        speed: Delay factor (1: normal pace, 0: check mode)
        fullscreen: True to show the main window in full screen
        screen: Index of the screen showing the main window (None: current one)
        web_url: URL of DataLab Web, to end each loop with the DataLab Web
         interlude (None: no interlude, unless its ID is in ``scenario_ids``), see
         :mod:`datalab.gui.showcase.scenarios.web_interlude`
    """

    scenario_ids: list[str] | None = None
    loop: bool = False
    speed: float = 1.0
    fullscreen: bool = False
    screen: int | None = None
    web_url: str | None = None

    @classmethod
    def from_execenv(cls) -> ShowcaseOptions:
        """Return the options given on the command line (``--showcase`` options)

        Returns:
            Showcase options
        """
        speed = execenv.showcase_speed
        return cls(
            scenario_ids=parse_scenario_ids(execenv.showcase),
            loop=execenv.showcase_loop,
            speed=1.0 if speed is None else max(speed, 0.0),
            fullscreen=execenv.showcase_fullscreen,
            screen=execenv.showcase_screen,
            web_url=execenv.showcase_web,
        )


@dataclass
class ScenarioResult:
    """Result of a scenario played by the showcase

    Args:
        scenario_id: Scenario ID
        status: ``'done'``, ``'skipped'`` or ``'failed'``
        message: Reason of the skip, or traceback of the failure
    """

    scenario_id: str
    status: str
    message: str = ""


def interleave(first: Sequence[Item], second: Sequence[Item]) -> list[Item]:
    """Interleave two sequences: a1, b1, a2, b2... then the remaining items

    Args:
        first: First sequence
        second: Second sequence

    Returns:
        Interleaved items
    """
    result: list[Item] = []
    for index in range(max(len(first), len(second))):
        result.extend(items[index] for items in (first, second) if index < len(items))
    return result


def get_showcase_scenarios(win: DLMainWindow) -> list[ShowcaseScenario]:
    """Return the available showcase scenarios

    The native scenarios alternate with the scenarios generated for the active
    application plugins (see :mod:`datalab.gui.showcase.applications`).

    Args:
        win: DataLab main window

    Returns:
        Scenarios, in playing order
    """
    try:
        generated = applications.create_application_scenarios(win)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Showcase scenarios of the applications cannot be created")
        generated = []
    return interleave(get_native_scenarios(), generated)


def select_scenarios(
    scenarios: Iterable[ShowcaseScenario], scenario_ids: Iterable[str] | None
) -> list[ShowcaseScenario]:
    """Select scenarios from their IDs (unknown IDs are logged, then ignored)

    Args:
        scenarios: Available scenarios
        scenario_ids: IDs of the scenarios to select, in this order (None: all)

    Returns:
        Selected scenarios
    """
    if scenario_ids is None:
        return list(scenarios)
    by_id = {scenario.id: scenario for scenario in scenarios}
    selected: list[ShowcaseScenario] = []
    for scenario_id in scenario_ids:
        scenario = by_id.get(scenario_id)
        if scenario is None:
            logger.warning("Unknown showcase scenario: %s", scenario_id)
        elif scenario not in selected:
            selected.append(scenario)
    return selected


def build_playlist(
    scenarios: Iterable[ShowcaseScenario], options: ShowcaseOptions
) -> list[ShowcaseScenario]:
    """Return the scenarios to play, including the DataLab Web interlude

    The DataLab Web interlude ends the playlist when a DataLab Web URL is given, or
    it is played where its ID is placed among the scenario IDs.

    Args:
        scenarios: Available scenarios
        options: Showcase options

    Returns:
        Scenarios to play, in this order
    """
    web = web_interlude.create_scenario(options.web_url)
    ids = options.scenario_ids
    if ids is not None and web.id in ids:
        return select_scenarios([*scenarios, web], ids)
    selected = select_scenarios(scenarios, ids)
    if options.web_url:
        selected.append(web)
    return selected


class ShowcasePlayer:
    """Showcase player: plays scenarios in the DataLab main window

    Each scenario is played in a clean workspace, after its title card. The demo
    mode of DataLab is enabled while playing: dialogs are shown for a while, then
    accepted. A scenario which fails is logged, then the showcase moves on to the
    next one: in a loop, the showcase only stops when the operator asks for it
    (Shift+F12, or Escape twice). Once stopped, the main window is given back in
    its previous state, with the demo mode disabled.

    Args:
        win: DataLab main window
        scenarios: Scenarios to play, in this order
        options: Showcase options (only one showcase is played at a time)
    """

    #: Player currently playing, if any
    active: ShowcasePlayer | None = None

    def __init__(
        self,
        win: DLMainWindow,
        scenarios: Iterable[ShowcaseScenario],
        options: ShowcaseOptions | None = None,
    ) -> None:
        self.win = win
        self.scenarios = list(scenarios)
        self.options = ShowcaseOptions() if options is None else options
        #: Context of the scenarios, while playing
        self.context: ShowcaseContext | None = None
        #: Results of the scenarios played (the last ones, in a loop)
        self.results: list[ScenarioResult] = []

    @property
    def dialog_delay(self) -> int:
        """Delay (ms) during which the demo mode shows each dialog"""
        speed = self.options.speed
        return 0 if speed == 0 else int(1000 * max(0.8, 1.2 * speed))

    def play(self) -> list[ScenarioResult]:
        """Play the scenarios (once, or in a loop until the operator stops it)

        Returns:
            Results of the scenarios played (empty if another showcase is playing)
        """
        if ShowcasePlayer.active is not None:
            logger.warning("A showcase is already playing")
            return []
        if not self.scenarios:
            logger.warning("No showcase scenario to play")
            return []
        ShowcasePlayer.active = self
        demo_state = (execenv.demo_mode, execenv.unattended, execenv.delay)
        window_state = None
        self.context = ctx = ShowcaseContext(self.win, self.options.speed)
        try:
            window_state = self.__show_window()
            execenv.enable_demo_mode(self.dialog_delay)
            ctx.enable_hotkeys()
            with ctx:
                ctx.start_warm_up()
                self.__play_loops(ctx)
        finally:
            execenv.demo_mode, execenv.unattended, execenv.delay = demo_state
            self.__restore_window(window_state)
            self.context = None
            ShowcasePlayer.active = None
        return self.results

    def __play_loops(self, ctx: ShowcaseContext) -> None:
        """Play the scenarios once, or in a loop"""
        while True:
            for scenario in self.scenarios:
                if ctx.stop_requested:
                    return
                self.results.append(self.__play_scenario(ctx, scenario))
                del self.results[:-MAX_RESULTS]
            if ctx.stop_requested:
                return
            ctx.end_card()
            if not self.options.loop or ctx.stop_requested:
                return
            # Soft reset between two loops
            ctx.reset_workspace()
            gc.collect()

    def __play_scenario(
        self, ctx: ShowcaseContext, scenario: ShowcaseScenario
    ) -> ScenarioResult:
        """Play a scenario in a clean workspace, after its title card"""
        logger.info("Showcase scenario: %s", scenario.id)
        try:
            scenario.check_requirements()
            ctx.begin_scenario(scenario)
            ctx.reset_workspace()
            if not ctx.check:
                # Another application may have been activated meanwhile
                bring_to_front(self.win)
            ctx.title_card(scenario.title, scenario.subtitle)
            scenario.run(ctx)
        except ScenarioSkipped as exc:
            logger.warning("Showcase scenario %s skipped: %s", scenario.id, exc)
            return ScenarioResult(scenario.id, "skipped", str(exc))
        except Exception:  # pylint: disable=broad-exception-caught
            message = traceback.format_exc()
            logger.error(
                "Showcase scenario %s failed:%s%s", scenario.id, chr(10), message
            )
            try:
                ctx.recover()
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception("Showcase recovery failed")
            return ScenarioResult(scenario.id, "failed", message)
        finally:
            ctx.end_scenario()
        return ScenarioResult(scenario.id, "done")

    def __show_window(self) -> tuple[QC.QByteArray, QC.Qt.WindowStates] | None:
        """Show the main window for the showcase

        Returns:
            Previous geometry and state of the main window, or None if unchanged
        """
        if self.options.speed == 0 or is_headless():
            return None
        win = self.win
        state = (win.saveGeometry(), win.windowState())
        screens = QG.QGuiApplication.screens()
        index = self.options.screen
        if index is not None:
            if not 0 <= index < len(screens):
                logger.warning("Showcase screen %d does not exist", index)
            elif screens[index] is not win.screen():
                win.showNormal()
                win.move(screens[index].availableGeometry().topLeft())
        if self.options.fullscreen:
            win.showFullScreen()
        else:
            win.showMaximized()
        bring_to_front(win)
        QW.QApplication.processEvents()
        return state

    def __restore_window(
        self, state: tuple[QC.QByteArray, QC.Qt.WindowStates] | None
    ) -> None:
        """Restore the geometry and the state of the main window"""
        if state is not None:
            geometry, window_state = state
            self.win.showNormal()
            self.win.restoreGeometry(geometry)
            self.win.setWindowState(window_state)
            self.win.show()


def play_showcase(
    win: DLMainWindow,
    options: ShowcaseOptions | None = None,
    scenarios: Iterable[ShowcaseScenario] | None = None,
) -> list[ScenarioResult]:
    """Play the showcase in a DataLab main window

    Args:
        win: DataLab main window
        options: Showcase options (default: all scenarios, once, at normal pace)
        scenarios: Available scenarios (default: :func:`get_showcase_scenarios`)

    Returns:
        Results of the scenarios played
    """
    options = ShowcaseOptions() if options is None else options
    if scenarios is None:
        scenarios = get_showcase_scenarios(win)
    selected = build_playlist(scenarios, options)
    return ShowcasePlayer(win, selected, options).play()


def play_from_command_line(win: DLMainWindow) -> list[ScenarioResult]:
    """Play the showcase requested on the command line (``--showcase`` options)

    Args:
        win: DataLab main window

    Returns:
        Results of the scenarios played
    """
    return play_showcase(win, ShowcaseOptions.from_execenv())


@contextmanager
def quiet_startup() -> Generator[None, None, None]:
    """Run the startup actions of DataLab without any question, for a showcase

    Startup questions (e.g. about the log files of a previous session) would block
    a showcase started from the command line, which may run unattended.

    Yields:
        None
    """
    state = (execenv.unattended, execenv.do_not_quit)
    execenv.unattended = execenv.do_not_quit = True
    try:
        yield
    finally:
        execenv.unattended, execenv.do_not_quit = state


def is_showcase_playing() -> bool:
    """Return True while a showcase is playing"""
    return ShowcasePlayer.active is not None
