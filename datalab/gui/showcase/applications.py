# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase scenarios of the application plugins
=============================================

The :mod:`datalab.gui.showcase.applications` module generates a showcase scenario
for each active application plugin (plugin with the ``APPLICATION`` capability):
the installed applications take part in the showcase without any specific code.

The demo of an application is derived from the contracts it declares (see
:mod:`datalab.gui.showcase.applications_plan`), and played in the real
Applications panel:

1. The application is presented with its name and description.
2. Its examples designed for a method are tried with their **Try with this
   example** command: the example is opened, the parameters of the method are
   shown, then the method runs. The main object created is selected and shown,
   with its table of results (if any). The next methods designed for the example
   may then run on the result.
3. Without such an example, the first tool of the application is opened from the
   panel. An instrument is driven: live view, a few acquisitions (with and
   without the **New group** option), then the window is closed, which
   disconnects the instrument.
4. The demo ends with an invitation to discover the plugin catalog.

The showcase player only calls :func:`create_application_scenarios`.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING

from guidata.qthelpers import is_qobject_valid
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.objects import SignalObj

from datalab.config import Conf, _
from datalab.env import execenv
from datalab.gui.plugins.applications import get_application_plugins
from datalab.gui.plugins.instrument import InstrumentWindow
from datalab.gui.showcase.applications_plan import (
    MAX_RESULT_ROWS,
    ApplicationDemoPlan,
    ExamplePlay,
    RecipeStep,
    ToolPlay,
    count_result_rows,
    main_output,
    plan_application_demo,
)
from datalab.gui.showcase.engine import ScenarioSkipped, ShowcaseScenario, plain_text
from datalab.plugins.recipe_binding import RecipeReadinessStatus

if TYPE_CHECKING:
    from sigima.objects import ImageObj

    from datalab.gui.main import DLMainWindow
    from datalab.gui.panel.base import BaseDataPanel
    from datalab.gui.plugins.applications import ApplicationPage
    from datalab.gui.showcase.engine import ShowcaseContext
    from datalab.plugins import PluginBase
    from datalab.plugins.recipes import RecipeOutcome

logger = logging.getLogger("datalab.showcase")

#: Prefix of the IDs of the scenarios generated for the application plugins
SCENARIO_ID_PREFIX = "app:"

#: Maximum waiting time (s) for the first frame of an instrument (whatever the
#: speed): the instrument may have to start (e.g. a simulated device server)
INSTRUMENT_FRAME_TIMEOUT = 120.0
#: Maximum waiting time (s) for an acquisition of an instrument
INSTRUMENT_CALL_TIMEOUT = 60.0
#: Maximum waiting time (s) for an instrument to close, once its window is closed
INSTRUMENT_CLOSE_TIMEOUT = 30.0
#: Number of acquisitions made with an instrument, before and after switching the
#: **New group** option
ACQUISITIONS_BEFORE_SWITCH = 2
ACQUISITIONS_AFTER_SWITCH = 1
#: Factor applied to the delay of the result dialogs, to let the audience read them
RESULT_DIALOG_DELAY_FACTOR = 2.5

#: Configuration options changed by the commands of the Applications panel and
#: of the instrument windows, restored at the end of a demo
RESTORED_OPTIONS = (
    "applications_selected_methods",
    "welcome_recent_applications",
    "instrument_new_group",
)


def find_application_plugin(plugin_id: str) -> PluginBase:
    """Return an active application plugin

    Args:
        plugin_id: Plugin ID

    Returns:
        Application plugin

    Raises:
        ScenarioSkipped: If no active application plugin has this ID
    """
    for plugin in get_application_plugins():
        if plugin.plugin_id == plugin_id:
            return plugin
    raise ScenarioSkipped(f"Application plugin {plugin_id} is not active")


def ensure_visible(widget: QW.QWidget) -> None:
    """Scroll the scroll areas containing a widget, so that it is visible

    Args:
        widget: Widget
    """
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QW.QScrollArea):
            parent.ensureWidgetVisible(widget)
        parent = parent.parentWidget()
    QW.QApplication.processEvents()


@contextmanager
def capture_outcomes(
    page: ApplicationPage,
) -> Generator[dict[str | None, RecipeOutcome], None, None]:
    """Record the outcomes shown by an application page, within the context

    The Applications panel shows the outcome of each command it runs on the page
    of the application: the outcomes are recorded by recipe ID (None for the
    commands which are not attached to a method).

    Args:
        page: Application page

    Yields:
        Outcomes shown, by recipe ID
    """
    outcomes: dict[str | None, RecipeOutcome] = {}
    show_outcome = page.show_outcome

    def record_outcome(outcome: RecipeOutcome, recipe_id: str | None = None) -> None:
        """Record an outcome, then show it"""
        outcomes[recipe_id] = outcome
        show_outcome(outcome, recipe_id)

    page.show_outcome = record_outcome
    try:
        yield outcomes
    finally:
        del page.show_outcome


@contextmanager
def longer_dialogs(factor: float) -> Generator[None, None, None]:
    """Show the dialogs longer in demo mode, within the context

    Args:
        factor: Factor applied to the delay of the demo mode

    Yields:
        None
    """
    delay = execenv.delay
    if delay > 0:
        execenv.delay = int(delay * factor)
    try:
        yield
    finally:
        execenv.delay = delay


def count_objects(win: DLMainWindow) -> int:
    """Return the number of signals and images of the workspace

    Args:
        win: DataLab main window

    Returns:
        Number of objects
    """
    return len(win.signalpanel) + len(win.imagepanel)


def find_panel_action(panel: BaseDataPanel, text: str) -> QW.QAction | None:
    """Return the action of a data panel which has a given text

    Args:
        panel: Data panel
        text: Action text

    Returns:
        Action, or None if the panel has no such action
    """
    for action in panel.actions():
        if action.text() == text:
            return action
    return None


class ApplicationStage:
    """Stage of the demo of an application: the Applications panel and the tools

    Within the context, the stage shows the page of the application in the
    Applications panel. A floating panel is placed over the main window at
    showcase speed, and hidden while the results are presented: it is never
    docked, nor undocked. On exit, the windows of the tools are closed, the panel
    is hidden (with its previous geometry) and the options changed by the
    commands of the panel are restored.

    Args:
        ctx: Showcase context
        plugin: Application plugin
    """

    def __init__(self, ctx: ShowcaseContext, plugin: PluginBase) -> None:
        self.ctx = ctx
        self.win = ctx.win
        self.plugin = plugin
        self.panel = ctx.win.applicationspanel
        self.dock = ctx.win.docks[self.panel]
        self.windows: list[QW.QWidget] = []
        self.__floating_geometry: QC.QRect | None = None
        self.__options: dict[str, object] = {}

    def __enter__(self) -> ApplicationStage:
        self.__options = {name: getattr(Conf, name).get() for name in RESTORED_OPTIONS}
        return self

    def __exit__(self, *args) -> None:
        try:
            self.close_tool_windows()
        finally:
            self.hide_panel()
            for name, value in self.__options.items():
                getattr(Conf, name).set(value)
            self.panel.SIG_WELCOME_PREFERENCES_CHANGED.emit()
            QW.QApplication.processEvents()

    # ------ Applications panel ---------------------------------------------------
    def stage_geometry(self) -> QC.QRect:
        """Return the geometry of the floating panel over the main window

        The panel is placed on the left of the main window, below its menu bar
        and above the captions.
        """
        geometry = self.win.geometry()
        top = geometry.top() + self.win.menuBar().height() + 6
        height = max(200, int(geometry.height() * 0.72) - self.win.menuBar().height())
        hint = max(self.panel.sizeHint().width(), self.panel.minimumSizeHint().width())
        width = min(hint, max(self.panel.minimumWidth(), geometry.width() // 3))
        return QC.QRect(geometry.left() + 6, top, width, height)

    def show_panel(self) -> ApplicationPage:
        """Show the page of the application in the Applications panel

        Returns:
            Page of the application
        """
        plugin_id = self.plugin.plugin_id
        if not self.dock.isVisible() or self.panel.current_plugin_id() != plugin_id:
            self.win.show_applications(plugin_id)
        if self.dock.isFloating() and not self.ctx.check:
            if self.__floating_geometry is None:
                self.__floating_geometry = QC.QRect(self.dock.geometry())
            # Placed before being painted where the user left it
            self.dock.setGeometry(self.stage_geometry())
            self.dock.raise_()
        QW.QApplication.processEvents()
        page = self.panel.current_page()
        self.ctx.checkpoint(
            page is not None and page.plugin is self.plugin and self.dock.isVisible(),
            f"The Applications panel does not show {plugin_id}",
        )
        return page

    def hide_panel(self, floating_only: bool = False) -> None:
        """Hide the Applications panel, where the user left it

        A floating panel gets its previous geometry back before being hidden: it
        is shown there again afterwards.

        Args:
            floating_only: True to hide the panel only if it floats (it would hide
             the results)
        """
        if not self.dock.isVisible() or (floating_only and not self.dock.isFloating()):
            return
        if self.__floating_geometry is not None and self.dock.isFloating():
            self.dock.setGeometry(self.__floating_geometry)
        self.dock.hide()
        QW.QApplication.processEvents()

    def hide_floating_panel(self) -> None:
        """Hide the Applications panel if it floats (it would hide the results)"""
        self.hide_panel(floating_only=True)

    def present(self, plan: ApplicationDemoPlan) -> None:
        """Present the application in the Applications panel

        Args:
            plan: Demo plan of the application
        """
        ctx = self.ctx
        self.show_panel()
        text = plan.name
        if plan.description:
            text += " — " + plan.description
        with ctx.spotlight(self.panel):
            ctx.say(text, hold=4.0)
            ctx.say(
                _(
                    "Applications extend DataLab for specific fields: their "
                    "methods, examples and tools are gathered in the Applications "
                    "panel"
                ),
                hold=3.5,
            )
            if not plan.examples and plan.tool is None:
                ctx.say(
                    _("Its methods run on your own data, selected in the workspace"),
                    hold=3.0,
                )

    # ------ Examples and methods -------------------------------------------------
    def play_example(self, play: ExamplePlay) -> None:
        """Try an example with its first method, then run its next methods

        Args:
            play: Example to try
        """
        ctx, win = self.ctx, self.win
        # The example replaces the workspace: no question when it is empty
        win.reset_all()
        page = self.show_panel()
        recipe = play.recipe
        page.select_recipe(recipe.recipe_id)
        button = page.recipe_details[recipe.recipe_id].example_buttons.get(
            play.example_id
        )
        ctx.checkpoint(
            button is not None, f"Example {play.example_id} has no Try command"
        )
        ensure_visible(button)
        text = _("Example: %s") % play.title
        if play.description:
            text += "\n" + play.description
        with ctx.spotlight(self.panel):
            ctx.say(text, hold=4.0)
            ctx.say(
                _("One click opens the example, then runs the method “%s” on it")
                % recipe.title,
                hold=2.5,
            )
            with capture_outcomes(page) as outcomes:
                ctx.click(button)
        outcome = outcomes.get(recipe.recipe_id)
        outputs = len(outcome.objects) if outcome is not None else 0
        ctx.checkpoint(
            count_objects(win) > outputs,
            f"Example {play.example_id} created no object",
        )
        self.check_outcome(outcome, recipe)
        self.present_outputs(outcome, recipe)
        for step in play.chained:
            if not self.run_on_result(step):
                break

    def check_outcome(self, outcome: RecipeOutcome | None, recipe: RecipeStep) -> None:
        """Check that a method run succeeded and created objects

        Args:
            outcome: Outcome of the method run, if any
            recipe: Method
        """
        self.ctx.checkpoint(
            outcome is not None, f"Method {recipe.recipe_id} did not succeed"
        )
        if outcome.diagnostics:
            levels = sorted(
                {diagnostic.level.value for diagnostic in outcome.diagnostics}
            )
            logger.info(
                "Showcase: %s: %d diagnostics (%s)",
                recipe.recipe_id,
                len(outcome.diagnostics),
                ", ".join(levels),
            )
        self.ctx.checkpoint(
            len(outcome.objects) >= 1, f"Method {recipe.recipe_id} created no object"
        )

    def present_outputs(self, outcome: RecipeOutcome, recipe: RecipeStep) -> None:
        """Show the main object created by a method run, and its results

        Args:
            outcome: Outcome of the method run
            recipe: Method
        """
        ctx, win = self.ctx, self.win
        main: SignalObj | ImageObj = main_output(outcome)
        panel = win.signalpanel if isinstance(main, SignalObj) else win.imagepanel
        self.hide_floating_panel()
        win.set_current_panel(panel)
        panel.objview.select_objects([main])
        QW.QApplication.processEvents()
        with ctx.spotlight(panel.objview, panel.plothandler.plotwidget):
            ctx.say(
                _("The method “%s” adds its results to the workspace, in a new group")
                % recipe.title,
                hold=4.0,
            )
        if outcome.results and count_result_rows(outcome) <= MAX_RESULT_ROWS:
            action = find_panel_action(panel, _("Show results") + "...")
            if action is not None and action.isEnabled():
                ctx.say(
                    _("Its measurements are attached to the result, in a table"),
                    hold=1.5,
                )
                with longer_dialogs(RESULT_DIALOG_DELAY_FACTOR):
                    ctx.trigger_action(action)
                ctx.close_stray_windows()

    def run_on_result(self, step: RecipeStep) -> bool:
        """Run a method on the result of the previous one, if it is ready

        Args:
            step: Method to run

        Returns:
            True if the method ran, False if it is not ready for the result
        """
        ctx = self.ctx
        readiness = self.plugin.assess_recipe(step.recipe_id)
        if readiness.status is not RecipeReadinessStatus.READY:
            logger.info(
                "Showcase: %s is not ready for the result (%s)",
                step.recipe_id,
                readiness.status.value,
            )
            return False
        page = self.show_panel()
        page.select_recipe(step.recipe_id)
        row = page.recipe_rows[step.recipe_id]
        ensure_visible(row)
        with ctx.spotlight(self.panel):
            ctx.say(_("Then, the method “%s” runs on this result") % step.title, 3.5)
            with capture_outcomes(page) as outcomes:
                ctx.click(row.run_button)
        outcome = outcomes.get(step.recipe_id)
        self.check_outcome(outcome, step)
        self.present_outputs(outcome, step)
        return True

    # ------ Tools ----------------------------------------------------------------
    def visible_windows(self) -> list[QW.QWidget]:
        """Return the visible windows, except the main window, docks and menus"""
        return [
            widget
            for widget in QW.QApplication.topLevelWidgets()
            if widget is not self.win
            and is_qobject_valid(widget)
            and widget.isVisible()
            and not isinstance(widget, (QW.QDockWidget, QW.QMenu))
        ]

    def place_tool_window(self, window: QW.QWidget) -> None:
        """Place a tool window over the right part of the main window, above the
        captions, at showcase speed

        Args:
            window: Tool window
        """
        if self.ctx.check:
            return
        geometry = self.win.geometry()
        window.resize(
            min(window.width(), int(geometry.width() * 0.62)),
            min(window.height(), int(geometry.height() * 0.66)),
        )
        splitter = window.findChild(QW.QSplitter)
        if isinstance(window, InstrumentWindow) and splitter is not None:
            # Room for the settings: the live view gets narrower first
            window.layout().activate()
            width = sum(splitter.sizes())
            splitter.setSizes([int(width * 0.56), width - int(width * 0.56)])
        frame = window.frameGeometry()
        left = max(geometry.left(), geometry.right() - frame.width() - 16)
        window.move(left, geometry.top() + self.win.menuBar().height() + 6)
        window.raise_()
        QW.QApplication.processEvents()

    def play_tool(self, tool: ToolPlay) -> None:
        """Open a tool from the Applications panel, then drive it

        Args:
            tool: Tool to open
        """
        ctx = self.ctx
        page = self.show_panel()
        page.refresh_readiness()
        button = page.tool_buttons.get(tool.tool_id)
        ctx.checkpoint(button is not None, f"Tool {tool.tool_id} has no Open command")
        ensure_visible(button)
        text = _("Tool: %s") % plain_text(tool.title).rstrip(".…")
        if tool.description:
            text += "\n" + tool.description
        before = self.visible_windows()
        with ctx.spotlight(self.panel):
            ctx.say(text, hold=4.0)
            ctx.click(button)
        opened = [window for window in self.visible_windows() if window not in before]
        self.windows.extend(opened)
        self.hide_floating_panel()
        instruments = [w for w in opened if isinstance(w, InstrumentWindow)]
        if tool.instrument and instruments:
            self.drive_instrument(instruments[0])
            return
        for window in opened:
            self.place_tool_window(window)
        if opened:
            with ctx.spotlight(*opened):
                ctx.say(_("The tool opens in its own window"), hold=4.0)
        self.close_tool_windows()

    def drive_instrument(self, window: InstrumentWindow) -> None:
        """Drive an instrument window: live view, acquisitions, then closing

        Args:
            window: Instrument window
        """
        ctx, win = self.ctx, self.win
        before = count_objects(win)
        self.place_tool_window(window)

        def shows_frame_or_error() -> bool:
            """Return True once the window shows a frame, or a failure"""
            return bool(window.items) or (
                window.is_idle() and bool(window.status_label.text())
            )

        with ctx.spotlight(window):
            ctx.say(_("Connecting to the instrument..."), hold=0.5)
            waiting = (
                _(
                    "The instrument window shows a live view, next to the "
                    "instrument settings"
                ),
                _("The instrument may take a few seconds to start..."),
            )
            for text in waiting:
                if ctx.wait_until(shows_frame_or_error, 6.0):
                    break
                ctx.say(text, hold=0.5)
            ctx.wait_until(shows_frame_or_error, INSTRUMENT_FRAME_TIMEOUT)
            connected = bool(window.items)
            if connected:
                self.show_live_view(window)
        if connected:
            # The workspace stays bright: the acquisitions show up in it
            with ctx.spotlight(window, win.tabwidget):
                self.acquire(window)
        with ctx.spotlight(window):
            if not connected:
                logger.warning(
                    "Showcase: instrument without frame: %s", window.status_label.text()
                )
                ctx.say(
                    _("This tool needs a connected instrument: it is only shown here"),
                    hold=4.0,
                )
            ctx.say(_("Closing the window disconnects the instrument"), hold=2.0)
            self.close_instrument(window)
        if not connected:
            return
        ctx.checkpoint(count_objects(win) > before, "The instrument acquired nothing")
        panel = (
            win.signalpanel if win.get_current_panel() == "signal" else win.imagepanel
        )
        with ctx.spotlight(panel.objview, panel.plothandler.plotwidget):
            ctx.say(_("The acquisitions are ready to be processed in DataLab"), 4.0)

    def show_live_view(self, window: InstrumentWindow) -> None:
        """Show the live view of an instrument for a while

        Args:
            window: Instrument window, showing a frame
        """
        ctx = self.ctx
        ctx.say(_("Live view: the instrument is refreshed continuously"), hold=0.5)
        ctx.click(window.live_button)
        ctx.pause(3.5)
        if window.live_button.isChecked():
            ctx.click(window.live_button)
        ctx.wait_until(window.is_idle, INSTRUMENT_CALL_TIMEOUT)

    def acquire(self, window: InstrumentWindow) -> None:
        """Acquire with an instrument, before and after switching the **New
        group** option

        Args:
            window: Instrument window, showing a frame
        """
        ctx = self.ctx
        ctx.say(_("Each acquisition is added to the workspace"), hold=0.5)
        self.click_acquire(window, ACQUISITIONS_BEFORE_SWITCH)
        ctx.click(window.new_group_box)
        if window.new_group_box.isChecked():
            text = _("With “New group”, each acquisition gets its own group")
        else:
            text = _(
                "Without “New group”, acquisitions accumulate in the current group"
            )
        ctx.say(text, hold=0.5)
        self.click_acquire(window, ACQUISITIONS_AFTER_SWITCH)
        ctx.pause(1.5)

    def click_acquire(self, window: InstrumentWindow, count: int) -> None:
        """Click on the **Acquire** button of an instrument window

        Args:
            window: Instrument window
            count: Number of acquisitions

        Raises:
            AssertionError: If an acquisition does not end in time
        """
        for _index in range(count):
            self.ctx.click(window.acquire_button)
            self.ctx.checkpoint(
                self.ctx.wait_until(window.is_idle, INSTRUMENT_CALL_TIMEOUT),
                "Instrument acquisition timed out",
            )
            self.ctx.pause(1.2)

    def close_instrument(self, window: InstrumentWindow) -> None:
        """Close an instrument window with its Close button, and wait for the
        instrument to be closed

        Args:
            window: Instrument window
        """
        buttons = window.findChild(QW.QDialogButtonBox)
        button = None if buttons is None else buttons.button(QW.QDialogButtonBox.Close)
        if button is not None and button.isVisible():
            self.ctx.click(button)
        else:
            window.close()
        if window in self.windows:
            self.windows.remove(window)
        future = window.close_future
        if future is not None and not self.ctx.wait_until(
            future.done, INSTRUMENT_CLOSE_TIMEOUT
        ):
            logger.warning("Showcase: the instrument did not close in time")

    def close_tool_windows(self) -> None:
        """Close the windows opened by the tools, waiting for the instruments"""
        while self.windows:
            window = self.windows.pop()
            if not is_qobject_valid(window):
                continue
            if isinstance(window, InstrumentWindow):
                self.close_instrument(window)
            else:
                window.close()
        QW.QApplication.processEvents()

    # ------ Plugin catalog ------------------------------------------------------
    def invite_to_catalog(self) -> None:
        """Invite the audience to discover the plugin catalog (Plugins menu)"""
        ctx, win = self.ctx, self.win
        self.hide_floating_panel()
        action = win.configure_plugins_action
        path = plain_text(win.plugins_menu.title()) + " > " + plain_text(action.text())
        with ctx.spotlight(win.menuBar()):
            ctx.say(
                _("Discover more applications in the plugin catalog: %s") % path,
                hold=1.0,
            )
            ctx.point_action(action)
            ctx.pause(2.5)


def play_application(ctx: ShowcaseContext, plugin_id: str) -> None:
    """Play the demo of an application plugin

    Args:
        ctx: Showcase context
        plugin_id: Plugin ID

    Raises:
        ScenarioSkipped: If the application plugin is no longer active
    """
    plugin = find_application_plugin(plugin_id)
    plan = plan_application_demo(plugin)
    with ApplicationStage(ctx, plugin) as stage:
        stage.present(plan)
        for play in plan.examples:
            stage.play_example(play)
        if plan.tool is not None:
            stage.play_tool(plan.tool)
        stage.invite_to_catalog()


def create_application_scenario(plugin: PluginBase) -> ShowcaseScenario:
    """Return the showcase scenario of an application plugin

    Args:
        plugin: Application plugin

    Returns:
        Scenario, whose ID is the plugin ID prefixed with ``app:``
    """
    info = plugin.info
    return ShowcaseScenario(
        id=SCENARIO_ID_PREFIX + plugin.plugin_id,
        title=info.name or plugin.plugin_id,
        subtitle=info.description or None,
        run=partial(play_application, plugin_id=plugin.plugin_id),
        duration_hint=plan_application_demo(plugin).duration_hint,
    )


def create_application_scenarios(
    win: DLMainWindow,  # pylint: disable=unused-argument
) -> list[ShowcaseScenario]:
    """Return the showcase scenarios of the active application plugins

    An application whose scenario cannot be created is left out (the error is
    logged), so that the other applications keep their scenario.

    Args:
        win: DataLab main window

    Returns:
        One scenario per active application plugin, in display order
    """
    scenarios: list[ShowcaseScenario] = []
    for plugin in get_application_plugins():
        try:
            scenarios.append(create_application_scenario(plugin))
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Showcase scenario of %s cannot be created", plugin)
    return scenarios
