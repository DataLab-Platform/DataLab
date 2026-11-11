# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase scenarios of the application plugins

The demo plans derived from the plugin contracts are checked without Qt; the
generated scenarios are played in check mode (speed 0) with test applications:
an application with a generated example and two methods, an application with an
instrument only, and an application with nothing to try.
"""

from __future__ import annotations

from collections.abc import Generator
from types import SimpleNamespace

import guidata.dataset as gds
import numpy as np
import pytest
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.objects import TableResult, create_signal

from datalab.config import Conf
from datalab.gui.showcase import ShowcaseContext, ShowcaseOptions, play_showcase
from datalab.gui.showcase import applications as showcase_applications
from datalab.gui.showcase.applications_plan import (
    MAX_RECIPE_RUNS,
    MAX_RESULT_ROWS,
    ApplicationDemoPlan,
    count_result_rows,
    main_output,
    plan_application_demo,
    plan_examples,
    plan_tool,
)
from datalab.plugins import PluginBase, PluginCapability, PluginInfo, PluginRegistry
from datalab.plugins.examples import PluginExample, PluginExampleData
from datalab.plugins.instruments import (
    InstrumentAcquisition,
    InstrumentFrame,
    PluginInstrument,
)
from datalab.plugins.recipes import (
    RecipeCardinality,
    RecipeDescriptor,
    RecipeInputSlot,
    RecipeObjectOutput,
    RecipeObjectType,
    RecipeOutcome,
    RecipeResultOutput,
)
from datalab.plugins.tools import PluginTool
from datalab.tests import datalab_test_app_context
from datalab.tests.features.plugins.applications_panel_test import _plugin

METHODS_ID = "org.example.showcase-methods"
INSTRUMENT_ID = "org.example.showcase-instrument"
APPLICATION = (PluginCapability.APPLICATION,)


class GainParameters(gds.DataSet):
    """Gain applied by the amplification method"""

    gain = gds.FloatItem("Gain", default=2.0)


def run_amplify(inputs, parameters, _context) -> RecipeOutcome:
    """Amplify signals; the peak of the last one is the scalar result"""
    outputs = []
    for index, signal in enumerate(inputs["signals"]):
        output = create_signal(
            f"Amplified {signal.title}", signal.x, signal.y * parameters.gain
        )
        outputs.append(RecipeObjectOutput(f"amplified-{index}", output))
    peak = TableResult(
        title="Peak", headers=["Peak"], data=[[float(outputs[-1].value.y.max())]]
    )
    return RecipeOutcome(
        objects=outputs,
        results=(RecipeResultOutput("peak", peak, anchor_id=outputs[-1].id),),
    )


def run_normalize(inputs, _parameters, _context) -> RecipeOutcome:
    """Normalize signals to their maximum"""
    return RecipeOutcome(
        objects=[
            RecipeObjectOutput(
                f"normalized-{index}",
                create_signal(
                    f"Normalized {signal.title}", signal.x, signal.y / signal.y.max()
                ),
            )
            for index, signal in enumerate(inputs["signals"])
        ]
    )


def _recipe(name: str, run, parameter_class=None) -> RecipeDescriptor:
    """Return a method of the test application, working on signals"""
    return RecipeDescriptor(
        recipe_id=f"{METHODS_ID}:{name}",
        plugin_version="1.0.0",
        title=name.capitalize(),
        version="1.0.0",
        inputs=(
            RecipeInputSlot("signals", RecipeObjectType.SIGNAL, RecipeCardinality.MANY),
        ),
        parameter_class=parameter_class,
        run=run,
    )


AMPLIFY = _recipe("amplify", run_amplify, GainParameters)
NORMALIZE = _recipe("normalize", run_normalize)
CAMPAIGN = PluginExample(
    id="campaign",
    title="Synthetic campaign",
    description="Three noisy pulses",
    recipe_ids=(AMPLIFY.recipe_id, NORMALIZE.recipe_id),
)
DATASET = PluginExample(
    id="raw", title="Raw dataset", resource="datalab:data/tests/reordering_test.h5"
)


class RampSettings(gds.DataSet):
    """Ramp generator settings"""

    slope = gds.FloatItem("Slope", default=1.0)


class RampInstrument(PluginInstrument):
    """Instrument generating ramps, recording its life cycle"""

    live_interval_ms = 50

    def __init__(self) -> None:
        super().__init__(RampSettings())
        self.events: list[str] = []

    def open(self) -> None:
        self.events.append("open")

    def close(self) -> None:
        self.events.append("close")

    def __ramp(self, title: str):
        x = np.linspace(0.0, 1.0, 11)
        return create_signal(title, x, self.settings.slope * x)

    def preview(self) -> InstrumentFrame:
        return InstrumentFrame((self.__ramp("Live"),))

    def acquire(self) -> InstrumentAcquisition:
        self.events.append("acquire")
        return InstrumentAcquisition("Ramp acquisition", [self.__ramp("Ramp")])


@pytest.fixture(name="applications")
def fixture_applications() -> Generator[SimpleNamespace, None, None]:
    """Give the classes of the test applications, forgotten after the test"""

    class MethodsApplication(PluginBase):
        """Application with a generated example and two methods"""

        PLUGIN_INFO = PluginInfo(
            id=METHODS_ID,
            name="Showcase methods",
            version="1.0.0",
            description="Amplify and normalize pulses",
            capabilities=APPLICATION,
        )
        RECIPES = (AMPLIFY, NORMALIZE)
        EXAMPLES = (CAMPAIGN, DATASET)

        @classmethod
        def materialize_example(cls, example_id: str) -> PluginExampleData | None:
            if example_id != CAMPAIGN.id:
                return None
            x = np.linspace(-5.0, 5.0, 101)
            signals = [
                create_signal(f"Pulse {index}", x, (index + 1) * np.exp(-(x**2)))
                for index in range(3)
            ]
            return PluginExampleData(signals, {AMPLIFY.recipe_id: {"gain": 3.0}})

        def create_actions(self) -> None:
            """No action: the Applications panel shows the declarations"""

    class InstrumentApplication(PluginBase):
        """Application with an instrument, and a tool needing a selection"""

        PLUGIN_INFO = PluginInfo(
            id=INSTRUMENT_ID,
            name="Showcase instrument",
            version="1.0.0",
            description="Generate ramps",
            capabilities=APPLICATION,
        )
        TOOLS = (
            PluginTool(
                id="inspect",
                title="Inspect...",
                launcher="inspect",
                selection="at_least_one",
            ),
            PluginTool(id="ramp", title="Ramp generator...", instrument="ramp"),
        )

        def __init__(self) -> None:
            super().__init__()
            self.instrument = RampInstrument()

        def ramp(self) -> RampInstrument:
            return self.instrument

        def inspect(self) -> None:
            raise AssertionError("A tool needing a selection is not demonstrated")

        def create_actions(self) -> None:
            """No action: the Applications panel shows the declarations"""

    classes = SimpleNamespace(
        methods=MethodsApplication, instrument=InstrumentApplication
    )
    try:
        yield classes
    finally:
        for cls in (MethodsApplication, InstrumentApplication):
            if cls in PluginRegistry.get_plugin_classes():
                PluginRegistry.get_plugin_classes().remove(cls)


def _play(win, scenario_id: str, speed: float = 0.0):
    """Play a scenario of an application (default: in check mode), and return its
    result"""
    # Once the application was asked to quit outside its event loop (e.g. by a
    # previous test), every dialog executed afterwards returns at once, rejected,
    # until the event loop runs again: it is run once here, so that the parameter
    # dialogs of the methods are accepted by the demo mode, whatever the test order.
    # `exit` (unlike `quit` with Qt 6) does not close the windows: closing the main
    # window would unregister the plugins.
    qt_app = QW.QApplication.instance()
    QC.QTimer.singleShot(0, lambda: qt_app.exit(0))
    qt_app.exec()
    scenarios = showcase_applications.create_application_scenarios(win)
    options = ShowcaseOptions(scenario_ids=[scenario_id], speed=speed)
    (result,) = play_showcase(win, options, scenarios)
    return result


# ------ Demo plans (no Qt) ---------------------------------------------------------
def test_examples_are_planned_in_declared_order() -> None:
    """Examples designed for known methods are tried in their declared order; the
    method runs left go to their next methods"""
    other = PluginExample(id="other", title="Other", recipe_ids=(NORMALIZE.recipe_id,))
    plays = plan_examples((DATASET, CAMPAIGN), (AMPLIFY, NORMALIZE))
    assert [(play.example_id, play.recipe.recipe_id) for play in plays] == [
        ("campaign", AMPLIFY.recipe_id)
    ]
    assert [step.recipe_id for step in plays[0].chained] == [NORMALIZE.recipe_id]
    assert plays[0].recipe.title == "Amplify"
    plays = plan_examples((CAMPAIGN, other, CAMPAIGN), (AMPLIFY, NORMALIZE))
    assert [play.example_id for play in plays] == ["campaign", "other"]
    assert sum(len(play.steps) for play in plays) == MAX_RECIPE_RUNS
    assert not plays[0].chained
    assert plan_examples((CAMPAIGN,), (NORMALIZE,))[0].recipe == plays[1].recipe
    assert plan_examples((CAMPAIGN,), ()) == ()
    assert plan_examples((CAMPAIGN,), (AMPLIFY,), max_runs=0) == ()


def test_tools_are_planned_instruments_first() -> None:
    """The first tool needing no selection is planned, instruments first"""
    launcher = PluginTool(id="wizard", title="Wizard...", launcher="wizard")
    selection = PluginTool(
        id="inspect", title="Inspect...", instrument="inspect", selection="exactly_one"
    )
    instrument = PluginTool(
        id="ramp", title="Ramp...", instrument="ramp", description="Ramps"
    )
    tool = plan_tool((selection, launcher, instrument))
    assert (tool.tool_id, tool.instrument, tool.description) == ("ramp", True, "Ramps")
    assert plan_tool((selection, launcher)).tool_id == "wizard"
    assert not plan_tool((selection, launcher)).instrument
    assert plan_tool((selection,)) is None


def test_application_plans_come_from_contracts() -> None:
    """Plans use the declared contracts only; broken declarations are left out"""
    tool = PluginTool(id="ramp", title="Ramp...", instrument="ramp")
    plugin = _plugin(
        "org.example.plan",
        "Plan application",
        recipes=(AMPLIFY, NORMALIZE),
        examples=(CAMPAIGN,),
        tools=(tool,),
    )
    plan = plan_application_demo(plugin)
    assert (plan.plugin_id, plan.name) == ("org.example.plan", "Plan application")
    assert plan.description == "Plan application description"
    assert [play.example_id for play in plan.examples] == ["campaign"]
    assert plan.tool is None, "Tools are only shown without example"
    assert plan.recipe_runs == 2
    assert plan.duration_hint > ApplicationDemoPlan("id", "name", "").duration_hint

    def broken() -> tuple:
        raise ValueError("Broken declaration")

    plugin.get_examples = broken
    plan = plan_application_demo(plugin)
    assert not plan.examples and plan.tool.tool_id == "ramp"
    plugin.get_tools = broken
    assert plan_application_demo(plugin) == ApplicationDemoPlan(
        "org.example.plan", "Plan application", "Plan application description"
    )


def test_main_output_holds_the_first_result() -> None:
    """The main object of a method run holds its first scalar result, or is the
    first object created"""
    x = np.linspace(0.0, 1.0, 5)
    first, second = create_signal("First", x, x), create_signal("Second", x, x)
    table = TableResult(title="Table", headers=["Value"], data=[[1.0]])
    objects = (RecipeObjectOutput("first", first), RecipeObjectOutput("second", second))
    results = (RecipeResultOutput("table", table, anchor_id="second"),)
    assert main_output(RecipeOutcome(objects, results)) is second
    assert main_output(RecipeOutcome(objects)) is first
    assert main_output(RecipeOutcome()) is None


def test_result_rows_are_counted() -> None:
    """Large tables of results are told apart from small ones"""
    x = np.linspace(0.0, 1.0, 5)
    objects = (RecipeObjectOutput("signal", create_signal("Signal", x, x)),)
    rows = [[float(index)] for index in range(MAX_RESULT_ROWS + 1)]
    table = TableResult(title="Table", headers=["Value"], data=rows)
    results = (RecipeResultOutput("table", table, anchor_id="signal"),)
    assert count_result_rows(RecipeOutcome(objects, results)) == MAX_RESULT_ROWS + 1
    assert count_result_rows(RecipeOutcome(objects)) == 0


def test_scenarios_are_generated_from_plugin_information() -> None:
    """Each scenario gets the ID, name and description of its application"""
    plugin = _plugin("org.example.ids", "IDs application", examples=())
    scenario = showcase_applications.create_application_scenario(plugin)
    assert scenario.id == "app:org.example.ids"
    assert (scenario.title, scenario.subtitle) == (
        "IDs application",
        "IDs application description",
    )
    assert scenario.duration_hint == ApplicationDemoPlan("", "", "").duration_hint


# ------ Scenarios played in DataLab (check mode) -----------------------------------
def test_application_example_scenario(applications: SimpleNamespace) -> None:
    """The example is tried with its first method, whose result is shown; the next
    method runs on this result. The panel and the options are left as they were"""
    selected = {"org.example.other": "org.example.other:method"}
    with (
        Conf.applications_selected_methods.context(selected),
        datalab_test_app_context(console=False) as win,
    ):
        plugin = applications.methods()
        plugin.register(win)
        recent = Conf.welcome_recent_applications.get()
        result = _play(win, "app:" + METHODS_ID)
        assert result.status == "done", result.message
        titles = [obj.title for obj in win.signalpanel.objmodel.get_all_objects()]
        assert titles[:3] == ["Pulse 0", "Pulse 1", "Pulse 2"]
        assert titles[3:6] == [f"Amplified Pulse {index}" for index in range(3)]
        # The method run on the result: the amplified signal holding the peak
        assert titles[6:] == ["Normalized Amplified Pulse 2"]
        amplified = win.signalpanel.objmodel.get_all_objects()[3]
        assert amplified.y.max() == pytest.approx(3.0)  # Example gain
        assert not win.docks[win.applicationspanel].isVisible()
        assert Conf.applications_selected_methods.get() == selected
        assert Conf.welcome_recent_applications.get() == recent


def test_reset_forgets_the_outcomes_of_the_applications(
    applications: SimpleNamespace,
) -> None:
    """The outcomes of the methods run by a loop are not shown in the next loop"""
    with datalab_test_app_context(console=False) as win:
        plugin = applications.methods()
        plugin.register(win)
        result = _play(win, "app:" + METHODS_ID)
        assert result.status == "done", result.message
        (page,) = [
            page
            for page in win.applicationspanel.application_pages
            if page.plugin.plugin_id == METHODS_ID
        ]
        labels = [row.outcome_label for row in page.recipe_rows.values()]
        assert any(not label.isHidden() for label in labels)
        ShowcaseContext(win, 0.0).reset_workspace()
        assert all(label.isHidden() for label in [*labels, page.outcome_label])


def test_application_instrument_scenario(applications: SimpleNamespace) -> None:
    """The instrument is opened from the panel, acquires with and without the New
    group option, then is closed. The option is left as it was"""
    with datalab_test_app_context(console=False) as win:
        plugin = applications.instrument()
        plugin.register(win)
        new_group = Conf.instrument_new_group.get()
        result = _play(win, "app:" + INSTRUMENT_ID)
        assert result.status == "done", result.message
        events = plugin.instrument.events
        assert events[0] == "open" and events[-1] == "close"
        assert events.count("acquire") == 3
        groups = win.signalpanel.objmodel.get_groups()
        assert sum(len(group.get_objects()) for group in groups) == 3
        assert len(groups) == 2, "The last acquisition goes to the current group"
        assert Conf.instrument_new_group.get() == new_group
        assert not [
            widget
            for widget in QW.QApplication.topLevelWidgets()
            if widget is not win and widget.isVisible()
        ]


def test_application_without_demo_scenario(registry_restored: list) -> None:
    """An application with nothing to try is only presented"""
    with datalab_test_app_context(console=False) as win:
        registry_restored.append(_plugin("org.example.page-only", "Page only"))
        result = _play(win, "app:org.example.page-only")
        assert result.status == "done", result.message
        assert not win.has_objects()


def test_floating_panel_is_given_back(registry_restored: list) -> None:
    """At showcase speed, the floating Applications panel is placed over the main
    window, then shown again where the user left it: it is never docked"""
    with datalab_test_app_context(console=False) as win:
        registry_restored.append(_plugin("org.example.page-only", "Page only"))
        dock = win.docks[win.applicationspanel]
        win.show_applications("org.example.page-only")
        QW.QApplication.processEvents()
        assert dock.isFloating()
        geometry = QC.QRect(dock.geometry())
        result = _play(win, "app:org.example.page-only", speed=0.05)
        assert result.status == "done", result.message
        assert dock.isFloating() and not dock.isVisible()
        win.show_applications()
        QW.QApplication.processEvents()
        assert dock.geometry() == geometry


def test_failing_example_does_not_stop_the_showcase(
    applications: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A method which fails makes the scenario fail, without any window left"""

    def fail(*_args) -> RecipeOutcome:
        raise ValueError("Method failure")

    with datalab_test_app_context(console=False) as win:
        plugin = applications.methods()
        plugin.register(win)
        monkeypatch.setattr(plugin, "start_recipe", lambda *_args, **_kwargs: fail())
        # Errors of plugin commands are reported in a message box: not shown here
        monkeypatch.setattr(
            "datalab.gui.plugins.applications.qt_handle_error_message",
            lambda *_args, **_kwargs: None,
        )
        result = _play(win, "app:" + METHODS_ID)
        assert result.status == "failed"
        assert "did not succeed" in result.message
        assert not win.docks[win.applicationspanel].isVisible()


@pytest.fixture(name="registry_restored")
def fixture_registry_restored() -> Generator[list, None, None]:
    """Give the list of registered plugins, restored after the test"""
    plugins = PluginRegistry.get_plugins()
    previous_plugins = list(plugins)
    try:
        yield plugins
    finally:
        plugins[:] = previous_plugins
