# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Demo plans of the application plugins
======================================

The :mod:`datalab.gui.showcase.applications_plan` module derives, from the
contracts declared by an application plugin, what the showcase demo of this
application plays (see :mod:`datalab.gui.showcase.applications`):

- the examples designed for a method (:class:`~datalab.plugins.PluginExample`
  with recipe IDs), in their declared order: each one is tried with the first
  method it was designed for, then, while the number of method runs allows it,
  with its next methods, run on the result (e.g. compute a spectrum, then
  normalize it);
- without such an example, the first tool which needs no selection
  (:class:`~datalab.plugins.tools.PluginTool`), instruments first;
- without example nor tool, only the presentation of the application.

The plugins do not declare anything specific to the showcase. This module does
not depend on Qt.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, TypeVar

from sigima.objects import GeometryResult, TableResult

from datalab.plugins.tools import ToolSelection

if TYPE_CHECKING:
    from sigima.objects import ImageObj, SignalObj

    from datalab.plugins import PluginBase
    from datalab.plugins.examples import PluginExample
    from datalab.plugins.recipes import RecipeDescriptor, RecipeOutcome
    from datalab.plugins.tools import PluginTool

logger = logging.getLogger("datalab.showcase")

#: Maximum number of examples tried in the demo of an application
MAX_EXAMPLES = 2
#: Maximum number of method runs in the demo of an application
MAX_RECIPE_RUNS = 2
#: Maximum number of rows of the results shown in a table: the results of a method
#: are always shown on the plot of its main object, larger tables are only shown
#: there (they would take long to show, and could not be read during the demo)
MAX_RESULT_ROWS = 30

#: Approximate durations (s) at normal pace, for the duration hint of a demo
PRESENTATION_DURATION = 20.0
RECIPE_RUN_DURATION = 18.0
TOOL_DURATION = 45.0

Item = TypeVar("Item")


@dataclasses.dataclass(frozen=True)
class RecipeStep:
    """Method run in the demo of an application

    Args:
        recipe_id: Namespaced recipe ID
        title: Recipe title
    """

    recipe_id: str
    title: str


@dataclasses.dataclass(frozen=True)
class ExamplePlay:
    """Example tried in the demo of an application

    The example is opened by its **Try with this example** command, which runs
    the first method; the next methods (if any) run on the result.

    Args:
        example_id: Plugin-local example ID
        title: Example title
        description: Example description
        steps: Methods run: the first one with the example, the next ones on the
         result of the previous one
    """

    example_id: str
    title: str
    description: str
    steps: tuple[RecipeStep, ...]

    @property
    def recipe(self) -> RecipeStep:
        """Method run with the example"""
        return self.steps[0]

    @property
    def chained(self) -> tuple[RecipeStep, ...]:
        """Methods run on the result, after the first one"""
        return self.steps[1:]


@dataclasses.dataclass(frozen=True)
class ToolPlay:
    """Tool opened in the demo of an application

    Args:
        tool_id: Plugin-local tool ID
        title: Tool title
        description: Tool description
        instrument: True if the tool is an instrument (shown by DataLab)
    """

    tool_id: str
    title: str
    description: str
    instrument: bool


@dataclasses.dataclass(frozen=True)
class ApplicationDemoPlan:
    """What the showcase demo of an application plays

    Args:
        plugin_id: Plugin ID
        name: Application name
        description: Application description
        examples: Examples tried, in this order
        tool: Tool opened (only without example)
    """

    plugin_id: str
    name: str
    description: str
    examples: tuple[ExamplePlay, ...] = ()
    tool: ToolPlay | None = None

    @property
    def recipe_runs(self) -> int:
        """Number of method runs"""
        return sum(len(play.steps) for play in self.examples)

    @property
    def duration_hint(self) -> float:
        """Approximate duration (s) of the demo at normal pace"""
        duration = PRESENTATION_DURATION + RECIPE_RUN_DURATION * self.recipe_runs
        if self.tool is not None:
            duration += TOOL_DURATION
        return duration


def plan_examples(
    examples: Iterable[PluginExample],
    recipes: Iterable[RecipeDescriptor],
    max_examples: int = MAX_EXAMPLES,
    max_runs: int = MAX_RECIPE_RUNS,
) -> tuple[ExamplePlay, ...]:
    """Return the examples tried in the demo of an application

    The examples designed for at least one known method are taken in their
    declared order, up to ``max_examples`` and ``max_runs``. Each example runs its
    first method; the method runs left are given to the next methods of the
    examples, in their declared order.

    Args:
        examples: Examples declared by the application
        recipes: Methods declared by the application
        max_examples: Maximum number of examples
        max_runs: Maximum number of method runs

    Returns:
        Examples tried, in this order
    """
    titles = {recipe.recipe_id: recipe.title for recipe in recipes}
    candidates: list[tuple[PluginExample, list[RecipeStep]]] = []
    for example in examples:
        steps = [
            RecipeStep(recipe_id, titles[recipe_id])
            for recipe_id in example.recipe_ids
            if recipe_id in titles
        ]
        if steps:
            candidates.append((example, steps))
    candidates = candidates[: max(0, min(max_examples, max_runs))]
    extra_runs = max_runs - len(candidates)
    plays: list[ExamplePlay] = []
    for example, steps in candidates:
        chained = steps[1 : 1 + max(0, extra_runs)]
        extra_runs -= len(chained)
        plays.append(
            ExamplePlay(
                example.id,
                example.title,
                example.description,
                (steps[0], *chained),
            )
        )
    return tuple(plays)


def plan_tool(tools: Iterable[PluginTool]) -> ToolPlay | None:
    """Return the tool opened in the demo of an application

    The first instrument which needs no selection is preferred to the first other
    tool which needs no selection, in the declared order.

    Args:
        tools: Tools declared by the application

    Returns:
        Tool opened, or None if no tool can be opened without selection
    """
    candidates = [tool for tool in tools if tool.selection is ToolSelection.NONE]
    candidates.sort(key=lambda tool: tool.instrument is None)
    if not candidates:
        return None
    tool = candidates[0]
    return ToolPlay(tool.id, tool.title, tool.description, tool.instrument is not None)


def _declared(getter: Callable[[], Sequence[Item]], what: str, name: str) -> tuple:
    """Return declarations of a plugin, or nothing if they are broken"""
    try:
        return tuple(getter())
    except Exception:  # pylint: disable=broad-exception-caught
        # Plugin declarations are third-party code: the demo leaves them out
        logger.exception("Showcase: the %s of %s cannot be used", what, name)
        return ()


def plan_application_demo(plugin: PluginBase) -> ApplicationDemoPlan:
    """Return what the showcase demo of an application plays

    Only the contracts declared by the application plugin are used: its
    information (name, description), examples, methods and tools. Broken
    declarations are logged and left out.

    Args:
        plugin: Application plugin

    Returns:
        Demo plan
    """
    info = plugin.info
    name = info.name or plugin.plugin_id
    examples = plan_examples(
        _declared(plugin.get_examples, "examples", name),
        _declared(plugin.get_recipes, "methods", name),
    )
    tool = None
    if not examples:
        tool = plan_tool(_declared(plugin.get_tools, "tools", name))
    return ApplicationDemoPlan(
        plugin.plugin_id, name, info.description or "", examples, tool
    )


def main_output(outcome: RecipeOutcome) -> SignalObj | ImageObj | None:
    """Return the main object created by a method run

    The object holding the first scalar result is the main one; without scalar
    result, the first object created is.

    Args:
        outcome: Outcome of the method run

    Returns:
        Main object, or None if no object was created
    """
    objects = {output.id: output.value for output in outcome.objects}
    for result in outcome.results:
        if result.anchor_id in objects:
            return objects[result.anchor_id]
    return outcome.objects[0].value if outcome.objects else None


def count_result_rows(outcome: RecipeOutcome) -> int:
    """Return the number of rows of the scalar results of a method run

    Args:
        outcome: Outcome of the method run

    Returns:
        Number of rows of the tables, and of the shapes of the geometric results
    """
    rows = 0
    for output in outcome.results:
        value = output.value
        if isinstance(value, TableResult):
            rows += len(value.data)
        elif isinstance(value, GeometryResult):
            rows += len(value.coords)
    return rows
