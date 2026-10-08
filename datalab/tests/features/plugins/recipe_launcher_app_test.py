# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Application test for the generic Desktop interaction running recipes."""

from __future__ import annotations

import guidata.dataset as gds
import numpy as np
import pytest
from qtpy import QtCore as QC
from sigima.objects import create_image

from datalab.gui.plugins.recipe_inputs import RecipeInputDialog
from datalab.plugins import PluginBase, PluginCapability, PluginInfo, PluginRegistry
from datalab.plugins.examples import PluginExample, PluginExampleData
from datalab.plugins.recipe_binding import RecipeReadinessStatus
from datalab.plugins.recipes import (
    RecipeDescriptor,
    RecipeDiagnostic,
    RecipeInputSlot,
    RecipeMetadataRequirement,
    RecipeObjectOutput,
    RecipeOutcome,
)
from datalab.tests import datalab_test_app_context


class FlatParameters(gds.DataSet):
    """Parameters changing the number of exposure levels required."""

    minimum_levels = gds.IntItem("Minimum exposure levels", default=3)


def _frame(title: str, role: str | None = None, exposure: float | None = None):
    """Create a small camera frame with optional role and exposure metadata."""
    image = create_image(title, np.full((4, 4), float(exposure or 1.0)))
    if role is not None:
        image.metadata["role"] = role
    if exposure is not None:
        image.metadata["exposure"] = exposure
    return image


def _ramp(levels: int, with_roles: bool = True) -> list:
    """Create one dark frame and two flat frames per exposure level."""
    frames = [_frame("Dark", "dark" if with_roles else None)]
    for level in range(1, levels + 1):
        for index in (1, 2):
            frames.append(
                _frame(f"Flat {level}.{index}", "flat" if with_roles else None, level)
            )
    return frames


def _suggest(candidates) -> dict[str, list]:
    """Assign frames from their role metadata."""
    return {
        role + "_frames": [
            obj for obj in candidates if obj.metadata.get("role") == role
        ]
        for role in ("dark", "flat")
        if any(obj.metadata.get("role") == role for obj in candidates)
    }


def _check(inputs, parameters: FlatParameters) -> list[RecipeDiagnostic]:
    """Require enough exposure levels among the flat frames."""
    levels = {obj.metadata["exposure"] for obj in inputs["flat_frames"]}
    if len(levels) < parameters.minimum_levels:
        message = f"Flat frames need {parameters.minimum_levels} exposure levels"
        return [RecipeDiagnostic("error", "too-few-levels", message)]
    return []


@pytest.fixture(name="received")
def fixture_received() -> list[tuple]:
    """Collect the inputs and parameters received by the recipe."""
    return []


@pytest.fixture(name="recipe")
def fixture_recipe(received: list[tuple]) -> RecipeDescriptor:
    """Return a dark/flat recipe averaging the flat frames."""

    def run(inputs, parameters, _context) -> RecipeOutcome:
        received.append((inputs, parameters))
        mean = np.mean([image.data for image in inputs["flat_frames"]], axis=0)
        output = create_image("Mean flat", mean)
        return RecipeOutcome(objects=(RecipeObjectOutput("mean_flat", output),))

    return RecipeDescriptor(
        recipe_id="org.example.launch:flats",
        plugin_version="1.0.0",
        title="Flat analysis",
        version="1.0.0",
        run=run,
        inputs=(
            RecipeInputSlot("dark_frames", "image", "many"),
            RecipeInputSlot(
                "flat_frames",
                "image",
                "many",
                min_count=2,
                metadata=(RecipeMetadataRequirement("exposure", "Exposure (s)"),),
            ),
        ),
        parameter_class=FlatParameters,
        suggest_bindings=_suggest,
        check_inputs=_check,
    )


def _plugin_class(recipe: RecipeDescriptor) -> type[PluginBase]:
    """Create a plugin exposing the recipe and a generated example for it."""
    example = PluginExample(id="ramp", title="Ramp", recipe_ids=(recipe.recipe_id,))

    class LaunchPlugin(PluginBase):
        """Plugin relying on the generic Desktop recipe interaction."""

        PLUGIN_INFO = PluginInfo(
            id="org.example.launch",
            name="Launch application",
            version="1.0.0",
            capabilities=(PluginCapability.APPLICATION,),
        )
        RECIPES = (recipe,)
        EXAMPLES = (example,)

        @classmethod
        def materialize_example(cls, example_id: str) -> PluginExampleData | None:
            cls.get_example(example_id)
            return PluginExampleData(
                _ramp(levels=2), {recipe.recipe_id: {"minimum_levels": 2}}
            )

        def create_actions(self) -> None:
            """Create no actions for this integration test."""

    return LaunchPlugin


def _add_and_select(win, frames: list) -> None:
    """Add frames to the image panel and select them."""
    for frame in frames:
        win.imagepanel.add_object(frame)
    win.set_current_panel("image")
    win.imagepanel.objview.select_objects(frames)


def test_generic_interaction_assigns_checks_and_runs(
    monkeypatch: pytest.MonkeyPatch,
    recipe: RecipeDescriptor,
    received: list[tuple],
) -> None:
    """Selections are assessed, completed by the user, checked, then run."""
    with datalab_test_app_context(console=False) as win:
        plugin_class = _plugin_class(recipe)
        plugin = plugin_class()
        plugin.register(win)
        messages: list[tuple[str, str]] = []
        monkeypatch.setattr(plugin, "show_info", lambda m: messages.append(("info", m)))
        monkeypatch.setattr(plugin, "show_error", lambda m: messages.append(("err", m)))
        monkeypatch.setattr(FlatParameters, "edit", lambda *_args, **_kwargs: True)
        shown_dialogs: list[RecipeInputDialog] = []

        def refuse_dialog(dialog: RecipeInputDialog) -> bool:
            shown_dialogs.append(dialog)
            return False

        monkeypatch.setattr(RecipeInputDialog, "exec", refuse_dialog)
        try:
            # Without selection, the user learns which inputs are expected
            assert plugin.assess_recipe(recipe.recipe_id).status is (
                RecipeReadinessStatus.NO_INPUT
            )
            assert plugin.start_recipe(recipe.recipe_id) is None
            assert messages[0][0] == "info"
            assert "Expected inputs" in messages[0][1]
            assert "exposure" in messages[0][1]

            # A complete, checked selection runs without any assignment dialog
            frames = _ramp(levels=3)
            _add_and_select(win, frames)
            assert plugin.assess_recipe(recipe.recipe_id).status is (
                RecipeReadinessStatus.READY
            )
            count = len(win.imagepanel)
            outcome = plugin.launch_recipe(recipe.recipe_id)
            assert isinstance(outcome, RecipeOutcome)
            assert shown_dialogs == []
            assert len(win.imagepanel) == count + 1
            inputs, parameters = received[-1]
            assert inputs["dark_frames"] == (frames[0],)
            assert inputs["flat_frames"] == tuple(frames[1:])
            assert parameters.minimum_levels == 3

            # Too few exposure levels: the dialog explains it and blocks the run
            win.imagepanel.objview.select_objects(frames[:5])
            readiness = plugin.assess_recipe(recipe.recipe_id)
            assert readiness.status is RecipeReadinessStatus.NOT_READY
            assert plugin.start_recipe(recipe.recipe_id) is None
            dialog = shown_dialogs[-1]
            assert not dialog.ok_button.isEnabled()
            assert "Flat frames need 3 exposure levels" in dialog.issues_label.text()
            assert len(received) == 1
        finally:
            plugin.unregister()
            if plugin_class in PluginRegistry.get_plugin_classes():
                PluginRegistry.get_plugin_classes().remove(plugin_class)


def test_generic_interaction_lets_the_user_assign_ambiguous_inputs(
    monkeypatch: pytest.MonkeyPatch,
    recipe: RecipeDescriptor,
    received: list[tuple],
) -> None:
    """Without role metadata, the user assigns frames in the input dialog."""
    with datalab_test_app_context(console=False) as win:
        plugin_class = _plugin_class(recipe)
        plugin = plugin_class()
        plugin.register(win)
        monkeypatch.setattr(FlatParameters, "edit", lambda *_args, **_kwargs: True)

        def assign(dialog: RecipeInputDialog) -> bool:
            assert not dialog.ok_button.isEnabled()
            for slot_id, rows in (
                ("dark_frames", [0]),
                ("flat_frames", [1, 2, 3, 4, 5, 6]),
            ):
                widget = dialog.lists[slot_id]
                for row in rows:
                    widget.item(row).setCheckState(QC.Qt.Checked)
            assert dialog.ok_button.isEnabled()
            return True

        monkeypatch.setattr(RecipeInputDialog, "exec", assign)
        try:
            frames = _ramp(levels=3, with_roles=False)
            _add_and_select(win, frames)
            assert plugin.assess_recipe(recipe.recipe_id).status is (
                RecipeReadinessStatus.NEEDS_ASSIGNMENT
            )
            assert isinstance(plugin.start_recipe(recipe.recipe_id), RecipeOutcome)
            inputs, _parameters = received[-1]
            assert inputs["dark_frames"] == (frames[0],)
            assert inputs["flat_frames"] == tuple(frames[1:])
        finally:
            plugin.unregister()
            if plugin_class in PluginRegistry.get_plugin_classes():
                PluginRegistry.get_plugin_classes().remove(plugin_class)


def test_trying_an_example_runs_its_recipe_with_example_values(
    monkeypatch: pytest.MonkeyPatch,
    recipe: RecipeDescriptor,
    received: list[tuple],
) -> None:
    """An example opens, then its recipe starts with the example's values."""
    with datalab_test_app_context(console=False) as win:
        plugin_class = _plugin_class(recipe)
        plugin = plugin_class()
        plugin.register(win)
        monkeypatch.setattr(FlatParameters, "edit", lambda *_args, **_kwargs: True)
        monkeypatch.setattr(plugin, "ask_yesno", lambda *_args, **_kwargs: True)
        try:
            with pytest.raises(ValueError, match="not designed for"):
                plugin.try_example("ramp", "org.example.launch:other")
            outcome = plugin.try_example("ramp", recipe.recipe_id)
            assert isinstance(outcome, RecipeOutcome)
            inputs, parameters = received[-1]
            # Two levels only: the run relies on the example's parameter values
            assert parameters.minimum_levels == 2
            assert len(inputs["flat_frames"]) == 4
            # Like standard processing, the run selects its last output
            selection = plugin.get_selected_objects()
            assert len(selection) == 1
            assert selection[0].title == "Mean flat"
        finally:
            plugin.unregister()
            if plugin_class in PluginRegistry.get_plugin_classes():
                PluginRegistry.get_plugin_classes().remove(plugin_class)
