# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Generic Desktop interaction running a plugin recipe."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Union

from sigima.objects import ImageObj, SignalObj

from datalab.config import _
from datalab.gui.plugins.recipe_inputs import RecipeInputDialog, format_inputs_html
from datalab.gui.plugins.recipe_runner import RecipeCommitError, RecipeRunner
from datalab.plugins.recipe_binding import (
    RecipeReadinessStatus,
    assess_recipe_inputs,
    create_recipe_parameters,
)
from datalab.plugins.recipes import (
    RecipeCancellationError,
    RecipeOutcome,
    RecipeValidationError,
)
from datalab.utils.qthelpers import qt_handle_error_message

if TYPE_CHECKING:
    from datalab.plugins import PluginBase

__all__ = ["RecipeLauncher"]

DataObject = Union[SignalObj, ImageObj]


class RecipeLauncher:
    """Assign inputs, check them, edit parameters, and run a plugin recipe.

    Args:
        plugin: registered plugin owning the recipe
    """

    def __init__(self, plugin: PluginBase) -> None:
        if plugin.main is None:
            raise RuntimeError("Plugin must be registered before starting a recipe")
        self.plugin = plugin
        self.main = plugin.main

    def start(
        self,
        recipe_id: str,
        objects: Sequence[DataObject] | None = None,
        parameter_values: Mapping[str, object] | None = None,
    ) -> RecipeOutcome | None:
        """Run a recipe on objects (default: current selection).

        Args:
            recipe_id: namespaced recipe ID
            objects: candidate objects (default: current selection)
            parameter_values: initial parameter values (default: values of the
             last opened example, when the objects come from it)

        Returns:
            Recipe outcome, or None if cancelled or failed
        """
        recipe = self.plugin.get_recipe(recipe_id)
        candidates = list(
            self.plugin.get_selected_objects() if objects is None else objects
        )
        if parameter_values is None:
            parameter_values = self.plugin.example_parameter_values(
                recipe_id, candidates
            )
        parameters = create_recipe_parameters(recipe, parameter_values)
        readiness = assess_recipe_inputs(recipe, candidates, parameters)
        if readiness.status is RecipeReadinessStatus.NO_INPUT:
            self.plugin.show_info(
                "<p>"
                + html.escape(
                    _("No selected object can be used by '%s'.") % recipe.title
                )
                + "</p><p>"
                + html.escape(_("Expected inputs:"))
                + "</p><p>"
                + format_inputs_html(recipe)
                + "</p>"
            )
            return None
        bindings = readiness.bindings
        if readiness.status is not RecipeReadinessStatus.READY:
            dialog = RecipeInputDialog(
                self.main,
                recipe,
                candidates,
                readiness,
                lambda edited: assess_recipe_inputs(
                    recipe, parameters=parameters, bindings=edited
                ),
            )
            if not dialog.exec():
                return None
            bindings = dialog.get_bindings()
        if parameters is not None and not parameters.edit(parent=self.main):
            return None
        inputs = {slot.id: tuple(bindings.get(slot.id, ())) for slot in recipe.inputs}
        try:
            return RecipeRunner(self.main).run(recipe, inputs, parameters)
        except (RecipeValidationError, RecipeCommitError) as exc:
            self.plugin.show_error(str(exc))
        except RecipeCancellationError:
            pass
        except Exception as exc:  # pylint: disable=broad-except
            # Recipe computations are third-party code: never crash DataLab
            qt_handle_error_message(self.main, exc, _("Running '%s'") % recipe.title)
        return None
