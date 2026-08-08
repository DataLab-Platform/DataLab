# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the headless plugin recipe contracts."""

from __future__ import annotations

import guidata.dataset as gds
import pytest
from sigima.objects import TableResult, create_signal

from datalab.plugins import PluginBase, PluginInfo, PluginRegistry
from datalab.recipes import (
    RecipeCancellationError,
    RecipeCardinality,
    RecipeDescriptor,
    RecipeDiagnostic,
    RecipeDiagnosticLevel,
    RecipeExecutionContext,
    RecipeInputSlot,
    RecipeObjectOutput,
    RecipeObjectType,
    RecipeOutcome,
    RecipeResultOutput,
)


class RecipeParameters(gds.DataSet):
    """Minimal parameters used to validate descriptor typing."""

    gain = gds.FloatItem("Gain", default=1.0)


def test_recipe_descriptor_normalizes_and_validates_contract() -> None:
    """Descriptors expose immutable typed slots and a namespaced identity."""

    def run_recipe(*_args, **_kwargs) -> RecipeOutcome:
        """Return an empty outcome for descriptor validation."""
        return RecipeOutcome()

    descriptor = RecipeDescriptor(
        recipe_id="org.example.camera:quick-check",
        title="Quick check",
        description="Run a minimal detector check.",
        version="1.2.0",
        inputs=[
            RecipeInputSlot(
                id="frames",
                object_type="image",
                cardinality="many",
            ),
            RecipeInputSlot(
                id="reference",
                object_type=RecipeObjectType.IMAGE,
                cardinality=RecipeCardinality.ONE,
                required=False,
            ),
        ],
        parameter_class=RecipeParameters,
        run=run_recipe,
    )

    assert descriptor.plugin_id == "org.example.camera"
    assert descriptor.local_id == "quick-check"
    assert isinstance(descriptor.inputs, tuple)
    assert descriptor.inputs[0].object_type is RecipeObjectType.IMAGE
    assert descriptor.inputs[0].cardinality is RecipeCardinality.MANY

    prerelease_descriptor = RecipeDescriptor(
        recipe_id="org.example.camera:preview",
        title="Preview",
        version="1.0.0rc1",
        run=run_recipe,
    )
    assert prerelease_descriptor.version == "1.0.0rc1"

    with pytest.raises(ValueError, match="namespaced"):
        RecipeDescriptor(
            recipe_id="quick-check",
            title="Quick check",
            version="1.0",
            run=run_recipe,
        )
    with pytest.raises(ValueError, match="Duplicate recipe input slot"):
        RecipeDescriptor(
            recipe_id="org.example.camera:duplicate-slots",
            title="Duplicate slots",
            version="1.0",
            inputs=(
                RecipeInputSlot("frames", "image", "one"),
                RecipeInputSlot("frames", "image", "many"),
            ),
            run=run_recipe,
        )
    with pytest.raises(TypeError, match="DataSet subclass"):
        RecipeDescriptor(
            recipe_id="org.example.camera:invalid-parameters",
            title="Invalid parameters",
            version="1.0",
            parameter_class=dict,
            run=run_recipe,
        )
    with pytest.raises(ValueError, match="plugin namespace"):
        RecipeDescriptor(
            recipe_id=":quick-check",
            title="Missing namespace",
            version="1.0",
            run=run_recipe,
        )
    with pytest.raises(ValueError, match="Invalid recipe version"):
        RecipeDescriptor(
            recipe_id="org.example.camera:invalid-version",
            title="Invalid version",
            version="not-a-version",
            run=run_recipe,
        )


def test_recipe_outcome_requires_named_objects_and_valid_anchors() -> None:
    """Scalar outputs reference named object outputs without GUI UUIDs."""
    signal = create_signal("Summary", x=[0.0, 1.0], y=[1.0, 2.0])
    table = TableResult(
        title="Summary metrics",
        headers=["Mean"],
        data=[[1.5]],
    )
    diagnostic = RecipeDiagnostic(
        level="warning",
        code="limited-sample-count",
        message="Only two samples were available.",
        details={"sample_count": 2},
    )

    outcome = RecipeOutcome(
        objects=[RecipeObjectOutput("summary", signal)],
        results=[RecipeResultOutput("metrics", table, anchor_id="summary")],
        diagnostics=[diagnostic],
    )

    assert isinstance(outcome.objects, tuple)
    assert isinstance(outcome.results, tuple)
    assert outcome.results[0].anchor_id == "summary"
    assert outcome.diagnostics[0].level is RecipeDiagnosticLevel.WARNING
    with pytest.raises(TypeError):
        outcome.diagnostics[0].details["sample_count"] = 3

    with pytest.raises(ValueError, match="unknown object output"):
        RecipeOutcome(
            objects=(RecipeObjectOutput("summary", signal),),
            results=(RecipeResultOutput("metrics", table, anchor_id="missing"),),
        )
    with pytest.raises(ValueError, match="Duplicate recipe object output"):
        RecipeOutcome(
            objects=(
                RecipeObjectOutput("summary", signal),
                RecipeObjectOutput("summary", signal),
            )
        )
    with pytest.raises(ValueError, match="Recipe diagnostic code"):
        RecipeDiagnostic(
            level="error",
            code="INVALID CODE",
            message="Invalid diagnostic code",
        )


def test_recipe_execution_context_reports_progress_and_cancellation() -> None:
    """Execution context stays independent from threading and UI technology."""
    progress_events: list[tuple[float, str | None]] = []
    cancelled = False
    context = RecipeExecutionContext(
        progress_callback=lambda progress, message: progress_events.append(
            (progress, message)
        ),
        cancellation_callback=lambda: cancelled,
    )

    context.report_progress(0.25, "Loading inputs")
    assert progress_events == [(0.25, "Loading inputs")]
    assert not context.is_cancelled

    cancelled = True
    assert context.is_cancelled
    with pytest.raises(RecipeCancellationError):
        context.raise_if_cancelled()
    for invalid_progress in (True, float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            context.report_progress(invalid_progress)


def test_plugin_exposes_only_owned_unique_recipes() -> None:
    """Plugin recipe declarations stay namespaced by their stable plugin ID."""

    def run_recipe(*_args, **_kwargs) -> RecipeOutcome:
        """Return an empty recipe outcome."""
        return RecipeOutcome()

    recipe = RecipeDescriptor(
        recipe_id="org.example.recipes:quick-check",
        title="Quick check",
        version="1.0",
        run=run_recipe,
    )

    class RecipePlugin(PluginBase):
        """Plugin exposing one headless recipe."""

        PLUGIN_INFO = PluginInfo(
            id="org.example.recipes",
            name="Recipe plugin",
        )
        RECIPES = (recipe,)

        def create_actions(self) -> None:
            """Create no actions for this contract test."""

    try:
        assert RecipePlugin.get_recipes() == (recipe,)

        RecipePlugin.RECIPES = (recipe, recipe)
        with pytest.raises(ValueError, match="Duplicate plugin recipe ID"):
            RecipePlugin.get_recipes()

        RecipePlugin.RECIPES = ("not-a-recipe",)
        with pytest.raises(TypeError, match="RecipeDescriptor"):
            RecipePlugin.get_recipes()

        RecipePlugin.RECIPES = (
            RecipeDescriptor(
                recipe_id="org.example.other:quick-check",
                title="Wrong owner",
                version="1.0",
                run=run_recipe,
            ),
        )
        with pytest.raises(ValueError, match="not owned by plugin"):
            RecipePlugin.get_recipes()
    finally:
        PluginRegistry.get_plugin_classes().remove(RecipePlugin)
