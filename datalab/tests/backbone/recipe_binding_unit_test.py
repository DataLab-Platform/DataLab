# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the host-independent recipe input binding and readiness."""

from __future__ import annotations

import guidata.dataset as gds
import numpy as np
import pytest
from sigima.objects import create_image, create_signal

from datalab.recipe_binding import (
    INPUT_CHECK_FAILED_CODE,
    RecipeInputIssueCode,
    RecipeReadinessStatus,
    assess_recipe_inputs,
    create_recipe_parameters,
    find_input_issues,
    propose_bindings,
)
from datalab.recipes import (
    RecipeDescriptor,
    RecipeDiagnostic,
    RecipeInputSlot,
    RecipeMetadataRequirement,
    RecipeOutcome,
    RecipeValidationError,
)


class ThresholdParameters(gds.DataSet):
    """Parameters whose value changes the input requirements."""

    minimum_levels = gds.IntItem("Minimum levels", default=2)


def _image(title: str, **metadata: object):
    """Create a small image carrying metadata."""
    image = create_image(title, np.zeros((2, 2)))
    image.metadata.update(metadata)
    return image


def _recipe(**kwargs) -> RecipeDescriptor:
    """Create a dark/flat recipe; keyword arguments override the descriptor."""
    values = {
        "recipe_id": "org.example.camera:flats",
        "plugin_version": "1.0.0",
        "title": "Flats",
        "version": "1.0.0",
        "run": lambda *_args: RecipeOutcome(),
        "inputs": (
            RecipeInputSlot("dark_frames", "image", "many"),
            RecipeInputSlot(
                "flat_frames",
                "image",
                "many",
                min_count=2,
                metadata=(
                    RecipeMetadataRequirement("exposure"),
                    RecipeMetadataRequirement("role", required=False),
                ),
            ),
        ),
    }
    values.update(kwargs)
    return RecipeDescriptor(**values)


def _suggest_by_role(candidates):
    """Assign images to slots from their role metadata."""
    return {
        "dark_frames": [
            obj for obj in candidates if obj.metadata.get("role") == "dark"
        ],
        "flat_frames": [
            obj for obj in candidates if obj.metadata.get("role") == "flat"
        ],
    }


def test_proposal_binds_unique_types_and_leaves_shared_types_to_the_user() -> None:
    """Only unambiguous assignments are made without suggestions."""
    images = [_image(f"Image {index}") for index in range(3)]
    signal = create_signal("Signal", [0.0, 1.0], [0.0, 1.0])

    single = _recipe(inputs=(RecipeInputSlot("frames", "image", "many"),))
    proposal = propose_bindings(single, [*images, signal, images[0]])
    assert proposal.bindings["frames"] == tuple(images)
    assert proposal.ambiguous_slots == ()

    one = _recipe(inputs=(RecipeInputSlot("frame", "image", "one"),))
    assert propose_bindings(one, images).ambiguous_slots == ("frame",)
    assert propose_bindings(one, images[:1]).bindings["frame"] == (images[0],)

    proposal = propose_bindings(_recipe(), images)
    assert proposal.ambiguous_slots == ("dark_frames", "flat_frames")
    assert all(objects == () for objects in proposal.bindings.values())


def test_proposal_validates_recipe_suggestions() -> None:
    """Suggestions name declared slots and objects among the candidates."""
    dark = _image("Dark", role="dark")
    flats = [_image(f"Flat {index}", role="flat", exposure=1.0) for index in (1, 2)]
    recipe = _recipe(suggest_bindings=_suggest_by_role)
    proposal = propose_bindings(recipe, [dark, *flats])
    assert proposal.bindings == {"dark_frames": (dark,), "flat_frames": tuple(flats)}
    assert proposal.ambiguous_slots == ()

    unknown = _recipe(suggest_bindings=lambda candidates: {"other": candidates})
    with pytest.raises(RecipeValidationError, match="unknown slot"):
        propose_bindings(unknown, [dark])
    outside = _recipe(suggest_bindings=lambda _candidates: {"dark_frames": [flats[0]]})
    with pytest.raises(RecipeValidationError, match="outside the candidates"):
        propose_bindings(outside, [dark])


def test_input_issues_follow_slot_declarations() -> None:
    """Counts, types, required metadata and duplicates are checked generically."""
    recipe = _recipe()
    dark = _image("Dark")
    flat = _image("Flat", exposure=1.0)
    unlabeled = _image("Unlabeled flat")
    signal = create_signal("Signal", [0.0, 1.0], [0.0, 1.0])

    issues = find_input_issues(recipe, {"dark_frames": (), "flat_frames": (flat,)})
    assert [(issue.code, issue.slot_id) for issue in issues] == [
        (RecipeInputIssueCode.MISSING, "dark_frames"),
        (RecipeInputIssueCode.TOO_FEW, "flat_frames"),
    ]
    assert issues[1].details == {"count": 1, "min_count": 2}

    issues = find_input_issues(
        recipe,
        {"dark_frames": (dark, signal), "flat_frames": (flat, unlabeled, dark)},
    )
    codes = [issue.code for issue in issues]
    assert codes == [
        RecipeInputIssueCode.WRONG_TYPE,
        RecipeInputIssueCode.DUPLICATE,
        RecipeInputIssueCode.MISSING_METADATA,
    ]
    assert issues[2].details["key"] == "exposure"
    assert issues[2].details["titles"] == ("Unlabeled flat", "Dark")

    one = _recipe(inputs=(RecipeInputSlot("frame", "image", "one"),))
    issues = find_input_issues(one, {"frame": (dark, flat)})
    assert [issue.code for issue in issues] == [RecipeInputIssueCode.TOO_MANY]
    issues = find_input_issues(_recipe(), {}, ambiguous_slots=("dark_frames",))
    assert [issue.code for issue in issues] == [
        RecipeInputIssueCode.AMBIGUOUS,
        RecipeInputIssueCode.MISSING,
    ]


def test_readiness_combines_generic_issues_and_recipe_checks() -> None:
    """Recipe checks run on complete inputs only, with the given parameters."""
    dark = _image("Dark", role="dark")
    flats = [_image(f"Flat {index}", role="flat", exposure=1.0) for index in (1, 2)]
    received: list[tuple] = []

    def check_inputs(inputs, parameters) -> list[RecipeDiagnostic]:
        received.append((inputs, parameters))
        levels = len({obj.metadata["exposure"] for obj in inputs["flat_frames"]})
        if levels < parameters.minimum_levels:
            return [RecipeDiagnostic("error", "too-few-levels", "Too few levels")]
        return [RecipeDiagnostic("warning", "short-ramp", "Short ramp")]

    recipe = _recipe(
        suggest_bindings=_suggest_by_role,
        check_inputs=check_inputs,
        parameter_class=ThresholdParameters,
    )
    signal = create_signal("Signal", [0.0, 1.0], [0.0, 1.0])
    assert (
        assess_recipe_inputs(recipe, [signal]).status is RecipeReadinessStatus.NO_INPUT
    )
    assert received == []

    unlabeled = [_image("Image A"), _image("Image B")]
    readiness = assess_recipe_inputs(_recipe(), unlabeled)
    assert readiness.status is RecipeReadinessStatus.NEEDS_ASSIGNMENT
    assert not readiness.can_run

    parameters = create_recipe_parameters(recipe)
    readiness = assess_recipe_inputs(recipe, [dark, *flats], parameters)
    assert readiness.status is RecipeReadinessStatus.NOT_READY
    assert [diagnostic.code for diagnostic in readiness.diagnostics] == [
        "too-few-levels"
    ]
    assert received[-1][1] is parameters

    parameters = create_recipe_parameters(recipe, {"minimum_levels": 1})
    readiness = assess_recipe_inputs(recipe, [dark, *flats], parameters)
    assert readiness.status is RecipeReadinessStatus.WARNINGS
    assert readiness.can_run

    readiness = assess_recipe_inputs(
        recipe, parameters=parameters, bindings={"dark_frames": [dark]}
    )
    assert readiness.status is RecipeReadinessStatus.NOT_READY
    assert [issue.code for issue in readiness.issues] == [RecipeInputIssueCode.MISSING]

    def failing_check(_inputs, _parameters):
        raise ValueError("Flat frames need an exposure ladder")

    failing = _recipe(suggest_bindings=_suggest_by_role, check_inputs=failing_check)
    readiness = assess_recipe_inputs(failing, [dark, *flats])
    assert readiness.status is RecipeReadinessStatus.NOT_READY
    assert readiness.diagnostics[0].code == INPUT_CHECK_FAILED_CODE
    assert readiness.diagnostics[0].message == "Flat frames need an exposure ladder"

    ready = _recipe(suggest_bindings=_suggest_by_role)
    assert assess_recipe_inputs(ready, [dark, *flats]).status is (
        RecipeReadinessStatus.READY
    )


def test_recipe_parameters_accept_only_declared_items() -> None:
    """Example values update declared parameters and reject unknown names."""
    recipe = _recipe(parameter_class=ThresholdParameters)
    assert create_recipe_parameters(recipe).minimum_levels == 2
    assert create_recipe_parameters(recipe, {"minimum_levels": 5}).minimum_levels == 5
    with pytest.raises(RecipeValidationError, match="Unknown parameter 'edit'"):
        create_recipe_parameters(recipe, {"edit": 1})
    assert create_recipe_parameters(_recipe()) is None
    with pytest.raises(RecipeValidationError, match="does not accept parameters"):
        create_recipe_parameters(_recipe(), {"minimum_levels": 1})
