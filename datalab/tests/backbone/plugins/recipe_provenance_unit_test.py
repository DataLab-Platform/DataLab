# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the description of the recipe run that produced an object."""

from __future__ import annotations

import guidata.dataset as gds
import numpy as np
import pytest

from datalab.plugins.recipe_provenance import (
    RECIPE_RUN_DESCRIPTION_VERSION,
    describe_recipe_run,
)
from datalab.plugins.recipes import RecipeDescriptor, RecipeOutcome

RUN = "11111111-1111-4111-8111-111111111111"
FRAME_1 = "22222222-2222-4222-8222-222222222221"
FRAME_2 = "22222222-2222-4222-8222-222222222222"
DARK = "33333333-3333-4333-8333-333333333333"
MAP = "44444444-4444-4444-8444-444444444444"
HOT = "55555555-5555-4555-8555-555555555555"


class DarkParameters(gds.DataSet):
    """Parameters of the test recipe."""

    threshold = gds.FloatItem("Hot pixel threshold", default=5.0)
    frames = gds.IntItem("Frames", default=2)
    weights = gds.FloatArrayItem("Weights", default=np.zeros(2))


def _parameters_json() -> str:
    parameters = DarkParameters()
    parameters.threshold = 7.5
    return gds.dataset_to_json(parameters)


def _payload(**changes: object) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "run_id": RUN,
        "plugin_id": "org.example.camera",
        "plugin_version": "1.2.0",
        "recipe_id": "org.example.camera:dark-current",
        "recipe_version": "1.0.0",
        "parameters_json": _parameters_json(),
        "input_uuids": {"frames": [FRAME_1, FRAME_2], "dark": [DARK]},
        "output_uuids": {"dark-map": MAP, "hot-pixels": HOT},
        "datalab_version": "1.4.1",
        "sigima_version": "1.4.0",
        "status": "completed",
        "started_at": "2026-05-01T10:00:00+00:00",
        "finished_at": "2026-05-01T10:00:02+00:00",
    }
    payload.update(changes)
    return payload


def _descriptor() -> RecipeDescriptor:
    return RecipeDescriptor(
        recipe_id="org.example.camera:dark-current",
        plugin_version="1.2.0",
        title="Dark current and hot pixels",
        version="1.0.0",
        parameter_class=DarkParameters,
        run=lambda _inputs, _parameters, _context: RecipeOutcome(),
    )


def test_describes_the_run_of_an_output() -> None:
    """An output is described with its role, inputs, siblings and parameters."""
    run = describe_recipe_run(MAP, _payload())
    assert run is not None
    assert run["version"] == RECIPE_RUN_DESCRIPTION_VERSION
    assert run["status"] == "ok"
    assert run["run_id"] == RUN
    assert run["recipe_id"] == "org.example.camera:dark-current"
    assert run["title"] is None
    assert run["role"] == "dark-map"
    assert run["inputs"] == [
        {"role": "frames", "position": 0, "object_uuid": FRAME_1},
        {"role": "frames", "position": 1, "object_uuid": FRAME_2},
        {"role": "dark", "position": 0, "object_uuid": DARK},
    ]
    assert run["outputs"] == [
        {"role": "dark-map", "object_uuid": MAP, "is_self": True},
        {"role": "hot-pixels", "object_uuid": HOT, "is_self": False},
    ]
    assert run["parameters"] == [
        {"name": "threshold", "label": None, "value": 7.5},
        {"name": "frames", "label": None, "value": 2},
        {
            "name": "weights",
            "label": None,
            "value": {"$ndarray": {"shape": [2], "dtype": "float64"}},
        },
    ]


@pytest.mark.parametrize(
    "payload",
    [None, "text", _payload(run_id="not-a-uuid"), _payload(parameters_json="{")],
)
def test_ignores_absent_or_unreadable_records(payload: object) -> None:
    """No description is given for an absent or malformed record."""
    assert describe_recipe_run(MAP, payload) is None


def test_ignores_objects_that_only_copied_the_record() -> None:
    """A copy of an output (another UUID) is not attributed to the recipe."""
    other = "66666666-6666-4666-8666-666666666666"
    assert describe_recipe_run(other, _payload()) is None


def test_reports_records_of_a_newer_schema() -> None:
    """A record written with an unknown schema is reported, not described."""
    run = describe_recipe_run(MAP, _payload(schema_version=2, extra="new"))
    assert run == {
        "version": RECIPE_RUN_DESCRIPTION_VERSION,
        "status": "unsupported",
        "schema_version": 2,
        "recipe_id": "org.example.camera:dark-current",
    }


def test_parameterless_recipe_has_no_parameter() -> None:
    """A parameterless recipe records JSON null: no parameter is listed."""
    run = describe_recipe_run(MAP, _payload(parameters_json="null"))
    assert run is not None and run["parameters"] == []


def test_resolver_gives_title_and_labels() -> None:
    """A loaded recipe gives its title and the labels of its parameters."""
    calls = []

    def resolve(plugin_id: str, recipe_id: str) -> RecipeDescriptor:
        calls.append((plugin_id, recipe_id))
        return _descriptor()

    run = describe_recipe_run(HOT, _payload(), resolve)
    assert calls == [("org.example.camera", "org.example.camera:dark-current")]
    assert run is not None
    assert run["title"] == "Dark current and hot pixels"
    assert run["role"] == "hot-pixels"
    assert [(p["name"], p["label"]) for p in run["parameters"]] == [
        ("threshold", "Hot pixel threshold"),
        ("frames", "Frames"),
        ("weights", "Weights"),
    ]


def test_resolver_errors_are_ignored() -> None:
    """When the recipe cannot be resolved, its identifiers are shown."""

    def resolve(_plugin_id: str, recipe_id: str) -> RecipeDescriptor:
        raise KeyError(recipe_id)

    run = describe_recipe_run(MAP, _payload(), resolve)
    assert run is not None
    assert run["title"] is None
    assert run["parameters"][0] == {"name": "threshold", "label": None, "value": 7.5}
