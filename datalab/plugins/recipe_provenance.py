# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Host-independent description of the recipe run that produced an object.

Every output of a plugin recipe holds the record of its run (see
:class:`~datalab.plugins.recipes.RecipeRunRecord`) in its metadata. When the
workspace provenance records no production for such an object, DataLab Desktop
and DataLab-Web show this record instead: both share this module and render its
description in their own user interface.

The record is local to the object: its data cannot be checked or recomputed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any, Optional

from datalab.plugins.recipes import (
    RECIPE_RUN_RECORD_SCHEMA_VERSION,
    RecipeDescriptor,
    RecipeRunRecord,
)

__all__ = [
    "RECIPE_RUN_DESCRIPTION_VERSION",
    "RecipeResolver",
    "describe_recipe_run",
]

#: Version of the description returned by :func:`describe_recipe_run`
RECIPE_RUN_DESCRIPTION_VERSION = 1

#: Return the descriptor of a loaded recipe from its plugin and recipe IDs
RecipeResolver = Callable[[str, str], Optional[RecipeDescriptor]]

_CLASS_KEYS = ("class_module", "class_name")

_LOGGER = logging.getLogger(__name__)


def _array_shape(data: Any) -> list[int]:
    """Return the shape of an array encoded as nested lists."""
    shape = []
    while isinstance(data, list):
        shape.append(len(data))
        data = data[0] if data else None
    return shape


def _parameter_value(value: Any) -> Any:
    """Return a parameter value, arrays being replaced by their type and shape."""
    if (
        isinstance(value, list)
        and len(value) == 3
        and value[0] == "array"
        and isinstance(value[2], str)
    ):
        return {"$ndarray": {"shape": _array_shape(value[1]), "dtype": value[2]}}
    return value


def _parameter_labels(descriptor: RecipeDescriptor | None) -> dict[str, str]:
    """Return the labels of the parameters of a loaded recipe, by item name."""
    if descriptor is None or descriptor.parameter_class is None:
        return {}
    instance = descriptor.parameter_class()
    labels = {}
    for item in instance.get_items():
        label = item.get_prop_value("display", instance, "label")
        if isinstance(label, str) and label:
            labels[item.get_name()] = label
    return labels


def _parameters(
    parameters_json: str, labels: Mapping[str, str]
) -> list[dict[str, Any]] | None:
    """Return the recorded parameters, in their order, or None if unreadable."""
    try:
        values = json.loads(parameters_json)
    except (TypeError, ValueError):
        return None
    if values is None:
        return []
    if not isinstance(values, dict):
        return None
    return [
        {"name": name, "label": labels.get(name), "value": _parameter_value(value)}
        for name, value in values.items()
        if name not in _CLASS_KEYS
    ]


def describe_recipe_run(
    object_uuid: str,
    payload: Any,
    resolve: RecipeResolver | None = None,
) -> dict[str, Any] | None:
    """Return the description of the recipe run that produced an object.

    Args:
        object_uuid: UUID of the object.
        payload: Recipe run record held by the object (metadata option
         ``RECIPE_RUN_RECORD_OPTION``), or None.
        resolve: Optional function returning the descriptor of a loaded recipe
         (title and labels of the parameters); its errors are ignored.

    Returns:
        None if the object holds no readable record of a run that produced it
        (e.g. a copy of an output); a description with ``status`` set to
        ``"unsupported"`` if the record was written with an unknown schema;
        otherwise, the description of the run (``status`` set to ``"ok"``).
    """
    if not isinstance(payload, Mapping):
        return None
    schema_version = payload.get("schema_version")
    if schema_version != RECIPE_RUN_RECORD_SCHEMA_VERSION:
        if isinstance(schema_version, int) and not isinstance(schema_version, bool):
            recipe_id = payload.get("recipe_id")
            return {
                "version": RECIPE_RUN_DESCRIPTION_VERSION,
                "status": "unsupported",
                "schema_version": schema_version,
                "recipe_id": recipe_id if isinstance(recipe_id, str) else None,
            }
        return None
    try:
        record = RecipeRunRecord.from_dict(payload)
    except (TypeError, ValueError) as exc:
        _LOGGER.debug("Unreadable recipe run record: %s", exc)
        return None
    role = next(
        (out_id for out_id, uid in record.output_uuids.items() if uid == object_uuid),
        None,
    )
    if role is None:
        return None
    descriptor = None
    if resolve is not None:
        try:
            descriptor = resolve(record.plugin_id, record.recipe_id)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("Recipe %s not resolved: %s", record.recipe_id, exc)
    try:
        labels = _parameter_labels(descriptor)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("Parameters of %s not resolved: %s", record.recipe_id, exc)
        labels = {}
    return {
        "version": RECIPE_RUN_DESCRIPTION_VERSION,
        "status": "ok",
        "run_id": record.run_id,
        "plugin_id": record.plugin_id,
        "plugin_version": record.plugin_version,
        "recipe_id": record.recipe_id,
        "recipe_version": record.recipe_version,
        "title": None if descriptor is None else descriptor.title,
        "role": role,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "datalab_version": record.datalab_version,
        "sigima_version": record.sigima_version,
        "parameters": _parameters(record.parameters_json, labels),
        "inputs": [
            {"role": slot_id, "position": position, "object_uuid": uid}
            for slot_id, uuids in record.input_uuids.items()
            for position, uid in enumerate(uuids)
        ],
        "outputs": [
            {"role": out_id, "object_uuid": uid, "is_self": uid == object_uuid}
            for out_id, uid in record.output_uuids.items()
        ],
    }
