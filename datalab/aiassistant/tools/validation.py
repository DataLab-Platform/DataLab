# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Validation of tool arguments against their JSON schema.

Tool schemas only use a small subset of JSON Schema: this module implements
this subset without any third-party dependency, so that the AI Assistant chat
and the MCP server validate arguments the same way:

- ``type`` (a name or a list of names, including ``null``),
- ``enum``,
- ``properties``, ``required`` and ``additionalProperties`` (boolean or schema),
- ``items``,
- ``minimum`` and ``maximum``.

Other keywords (``description``, ``default``...) are ignored.
"""

from __future__ import annotations

import math
from typing import Any


def _matches_type(value: Any, type_name: str) -> bool:
    """Return True if ``value`` matches the JSON type ``type_name``."""
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        return isinstance(value, float) and math.isfinite(value) and value.is_integer()
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, (list, tuple))
    return True  # Unknown type names are not checked


def _validate(value: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    """Validate ``value`` against ``schema``, appending errors to ``errors``."""
    if not isinstance(schema, dict):
        return
    type_spec = schema.get("type")
    if type_spec is not None:
        type_names = [type_spec] if isinstance(type_spec, str) else list(type_spec)
        if not any(_matches_type(value, name) for name in type_names):
            errors.append(
                f"{path}: expected {' or '.join(type_names)}, "
                f"got {type(value).__name__}"
            )
            return
    if "enum" in schema and value not in schema["enum"]:
        choices = ", ".join(repr(choice) for choice in schema["enum"])
        errors.append(f"{path}: {value!r} is not one of {choices}")
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value!r} is lower than {schema['minimum']!r}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value!r} is greater than {schema['maximum']!r}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required property {name!r}")
        additional = schema.get("additionalProperties", True)
        for name, item in value.items():
            item_path = f"{path}.{name}"
            if name in properties:
                _validate(item, properties[name], item_path, errors)
            elif additional is False:
                valid = ", ".join(sorted(properties)) or "<none>"
                errors.append(
                    f"{path}: unexpected property {name!r} (valid properties: {valid})"
                )
            elif isinstance(additional, dict):
                _validate(item, additional, item_path, errors)
    if isinstance(value, (list, tuple)) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{index}]", errors)


def validate_arguments(schema: dict[str, Any], arguments: Any) -> list[str]:
    """Validate tool arguments against the JSON schema of the tool.

    Args:
        schema: JSON schema of the tool parameters.
        arguments: Arguments to validate.

    Returns:
        Human-readable errors. An empty list means the arguments are valid.
    """
    errors: list[str] = []
    _validate(arguments, schema, "arguments", errors)
    return errors
