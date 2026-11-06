# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Safe decoding of parameter sets stored in workspaces

DataLab stores guidata parameter sets in workspaces as JSON documents naming
their class (``class_module`` and ``class_name``). guidata's permissive decoder
imports the named module and calls the named object, so a workspace received
from someone else could choose code to run. :func:`decode_dataset_json` only
builds :class:`guidata.dataset.DataSet` subclasses that DataLab already knows.
"""

from __future__ import annotations

import importlib
import json
import sys

import guidata.dataset as gds
from guidata.io.jsonfmt import JSONReader

#: Packages whose modules may be imported to find a stored parameter class
TRUSTED_PACKAGES = ("datalab", "guidata", "sigima", "sigimax")


class ParameterClassError(ValueError):
    """Stored parameters name a class that may not be used"""


def _may_import(module_name: str) -> bool:
    """Return True if *module_name* may be imported to find a parameter class.

    Only public modules of DataLab's own packages qualify: private modules
    (e.g. ``__main__``) and test modules are never imported for a file.
    """
    parts = module_name.split(".")
    return (
        parts[0] in TRUSTED_PACKAGES
        and all(part.isidentifier() and not part.startswith("_") for part in parts)
        and "tests" not in parts
    )


def resolve_dataset_class(module_name: object, class_name: object) -> type:
    """Return the parameter class named by stored parameters.

    The class must be a :class:`guidata.dataset.DataSet` subclass, defined in a
    module already loaded (DataLab, Sigima, guidata, installed plugins) or in a
    public module of DataLab's own packages, which is then imported. Nothing else
    is imported or called.

    Args:
        module_name: Value of the ``class_module`` field
        class_name: Value of the ``class_name`` field

    Returns:
        The parameter class

    Raises:
        ParameterClassError: If the class may not be used
    """
    if not isinstance(module_name, str) or not isinstance(class_name, str):
        raise ParameterClassError("Stored parameters do not name their class")
    module = sys.modules.get(module_name)
    if module is None:
        if not _may_import(module_name):
            raise ParameterClassError(
                f"Stored parameters name module {module_name!r}, "
                "which DataLab does not use"
            )
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ParameterClassError(
                f"Stored parameters name missing module {module_name!r}"
            ) from exc
    cls = getattr(module, class_name, None) if class_name.isidentifier() else None
    if not (isinstance(cls, type) and issubclass(cls, gds.DataSet)):
        raise ParameterClassError(
            f"{module_name}.{class_name} is not a parameter set class"
        )
    return cls


def decode_dataset_json(json_str: str) -> gds.DataSet:
    """Decode parameters serialized by :func:`guidata.dataset.dataset_to_json`.

    The class is resolved by :func:`resolve_dataset_class`. Fields are then read
    as guidata does: unknown fields are ignored and missing fields keep their
    default values, so parameters saved by earlier versions stay readable.

    Args:
        json_str: JSON document stored with an object, a result or a History
         action

    Returns:
        The decoded parameter set

    Raises:
        ParameterClassError: If the stored class may not be used
        ValueError: If *json_str* is not valid JSON
    """
    payload = json.loads(json_str)
    if not isinstance(payload, dict):
        raise ParameterClassError("Stored parameters are not a JSON object")
    cls = resolve_dataset_class(payload.get("class_module"), payload.get("class_name"))
    param = cls()
    param.deserialize(JSONReader(json_str))
    return param
