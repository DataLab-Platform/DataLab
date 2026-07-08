# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
History core utilities: schema constants, JSON codec.
"""

from __future__ import annotations

import importlib
import json
import logging
import warnings
from copy import deepcopy
from typing import Any

import numpy as np
from guidata.dataset.conv import dataset_to_json, json_to_dataset
from guidata.dataset.datatypes import DataSet
from qtpy import QtCore as QC
from sigima.objects.base import BaseROI

from datalab.config import _

_logger = logging.getLogger(__name__)
_TRUSTED_ROI_MODULE_PREFIX = "sigima."

# Schema versions for persisted history sessions/actions. Both start at 1.
# Bump the relevant constant (and add the corresponding optional field
# handling in serialize/deserialize) when the on-disk layout evolves.
HISTORY_SCHEMA_VERSION = 1
HISTORY_ACTION_SCHEMA_VERSION = 1
# Keys used in the kwargs dict to mark DataSet payloads, so that the
# serialization layer can round-trip them as JSON strings instead of pickling
# arbitrary Python objects.
_DATASET_MARKER = "__dataset_json__"
_DATASET_LIST_MARKER = "__dataset_list_json__"
_ROI_MARKER = "__roi_json__"


def get_datetime_str() -> str:
    """Return current date and time as a string"""
    return QC.QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")


def _numpy_to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _numpy_to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_to_json_safe(i) for i in obj]
    return obj


def _encode_roi(roi: Any) -> str:
    """Encode a sigima ROI object to a JSON string via ``to_dict()``."""
    if not isinstance(roi, BaseROI):
        raise TypeError(f"Expected BaseROI instance, got {type(roi)!r}")
    roi_dict = _numpy_to_json_safe(roi.to_dict())
    # Store the concrete class so we can reconstruct on decode.
    payload = {
        "module": type(roi).__module__,
        "class": type(roi).__qualname__,
        "data": roi_dict,
    }
    return json.dumps(payload)


def _decode_roi(encoded: str) -> Any:
    """Decode a JSON string back to a sigima ROI object.

    Only classes from trusted ``sigima.`` modules that are actual
    :class:`sigima.objects.base.BaseROI` subclasses are allowed.

    Raises:
        ValueError: If the module is not a trusted sigima module or the
            resolved class is not a BaseROI subclass.
    """
    payload = json.loads(encoded)
    module_name = payload["module"]
    class_name = payload["class"]

    if not module_name.startswith(_TRUSTED_ROI_MODULE_PREFIX):
        raise ValueError(
            f"Untrusted ROI module {module_name!r}: "
            f"only modules under {_TRUSTED_ROI_MODULE_PREFIX!r} are allowed"
        )

    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)

    if not (isinstance(cls, type) and issubclass(cls, BaseROI)):
        raise ValueError(f"{module_name}.{class_name} is not a BaseROI subclass")

    return cls.from_dict(payload["data"])


def _encode_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Encode kwargs for HDF5 storage: replace ``DataSet``, ``list[DataSet]``,
    and sigima ROI values with marker dicts holding their JSON representation.

    All other values must already be HDF5-friendly primitives (str, int, float,
    bool, list/tuple of the same).

    Args:
        kwargs: Raw kwargs dict (may contain ``DataSet`` or ROI instances).

    Returns:
        A new dict with special values wrapped in marker dicts.
    """
    encoded: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, DataSet):
            encoded[key] = {_DATASET_MARKER: dataset_to_json(value)}
        elif isinstance(value, BaseROI):
            encoded[key] = {_ROI_MARKER: _encode_roi(value)}
        elif (
            isinstance(value, list)
            and value
            and all(isinstance(item, DataSet) for item in value)
        ):
            encoded[key] = {
                _DATASET_LIST_MARKER: [dataset_to_json(item) for item in value]
            }
        else:
            encoded[key] = value
    return encoded


def _decode_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`_encode_kwargs`."""
    decoded: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, dict) and _DATASET_MARKER in value:
            try:
                decoded[key] = json_to_dataset(value[_DATASET_MARKER])
            except (TypeError, ValueError, KeyError):
                warnings.warn(
                    _("Failed to deserialize history DataSet kwarg %r.") % key
                )
                decoded[key] = None
        elif isinstance(value, dict) and _ROI_MARKER in value:
            try:
                decoded[key] = _decode_roi(value[_ROI_MARKER])
            except Exception as exc:
                raise ValueError(
                    f"Failed to deserialize history ROI kwarg {key!r}: {exc}"
                ) from exc
        elif isinstance(value, dict) and _DATASET_LIST_MARKER in value:
            try:
                decoded[key] = [
                    json_to_dataset(item) for item in value[_DATASET_LIST_MARKER]
                ]
            except (TypeError, ValueError, KeyError):
                warnings.warn(
                    _("Failed to deserialize history DataSet-list kwarg %r.") % key
                )
                decoded[key] = []
        else:
            decoded[key] = value
    return decoded


def _copy_history_value(value: Any) -> Any:
    """Return an independent copy of a history-serializable value."""
    if callable(value):
        raise TypeError("History duplication does not support callable kwargs")
    if isinstance(value, DataSet):
        return json_to_dataset(dataset_to_json(value))
    if isinstance(value, BaseROI):
        return _decode_roi(_encode_roi(value))
    if isinstance(value, dict):
        return {key: _copy_history_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_history_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_history_value(item) for item in value)
    return deepcopy(value)
