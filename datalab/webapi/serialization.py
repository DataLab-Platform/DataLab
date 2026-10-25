# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Web API Serialization
=====================

Binary serialization of DataLab objects using NumPy's NPZ format.

This module handles the data plane of the Web API, efficiently transferring
large numerical arrays between DataLab and clients. The NPZ format (signals and
images with their labels, units, uncertainties, uniform or non-uniform
coordinates and metadata) is implemented by Sigima
(:mod:`sigima.io.npz`, where it is described), so that DataLab and its clients
(e.g. DataLab-Kernel) share one implementation: :func:`serialize_object_to_npz`
and :func:`deserialize_object_from_npz` are those of :mod:`sigima.io`. The
metadata returned by the API for an object (:func:`object_to_metadata`) are
specific to DataLab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
from sigima.io import deserialize_object_from_npz, serialize_object_to_npz

from datalab.objectmodel import get_uuid

if TYPE_CHECKING:
    from sigima.objects import ImageObj, SignalObj

    DataObject = Union[SignalObj, ImageObj]

__all__ = [
    "deserialize_object_from_npz",
    "object_to_metadata",
    "serialize_object_to_npz",
]


def _to_python_scalar(value):
    """Convert numpy scalar types to Python native types for JSON serialization.

    Args:
        value: Any value, possibly a numpy scalar.

    Returns:
        Python native type if input was a numpy scalar, otherwise unchanged.
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def object_to_metadata(obj: DataObject, name: str) -> dict:
    """Extract metadata from an object for API responses.

    Args:
        obj: The object to extract metadata from.
        name: The object name in the workspace.

    Returns:
        Dictionary suitable for ObjectMetadata schema.
    """
    obj_type = type(obj).__name__

    if obj_type == "SignalObj":
        return {
            "name": name,
            "uuid": get_uuid(obj),
            "type": "signal",
            "shape": list(obj.y.shape),
            "dtype": str(obj.y.dtype),
            "title": obj.title,
            "xlabel": obj.xlabel,
            "ylabel": obj.ylabel,
            "xunit": obj.xunit,
            "yunit": obj.yunit,
            "attributes": {},
        }
    if obj_type == "ImageObj":
        return {
            "name": name,
            "uuid": get_uuid(obj),
            "type": "image",
            "shape": list(obj.data.shape),
            "dtype": str(obj.data.dtype),
            "title": obj.title,
            "xlabel": obj.xlabel,
            "ylabel": obj.ylabel,
            "zlabel": obj.zlabel,
            "xunit": obj.xunit,
            "yunit": obj.yunit,
            "zunit": obj.zunit,
            "attributes": {
                "x0": _to_python_scalar(obj.x0),
                "y0": _to_python_scalar(obj.y0),
                "dx": _to_python_scalar(obj.dx),
                "dy": _to_python_scalar(obj.dy),
            },
        }
    raise TypeError(f"Unsupported object type: {obj_type}")
