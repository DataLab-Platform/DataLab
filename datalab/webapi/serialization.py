# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Web API Serialization
=====================

Binary serialization of DataLab objects using NumPy's NPZ format.

This module handles the data plane of the Web API, efficiently transferring
large numerical arrays between DataLab and clients.

Format Specification (V1)
-------------------------

The NPZ archive contains:

For SignalObj:
    - ``x.npy``: X coordinates (1D float64 array)
    - ``y.npy``: Y data (1D float64 array)
    - ``dx.npy``: X uncertainties (optional)
    - ``dy.npy``: Y uncertainties (optional)
    - ``metadata.json``: JSON-encoded metadata

For ImageObj:
    - ``data.npy``: Image data (2D array, preserves dtype)
    - ``metadata.json``: JSON-encoded metadata

Metadata JSON includes:
    - ``type``: "signal" or "image"
    - ``title``, ``xlabel``, ``ylabel``, etc.
    - ``x0``, ``y0``, ``dx``, ``dy`` (for images)
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import TYPE_CHECKING, Union

import numpy as np
from sigima.objects import ImageObj, SignalObj

if TYPE_CHECKING:
    DataObject = Union[SignalObj, ImageObj]


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


def _make_json_serializable(value):
    """Recursively convert a value to be JSON serializable.

    Handles numpy arrays, numpy scalars, nested dicts, and lists.

    Args:
        value: Any value that may contain numpy types.

    Returns:
        JSON-serializable value.
    """
    if isinstance(value, dict):
        return {k: _make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _serialize_obj_metadata(obj_metadata: dict) -> dict:
    """Convert object metadata to JSON-serializable format.

    Handles numpy arrays (from GeometryResult.coords, TableResult, etc.)
    by converting them to nested lists, and numpy scalar types (int64, float64)
    to Python native types.

    Args:
        obj_metadata: The object's metadata dictionary.

    Returns:
        JSON-serializable dictionary.
    """
    return _make_json_serializable(obj_metadata)


def _deserialize_obj_metadata(serialized: dict) -> dict:
    """Convert deserialized metadata back to original types.

    Converts nested lists back to numpy arrays where appropriate
    (for coords fields in geometry results).

    Args:
        serialized: The JSON-deserialized metadata.

    Returns:
        Metadata with numpy arrays restored.
    """
    result = {}
    for key, value in serialized.items():
        if isinstance(value, dict):
            # Recursively handle nested dicts
            result[key] = _deserialize_obj_metadata(value)
        elif isinstance(value, list) and key == "coords":
            # Convert coords back to numpy array
            result[key] = np.array(value)
        else:
            result[key] = value
    return result


def serialize_object_to_npz(obj: DataObject, *, compress: bool = True) -> bytes:
    """Serialize a SignalObj or ImageObj to NPZ format.

    Args:
        obj: The object to serialize.
        compress: If True (default), use ZIP deflate compression.
            Set to False for faster serialization at the cost of larger size.
            For incompressible data (random images), False can be 10-30x faster.

    Returns:
        Bytes containing the NPZ archive.

    Raises:
        TypeError: If object type is not supported.
    """
    buffer = io.BytesIO()

    # Detect object type
    obj_type = type(obj).__name__

    if obj_type == "SignalObj":
        _serialize_signal(obj, buffer, compress=compress)
    elif obj_type == "ImageObj":
        _serialize_image(obj, buffer, compress=compress)
    else:
        raise TypeError(f"Unsupported object type: {obj_type}")

    return buffer.getvalue()


def _serialize_signal(obj, buffer: io.BytesIO, *, compress: bool = True) -> None:
    """Serialize a SignalObj to NPZ format."""
    # Build metadata dict
    metadata = {
        "type": "signal",
        "title": getattr(obj, "title", None),
        "xlabel": getattr(obj, "xlabel", None),
        "ylabel": getattr(obj, "ylabel", None),
        "xunit": getattr(obj, "xunit", None),
        "yunit": getattr(obj, "yunit", None),
    }
    # Include object metadata (contains Geometry_*, Table_* results)
    obj_metadata = getattr(obj, "metadata", None)
    if obj_metadata:
        metadata["obj_metadata"] = _serialize_obj_metadata(obj_metadata)

    # Create zip archive
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(buffer, "w", compression) as zf:
        # Write arrays
        _write_array_to_zip(zf, "x.npy", obj.x)
        _write_array_to_zip(zf, "y.npy", obj.y)

        if obj.dx is not None:
            _write_array_to_zip(zf, "dx.npy", obj.dx)
        if obj.dy is not None:
            _write_array_to_zip(zf, "dy.npy", obj.dy)

        # Write metadata
        zf.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False))


def _serialize_image(obj, buffer: io.BytesIO, *, compress: bool = True) -> None:
    """Serialize an ImageObj to NPZ format."""
    # Build metadata dict - use _to_python_scalar for numeric values that may be
    # numpy scalars after HDF5 deserialization
    metadata = {
        "type": "image",
        "title": getattr(obj, "title", None),
        "xlabel": getattr(obj, "xlabel", None),
        "ylabel": getattr(obj, "ylabel", None),
        "zlabel": getattr(obj, "zlabel", None),
        "xunit": getattr(obj, "xunit", None),
        "yunit": getattr(obj, "yunit", None),
        "zunit": getattr(obj, "zunit", None),
        "x0": _to_python_scalar(getattr(obj, "x0", 0.0)),
        "y0": _to_python_scalar(getattr(obj, "y0", 0.0)),
        "dx": _to_python_scalar(getattr(obj, "dx", 1.0)),
        "dy": _to_python_scalar(getattr(obj, "dy", 1.0)),
    }
    # Include object metadata (contains Geometry_*, Table_* results)
    obj_metadata = getattr(obj, "metadata", None)
    if obj_metadata:
        metadata["obj_metadata"] = _serialize_obj_metadata(obj_metadata)

    # Create zip archive
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(buffer, "w", compression) as zf:
        _write_array_to_zip(zf, "data.npy", obj.data)
        zf.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False))


def _write_array_to_zip(zf: zipfile.ZipFile, name: str, arr: np.ndarray) -> None:
    """Write a numpy array to a zip file."""
    arr_buffer = io.BytesIO()
    np.save(arr_buffer, arr, allow_pickle=False)
    zf.writestr(name, arr_buffer.getvalue())


def deserialize_object_from_npz(data: bytes) -> DataObject:
    """Deserialize a SignalObj or ImageObj from NPZ format.

    Args:
        data: Bytes containing the NPZ archive.

    Returns:
        SignalObj or ImageObj.

    Raises:
        ValueError: If the archive format is invalid.
    """
    buffer = io.BytesIO(data)

    with zipfile.ZipFile(buffer, "r") as zf:
        # Read metadata
        if "metadata.json" not in zf.namelist():
            raise ValueError("Invalid NPZ format: missing metadata.json")

        metadata = json.loads(zf.read("metadata.json"))
        obj_type = metadata.get("type")

        if obj_type == "signal":
            return _deserialize_signal(zf, metadata)
        if obj_type == "image":
            return _deserialize_image(zf, metadata)

        raise ValueError(f"Unknown object type in NPZ: {obj_type}")


def _read_array_from_zip(zf: zipfile.ZipFile, name: str) -> np.ndarray | None:
    """Read a numpy array from a zip file."""
    if name not in zf.namelist():
        return None
    arr_buffer = io.BytesIO(zf.read(name))
    return np.load(arr_buffer, allow_pickle=False)


def _deserialize_signal(zf: zipfile.ZipFile, metadata: dict) -> DataObject:
    """Deserialize a SignalObj from NPZ archive."""
    x = _read_array_from_zip(zf, "x.npy")
    y = _read_array_from_zip(zf, "y.npy")
    dx = _read_array_from_zip(zf, "dx.npy")
    dy = _read_array_from_zip(zf, "dy.npy")

    if x is None or y is None:
        raise ValueError("Invalid signal NPZ: missing x.npy or y.npy")

    obj = SignalObj()
    obj.set_xydata(x, y, dx=dx, dy=dy)

    # Set metadata
    if metadata.get("title"):
        obj.title = metadata["title"]
    if metadata.get("xlabel"):
        obj.xlabel = metadata["xlabel"]
    if metadata.get("ylabel"):
        obj.ylabel = metadata["ylabel"]
    if metadata.get("xunit"):
        obj.xunit = metadata["xunit"]
    if metadata.get("yunit"):
        obj.yunit = metadata["yunit"]

    # Restore object metadata (Geometry_*, Table_* results)
    if metadata.get("obj_metadata"):
        obj.metadata = _deserialize_obj_metadata(metadata["obj_metadata"])

    return obj


def _deserialize_image(zf: zipfile.ZipFile, metadata: dict) -> DataObject:
    """Deserialize an ImageObj from NPZ archive."""
    data = _read_array_from_zip(zf, "data.npy")
    if data is None:
        raise ValueError("Invalid image NPZ: missing data.npy")

    obj = ImageObj()
    obj.data = data

    # Set metadata
    if metadata.get("title"):
        obj.title = metadata["title"]
    if metadata.get("xlabel"):
        obj.xlabel = metadata["xlabel"]
    if metadata.get("ylabel"):
        obj.ylabel = metadata["ylabel"]
    if metadata.get("zlabel"):
        obj.zlabel = metadata["zlabel"]
    if metadata.get("xunit"):
        obj.xunit = metadata["xunit"]
    if metadata.get("yunit"):
        obj.yunit = metadata["yunit"]
    if metadata.get("zunit"):
        obj.zunit = metadata["zunit"]

    # Set coordinate info
    obj.x0 = metadata.get("x0", 0.0)
    obj.y0 = metadata.get("y0", 0.0)
    obj.dx = metadata.get("dx", 1.0)
    obj.dy = metadata.get("dy", 1.0)

    # Restore object metadata (Geometry_*, Table_* results)
    if metadata.get("obj_metadata"):
        obj.metadata = _deserialize_obj_metadata(metadata["obj_metadata"])

    return obj


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
            "type": "signal",
            "shape": list(obj.y.shape),
            "dtype": str(obj.y.dtype),
            "title": getattr(obj, "title", None),
            "xlabel": getattr(obj, "xlabel", None),
            "ylabel": getattr(obj, "ylabel", None),
            "xunit": getattr(obj, "xunit", None),
            "yunit": getattr(obj, "yunit", None),
            "attributes": {},
        }
    if obj_type == "ImageObj":
        return {
            "name": name,
            "type": "image",
            "shape": list(obj.data.shape),
            "dtype": str(obj.data.dtype),
            "title": getattr(obj, "title", None),
            "xlabel": getattr(obj, "xlabel", None),
            "ylabel": getattr(obj, "ylabel", None),
            "zlabel": getattr(obj, "zlabel", None),
            "xunit": getattr(obj, "xunit", None),
            "yunit": getattr(obj, "yunit", None),
            "zunit": getattr(obj, "zunit", None),
            "attributes": {
                "x0": _to_python_scalar(getattr(obj, "x0", 0.0)),
                "y0": _to_python_scalar(getattr(obj, "y0", 0.0)),
                "dx": _to_python_scalar(getattr(obj, "dx", 1.0)),
                "dy": _to_python_scalar(getattr(obj, "dy", 1.0)),
            },
        }
    raise TypeError(f"Unsupported object type: {obj_type}")
