# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Geometry transformation utilities
=================================

This module provides utilities to handle geometry result transformations
that were removed from Sigima. It replaces the removed `transform_shapes`
method functionality.
"""

from __future__ import annotations

import numpy as np
from sigima.objects import ImageObj, SignalObj

from datalab.adapters_metadata import GeometryAdapter


def apply_geometry_transform(
    obj: SignalObj | ImageObj, operation: str, param_dict: dict | None = None
) -> None:
    """Apply a geometric transformation to all geometry results in an object.

    This replaces the removed `transform_shapes` method from Sigima.

    Args:
        obj: The object containing geometry results to transform
        operation: The transformation operation name
                  ("rotate_90", "rotate_270", "hflip", "vflip", "transpose", "rotate")
        param_dict: Optional parameters for the transformation (e.g., angle for rotate)
    """
    # Map operation names to transformation functions
    if operation == "rotate_90":
        transform_func = rotate_90_coords
    elif operation == "rotate_270":
        transform_func = rotate_270_coords
    elif operation == "hflip":
        transform_func = hflip_coords
    elif operation == "vflip":
        transform_func = vflip_coords
    elif operation == "transpose":
        transform_func = transpose_coords
    elif operation == "rotate":
        if param_dict and "angle" in param_dict:
            # For custom rotation with angle
            angle_rad = np.deg2rad(param_dict["angle"])

            def transform_func(coords, obj_ref):
                return rotate_coords(coords, obj_ref, angle_rad)
        else:
            raise ValueError("rotate operation requires 'angle' parameter")
    elif operation == "translate":
        if param_dict and "dx" in param_dict and "dy" in param_dict:
            dx, dy = param_dict["dx"], param_dict["dy"]

            def transform_func(coords, obj_ref):
                return translate_coords(coords, obj_ref, dx, dy)
        else:
            raise ValueError("translate operation requires 'dx' and 'dy' parameters")
    else:
        raise ValueError(f"'{operation}' is not a valid computation function.")

    # Get all geometry results from the object
    for adapter in GeometryAdapter.iterate_from_obj(obj):
        geometry = adapter.geometry
        if geometry.coords is not None and len(geometry.coords) > 0:
            # Apply transformation to coordinates
            transform_func(geometry.coords, obj)

            # Update the geometry result in the object
            adapter.add_to(obj)


def remove_all_geometry_results(obj: SignalObj | ImageObj) -> None:
    """Remove all geometry results from an object.

    This replaces the removed `remove_all_shapes` method from Sigima.

    Args:
        obj: The object to remove geometry results from
    """
    # Get all geometry result keys to remove
    keys_to_remove = []
    for key in obj.metadata.keys():
        if GeometryAdapter.match(key, obj.metadata[key]):
            keys_to_remove.append(key)

    # Remove geometry results and related metadata
    for key in keys_to_remove:
        base_key = key[: -len(GeometryAdapter.ARRAY_SUFFIX)]
        obj.metadata.pop(key, None)
        obj.metadata.pop(f"{base_key}{GeometryAdapter.TITLE_SUFFIX}", None)
        obj.metadata.pop(f"{base_key}{GeometryAdapter.SHAPE_SUFFIX}", None)
        obj.metadata.pop(f"{base_key}{GeometryAdapter.ADDLABEL_SUFFIX}", None)


# Coordinate transformation functions for specific image operations
def translate_coords(
    coords: np.ndarray, obj: ImageObj, delta_x: float, delta_y: float
) -> None:
    """Apply translation to coordinates.

    Args:
        coords: Coordinate array to modify in-place
        obj: Image object (for context)
        delta_x: Translation in X direction
        delta_y: Translation in Y direction
    """
    coords[:, ::2] += delta_x  # X coordinates
    coords[:, 1::2] += delta_y  # Y coordinates


def rotate_90_coords(coords: np.ndarray, obj: ImageObj) -> None:
    """Apply 90° rotation to coordinates.

    Args:
        coords: Coordinate array to modify in-place
        obj: Image object for dimensions
    """
    # 90° rotation: (x, y) -> (height - y, x)
    height = obj.height
    x_coords = coords[:, ::2].copy()
    y_coords = coords[:, 1::2].copy()
    coords[:, ::2] = height - y_coords  # New X = height - old Y
    coords[:, 1::2] = x_coords  # New Y = old X


def rotate_180_coords(coords: np.ndarray, obj: ImageObj) -> None:
    """Apply 180° rotation to coordinates.

    Args:
        coords: Coordinate array to modify in-place
        obj: Image object for dimensions
    """
    # 180° rotation: (x, y) -> (width - x, height - y)
    width, height = obj.width, obj.height
    coords[:, ::2] = width - coords[:, ::2]  # New X = width - old X
    coords[:, 1::2] = height - coords[:, 1::2]  # New Y = height - old Y


def rotate_270_coords(coords: np.ndarray, obj: ImageObj) -> None:
    """Apply 270° rotation to coordinates.

    Args:
        coords: Coordinate array to modify in-place
        obj: Image object for dimensions
    """
    # 270° rotation: (x, y) -> (y, width - x)
    width = obj.width
    x_coords = coords[:, ::2].copy()
    y_coords = coords[:, 1::2].copy()
    coords[:, ::2] = y_coords  # New X = old Y
    coords[:, 1::2] = width - x_coords  # New Y = width - old X


def hflip_coords(coords: np.ndarray, obj: ImageObj) -> None:
    """Apply horizontal flip to coordinates.

    Args:
        coords: Coordinate array to modify in-place
        obj: Image object for dimensions
    """
    # Horizontal flip: (x, y) -> (width - x, y)
    coords[:, ::2] = obj.width - coords[:, ::2]


def vflip_coords(coords: np.ndarray, obj: ImageObj) -> None:
    """Apply vertical flip to coordinates.

    Args:
        coords: Coordinate array to modify in-place
        obj: Image object for dimensions
    """
    # Vertical flip: (x, y) -> (x, height - y)
    coords[:, 1::2] = obj.height - coords[:, 1::2]


def transpose_coords(coords: np.ndarray, obj: ImageObj) -> None:
    """Apply transpose (diagonal flip) to coordinates.

    Args:
        coords: Coordinate array to modify in-place
        obj: Image object (for context)
    """
    # Transpose: (x, y) -> (y, x)
    x_coords = coords[:, ::2].copy()
    y_coords = coords[:, 1::2].copy()
    coords[:, ::2] = y_coords  # New X = old Y
    coords[:, 1::2] = x_coords  # New Y = old X


def rotate_coords(coords: np.ndarray, obj: ImageObj, angle_rad: float) -> None:
    """Apply custom rotation to coordinates around image center.

    Args:
        coords: Coordinate array to modify in-place
        obj: Image object for dimensions and center calculation
        angle_rad: Rotation angle in radians
    """
    # Get image center
    cx, cy = obj.width / 2, obj.height / 2

    # Apply rotation around center
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Extract coordinates
    x_coords = coords[:, ::2]
    y_coords = coords[:, 1::2]

    # Translate to origin, rotate, translate back
    x_centered = x_coords - cx
    y_centered = y_coords - cy

    x_rotated = x_centered * cos_a - y_centered * sin_a
    y_rotated = x_centered * sin_a + y_centered * cos_a

    coords[:, ::2] = x_rotated + cx
    coords[:, 1::2] = y_rotated + cy
