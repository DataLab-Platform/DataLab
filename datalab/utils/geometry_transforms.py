# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Geometry transformation utilities
=================================

This module provides utilities to handle geometry result transformations
using the comprehensive Sigima transformation system.
"""

from __future__ import annotations

from sigima.objects import ImageObj, SignalObj
from sigima.proc import scalar

from datalab.adapters_metadata import GeometryAdapter


def apply_geometry_transform(
    obj: SignalObj | ImageObj, operation: str, param_dict: dict | None = None
) -> None:
    """Apply a geometric transformation to all geometry results in an object.

    This uses the Sigima transformation system for proper geometric operations.

    Args:
        obj: The object containing geometry results to transform
        operation: The transformation operation name
        param_dict: Optional parameters for the transformation (e.g., angle for rotate)
    """
    # Determine if this is an image or signal object
    is_image = isinstance(obj, ImageObj)

    # Select the appropriate mapping based on object type
    if is_image:
        operation_map = scalar.IMAGE_GEOMETRY_UPDATE_MAP
    else:
        operation_map = scalar.SIGNAL_GEOMETRY_UPDATE_MAP

    if operation not in operation_map:
        if operation in ["translate", "scale"]:
            # Handle operations not in the standard maps
            _apply_custom_operation(obj, operation, param_dict)
            return
        raise ValueError(f"'{operation}' is not a valid transformation operation.")

    # Get transformation function and parameter builder
    transform_func, param_builder = operation_map[operation]

    # Transform all geometry results in the object
    for adapter in GeometryAdapter.iterate_from_obj(obj):
        geometry = adapter.geometry
        if geometry.coords is not None and len(geometry.coords) > 0:
            # Build parameters for the transformation
            if param_dict:
                # Create a parameter object if needed
                if operation == "rotate" and "angle" in param_dict:
                    import sigima.proc.image as sigima_image

                    param_obj = sigima_image.RotateParam()
                    param_obj.angle = param_dict["angle"]
                    args = param_builder(geometry, param_obj)
                else:
                    args = param_builder(geometry)
            else:
                args = param_builder(geometry)

            # Apply the transformation
            transformed_geometry = transform_func(*args)

            # Update the geometry result in the object
            adapter.geometry = transformed_geometry
            adapter.add_to(obj)


def _apply_custom_operation(
    obj: SignalObj | ImageObj, operation: str, param_dict: dict | None = None
) -> None:
    """Apply custom operations not in the standard Sigima maps."""
    if operation == "translate":
        if not param_dict or "dx" not in param_dict:
            raise ValueError("translate operation requires 'dx' parameter")
        dx = param_dict["dx"]
        dy = param_dict.get("dy", 0.0)  # Default to 0 for signals

        for adapter in GeometryAdapter.iterate_from_obj(obj):
            geometry = adapter.geometry
            if geometry.coords is not None and len(geometry.coords) > 0:
                if isinstance(obj, ImageObj):
                    transformed_geometry = scalar.translate(geometry, dx, dy)
                else:
                    # For signals, only translate in X direction
                    transformed_geometry = scalar.translate_1d(geometry, dx)

                adapter.geometry = transformed_geometry
                adapter.add_to(obj)

    elif operation == "scale":
        if not param_dict:
            raise ValueError("scale operation requires parameters")

        for adapter in GeometryAdapter.iterate_from_obj(obj):
            geometry = adapter.geometry
            if geometry.coords is not None and len(geometry.coords) > 0:
                if isinstance(obj, ImageObj):
                    sx = param_dict.get("sx", 1.0)
                    sy = param_dict.get("sy", 1.0)
                    center = param_dict.get("center", None)
                    transformed_geometry = scalar.scale(geometry, sx, sy, center)
                else:
                    # For signals, only scale in X direction
                    factor = param_dict.get("factor", 1.0)
                    center_x = param_dict.get("center_x", None)
                    transformed_geometry = scalar.scale_1d(geometry, factor, center_x)

                adapter.geometry = transformed_geometry
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
