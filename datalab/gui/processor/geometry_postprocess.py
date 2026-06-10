# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Geometry post-processing for image geometric transformations
============================================================

This module provides post-processing of geometry metadata (e.g. detected shapes,
analysis results) after geometric image transformations (flip, rotate, transpose).

It is intentionally kept in a separate module with no dependency on
:mod:`datalab.gui.processor.base` or any Qt/PlotPy module, so that the
:class:`GeometricTransformWrapper` class can be safely pickled and unpickled in
multiprocessing worker processes without triggering the DataLab GUI import chain.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import importlib
from typing import Literal

import numpy as np
import sigima.proc.image as sipi
from sigima.objects import ImageObj
from sigima.proc.decorator import ComputationMetadata

# Import GeometryAdapter directly from its module (not from the package __init__)
# to avoid pulling in datalab.adapters_metadata.common -> datalab.config -> PlotPy/Qt
from datalab.adapters_metadata.geometry_adapter import GeometryAdapter


def apply_geometry_transform(
    obj: ImageObj,
    operation: Literal[
        "translate", "scale", "rotate90", "rotate270", "fliph", "flipv", "transpose"
    ],
    **kwargs,
) -> None:
    """Apply a geometric transformation to all geometry results in an object.

    This uses the Sigima transformation system for proper geometric operations.
    For image objects, rotations are performed around the image center to match
    how the image data is transformed.

    Args:
        obj: The object containing geometry results to transform
        operation: The transformation operation name
        **kwargs: Optional parameters for the transformation (e.g., angle for rotate)
    """
    assert operation in [
        "translate",
        "scale",
        "rotate90",
        "rotate270",
        "fliph",
        "flipv",
        "transpose",
    ], f"Unknown operation: {operation}"
    if operation == "translate":
        if not kwargs or "dx" not in kwargs or "dy" not in kwargs:
            raise ValueError("translate operation requires 'dx' and 'dy' parameters")
        dx, dy = kwargs["dx"], kwargs["dy"]
    elif operation == "scale":
        if not kwargs or "sx" not in kwargs or "sy" not in kwargs:
            raise ValueError("scale operation requires 'sx' and 'sy' parameters")
        sx, sy = kwargs["sx"], kwargs["sy"]
    for adapter in list(GeometryAdapter.iterate_from_obj(obj)):
        geom = adapter.result
        assert geom is not None, "Geometry should not be None"
        assert len(geom.coords) > 0, "Geometry coordinates should not be empty"
        if operation == "translate":
            tr_geom = sipi.transformer.translate(geom, dx, dy)
        elif operation == "scale":
            tr_geom = sipi.transformer.scale(geom, sx, sy, (obj.xc, obj.yc))
        elif operation == "rotate90":
            tr_geom = sipi.transformer.rotate(geom, -np.pi / 2, (obj.xc, obj.yc))
        elif operation == "rotate270":
            tr_geom = sipi.transformer.rotate(geom, np.pi / 2, (obj.xc, obj.yc))
        elif operation == "fliph":
            tr_geom = sipi.transformer.fliph(geom, obj.xc)
        elif operation == "flipv":
            tr_geom = sipi.transformer.flipv(geom, obj.yc)
        elif operation == "transpose":
            tr_geom = sipi.transformer.transpose(geom)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Remove the old geometry and add the transformed one
        adapter.remove_from(obj)
        tr_adapter = GeometryAdapter(tr_geom)
        tr_adapter.add_to(obj)


class GeometricTransformWrapper:
    """Pickleable wrapper for geometric transformation functions.

    This class creates a callable wrapper that can be pickled, unlike nested functions.
    Instead of storing the function directly, it stores the module path and function
    name to allow proper pickling.
    """

    def __init__(self, func, operation: str):
        self.operation = operation

        # Store function reference for execution
        self.func = func

        # Store function module and name for pickling
        self.__module__ = func.__module__
        self.__qualname__ = func.__qualname__
        self.__annotations__ = getattr(func, "__annotations__", {})
        self.__name__ = getattr(func, "__name__", str(func))

        # Copy the __wrapped__ attribute if it exists (for Sigima compatibility)
        # Note: We don't copy __wrapped__ as it may contain unpickleable references
        # The wrapper functionality will still work without it

        # Copy Sigima computation metadata (required for validation)
        computation_metadata_attr = "__computation_function_metadata"
        if hasattr(func, computation_metadata_attr):
            setattr(
                self,
                computation_metadata_attr,
                getattr(func, computation_metadata_attr),
            )

    def __call__(self, src_obj, param=None):
        """Call the wrapped function and apply geometry transformations."""
        # Call the original function
        if param is not None:
            dst_obj = self.func(src_obj, param)
        else:
            dst_obj = self.func(src_obj)
        apply_geometry_transform(dst_obj, operation=self.operation)
        return dst_obj

    def __getstate__(self):
        """Custom pickling: exclude the function reference."""
        # Build state manually to avoid any problematic attributes
        state = {
            "operation": self.operation,
            "__module__": self.__module__,
            "__qualname__": self.__qualname__,
            "__annotations__": self.__annotations__,
            "__name__": self.__name__,
        }

        # Store function information for reconstruction
        if hasattr(self, "func"):
            state["_func_module"] = self.func.__module__
            state["_func_name"] = self.func.__name__

        # Note: We don't copy __wrapped__ as it may contain unpickleable references

        # Copy computation metadata safely
        computation_metadata_attr = "__computation_function_metadata"
        if hasattr(self, computation_metadata_attr):
            metadata = getattr(self, computation_metadata_attr)
            # Store as a dict to avoid any pickling issues with the object itself
            if hasattr(metadata, "__dict__"):
                state[computation_metadata_attr] = metadata.__dict__.copy()
            else:
                state[computation_metadata_attr] = metadata

        return state

    def __setstate__(self, state):
        """Custom unpickling: restore the function reference."""
        self.__dict__.update(state)
        # Restore function from module and name
        if "_func_module" in state and "_func_name" in state:
            module = importlib.import_module(state["_func_module"])
            self.func = getattr(module, state["_func_name"])

        # Reconstruct computation metadata if it was stored as dict
        computation_metadata_attr = "__computation_function_metadata"
        if computation_metadata_attr in state:
            metadata_dict = state[computation_metadata_attr]
            if isinstance(metadata_dict, dict):
                metadata = ComputationMetadata(**metadata_dict)
                setattr(self, computation_metadata_attr, metadata)
