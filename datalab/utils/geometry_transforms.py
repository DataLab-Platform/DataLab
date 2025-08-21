# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Geometry transformation utilities
=================================

This module provides utilities to handle geometry result transformations
using the comprehensive Sigima transformation system.
"""

from __future__ import annotations

import numpy as np
from sigima.objects import ImageObj
from sigima.proc.transformations import transformer

from datalab.adapters_metadata import GeometryAdapter


def apply_geometry_transform(
    obj: ImageObj, operation: str, param_dict: dict | None = None
) -> None:
    """Apply a geometric transformation to all geometry results in an object.

    This uses the Sigima transformation system for proper geometric operations.
    For image objects, rotations are performed around the image center to match
    how the image data is transformed.

    Args:
        obj: The object containing geometry results to transform
        operation: The transformation operation name
        param_dict: Optional parameters for the transformation (e.g., angle for rotate)
    """
    for adapter in list(GeometryAdapter.iterate_from_obj(obj)):
        geometry = adapter.geometry
        assert geometry is not None, "Geometry should not be None"
        assert len(geometry.coords) > 0, "Geometry coordinates should not be empty"
        if operation == "translate":
            if not param_dict or "dx" not in param_dict or "dy" not in param_dict:
                raise ValueError(
                    "translate operation requires 'dx' and 'dy' parameters"
                )
            dx = param_dict["dx"]
            dy = param_dict["dy"]
            tr_geometry = transformer.translate(geometry, dx, dy)
        elif operation == "scale":
            if not param_dict or "sx" not in param_dict or "sy" not in param_dict:
                raise ValueError("scale operation requires 'sx' and 'sy' parameters")
            sx = param_dict["sx"]
            sy = param_dict["sy"]
            center = param_dict.get("center", (0.0, 0.0))  # Default to (0, 0)
            tr_geometry = transformer.scale(geometry, sx, sy, center)
        elif operation == "rotate90":
            tr_geometry = transformer.rotate(geometry, -np.pi / 2, (obj.xc, obj.yc))
        elif operation == "rotate270":
            tr_geometry = transformer.rotate(geometry, np.pi / 2, (obj.xc, obj.yc))
        elif operation == "fliph":
            tr_geometry = transformer.fliph(geometry, obj.xc)
        elif operation == "flipv":
            tr_geometry = transformer.flipv(geometry, obj.yc)
        elif operation == "transpose":
            tr_geometry = transformer.transpose(geometry)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Remove the old geometry and add the transformed one
        adapter.remove_from(obj)
        tr_adapter = GeometryAdapter(tr_geometry)
        tr_adapter.add_to(obj)
