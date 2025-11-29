# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI Coordinate Utilities
=========================

This module provides utility functions for rounding ROI coordinates to appropriate
precision based on the sampling characteristics of signals and images.

These functions are used when converting interactive PlotPy shapes to ROI objects
to ensure coordinates are displayed with reasonable precision.
"""

from __future__ import annotations

import numpy as np
from sigima.objects import ImageObj, ROI1DParam, ROI2DParam, SignalObj


def round_signal_coords(
    obj: SignalObj, coords: list[float], precision_factor: float = 0.1
) -> list[float]:
    """Round signal coordinates to appropriate precision based on sampling period.

    Rounds to a fraction of the median sampling period to avoid excessive decimal
    places while maintaining reasonable precision.

    Args:
        obj: signal object
        coords: coordinates to round
        precision_factor: fraction of sampling period to use as rounding precision.
            Default is 0.1 (1/10th of sampling period).

    Returns:
        Rounded coordinates
    """
    if len(obj.x) < 2:
        # Cannot compute sampling period, return coords as-is
        return coords
    # Compute median sampling period
    sampling_period = float(np.median(np.diff(obj.x)))
    if sampling_period == 0:
        # Avoid division by zero for constant x arrays
        return coords
    # Round to specified fraction of sampling period
    precision = sampling_period * precision_factor
    # Determine number of decimal places
    if precision > 0:
        decimals = max(0, int(-np.floor(np.log10(precision))))
        return [round(c, decimals) for c in coords]
    return coords


def round_image_coords(
    obj: ImageObj, coords: list[float], precision_factor: float = 0.1
) -> list[float]:
    """Round image coordinates to appropriate precision based on pixel spacing.

    Rounds to a fraction of the pixel spacing to avoid excessive decimal places
    while maintaining reasonable precision. Uses separate precision for X and Y.

    Args:
        obj: image object
        coords: flat list of coordinates [x0, y0, x1, y1, ...] to round
        precision_factor: fraction of pixel spacing to use as rounding precision.
            Default is 0.1 (1/10th of pixel spacing).

    Returns:
        Rounded coordinates

    Raises:
        ValueError: if coords does not contain an even number of elements
    """
    if len(coords) % 2 != 0:
        raise ValueError("coords must contain an even number of elements (x, y pairs).")
    if len(coords) == 0:
        return coords

    rounded = list(coords)
    if obj.is_uniform_coords:
        # Use dx, dy for uniform coordinates
        precision_x = abs(obj.dx) * precision_factor
        precision_y = abs(obj.dy) * precision_factor
    else:
        # Compute average spacing for non-uniform coordinates
        if len(obj.xcoords) > 1:
            avg_dx = float(np.mean(np.abs(np.diff(obj.xcoords))))
            precision_x = avg_dx * precision_factor
        else:
            precision_x = 0
        if len(obj.ycoords) > 1:
            avg_dy = float(np.mean(np.abs(np.diff(obj.ycoords))))
            precision_y = avg_dy * precision_factor
        else:
            precision_y = 0

    # Round X coordinates (even indices)
    if precision_x > 0:
        decimals_x = max(0, int(-np.floor(np.log10(precision_x))))
        for i in range(0, len(rounded), 2):
            rounded[i] = round(rounded[i], decimals_x)

    # Round Y coordinates (odd indices)
    if precision_y > 0:
        decimals_y = max(0, int(-np.floor(np.log10(precision_y))))
        for i in range(1, len(rounded), 2):
            rounded[i] = round(rounded[i], decimals_y)

    return rounded


def round_signal_roi_param(
    obj: SignalObj, param: ROI1DParam, precision_factor: float = 0.1
) -> None:
    """Round signal ROI parameter coordinates in-place.

    Args:
        obj: signal object
        param: ROI parameter to round (modified in-place)
        precision_factor: fraction of sampling period to use as rounding precision
    """
    coords = round_signal_coords(obj, [param.xmin, param.xmax], precision_factor)
    param.xmin, param.xmax = coords


def round_image_roi_param(
    obj: ImageObj, param: ROI2DParam, precision_factor: float = 0.1
) -> None:
    """Round image ROI parameter coordinates in-place.

    Args:
        obj: image object
        param: ROI parameter to round (modified in-place)
        precision_factor: fraction of pixel spacing to use as rounding precision
    """
    if param.geometry == "rectangle":
        # Round x0, y0, dx, dy
        x0, y0, x1, y1 = param.x0, param.y0, param.x0 + param.dx, param.y0 + param.dy
        coords = round_image_coords(obj, [x0, y0, x1, y1], precision_factor)
        param.x0, param.y0 = coords[0], coords[1]
        # Round dx and dy to avoid floating-point errors in subtraction
        dx_dy_rounded = round_image_coords(
            obj, [coords[2] - coords[0], coords[3] - coords[1]], precision_factor
        )
        param.dx = dx_dy_rounded[0]
        param.dy = dx_dy_rounded[1]
    elif param.geometry == "circle":
        # Round xc, yc, r
        coords = round_image_coords(obj, [param.xc, param.yc], precision_factor)
        param.xc, param.yc = coords
        # Round radius using X precision
        r_rounded = round_image_coords(obj, [param.r, 0], precision_factor)[0]
        param.r = r_rounded
    elif param.geometry == "polygon":
        # Round polygon points
        rounded = round_image_coords(obj, param.points.tolist(), precision_factor)
        param.points = np.array(rounded)
