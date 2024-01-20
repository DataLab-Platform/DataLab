# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
.. Coordinates Algorithms (see parent package :mod:`cdl.algorithms`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np


def circle_center_radius_to_diameter(
    xc: float, yc: float, r: float
) -> tuple[float, float, float, float]:
    """Convert circle center and radius to X diameter coordinates

    Args:
        xc: Circle center X coordinate
        yc: Circle center Y coordinate
        r: Circle radius

    Returns:
        tuple: Circle X diameter coordinates
    """
    return xc - r, yc, xc + r, yc


def ellipse_center_axes_angle_to_diameters(
    xc: float, yc: float, a: float, b: float, theta: float
) -> tuple[float, float, float, float, float, float, float, float]:
    """Convert ellipse center, axes and angle to X/Y diameters coordinates

    Args:
        xc: Ellipse center X coordinate
        yc: Ellipse center Y coordinate
        a: Ellipse greater axis
        b: Ellipse smaller axis
        theta: Ellipse angle

    Returns:
        Ellipse X/Y diameters coordinates
    """
    dxa, dya = a * np.cos(theta), a * np.sin(theta)
    dxb, dyb = b * np.sin(theta), b * np.cos(theta)
    x0, y0, x1, y1 = xc - dxa, yc - dya, xc + dxa, yc + dya
    x2, y2, x3, y3 = xc - dxb, yc - dyb, xc + dxb, yc + dyb
    return x0, y0, x1, y1, x2, y2, x3, y3
