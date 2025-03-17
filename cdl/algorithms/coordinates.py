# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Coordinates Algorithms (see parent package :mod:`cdl.algorithms`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import Literal

import numpy as np


def circle_to_diameter(
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


def array_circle_to_diameter(data: np.ndarray) -> np.ndarray:
    """Convert circle center and radius to X diameter coordinates (array version)

    Args:
        data: Circle center and radius, in the form of a 2D array (N, 3)

    Returns:
        Circle X diameter coordinates, in the form of a 2D array (N, 4)
    """
    xc, yc, r = data[:, 0], data[:, 1], data[:, 2]
    x_start = xc - r
    x_end = xc + r
    result = np.column_stack((x_start, yc, x_end, yc)).astype(float)
    return result


def circle_to_center_radius(
    x0: float, y0: float, x1: float, y1: float
) -> tuple[float, float, float]:
    """Convert circle X diameter coordinates to center and radius

    Args:
        x0: Diameter start X coordinate
        y0: Diameter start Y coordinate
        x1: Diameter end X coordinate
        y1: Diameter end Y coordinate

    Returns:
        tuple: Circle center and radius
    """
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    r = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2
    return xc, yc, r


def array_circle_to_center_radius(data: np.ndarray) -> np.ndarray:
    """Convert circle X diameter coordinates to center and radius (array version)

    Args:
        data: Circle X diameter coordinates, in the form of a 2D array (N, 4)

    Returns:
        Circle center and radius, in the form of a 2D array (N, 3)
    """
    x0, y0, x1, y1 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    r = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2
    result = np.column_stack((xc, yc, r)).astype(float)
    return result


def ellipse_to_diameters(
    xc: float, yc: float, a: float, b: float, theta: float
) -> tuple[float, float, float, float, float, float, float, float]:
    """Convert ellipse center, axes and angle to X/Y diameters coordinates

    Args:
        xc: Ellipse center X coordinate
        yc: Ellipse center Y coordinate
        a: Ellipse half larger axis
        b: Ellipse half smaller axis
        theta: Ellipse angle

    Returns:
        Ellipse X/Y diameters (major/minor axes) coordinates
    """
    dxa, dya = a * np.cos(theta), a * np.sin(theta)
    dxb, dyb = b * np.sin(theta), b * np.cos(theta)
    x0, y0, x1, y1 = xc - dxa, yc - dya, xc + dxa, yc + dya
    x2, y2, x3, y3 = xc - dxb, yc - dyb, xc + dxb, yc + dyb
    return x0, y0, x1, y1, x2, y2, x3, y3


def array_ellipse_to_diameters(data: np.ndarray) -> np.ndarray:
    """Convert ellipse center, axes and angle to X/Y diameters coordinates
    (array version)

    Args:
        data: Ellipse center, axes and angle, in the form of a 2D array (N, 5)

    Returns:
        Ellipse X/Y diameters (major/minor axes) coordinates,
         in the form of a 2D array (N, 8)
    """
    xc, yc, a, b, theta = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    dxa, dya = a * np.cos(theta), a * np.sin(theta)
    dxb, dyb = b * np.sin(theta), b * np.cos(theta)
    x0, y0, x1, y1 = xc - dxa, yc - dya, xc + dxa, yc + dya
    x2, y2, x3, y3 = xc - dxb, yc - dyb, xc + dxb, yc + dyb
    result = np.column_stack((x0, y0, x1, y1, x2, y2, x3, y3)).astype(float)
    return result


def ellipse_to_center_axes_angle(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
) -> tuple[float, float, float, float, float]:
    """Convert ellipse X/Y diameters coordinates to center, axes and angle

    Args:
        x0: major axis start X coordinate
        y0: major axis start Y coordinate
        x1: major axis end X coordinate
        y1: major axis end Y coordinate
        x2: minor axis start X coordinate
        y2: minor axis start Y coordinate
        x3: minor axis end X coordinate
        y3: minor axis end Y coordinate

    Returns:
        Ellipse center, axes and angle
    """
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    a = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2
    b = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2) / 2
    theta = np.arctan2(y1 - y0, x1 - x0)
    return xc, yc, a, b, theta


def array_ellipse_to_center_axes_angle(data: np.ndarray) -> np.ndarray:
    """Convert ellipse X/Y diameters coordinates to center, axes and angle
    (array version)

    Args:
        data: Ellipse X/Y diameters coordinates, in the form of a 2D array (N, 8)

    Returns:
        Ellipse center, axes and angle, in the form of a 2D array (N, 5)
    """
    x0, y0, x1, y1, x2, y2, x3, y3 = (
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        data[:, 4],
        data[:, 5],
        data[:, 6],
        data[:, 7],
    )
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    a = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2
    b = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2) / 2
    theta = np.arctan2(y1 - y0, x1 - x0)
    result = np.column_stack((xc, yc, a, b, theta)).astype(float)
    return result


def cartesian2polar(
    x: np.ndarray, y: np.ndarray, unit: Literal["rad", "deg"] = "rad"
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to polar coordinates.

    Args:
        x: Cartesian x-coordinate.
        y: Cartesian y-coordinate.
        unit: Unit of the angle ('rad' or 'deg').

    Returns:
        Polar coordinates (r, theta) where r is the radius and theta is the angle.
    """
    assert x.shape == y.shape, "x and y must have the same shape"
    assert unit in ["rad", "deg"], "unit must be 'rad' or 'deg'"
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if unit == "deg":
        theta = np.rad2deg(theta)
    return r, theta


def polar2cartesian(
    r: np.ndarray, theta: np.ndarray, unit: Literal["rad", "deg"] = "rad"
) -> tuple[np.ndarray, np.ndarray]:
    """Convert polar coordinates to Cartesian coordinates.

    Args:
        r: Polar radius.
        theta: Polar angle.
        unit: Unit of the angle ('rad' or 'deg').

    Returns:
        Cartesian coordinates (x, y) where x is the x-coordinate and y is the
        y-coordinate.
    """
    assert r.shape == theta.shape, "r and theta must have the same shape"
    assert unit in ["rad", "deg"], "unit must be 'rad' or 'deg'"
    if unit == "deg":
        theta = np.deg2rad(theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y
