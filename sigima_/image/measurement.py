# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Measurement computation module
------------------------------

This module provides tools for extracting quantitative information from images,
such as object centroids, enclosing circles, and region-based statistics.

Main features include:
- Centroid and enclosing circle computation
- Region/property measurements
- Statistical analysis of image regions

These functions are useful for image quantification and morphometric analysis.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# - All `guidata.dataset.DataSet` parameter classes must also be imported
#   in the `sigima_.param` module.
# - All functions decorated by `computation_function` must be imported in the upper
#   level `sigima_.image` module.

from __future__ import annotations

import numpy as np
from numpy import ma

import sigima_.algorithms.image as alg
from sigima_ import computation_function
from sigima_.base import calc_resultproperties
from sigima_.image.base import calc_resultshape
from sigima_.obj.base import ResultProperties, ResultShape
from sigima_.obj.image import ImageObj


def get_centroid_coords(data: np.ndarray) -> np.ndarray:
    """Return centroid coordinates
    with :py:func:`sigima_.algorithms.image.get_centroid_fourier`

    Args:
        data: input data

    Returns:
        Centroid coordinates
    """
    y, x = alg.get_centroid_fourier(data)
    return np.array([(x, y)])


@computation_function()
def centroid(image: ImageObj) -> ResultShape | None:
    """Compute centroid
    with :py:func:`sigima_.algorithms.image.get_centroid_fourier`

    Args:
        image: input image

    Returns:
        Centroid coordinates
    """
    return calc_resultshape("centroid", "marker", image, get_centroid_coords)


def get_enclosing_circle_coords(data: np.ndarray) -> np.ndarray:
    """Return diameter coords for the circle contour enclosing image
    values above threshold (FWHM)

    Args:
        data: input data

    Returns:
        Diameter coords
    """
    x, y, r = alg.get_enclosing_circle(data)
    return np.array([[x, y, r]])


@computation_function()
def enclosing_circle(image: ImageObj) -> ResultShape | None:
    """Compute minimum enclosing circle
    with :py:func:`sigima_.algorithms.image.get_enclosing_circle`

    Args:
        image: input image

    Returns:
        Diameter coords
    """
    return calc_resultshape(
        "enclosing_circle", "circle", image, get_enclosing_circle_coords
    )


def __calc_snr_without_warning(data: np.ndarray) -> float:
    """Calculate SNR based on <z>/σ(z), ignoring warnings

    Args:
        data: input data

    Returns:
        Signal-to-noise ratio
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = ma.mean(data) / ma.std(data)
    return snr


@computation_function()
def stats(obj: ImageObj) -> ResultProperties:
    """Compute statistics on an image

    Args:
        obj: input image object

    Returns:
        Result properties
    """
    statfuncs = {
        "min(z) = %g {.zunit}": ma.min,
        "max(z) = %g {.zunit}": ma.max,
        "<z> = %g {.zunit}": ma.mean,
        "median(z) = %g {.zunit}": ma.median,
        "σ(z) = %g {.zunit}": ma.std,
        "<z>/σ(z)": __calc_snr_without_warning,
        "peak-to-peak(z) = %g {.zunit}": ma.ptp,
        "Σ(z) = %g {.zunit}": ma.sum,
    }
    return calc_resultproperties("stats", obj, statfuncs)
