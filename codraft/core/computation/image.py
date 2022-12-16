# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT Computation / Image module
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import numpy as np
import scipy.ndimage as spi
import scipy.ndimage.filters as spf
import scipy.spatial as spt
from numpy import ma
from skimage import measure


def scale_data_to_min_max(data: np.ndarray, zmin, zmax):
    """Scale array `data` to fit [zmin, zmax] dynamic range"""
    dmin = data.min()
    dmax = data.max()
    fdata = np.array(data, dtype=float)
    fdata -= dmin
    fdata *= float(zmax - zmin) / (dmax - dmin)
    fdata += float(zmin)
    return np.array(fdata, data.dtype)


def flatfield(rawdata: np.ndarray, flatdata: np.ndarray, threshold: float = None):
    """Compute flat-field correction"""
    dtemp = np.array(rawdata, dtype=np.float64, copy=True) * flatdata.mean()
    dunif = np.array(flatdata, dtype=np.float64, copy=True)
    dunif[dunif == 0] = 1.0
    dcorr_all = np.array(dtemp / dunif, dtype=rawdata.dtype)
    dcorr = np.array(rawdata, copy=True)
    dcorr[rawdata > threshold] = dcorr_all[rawdata > threshold]
    return dcorr


def get_centroid_fourier(data: np.ndarray):
    """Return image centroid using Fourier algorithm"""
    # Fourier transform method as discussed by Weisshaar et al.
    # (http://www.mnd-umwelttechnik.fh-wiesbaden.de/pig/weisshaar_u5.pdf)
    rows, cols = data.shape
    if rows == 1 or cols == 1:
        return 0, 0

    i = np.arange(0, rows).reshape(1, rows)
    sin_a = np.sin((i - 1) * 2 * np.pi / (rows - 1)).T
    cos_a = np.cos((i - 1) * 2 * np.pi / (rows - 1)).T

    j = np.arange(0, cols).reshape(cols, 1)
    sin_b = np.sin((j - 1) * 2 * np.pi / (cols - 1)).T
    cos_b = np.cos((j - 1) * 2 * np.pi / (cols - 1)).T

    a = (cos_a * data).sum()
    b = (sin_a * data).sum()
    c = (data * cos_b).sum()
    d = (data * sin_b).sum()

    rphi = (0 if b > 0 else 2 * np.pi) if a > 0 else np.pi
    cphi = (0 if d > 0 else 2 * np.pi) if c > 0 else np.pi

    if a * c == 0.0:
        return 0, 0

    row = (np.arctan(b / a) + rphi) * (rows - 1) / (2 * np.pi) + 1
    col = (np.arctan(d / c) + cphi) * (cols - 1) / (2 * np.pi) + 1
    try:
        row = int(row)
    except ma.MaskError:
        row = np.nan
    try:
        col = int(col)
    except ma.MaskError:
        col = np.nan
    return row, col


def get_enclosing_circle(data: np.ndarray, level: float = 0.5):
    """Return (x, y, radius) for the circle contour enclosing image
    values above threshold relative level (.5 means FWHM)

    Raise ValueError if no contour was found"""
    if not isinstance(level, float) or level < 0.0 or level > 1.0:
        raise ValueError("Level must be a float between 0. and 1.")
    data_th = data.copy()
    data_th[data <= data.max() * level] = 0.0
    contours = measure.find_contours(data_th)
    model = measure.CircleModel()
    result = None
    max_radius = 1.0
    for contour in contours:
        if model.estimate(contour):
            yc, xc, radius = model.params
            if radius > max_radius:
                result = (int(xc), int(yc), radius)
                max_radius = radius
    if result is None:
        raise ValueError("No contour was found")
    return result


def distance_matrix(coords: list) -> np.ndarray:
    """Return distance matrix from coords"""
    return np.triu(spt.distance.cdist(coords, coords, "euclidean"))


def get_2d_peaks_coords(
    data: np.ndarray, size: int = None, threshold: float = 0.5
) -> np.ndarray:
    """Detect peaks in image data, return coordinates.

    If neighborhoods size is None, default value is the highest value
    between 50 pixels and the 1/40th of the smallest image dimension.

    Detection threshold is relative to difference between data maximum and minimum.
    """
    if size is None:
        size = max(min(data.shape) // 40, 50)
    data_max = spf.maximum_filter(data, size)
    data_min = spf.minimum_filter(data, size)
    data_diff = data_max - data_min
    abs_threshold = (data_diff.max() - data_diff.min()) * threshold
    diff = (data_max - data_min) > abs_threshold
    maxima = data == data_max
    maxima[diff == 0] = 0
    labeled, _num_objects = spi.label(maxima)
    slices = spi.find_objects(labeled)
    coords = []
    for dy, dx in slices:
        x_center = int(0.5 * (dx.start + dx.stop - 1))
        y_center = int(0.5 * (dy.start + dy.stop - 1))
        coords.append((x_center, y_center))
    if len(coords) > 1:
        # Eventually removing duplicates
        dist = distance_matrix(coords)
        for index in reversed(np.unique(np.where((dist < size) & (dist > 0))[1])):
            coords.pop(index)
    return np.array(coords)


def get_contour_shapes(data: np.ndarray, shape: str = "ellipse") -> np.ndarray:
    """Find iso-valued contours in a 2D array, then fit contours
    with shape ('ellipse' or 'circle')
    Return NumPy array containing coordinates of shapes."""
    # pylint: disable=too-many-locals
    level = (float(np.nanmin(data)) + float(np.nanmax(data))) / 2.0
    contours = measure.find_contours(data, level=level)
    coords = []
    for contour in contours:
        if shape == "circle":
            model = measure.CircleModel()
            if model.estimate(contour):
                yc, xc, r = model.params
                if r <= 1.0:
                    continue
                coords.append([xc - r, yc, xc + r, yc])
        elif shape == "ellipse":
            model = measure.EllipseModel()
            if model.estimate(contour):
                yc, xc, b, a, theta = model.params
                if a <= 1.0 or b <= 1.0:
                    continue
                dxa, dya = a * np.cos(theta), a * np.sin(theta)
                dxb, dyb = b * np.sin(theta), b * np.cos(theta)
                x1, y1, x2, y2 = xc - dxa, yc - dya, xc + dxa, yc + dya
                x3, y3, x4, y4 = xc - dxb, yc - dyb, xc + dxb, yc + dyb
                coords.append([x1, y1, x2, y2, x3, y3, x4, y4])
        else:
            raise NotImplementedError(f"Invalid contour model {model}")
    return np.array(coords)
