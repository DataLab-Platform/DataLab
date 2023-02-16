# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT Computation / Image module
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import cv2
import numpy as np
import scipy.ndimage as spi
import scipy.ndimage.filters as spf
import scipy.spatial as spt
from numpy import ma
from skimage import exposure, feature, measure, transform


def scale_data_to_min_max(data: np.ndarray, zmin, zmax):
    """Scale array `data` to fit [zmin, zmax] dynamic range"""
    dmin = data.min()
    dmax = data.max()
    fdata = np.array(data, dtype=float)
    fdata -= dmin
    fdata *= float(zmax - zmin) / (dmax - dmin)
    fdata += float(zmin)
    return np.array(fdata, data.dtype)


BINNING_OPERATIONS = ("sum", "average", "median", "min", "max")


def binning(
    data: np.ndarray, binning_x: int, binning_y: int, operation: str, dtype=None
) -> np.ndarray:
    """Perform image pixel binning"""
    ny, nx = data.shape
    shape = (ny // binning_y, binning_y, nx // binning_x, binning_x)
    try:
        bdata = data[: ny - ny % binning_y, : nx - nx % binning_x].reshape(shape)
    except ValueError as err:
        raise ValueError("Binning is not a multiple of image dimensions") from err
    if operation == "sum":
        bdata = np.array(bdata, dtype=np.float64).sum(axis=(-1, 1))
    elif operation == "average":
        bdata = bdata.mean(axis=(-1, 1))
    elif operation == "median":
        bdata = ma.median(bdata, axis=(-1, 1))
    elif operation == "min":
        bdata = bdata.min(axis=(-1, 1))
    elif operation == "max":
        bdata = bdata.max(axis=(-1, 1))
    else:
        valid = ", ".join(BINNING_OPERATIONS)
        raise ValueError(f"Invalid operation {operation} (valid values: {valid})")
    return np.array(bdata, dtype=data.dtype if dtype is None else np.dtype(dtype))


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


def get_absolute_level(data: np.ndarray, level: float):
    """Return absolute level"""
    if not isinstance(level, float) or level < 0.0 or level > 1.0:
        raise ValueError("Level must be a float between 0. and 1.")
    return (float(np.nanmin(data)) + float(np.nanmax(data))) * level


def get_enclosing_circle(data: np.ndarray, level: float = 0.5):
    """Return (x, y, radius) for the circle contour enclosing image
    values above threshold relative level (.5 means FWHM)

    Raise ValueError if no contour was found"""
    data_th = data.copy()
    data_th[data <= get_absolute_level(data, level)] = 0.0
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
    data: np.ndarray, size: int = None, level: float = 0.5
) -> np.ndarray:
    """Detect peaks in image data, return coordinates.

    If neighborhoods size is None, default value is the highest value
    between 50 pixels and the 1/40th of the smallest image dimension.

    Detection threshold level is relative to difference
    between data maximum and minimum values.
    """
    if size is None:
        size = max(min(data.shape) // 40, 50)
    data_max = spf.maximum_filter(data, size)
    data_min = spf.minimum_filter(data, size)
    data_diff = data_max - data_min
    diff = (data_max - data_min) > get_absolute_level(data_diff, level)
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


def get_contour_shapes(
    data: np.ndarray, shape: str = "ellipse", level: float = 0.5
) -> np.ndarray:
    """Find iso-valued contours in a 2D array, above relative level (.5 means FWHM),
    then fit contours with shape ('ellipse' or 'circle')

    Return NumPy array containing coordinates of shapes."""
    # pylint: disable=too-many-locals
    contours = measure.find_contours(data, level=get_absolute_level(data, level))
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


def get_hough_circle_peaks(
    data: np.ndarray,
    min_radius: float = None,
    max_radius: float = None,
    nb_radius: int = None,
    min_distance: int = 1,
) -> np.ndarray:
    """Detect peaks in image from circle Hough transform, return circle coordinates."""
    assert min_radius is not None and max_radius is not None and max_radius > min_radius
    if nb_radius is None:
        nb_radius = max_radius - min_radius + 1
    hough_radii = np.arange(
        min_radius, max_radius + 1, (max_radius - min_radius + 1) // nb_radius
    )
    hough_res = transform.hough_circle(data, hough_radii)
    _accums, cx, cy, radii = transform.hough_circle_peaks(
        hough_res, hough_radii, min_xdistance=min_distance, min_ydistance=min_distance
    )
    return np.vstack([cx - radii, cy, cx + radii, cy]).T


def __blobs_to_coords(blobs: np.ndarray) -> np.ndarray:
    """Convert blobs to coordinates"""
    cy, cx, radii = blobs.T
    coords = np.vstack([cx - radii, cy, cx + radii, cy]).T
    return coords


def find_blobs_doh(
    data: np.ndarray,
    min_sigma: float = 1,
    max_sigma: float = 30,
    overlap: float = 0.5,
    log_scale: bool = False,
    threshold_rel: float = 0.2,
) -> np.ndarray:
    """
    Finds blobs in the given grayscale image using the Determinant of Hessian
    (DoH) method.

    Args:
        data: The grayscale input image.
        min_sigma: The minimum blob radius in pixels.
        max_sigma: The maximum blob radius in pixels.
        overlap: The minimum overlap ratio between blobs.
        log_scale: Whether to detect blobs on a log scale.
        threshold_rel: The threshold relative to the maximum intensity value.

    Returns:
        An array of blob coordinates and radii, with shape (N, 4).
    """
    # Use scikit-image's Determinant of Hessian (DoH) method to detect blobs
    blobs = feature.blob_doh(
        data,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=int(max_sigma - min_sigma + 1),
        threshold=None,
        threshold_rel=threshold_rel,
        overlap=overlap,
        log_scale=log_scale,
    )
    return __blobs_to_coords(blobs)


def remove_overlapping_disks(coords: np.ndarray) -> np.ndarray:
    """Remove overlapping disks among coordinates"""
    # Get the radii of each disk from the coordinates
    radii = coords[:, 2]
    # Calculate the distance between the center of each pair of disks
    dist = np.sqrt(np.sum((coords[:, None, :2] - coords[:, :2]) ** 2, axis=-1))
    # Create a boolean mask where the distance between the centers
    # is less than the sum of the radii
    mask = dist < (radii[:, None] + radii)
    # Find the indices of overlapping disks
    overlapping_indices = np.argwhere(mask)
    # Remove the smaller disk from each overlapping pair
    for i, j in overlapping_indices:
        if i != j:
            if radii[i] < radii[j]:
                coords[i] = [np.nan, np.nan, np.nan]
            else:
                coords[j] = [np.nan, np.nan, np.nan]
    # Remove rows with NaN values
    coords = coords[~np.isnan(coords).any(axis=1)]
    return coords


def find_blobs_opencv(
    data: np.ndarray,
    min_threshold: float = None,
    max_threshold: float = None,
    min_repeatability: int = None,
    min_dist_between_blobs: float = None,
    filter_by_color: bool = None,
    blob_color: int = None,
    filter_by_area: bool = None,
    min_area: float = None,
    max_area: float = None,
    filter_by_circularity: bool = None,
    min_circularity: float = None,
    max_circularity: float = None,
    filter_by_inertia: bool = None,
    min_inertia_ratio: float = None,
    max_inertia_ratio: float = None,
    filter_by_convexity: bool = None,
    min_convexity: float = None,
    max_convexity: float = None,
) -> np.ndarray:
    """
    Finds blobs in the given grayscale image using OpenCV's SimpleBlobDetector.

    Args:
        data: The grayscale input image.
        min_sigma: The minimum blob radius in pixels.
        max_sigma: The maximum blob radius in pixels.
        overlap: The minimum overlap ratio between blobs.
        log_scale: Whether to detect blobs on a log scale.
        threshold_rel: The threshold relative to the maximum intensity value.

    Returns:
        An array of blob coordinates and radii, with shape (N, 4).
    """
    params = cv2.SimpleBlobDetector_Params()
    if min_threshold is not None:
        params.minThreshold = min_threshold
    if max_threshold is not None:
        params.maxThreshold = max_threshold
    if min_repeatability is not None:
        params.minRepeatability = min_repeatability
    if min_dist_between_blobs is not None:
        params.minDistBetweenBlobs = min_dist_between_blobs
    if filter_by_color is not None:
        params.filterByColor = filter_by_color
    if blob_color is not None:
        params.blobColor = blob_color
    if filter_by_area is not None:
        params.filterByArea = filter_by_area
    if min_area is not None:
        params.minArea = min_area
    if max_area is not None:
        params.maxArea = max_area
    if filter_by_circularity is not None:
        params.filterByCircularity = filter_by_circularity
    if min_circularity is not None:
        params.minCircularity = min_circularity
    if max_circularity is not None:
        params.maxCircularity = max_circularity
    if filter_by_inertia is not None:
        params.filterByInertia = filter_by_inertia
    if min_inertia_ratio is not None:
        params.minInertiaRatio = min_inertia_ratio
    if max_inertia_ratio is not None:
        params.maxInertiaRatio = max_inertia_ratio
    if filter_by_convexity is not None:
        params.filterByConvexity = filter_by_convexity
    if min_convexity is not None:
        params.minConvexity = min_convexity
    if max_convexity is not None:
        params.maxConvexity = max_convexity
    detector = cv2.SimpleBlobDetector_create(params)
    image = exposure.rescale_intensity(data, out_range=np.uint8)
    keypoints = detector.detect(image)
    if keypoints:
        coords = cv2.KeyPoint_convert(keypoints)
        radii = 0.5 * np.array([kp.size for kp in keypoints])
        blobs = np.vstack([coords[:, 1], coords[:, 0], radii]).T
        blobs = remove_overlapping_disks(blobs)
    else:
        blobs = np.array([]).reshape((0, 3))
    return __blobs_to_coords(blobs)
