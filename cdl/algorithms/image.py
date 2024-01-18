# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
.. Image Processing Algorithms (see parent package :mod:`cdl.algorithms`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import cv2
import numpy as np
import scipy.ndimage as spi
import scipy.ndimage.filters as spf
import scipy.spatial as spt
from numpy import ma
from skimage import exposure, feature, measure, transform


def scale_data_to_min_max(
    data: np.ndarray, zmin: float | int, zmax: float | int
) -> np.ndarray:
    """Scale array `data` to fit [zmin, zmax] dynamic range

    Args:
        data (numpy.ndarray): Input data
        zmin (float|int): Minimum value of output data
        zmax (float|int): Maximum value of output data

    Returns:
        np.ndarray: Scaled data
    """
    dmin = data.min()
    dmax = data.max()
    fdata = np.array(data, dtype=float)
    fdata -= dmin
    fdata *= float(zmax - zmin) / (dmax - dmin)
    fdata += float(zmin)
    return np.array(fdata, data.dtype)


def z_fft(z: np.ndarray, shift: bool = True) -> np.ndarray:
    """Compute FFT of complex array `z`

    Args:
        z (numpy.ndarray): Input data
        shift (bool | None): Shift zero frequency to center (default: True)

    Returns:
        np.ndarray: FFT of input data
    """
    z1 = np.fft.fft2(z)
    if shift:
        z1 = np.fft.fftshift(z1)
    return z1


def z_ifft(z: np.ndarray, shift: bool = True) -> np.ndarray:
    """Compute inverse FFT of complex array `z`

    Args:
        z (numpy.ndarray): Input data
        shift (bool | None): Shift zero frequency to center (default: True)

    Returns:
        np.ndarray: Inverse FFT of input data
    """
    if shift:
        z = np.fft.ifftshift(z)
    z1 = np.fft.ifft2(z)
    return z1


BINNING_OPERATIONS = ("sum", "average", "median", "min", "max")


def binning(
    data: np.ndarray, binning_x: int, binning_y: int, operation: str, dtype=None
) -> np.ndarray:
    """Perform image pixel binning

    Args:
        data (numpy.ndarray): Input data
        binning_x (int): Binning factor along x-axis
        binning_y (int): Binning factor along y-axis
        operation (str): Binning operation (sum, average, median, min, max)
        dtype (str | None): Output data type (default: None, i.e. same as input)

    Returns:
        np.ndarray: Binned data
    """
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


def flatfield(
    rawdata: np.ndarray, flatdata: np.ndarray, threshold: float | None = None
) -> np.ndarray:
    """Compute flat-field correction

    Args:
        rawdata (numpy.ndarray): Raw data
        flatdata (numpy.ndarray): Flat-field data
        threshold (float | None): Threshold for flat-field correction (default: None)

    Returns:
        np.ndarray: Flat-field corrected data
    """
    dtemp = np.array(rawdata, dtype=np.float64, copy=True) * flatdata.mean()
    dunif = np.array(flatdata, dtype=np.float64, copy=True)
    dunif[dunif == 0] = 1.0
    dcorr_all = np.array(dtemp / dunif, dtype=rawdata.dtype)
    dcorr = np.array(rawdata, copy=True)
    dcorr[rawdata > threshold] = dcorr_all[rawdata > threshold]
    return dcorr


def get_centroid_fourier(data: np.ndarray) -> tuple[float, float]:
    """Return image centroid using Fourier algorithm

    Args:
        data (numpy.ndarray): Input data

    Returns:
        tuple[float, float]: Centroid coordinates
    """
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


def get_absolute_level(data: np.ndarray, level: float) -> float:
    """Return absolute level

    Args:
        data (numpy.ndarray): Input data
        level (float): Relative level (0.0 to 1.0)

    Returns:
        float: Absolute level
    """
    if not isinstance(level, float) or level < 0.0 or level > 1.0:
        raise ValueError("Level must be a float between 0. and 1.")
    return (float(np.nanmin(data)) + float(np.nanmax(data))) * level


def get_enclosing_circle(
    data: np.ndarray, level: float = 0.5
) -> tuple[int, int, float]:
    """Return (x, y, radius) for the circle contour enclosing image
    values above threshold relative level (.5 means FWHM)

    Args:
        data (numpy.ndarray): Input data
        level (float | None): Relative level (default: 0.5)

    Returns:
        tuple[int, int, float]: (x, y, radius)

    Raises:
        ValueError: No contour was found
    """
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
    """Return distance matrix from coords

    Args:
        coords (list): List of coordinates

    Returns:
        np.ndarray: Distance matrix
    """
    return np.triu(spt.distance.cdist(coords, coords, "euclidean"))


def get_2d_peaks_coords(
    data: np.ndarray, size: int | None = None, level: float = 0.5
) -> np.ndarray:
    """Detect peaks in image data, return coordinates.

    If neighborhoods size is None, default value is the highest value
    between 50 pixels and the 1/40th of the smallest image dimension.

    Detection threshold level is relative to difference
    between data maximum and minimum values.

    Args:
        data (numpy.ndarray): Input data
        size (int | None): Neighborhood size (default: None)
        level (float | None): Relative level (default: 0.5)

    Returns:
        np.ndarray: Coordinates of peaks
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


def is_circle_outside_mask(array: np.ma.MaskedArray, xc: int, yc: int, r: int) -> bool:
    """
    Check if a circular disk is outside the masked areas of a NumPy masked array.

    Args:
        array: The masked array.
        xc: The x-coordinate of the disk's center.
        yc: The y-coordinate of the disk's center.
        r: The radius of the disk.

    Returns:
        True if the disk is outside the masked areas, False otherwise.
    """
    y, x = np.ogrid[: array.shape[0], : array.shape[1]]
    mask = (y - yc) ** 2 + (x - xc) ** 2 <= r**2
    return array.mask[mask].all()


def is_ellipse_outside_mask(
    array: np.ma.MaskedArray, xc: int, yc: int, a: int, b: int, theta: float
) -> bool:
    """
    Check if an elliptical disk is outside the masked areas of a NumPy masked array.

    Args:
        array: The masked array.
        xc: The x-coordinate of the ellipse's center.
        yc: The y-coordinate of the ellipse's center.
        a: The full length of the major axis of the ellipse.
        b: The full length of the minor axis of the ellipse.
        theta: The rotation angle of the ellipse in radians.

    Returns:
        True if the ellipse is outside the masked areas, False otherwise.
    """
    y, x = np.ogrid[: array.shape[0], : array.shape[1]]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Coordinate transformation
    x_prime = cos_theta * (x - xc) + sin_theta * (y - yc)
    y_prime = -sin_theta * (x - xc) + cos_theta * (y - yc)

    # Adjust for semi-major and semi-minor axes
    a /= 2
    b /= 2

    # Ellipse equation
    mask = (x_prime**2 / a**2) + (y_prime**2 / b**2) <= 1

    # Check if any point within the ellipse is masked
    return array.mask[mask].all()


def get_contour_shapes(
    data: np.ndarray, shape: str = "ellipse", level: float = 0.5
) -> np.ndarray:
    """Find iso-valued contours in a 2D array, above relative level (.5 means FWHM),
    then fit contours with shape ('ellipse' or 'circle')

    Args:
        data: Input data
        shape: Shape to fit. Valid values: 'circle', 'ellipse', 'polygon'.
         (default: 'ellipse')
        level: Relative level (default: 0.5)

    Returns:
        Coordinates of shapes
    """
    # pylint: disable=too-many-locals
    assert shape in ("circle", "ellipse", "polygon")
    contours = measure.find_contours(data, level=get_absolute_level(data, level))
    coords = []
    for contour in contours:
        if shape == "circle":
            model = measure.CircleModel()
            if model.estimate(contour):
                yc, xc, r = model.params
                if r <= 1.0 or is_circle_outside_mask(data, xc, yc, r):
                    continue
                coords.append([xc - r, yc, xc + r, yc])
        elif shape == "ellipse":
            model = measure.EllipseModel()
            if model.estimate(contour):
                yc, xc, b, a, theta = model.params
                if (
                    a <= 1.0
                    or b <= 1.0
                    or is_ellipse_outside_mask(data, xc, yc, a, b, theta)
                ):
                    continue
                dxa, dya = a * np.cos(theta), a * np.sin(theta)
                dxb, dyb = b * np.sin(theta), b * np.cos(theta)
                x1, y1, x2, y2 = xc - dxa, yc - dya, xc + dxa, yc + dya
                x3, y3, x4, y4 = xc - dxb, yc - dyb, xc + dxb, yc + dyb
                coords.append([x1, y1, x2, y2, x3, y3, x4, y4])
        elif shape == "polygon":
            # `contour` is a (N, 2) array (rows, cols): we need to convert it
            # to a list of x, y coordinates flattened in a single list
            coords.append(contour[:, ::-1].flatten())
        else:
            raise NotImplementedError(f"Invalid contour model {model}")
    if shape == "polygon":
        # `coords` is a list of arrays of shape (N, 2) where N is the number of points
        # that can vary from one array to another, so we need to padd with NaNs each
        # array to get a regular array:
        max_len = max(coord.shape[0] for coord in coords)
        arr = np.full((len(coords), max_len), np.nan)
        for i_row, coord in enumerate(coords):
            arr[i_row, : coord.shape[0]] = coord
        return arr
    return np.array(coords)


def get_hough_circle_peaks(
    data: np.ndarray,
    min_radius: float | None = None,
    max_radius: float | None = None,
    nb_radius: int | None = None,
    min_distance: int = 1,
) -> np.ndarray:
    """Detect peaks in image from circle Hough transform, return circle coordinates.

    Args:
        data (numpy.ndarray): Input data
        min_radius (float | None): Minimum radius (default: None)
        max_radius (float | None): Maximum radius (default: None)
        nb_radius (int | None): Number of radii (default: None)
        min_distance (int | None): Minimum distance between circles (default: 1)

    Returns:
        np.ndarray: Coordinates of circles
    """
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
    """Convert blobs to coordinates

    Args:
        blobs (numpy.ndarray): Blobs

    Returns:
        np.ndarray: Coordinates
    """
    cy, cx, radii = blobs.T
    coords = np.vstack([cx - radii, cy, cx + radii, cy]).T
    return coords


def find_blobs_dog(
    data: np.ndarray,
    min_sigma: float = 1,
    max_sigma: float = 30,
    overlap: float = 0.5,
    threshold_rel: float = 0.2,
    exclude_border: bool = True,
) -> np.ndarray:
    """
    Finds blobs in the given grayscale image using the Difference of Gaussians
    (DoG) method.

    Args:
        data (numpy.ndarray): The grayscale input image.
        min_sigma (float | None): The minimum blob radius in pixels.
        max_sigma (float | None): The maximum blob radius in pixels.
        overlap (float | None): The minimum overlap between two blobs in
            pixels. For instance, if two blobs are detected with radii of 10 and
            12 respectively, and the ``overlap`` is set to 0.5, then the area of
            the smaller blob will be ignored and only the area of the larger blob
            will be returned.
        threshold_rel (float | None): The absolute lower bound for scale space
            maxima. Local maxima smaller than ``threshold_rel`` are ignored. Reduce
            this to detect blobs with less intensities.
        exclude_border (bool | None): If ``True``, exclude blobs from
            detection if they are too close to the border of the image. Border
            size is ``min_sigma``.

    Returns:
        np.ndarray: Coordinates of blobs
    """
    # Use scikit-image's Difference of Gaussians (DoG) method
    blobs = feature.blob_dog(
        data,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        overlap=overlap,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
    )
    return __blobs_to_coords(blobs)


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
        data (numpy.ndarray): The grayscale input image.
        min_sigma (float | None): The minimum blob radius in pixels.
        max_sigma (float | None): The maximum blob radius in pixels.
        overlap (float | None): The minimum overlap between two blobs in
            pixels. For instance, if two blobs are detected with radii of 10 and
            12 respectively, and the ``overlap`` is set to 0.5, then the area of
            the smaller blob will be ignored and only the area of the larger blob
            will be returned.
        log_scale (bool | None): If ``True``, the radius of each blob is
            returned as ``sqrt(sigma)`` for each detected blob.
        threshold_rel (float | None): The absolute lower bound for scale space
            maxima. Local maxima smaller than ``threshold_rel`` are ignored. Reduce
            this to detect blobs with less intensities.

    Returns:
        np.ndarray: Coordinates of blobs
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


def find_blobs_log(
    data: np.ndarray,
    min_sigma: float = 1,
    max_sigma: float = 30,
    overlap: float = 0.5,
    log_scale: bool = False,
    threshold_rel: float = 0.2,
    exclude_border: bool = True,
) -> np.ndarray:
    """Finds blobs in the given grayscale image using the Laplacian of Gaussian
    (LoG) method.

    Args:
        data (numpy.ndarray): The grayscale input image.
        min_sigma (float | None): The minimum blob radius in pixels.
        max_sigma (float | None): The maximum blob radius in pixels.
        overlap (float | None): The minimum overlap between two blobs in
            pixels. For instance, if two blobs are detected with radii of 10 and
            12 respectively, and the ``overlap`` is set to 0.5, then the area of
            the smaller blob will be ignored and only the area of the larger blob
            will be returned.
        log_scale (bool | None): If ``True``, the radius of each blob is
            returned as ``sqrt(sigma)`` for each detected blob.
        threshold_rel (float | None): The absolute lower bound for scale space
            maxima. Local maxima smaller than ``threshold_rel`` are ignored. Reduce
            this to detect blobs with less intensities.
        exclude_border (bool | None): If ``True``, exclude blobs from
            detection if they are too close to the border of the image. Border
            size is ``min_sigma``.

    Returns:
        np.ndarray: Coordinates of blobs
    """
    # Use scikit-image's Laplacian of Gaussian (LoG) method to detect blobs
    blobs = feature.blob_log(
        data,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=int(max_sigma - min_sigma + 1),
        threshold=None,
        threshold_rel=threshold_rel,
        overlap=overlap,
        log_scale=log_scale,
        exclude_border=exclude_border,
    )
    return __blobs_to_coords(blobs)


def remove_overlapping_disks(coords: np.ndarray) -> np.ndarray:
    """Remove overlapping disks among coordinates

    Args:
        coords (numpy.ndarray): The coordinates of the disks

    Returns:
        np.ndarray: The coordinates of the disks with overlapping disks removed
    """
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
    min_threshold: float | None = None,
    max_threshold: float | None = None,
    min_repeatability: int | None = None,
    min_dist_between_blobs: float | None = None,
    filter_by_color: bool | None = None,
    blob_color: int | None = None,
    filter_by_area: bool | None = None,
    min_area: float | None = None,
    max_area: float | None = None,
    filter_by_circularity: bool | None = None,
    min_circularity: float | None = None,
    max_circularity: float | None = None,
    filter_by_inertia: bool | None = None,
    min_inertia_ratio: float | None = None,
    max_inertia_ratio: float | None = None,
    filter_by_convexity: bool | None = None,
    min_convexity: float | None = None,
    max_convexity: float | None = None,
) -> np.ndarray:
    """
    Finds blobs in the given grayscale image using OpenCV's SimpleBlobDetector.

    Args:
        data (numpy.ndarray): The grayscale input image.
        min_threshold (float | None): The minimum blob intensity.
        max_threshold (float | None): The maximum blob intensity.
        min_repeatability (int | None): The minimum number of times a blob is
            detected before it is reported.
        min_dist_between_blobs (float | None): The minimum distance between
            blobs.
        filter_by_color (bool | None): If ``True``, blobs are filtered by color.
        blob_color (int | None): The color of the blobs to filter by.
        filter_by_area (bool | None): If ``True``, blobs are filtered by area.
        min_area (float | None): The minimum blob area.
        max_area (float | None): The maximum blob area.
        filter_by_circularity (bool | None): If ``True``, blobs are filtered by
            circularity.
        min_circularity (float | None): The minimum blob circularity.
        max_circularity (float | None): The maximum blob circularity.
        filter_by_inertia (bool | None): If ``True``, blobs are filtered by
            inertia.
        min_inertia_ratio (float | None): The minimum blob inertia ratio.
        max_inertia_ratio (float | None): The maximum blob inertia ratio.
        filter_by_convexity (bool | None): If ``True``, blobs are filtered by
            convexity.
        min_convexity (float | None): The minimum blob convexity.
        max_convexity (float | None): The maximum blob convexity.

    Returns:
        np.ndarray: Coordinates of blobs
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
