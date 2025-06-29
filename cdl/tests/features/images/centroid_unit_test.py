# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image centroid computation test

Comparing different algorithms for centroid calculation:

- SciPy (measurements.center_of_mass)
- OpenCV (moments)
- Method based on moments
- Method based on Fourier (DataLab)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

import time

import numpy as np
import pytest
import scipy.ndimage as spi
from guidata.qthelpers import qt_app_context
from numpy import ma
from plotpy.builder import make

import sigima_.algorithms.image as alg
import sigima_.computation.image as sigima_image
import sigima_.obj
from cdl.config import _
from cdl.env import execenv
from sigima_.tests.data import (
    create_noisygauss_image,
    get_laser_spot_data,
)
from sigima_.tests.helpers import check_scalar_result
from sigima_.tests.vistools import view_image_items


def get_centroid_from_moments(data):
    """Computing centroid from image moments"""
    y, x = np.ogrid[: data.shape[0], : data.shape[1]]
    imx, imy = data.sum(axis=0)[None, :], data.sum(axis=1)[:, None]
    m00 = np.array(data, dtype=float).sum() or 1.0
    m10 = (np.array(imx, dtype=float) * x).sum() / m00
    m01 = (np.array(imy, dtype=float) * y).sum() / m00
    return int(m01), int(m10)


def get_centroid_with_cv2(data):
    """Compute centroid from moments with OpenCV"""
    import cv2  # pylint: disable=import-outside-toplevel

    m = cv2.moments(data)
    col = int(m["m10"] / m["m00"])
    row = int(m["m01"] / m["m00"])
    return row, col


def add_xcursor(items, x, y, title):
    """Added X cursor to plot"""
    label = "  " + f"{_('Centroid')}[{title}] (x=%s, y=%s)"
    execenv.print(label % (x, y))
    cursor = make.xcursor(x, y, label=label)
    cursor.setTitle(title)
    items.append(cursor)


def compare_centroid_funcs(data):
    """Compare centroid methods"""
    items = []
    items += [make.image(data, interpolation="nearest", eliminate_outliers=2.0)]
    # Computing centroid coordinates
    for name, func in (
        ("SciPy", spi.center_of_mass),
        ("OpenCV", get_centroid_with_cv2),
        ("Moments", get_centroid_from_moments),
        ("Fourier", alg.get_centroid_fourier),
    ):
        try:
            t0 = time.time()
            row, col = func(data)
            dt = time.time() - t0
            add_xcursor(items, col, row, name)
            execenv.print(f"    Calculation time: {int(dt * 1e3):d} ms")
        except ImportError:
            execenv.print(f"    Unable to compute {name}: missing module")
    view_image_items(items)


def test_centroid_graphically():
    """Centroid test comparing different methods and showing results"""
    with qt_app_context():
        for data in get_laser_spot_data():
            execenv.print(f"Data[dtype={data.dtype},shape={data.shape}]")
            # Testing with masked arrays
            compare_centroid_funcs(data.view(ma.MaskedArray))


def __check_centroid(image, expected_x, expected_y):
    """Check centroid computation"""
    df = sigima_image.centroid(image).to_dataframe()
    check_scalar_result("Centroid X", df.x[0], expected_x, atol=1.0)
    check_scalar_result("Centroid Y", df.y[0], expected_y, atol=1.0)


@pytest.mark.validation
def test_image_centroid():
    """Test centroid computation"""
    param = sigima_.obj.NewImageParam.create(height=500, width=500)
    image = create_noisygauss_image(param, center=(-2.0, 3.0), add_annotations=True)
    circle_roi = sigima_.obj.create_image_roi("circle", [200, 325, 10])
    for roi, x0, y0 in (
        (None, 0, 0),
        (None, 100, 100),
        (circle_roi, 0, 0),
        (circle_roi, 100, 100),  # Test for regression like #106
    ):
        image.roi, image.x0, image.y0 = roi, x0, y0
        __check_centroid(image, 200.0 + x0, 325.0 + y0)


if __name__ == "__main__":
    test_centroid_graphically()
    test_image_centroid()
