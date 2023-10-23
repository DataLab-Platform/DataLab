# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

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
import scipy.ndimage as spi
from numpy import ma
from plotpy.builder import make

from cdl.algorithms.image import get_centroid_fourier
from cdl.config import _
from cdl.env import execenv
from cdl.tests.data import get_laser_spot_data
from cdl.utils.qthelpers import qt_app_context
from cdl.utils.vistools import view_image_items


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
    label = "  " + f'{_("Centroid")}[{title}] (x=%s, y=%s)'
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
        ("Fourier", get_centroid_fourier),
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


def centroid_test():
    """Centroid test"""
    with qt_app_context():
        for data in get_laser_spot_data():
            execenv.print(f"Data[dtype={data.dtype},shape={data.shape}]")
            # Testing with masked arrays
            compare_centroid_funcs(data.view(ma.MaskedArray))


if __name__ == "__main__":
    centroid_test()
